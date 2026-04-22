"""
features/maze_geometry.py

阶段 2 — 迷宫游戏的【通道几何 + 解路径】模块。

输入：maze_mask.png（前景=墙壁线条，背景=黑）。
输出：MazeGeometry dataclass，含
    - canvas_mask         : 外框内部区域（前景=可绘制画布）
    - wall_mask           : 原墙壁掩码（与 canvas_mask 同尺寸）
    - channel_mask        : 可行走通道（前景=通道内部，墙壁与画布外均为 0）
    - channel_skeleton    : 通道骨架（1 px 宽）
    - channel_dist        : 距离变换 = 每个非通道像素到最近通道的 L2 距离
    - entry_xy, exit_xy   : 入口 / 出口锚点 (x, y)（自动检测 + 简化版兼容）
    - solution_polyline   : 沿解路径的有序 (x, y) 数组（在骨架像素上）
    - solution_skeleton_mask : 解路径骨架（1 px 宽）
    - solution_channel_mask  : 解路径通道（解骨架膨胀至通道半宽）

设计要点：
1. floodFill 从画布外（左上角 (0,0)）灌入背景，取反 → canvas_mask；
2. wall_mask 略膨胀 r_wall 像素再从 canvas_mask 中扣除，避免墙壁微缝导致通道串连；
3. 入口 / 出口：
    a. 简化版（默认）—— 文档 4.3.2 给出的"内框右上 / 左下 105×105 矩形"定位；
    b. 通用版（备用）—— 找外框 ring ∩ channel_mask 的连通域。
4. 最短路径：把 channel_skeleton 的所有前景像素当节点，8 邻接，用堆 Dijkstra
   求 entry_anchor → exit_anchor 的最短路径。骨架像素 ≲ 数千，瞬间出结果。

调用方一般只需 `build_maze_geometry(mask_path)` 即可；可视化用 `visualize_maze_geometry`。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
except ImportError as e:
    raise ImportError(
        "需要 scikit-image 提供 skeletonize：pip install scikit-image"
    ) from e


# =====================================================================
# 数据类型
# =====================================================================

Point = Tuple[int, int]  # (x, y)


@dataclass
class MazeGeometry:
    """打包所有从 maze_mask.png 派生的几何结构。"""

    # 路径来源
    mask_path: str

    # 基础掩码（uint8，0/1）
    wall_mask: np.ndarray
    canvas_mask: np.ndarray
    channel_mask: np.ndarray

    # 通道骨架与距离变换
    channel_skeleton: np.ndarray  # uint8, 0/1
    channel_dist: np.ndarray  # float32, 非通道像素到最近通道像素的 L2 距离

    # 入口 / 出口锚点（画布坐标，xy）
    entry_xy: Point
    exit_xy: Point

    # 解路径
    solution_polyline: np.ndarray  # (N, 2) int，沿路径的有序 (x, y)
    solution_skeleton_mask: np.ndarray  # uint8, 0/1
    solution_channel_mask: np.ndarray  # uint8, 0/1

    # 元信息
    canvas_hw: Tuple[int, int] = field(default=(0, 0))  # (h, w)
    inner_bbox: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0))  # x1,y1,x2,y2
    channel_half_width_px: int = 28
    r_wall: int = 2
    solution_length_px: int = 0


# =====================================================================
# 基础几何构造
# =====================================================================

def _binarize_mask(path: str, thresh: int = 127) -> np.ndarray:
    """读取灰度 PNG 并按 >thresh 二值化为 0/1 uint8。"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取掩码: {path}")
    return (img > thresh).astype(np.uint8)


def _floodfill_canvas(wall_mask: np.ndarray) -> np.ndarray:
    """
    从图像四角的背景像素开始 floodFill，将"画布外"标记掉；
    取反即得到外框 + 内部 = canvas_mask。

    实现要点：
    - cv2.floodFill 需要比图大 2 像素的 padded mask，但我们只用其
      改写后的源图像即可。
    - 假设 (0, 0) 是黑色（画布外）。
    """
    h, w = wall_mask.shape[:2]
    # filled 中：墙=255，背景=0
    filled = (wall_mask > 0).astype(np.uint8) * 255
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    # 从四角灌入背景。floodFill 把可达背景改写成 128。
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if filled[seed[1], seed[0]] == 0:
            cv2.floodFill(filled, ff_mask, seed, 128)
    # canvas_mask = NOT outside  ⇒  既包含外框墙也包含内部
    canvas_mask = (filled != 128).astype(np.uint8)
    return canvas_mask


def _build_channel_mask(
    canvas_mask: np.ndarray,
    wall_mask: np.ndarray,
    r_wall: int = 2,
) -> np.ndarray:
    """channel_mask = canvas_mask AND NOT dilate(wall_mask, r_wall)。"""
    if r_wall <= 0:
        wall_dilated = wall_mask
    else:
        k_size = 2 * r_wall + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        wall_dilated = cv2.dilate(wall_mask * 255, kernel)
        wall_dilated = (wall_dilated > 0).astype(np.uint8)
    channel = ((canvas_mask > 0) & (wall_dilated == 0)).astype(np.uint8)
    return channel


def _channel_distance_transform(channel_mask: np.ndarray) -> np.ndarray:
    """每个非通道像素到最近通道像素的 L2 距离（float32）。"""
    # cv2.distanceTransform 输入：前景=非 0 的像素到 0 像素的距离。
    # 我们想得到"非通道→最近通道"距离，所以把 channel_mask 取反作前景。
    inv = (channel_mask == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return dist.astype(np.float32)


# =====================================================================
# 入口 / 出口检测
# =====================================================================

def _inner_bbox_from_walls(wall_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """墙壁前景的外接矩形 (x1, y1, x2, y2)，闭区间。"""
    ys, xs = np.where(wall_mask > 0)
    if len(xs) == 0:
        h, w = wall_mask.shape[:2]
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _detect_entry_exit_simple(
    channel_mask: np.ndarray,
    inner_bbox: Tuple[int, int, int, int],
    corner_size: int = 105,
) -> Tuple[Point, Point]:
    """
    简化版（针对 maze_mask.png）：
      入口 = 内框右上 corner_size × corner_size 矩形内通道像素的质心
      出口 = 内框左下 corner_size × corner_size 矩形内通道像素的质心
    """
    x1, y1, x2, y2 = inner_bbox

    # 右上区域：x ∈ [x2 - corner_size, x2], y ∈ [y1, y1 + corner_size]
    rt_x1 = max(x1, x2 - corner_size)
    rt_y1 = y1
    rt_x2 = x2
    rt_y2 = min(y2, y1 + corner_size)

    # 左下区域：x ∈ [x1, x1 + corner_size], y ∈ [y2 - corner_size, y2]
    lb_x1 = x1
    lb_y1 = max(y1, y2 - corner_size)
    lb_x2 = min(x2, x1 + corner_size)
    lb_y2 = y2

    def _centroid(box):
        bx1, by1, bx2, by2 = box
        sub = channel_mask[by1 : by2 + 1, bx1 : bx2 + 1]
        ys, xs = np.where(sub > 0)
        if len(xs) == 0:
            # 没找到通道像素时退化为子矩形中心
            return (bx1 + bx2) // 2, (by1 + by2) // 2
        return int(xs.mean() + bx1), int(ys.mean() + by1)

    entry = _centroid((rt_x1, rt_y1, rt_x2, rt_y2))
    exit_ = _centroid((lb_x1, lb_y1, lb_x2, lb_y2))
    return entry, exit_


def _detect_entry_exit_frame_ring(
    channel_mask: np.ndarray,
    inner_bbox: Tuple[int, int, int, int],
    ring_width: int = 3,
) -> Optional[Tuple[Point, Point]]:
    """
    通用版（备用）：
      取内框 1 圈 ring（宽 ring_width 像素），与 channel_mask 取交，
      连通域分析。若恰好 2 个连通域，则返回 (右上质心, 左下质心)；
      否则返回 None 让上层 fallback。
    """
    x1, y1, x2, y2 = inner_bbox
    ring = np.zeros_like(channel_mask)
    rw = ring_width
    # 上下边
    ring[y1 : y1 + rw + 1, x1 : x2 + 1] = 1
    ring[y2 - rw : y2 + 1, x1 : x2 + 1] = 1
    # 左右边
    ring[y1 : y2 + 1, x1 : x1 + rw + 1] = 1
    ring[y1 : y2 + 1, x2 - rw : x2 + 1] = 1
    openings = (ring > 0) & (channel_mask > 0)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        openings.astype(np.uint8), connectivity=8
    )
    # n 包含背景 0，故有效连通域数 = n - 1
    if n - 1 != 2:
        return None
    # 按位置：右上 = y 小且 x 大；左下 = y 大且 x 小
    cents = [(int(centroids[i, 0]), int(centroids[i, 1])) for i in range(1, n)]
    cents_sorted = sorted(cents, key=lambda p: (p[1] - p[0]))  # y - x 越小越右上
    entry = cents_sorted[0]
    exit_ = cents_sorted[-1]
    return entry, exit_


# =====================================================================
# 骨架最短路径
# =====================================================================

def _nearest_skeleton_pixel(
    skeleton: np.ndarray, anchor_xy: Point
) -> Tuple[int, int]:
    """在骨架中找到距 anchor_xy 最近的前景像素，返回 (x, y)。"""
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        raise RuntimeError("通道骨架为空，无法定位锚点。")
    ax, ay = anchor_xy
    d2 = (xs - ax) ** 2 + (ys - ay) ** 2
    idx = int(np.argmin(d2))
    return int(xs[idx]), int(ys[idx])


# 8 邻接偏移：(dx, dy, weight)
_NEIGHBORS_8 = [
    (-1, -1, 1.41421356),
    (0, -1, 1.0),
    (1, -1, 1.41421356),
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (-1, 1, 1.41421356),
    (0, 1, 1.0),
    (1, 1, 1.41421356),
]


def _dijkstra_on_skeleton(
    skeleton: np.ndarray,
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    在 1 px 宽骨架像素上跑 Dijkstra（8 邻接，权重=欧氏距离）。
    返回从 start_xy 到 goal_xy 的最短路径像素列表 [(x, y), ...]。
    """
    h, w = skeleton.shape[:2]
    skel = (skeleton > 0)
    sx, sy = start_xy
    gx, gy = goal_xy
    if not skel[sy, sx] or not skel[gy, gx]:
        raise RuntimeError(
            f"start/goal 不在骨架上：start={start_xy}, goal={goal_xy}"
        )

    INF = float("inf")
    dist = np.full((h, w), INF, dtype=np.float64)
    parent = np.full((h, w, 2), -1, dtype=np.int32)
    dist[sy, sx] = 0.0

    pq: List[Tuple[float, int, int]] = [(0.0, sx, sy)]
    while pq:
        d, x, y = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        if (x, y) == (gx, gy):
            break
        for dx, dy, wgt in _NEIGHBORS_8:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if not skel[ny, nx]:
                continue
            nd = d + wgt
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                parent[ny, nx, 0] = x
                parent[ny, nx, 1] = y
                heapq.heappush(pq, (nd, nx, ny))

    if dist[gy, gx] == INF:
        raise RuntimeError(
            f"骨架上 start→goal 不连通，请检查 channel_mask / r_wall 设置。"
        )

    # 回溯
    path: List[Tuple[int, int]] = []
    cx, cy = gx, gy
    while not (cx == sx and cy == sy):
        path.append((cx, cy))
        px, py = int(parent[cy, cx, 0]), int(parent[cy, cx, 1])
        if px < 0:
            break
        cx, cy = px, py
    path.append((sx, sy))
    path.reverse()
    return path


# =====================================================================
# 主入口
# =====================================================================

def build_maze_geometry(
    mask_path: str,
    *,
    r_wall: int = 2,
    channel_half_width_px: int = 28,
    entry_corner_size: int = 105,
    entry_xy: Optional[Point] = None,
    exit_xy: Optional[Point] = None,
    use_frame_ring_first: bool = False,
) -> MazeGeometry:
    """
    构造迷宫几何对象。

    参数：
        mask_path             : maze_mask.png 路径（前景=墙壁）。
        r_wall                : 墙壁膨胀半径（消除细缝），默认 2 px。
        channel_half_width_px : solution_skeleton 膨胀半径（≈ 实测通道半宽 28 px）。
                                kernel 直径 = 2 * r + 1 ⇒ 通道宽度约 57 px，与实测一致。
        entry_corner_size     : 简化版入口/出口检测用的方框大小，默认 105 px。
        entry_xy / exit_xy    : 手动指定锚点（覆盖自动检测），仅在自动失败时使用。
        use_frame_ring_first  : True 时优先尝试通用 frame_ring 检测；
                                默认 False（直接用简化版，更稳定）。
    """
    # ---------- 1. 读墙壁 + canvas + channel ----------
    wall_mask = _binarize_mask(mask_path)
    canvas_mask = _floodfill_canvas(wall_mask)
    channel_mask = _build_channel_mask(canvas_mask, wall_mask, r_wall=r_wall)

    h, w = wall_mask.shape[:2]
    inner_bbox = _inner_bbox_from_walls(wall_mask)

    # ---------- 2. 通道骨架 + 距离变换 ----------
    channel_skeleton = skeletonize(channel_mask > 0).astype(np.uint8)
    channel_dist = _channel_distance_transform(channel_mask)

    # ---------- 3. 入口 / 出口 ----------
    if entry_xy is not None and exit_xy is not None:
        entry, exit_ = entry_xy, exit_xy
    else:
        entry = exit_ = None
        if use_frame_ring_first:
            ring = _detect_entry_exit_frame_ring(channel_mask, inner_bbox)
            if ring is not None:
                entry, exit_ = ring
        if entry is None:
            entry, exit_ = _detect_entry_exit_simple(
                channel_mask, inner_bbox, corner_size=entry_corner_size
            )

    # ---------- 4. 解路径（在骨架上跑 dijkstra） ----------
    entry_anchor = _nearest_skeleton_pixel(channel_skeleton, entry)
    exit_anchor = _nearest_skeleton_pixel(channel_skeleton, exit_)
    path_pixels = _dijkstra_on_skeleton(
        channel_skeleton, entry_anchor, exit_anchor
    )
    solution_polyline = np.array(path_pixels, dtype=np.int32)  # (N, 2)

    solution_skeleton_mask = np.zeros((h, w), dtype=np.uint8)
    solution_skeleton_mask[
        solution_polyline[:, 1], solution_polyline[:, 0]
    ] = 1

    # ---------- 5. 解路径通道（膨胀至通道宽） ----------
    r = max(1, int(channel_half_width_px))
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    solution_channel_mask = cv2.dilate(solution_skeleton_mask * 255, kernel)
    solution_channel_mask = (solution_channel_mask > 0).astype(np.uint8)
    # 限制在 channel_mask 内部，避免把墙壁也算进通道
    solution_channel_mask = (solution_channel_mask & channel_mask).astype(np.uint8)

    return MazeGeometry(
        mask_path=str(mask_path),
        wall_mask=wall_mask,
        canvas_mask=canvas_mask,
        channel_mask=channel_mask,
        channel_skeleton=channel_skeleton,
        channel_dist=channel_dist,
        entry_xy=entry,
        exit_xy=exit_,
        solution_polyline=solution_polyline,
        solution_skeleton_mask=solution_skeleton_mask,
        solution_channel_mask=solution_channel_mask,
        canvas_hw=(h, w),
        inner_bbox=inner_bbox,
        channel_half_width_px=r,
        r_wall=int(r_wall),
        solution_length_px=int(len(path_pixels)),
    )


# =====================================================================
# 可视化（目视验证用）
# =====================================================================

def visualize_maze_geometry(geom: MazeGeometry, out_path: str) -> None:
    """
    输出一张 BGR 叠加图：
      - 灰色  : channel_mask（全通道）
      - 暗黄  : channel_skeleton（全骨架）
      - 浅绿  : solution_channel_mask（解路径通道）
      - 鲜红  : solution_skeleton_mask（解路径骨架）
      - 白色  : 墙壁（原 wall_mask）
      - 绿圈  : entry_xy
      - 红圈  : exit_xy
      - 蓝框  : inner_bbox（墙壁外接矩形）
    """
    h, w = geom.canvas_hw
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # channel
    vis[geom.channel_mask > 0] = (60, 60, 60)
    # 全骨架（暗黄）
    vis[geom.channel_skeleton > 0] = (0, 100, 100)
    # 解路径通道（浅绿）— 覆盖原 channel
    sol_c = (geom.solution_channel_mask > 0)
    vis[sol_c] = (60, 200, 80)
    # 解路径骨架（红） — 最上层
    vis[geom.solution_skeleton_mask > 0] = (50, 50, 230)
    # 墙壁
    vis[geom.wall_mask > 0] = (255, 255, 255)

    # bbox 蓝框
    x1, y1, x2, y2 = geom.inner_bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)

    # entry / exit
    cv2.circle(vis, tuple(int(v) for v in geom.entry_xy), 22, (0, 255, 80), 3)
    cv2.putText(vis, "ENTRY", (geom.entry_xy[0] - 28, geom.entry_xy[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)
    cv2.circle(vis, tuple(int(v) for v in geom.exit_xy), 22, (50, 50, 255), 3)
    cv2.putText(vis, "EXIT", (geom.exit_xy[0] - 20, geom.exit_xy[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, vis)


# =====================================================================
# 命令行
# =====================================================================

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="迷宫通道几何 + 解路径")
    parser.add_argument("--mask", required=True, help="maze_mask.png 路径")
    parser.add_argument("--vis", default=None, help="可视化叠加图输出路径")
    parser.add_argument("--r_wall", type=int, default=2)
    parser.add_argument("--r_channel", type=int, default=28,
                        help="solution_channel_mask 膨胀半径（默认 28，约通道半宽）")
    parser.add_argument("--corner", type=int, default=105)
    args = parser.parse_args()

    geom = build_maze_geometry(
        args.mask,
        r_wall=args.r_wall,
        channel_half_width_px=args.r_channel,
        entry_corner_size=args.corner,
    )
    info = {
        "canvas_hw": geom.canvas_hw,
        "inner_bbox": geom.inner_bbox,
        "channel_area": int(geom.channel_mask.sum()),
        "channel_skeleton_length": int(geom.channel_skeleton.sum()),
        "solution_path_length_px": geom.solution_length_px,
        "solution_channel_area": int(geom.solution_channel_mask.sum()),
        "entry_xy": geom.entry_xy,
        "exit_xy": geom.exit_xy,
        "channel_half_width_px": geom.channel_half_width_px,
        "r_wall": geom.r_wall,
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if args.vis:
        visualize_maze_geometry(geom, args.vis)
        print(f"[maze_geom] 可视化已保存到: {args.vis}")