"""
features/maze_geometry.py  (v2.1 — 新增圆形迷宫支持)

阶段 2/3 — 迷宫游戏的【通道几何 + 解路径】模块。
支持 game_type='maze'（方形迷宫）和 game_type='circle'（圆形迷宫）。

输入：maze_mask.png 或 circle_mask.png（前景=墙壁线条，背景=黑）。
输出：MazeGeometry dataclass，含
    - canvas_mask         : 可绘制画布内部区域
    - wall_mask           : 原墙壁掩码
    - channel_mask        : 可行走通道（前景=通道内部）
    - channel_skeleton    : 通道骨架（1 px 宽）
    - channel_dist        : 非通道像素到最近通道的 L2 距离
    - entry_xy, exit_xy   : 入口 / 出口锚点 (x, y)
    - solution_polyline   : 沿解路径的有序 (x, y) 数组（骨架像素）
    - solution_skeleton_mask : 解路径骨架（1 px 宽）
    - solution_channel_mask  : 解路径通道（解骨架膨胀至通道半宽）
    - circle_meta         : 圆形迷宫专属元信息（dict，仅 game_type='circle' 时非 None）

方形迷宫设计要点（原 v2.0）：
    1. floodFill 从画布角外背景→取反→canvas_mask
    2. 墙壁膨胀 r_wall px 再从 canvas_mask 扣除，避免细缝串连
    3. 入口/出口：简化版（固定 corner_size 矩形）+ 通用版（frame-ring 连通域）备用

圆形迷宫设计要点（v2.1 新增）：
    1. 最小二乘拟合外环圆 → 填充圆盘 → canvas_mask（避免 floodFill 从缺口漏入）
    2. 过滤 channel_mask 最大连通域，去除中心装饰小圆形成的孤立伪通道
    3. 外环角度扫描找 True-run（缺口=通道可见处）→ 入口/出口
    4. BFS Dijkstra 在骨架上求最短路（圆形骨架含环状拓扑，BFS 自然处理）
    5. 不做 Douglas-Peucker 简化，保留曲线骨架像素序列

调用：`build_maze_geometry(mask_path, game_type='maze'|'circle')`
可视化：`visualize_maze_geometry(geom, out_path)`
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
except ImportError as e:
    raise ImportError("需要 scikit-image：pip install scikit-image") from e


# =====================================================================
# 数据类型
# =====================================================================

Point = Tuple[int, int]  # (x, y)


@dataclass
class MazeGeometry:
    """打包所有从 mask.png 派生的几何结构（方形/圆形迷宫共用）。"""

    mask_path: str

    # 基础掩码（uint8，0/1）
    wall_mask: np.ndarray
    canvas_mask: np.ndarray
    channel_mask: np.ndarray

    # 骨架与距离变换
    channel_skeleton: np.ndarray   # uint8, 0/1
    channel_dist: np.ndarray       # float32, 非通道像素到最近通道像素的 L2 距离

    # 入口 / 出口
    entry_xy: Point
    exit_xy: Point

    # 解路径
    solution_polyline: np.ndarray         # (N, 2) int，有序 (x, y)
    solution_skeleton_mask: np.ndarray    # uint8, 0/1
    solution_channel_mask: np.ndarray     # uint8, 0/1

    # 通用元信息
    canvas_hw: Tuple[int, int] = field(default=(0, 0))
    inner_bbox: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0))  # x1,y1,x2,y2（方形用）
    channel_half_width_px: int = 28
    r_wall: int = 2
    solution_length_px: int = 0

    # 圆形迷宫专属元信息（方形时为 None）
    circle_meta: Optional[Dict[str, Any]] = field(default=None)


# =====================================================================
# 共用基础函数
# =====================================================================

def _binarize_mask(path: str, thresh: int = 127) -> np.ndarray:
    """读取灰度 PNG → 0/1 uint8 二值图（前景=墙壁）。"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取掩码: {path}")
    return (img > thresh).astype(np.uint8)


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
    return ((canvas_mask > 0) & (wall_dilated == 0)).astype(np.uint8)


def _channel_distance_transform(channel_mask: np.ndarray) -> np.ndarray:
    """非通道像素到最近通道像素的 L2 距离（float32）。"""
    inv = (channel_mask == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return dist.astype(np.float32)


def _inner_bbox_from_walls(wall_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """墙壁前景的外接矩形 (x1, y1, x2, y2)，闭区间。"""
    ys, xs = np.where(wall_mask > 0)
    if len(xs) == 0:
        h, w = wall_mask.shape[:2]
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# =====================================================================
# 方形迷宫：canvas_mask 与入口/出口
# =====================================================================

def _floodfill_canvas(wall_mask: np.ndarray) -> np.ndarray:
    """
    从图像四角背景 floodFill，把"画布外"标记为 128；
    取反得到外框+内部 = canvas_mask。
    """
    h, w = wall_mask.shape[:2]
    filled = (wall_mask > 0).astype(np.uint8) * 255
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if filled[seed[1], seed[0]] == 0:
            cv2.floodFill(filled, ff_mask, seed, 128)
    return (filled != 128).astype(np.uint8)


def _detect_entry_exit_simple(
    channel_mask: np.ndarray,
    inner_bbox: Tuple[int, int, int, int],
    corner_size: int = 105,
) -> Tuple[Point, Point]:
    """
    方形迷宫简化版入口/出口检测：
      入口 = 内框右上 corner_size×corner_size 矩形内通道像素的质心
      出口 = 内框左下 corner_size×corner_size 矩形内通道像素的质心
    """
    x1, y1, x2, y2 = inner_bbox

    rt_x1 = max(x1, x2 - corner_size)
    rt_y1, rt_x2, rt_y2 = y1, x2, min(y2, y1 + corner_size)
    lb_x1, lb_y1, lb_x2 = x1, max(y1, y2 - corner_size), min(x2, x1 + corner_size)
    lb_y2 = y2

    def _centroid(box):
        bx1, by1, bx2, by2 = box
        sub = channel_mask[by1:by2 + 1, bx1:bx2 + 1]
        ys, xs = np.where(sub > 0)
        if len(xs) == 0:
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
    方形迷宫通用版（备用）：
    取内框 1 圈 ring，与 channel_mask 取交，连通域分析。
    恰好 2 个连通域时返回 (右上质心, 左下质心)，否则返回 None。
    """
    x1, y1, x2, y2 = inner_bbox
    ring = np.zeros_like(channel_mask)
    rw = ring_width
    ring[y1:y1 + rw + 1, x1:x2 + 1] = 1
    ring[y2 - rw:y2 + 1, x1:x2 + 1] = 1
    ring[y1:y2 + 1, x1:x1 + rw + 1] = 1
    ring[y1:y2 + 1, x2 - rw:x2 + 1] = 1
    openings = (ring > 0) & (channel_mask > 0)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        openings.astype(np.uint8), connectivity=8
    )
    if n - 1 != 2:
        return None
    cents = [(int(centroids[i, 0]), int(centroids[i, 1])) for i in range(1, n)]
    cents_sorted = sorted(cents, key=lambda p: (p[1] - p[0]))
    return cents_sorted[0], cents_sorted[-1]


# =====================================================================
# 圆形迷宫：专属几何函数
# =====================================================================

def _fit_outer_circle_lsq(
    wall_mask: np.ndarray,
    outer_percentile: float = 90.0,
) -> Tuple[float, float, float]:
    """
    最小二乘拟合外环圆。

    算法：取距 bbox 质心最远的 (100 - outer_percentile)% 像素（即外环像素），
    用代数圆拟合 (x-cx)^2+(y-cy)^2=r^2 的线性化形式求解 cx, cy, r。

    参数：
        wall_mask        : 圆形迷宫的墙壁二值掩码
        outer_percentile : 阈值百分位，仅取 >= 此百分位距离的像素参与拟合

    返回：(cx, cy, r_out)，单位像素
    """
    ys, xs = np.where(wall_mask > 0)
    if len(xs) == 0:
        h, w = wall_mask.shape[:2]
        return float(w) / 2, float(h) / 2, min(h, w) / 2.0

    # 粗略质心，用于筛选外环像素
    cx0, cy0 = float(xs.mean()), float(ys.mean())
    r_all = np.sqrt((xs - cx0) ** 2 + (ys - cy0) ** 2)
    thresh = np.percentile(r_all, outer_percentile)
    outer = r_all >= thresh

    ox = xs[outer].astype(np.float64)
    oy = ys[outer].astype(np.float64)

    # 线性化：a*x + b*y + c = x^2 + y^2，其中 a=2cx, b=2cy, c=r^2-cx^2-cy^2
    A = np.column_stack([ox, oy, np.ones(len(ox))])
    b_vec = ox ** 2 + oy ** 2
    result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    cx_fit = result[0] / 2.0
    cy_fit = result[1] / 2.0
    r_fit = float(np.sqrt(max(0.0, result[2] + cx_fit ** 2 + cy_fit ** 2)))
    return cx_fit, cy_fit, r_fit


def _make_disk_canvas_mask(
    h: int, w: int, cx: float, cy: float, r: float
) -> np.ndarray:
    """返回以 (cx, cy) 为圆心、半径 r 的填充圆盘掩码（uint8, 0/1）。"""
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2).astype(np.uint8)


def _filter_largest_cc(mask: np.ndarray) -> np.ndarray:
    """
    仅保留最大连通域。

    用途：圆形迷宫中，中心装饰小圆会在 channel_mask 内形成一个
    孤立的封闭区域（面积约 4000 px），过滤后只剩主环通道系统。
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return mask.copy()
    # stats[0] 是背景，从 1 开始取面积最大的前景连通域
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    return (labels == best_label).astype(np.uint8)


def _detect_entry_exit_circle(
    channel_mask: np.ndarray,
    cx: float,
    cy: float,
    r_out: float,
    scan_r_offset: float = -5.0,
    angle_step_deg: float = 1.0,
    min_gap_deg: float = 10.0,
) -> Optional[Tuple[Point, Point]]:
    """
    圆形迷宫入口/出口自动检测。

    原理：在外环内侧（半径 r_out + scan_r_offset）做角度扫描。
    - 非缺口角度：处于外环墙壁内 → channel_mask=False
    - 缺口角度（入口/出口）：墙壁有孔 → channel_mask=True
    找 True-run（连续 True 的角度段），每段即一个缺口。

    参数：
        scan_r_offset : 扫描半径相对 r_out 的偏移，负值表示向内。
                        默认 -5 px，使扫描点处于外环墙壁厚度内。
        min_gap_deg   : 最小缺口弧长（度），用于过滤小噪声。

    返回：(entry_xy, exit_xy) 元组，按 x+y 升序排列（左上=入口）；
          无法检测到恰好 2 个缺口时返回 None。
    """
    r_scan = r_out + scan_r_offset
    if r_scan <= 0:
        return None

    h, w = channel_mask.shape[:2]
    n_steps = int(round(360.0 / angle_step_deg))

    # 角度扫描 → 布尔数组
    thetas_deg = np.arange(n_steps) * angle_step_deg  # 0..359°
    on = np.zeros(n_steps, dtype=bool)
    for i, theta in enumerate(thetas_deg):
        rad = np.deg2rad(theta)
        xi = int(round(cx + r_scan * np.cos(rad)))
        yi = int(round(cy + r_scan * np.sin(rad)))
        if 0 <= xi < w and 0 <= yi < h:
            on[i] = channel_mask[yi, xi] > 0

    # 用 diff 找 True-run（缺口）
    on_int = on.astype(np.int8)
    padded = np.append(on_int, on_int[0])           # 首尾相接，处理环绕
    diff = np.diff(padded.astype(np.int32))
    rising  = np.where(diff > 0)[0]  # False→True  : True-run 在 rising+1 开始
    falling = np.where(diff < 0)[0]  # True→False  : True-run 在 falling 结束

    if len(rising) == 0 or len(falling) == 0:
        return None

    gaps: List[float] = []  # True-run 中心角度（度）
    for r_idx in rising:
        f_after = falling[falling > r_idx]
        if len(f_after) > 0:
            f = int(f_after[0])
        else:
            # 缺口跨过 0°/360° 边界
            f = int(falling[0]) + n_steps
        true_start = r_idx + 1
        true_end   = f          # inclusive
        run_len    = true_end - true_start + 1
        run_deg    = run_len * angle_step_deg
        if run_deg >= min_gap_deg:
            mid_idx = (true_start + true_end) / 2.0
            mid_deg = (mid_idx * angle_step_deg) % 360.0
            gaps.append(mid_deg)

    if len(gaps) != 2:
        return None

    def _theta_to_xy(theta_deg: float) -> Point:
        rad = np.deg2rad(theta_deg)
        xi = int(round(cx + r_scan * np.cos(rad)))
        yi = int(round(cy + r_scan * np.sin(rad)))
        # 限制在图像范围内
        xi = max(0, min(w - 1, xi))
        yi = max(0, min(h - 1, yi))
        return xi, yi

    p1 = _theta_to_xy(gaps[0])
    p2 = _theta_to_xy(gaps[1])

    # 按 x+y 升序：左上（x+y 较小）= 入口；右下（x+y 较大）= 出口
    if p1[0] + p1[1] <= p2[0] + p2[1]:
        return p1, p2
    else:
        return p2, p1


# =====================================================================
# 骨架最短路径（方形/圆形共用）
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


_NEIGHBORS_8 = [
    (-1, -1, 1.41421356), (0, -1, 1.0), (1, -1, 1.41421356),
    (-1,  0, 1.0),                       (1,  0, 1.0),
    (-1,  1, 1.41421356), (0,  1, 1.0), (1,  1, 1.41421356),
]


def _dijkstra_on_skeleton(
    skeleton: np.ndarray,
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    在 1 px 宽骨架上跑 Dijkstra（8 邻接，权重=欧氏距离）。
    返回从 start_xy 到 goal_xy 的最短路径像素列表 [(x, y), ...]。

    注意：圆形迷宫骨架含环状拓扑（同心环为 cycle），Dijkstra 自然选择最短路。
    """
    h, w = skeleton.shape[:2]
    skel = skeleton > 0
    sx, sy = start_xy
    gx, gy = goal_xy

    if not skel[sy, sx]:
        raise RuntimeError(f"start {start_xy} 不在骨架上。")
    if not skel[gy, gx]:
        raise RuntimeError(f"goal {goal_xy} 不在骨架上。")

    INF = float("inf")
    dist   = np.full((h, w), INF, dtype=np.float64)
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
        raise RuntimeError("骨架上 start→goal 不连通，请检查 channel_mask / r_wall。")

    path: List[Tuple[int, int]] = []
    cx2, cy2 = gx, gy
    while not (cx2 == sx and cy2 == sy):
        path.append((cx2, cy2))
        px, py = int(parent[cy2, cx2, 0]), int(parent[cy2, cx2, 1])
        if px < 0:
            break
        cx2, cy2 = px, py
    path.append((sx, sy))
    path.reverse()
    return path


def _dilate_to_solution_channel(
    solution_skeleton_mask: np.ndarray,
    channel_mask: np.ndarray,
    channel_half_width_px: int,
) -> np.ndarray:
    """解路径骨架膨胀至通道宽，限制在 channel_mask 内。"""
    r = max(1, int(channel_half_width_px))
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sol_ch = cv2.dilate(solution_skeleton_mask * 255, kernel)
    sol_ch = (sol_ch > 0).astype(np.uint8)
    return (sol_ch & channel_mask).astype(np.uint8)


# =====================================================================
# 方形迷宫：完整几何构造
# =====================================================================

def _build_square_geometry(
    wall_mask: np.ndarray,
    *,
    r_wall: int = 2,
    channel_half_width_px: int = 28,
    entry_corner_size: int = 105,
    entry_xy: Optional[Point] = None,
    exit_xy: Optional[Point] = None,
    use_frame_ring_first: bool = False,
) -> MazeGeometry:
    """方形迷宫几何构造（原 build_maze_geometry 主体）。"""
    h, w = wall_mask.shape[:2]
    canvas_mask = _floodfill_canvas(wall_mask)
    channel_mask = _build_channel_mask(canvas_mask, wall_mask, r_wall=r_wall)
    inner_bbox = _inner_bbox_from_walls(wall_mask)

    channel_skeleton = skeletonize(channel_mask > 0).astype(np.uint8)
    channel_dist = _channel_distance_transform(channel_mask)

    # 入口/出口
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

    # 解路径
    entry_anchor = _nearest_skeleton_pixel(channel_skeleton, entry)
    exit_anchor  = _nearest_skeleton_pixel(channel_skeleton, exit_)
    path_pixels  = _dijkstra_on_skeleton(channel_skeleton, entry_anchor, exit_anchor)
    solution_polyline = np.array(path_pixels, dtype=np.int32)

    sol_skel = np.zeros((h, w), dtype=np.uint8)
    sol_skel[solution_polyline[:, 1], solution_polyline[:, 0]] = 1
    sol_ch = _dilate_to_solution_channel(sol_skel, channel_mask, channel_half_width_px)

    return MazeGeometry(
        mask_path="",
        wall_mask=wall_mask,
        canvas_mask=canvas_mask,
        channel_mask=channel_mask,
        channel_skeleton=channel_skeleton,
        channel_dist=channel_dist,
        entry_xy=entry,
        exit_xy=exit_,
        solution_polyline=solution_polyline,
        solution_skeleton_mask=sol_skel,
        solution_channel_mask=sol_ch,
        canvas_hw=(h, w),
        inner_bbox=inner_bbox,
        channel_half_width_px=channel_half_width_px,
        r_wall=int(r_wall),
        solution_length_px=int(len(path_pixels)),
        circle_meta=None,
    )


# =====================================================================
# 圆形迷宫：完整几何构造（v2.1 新增）
# =====================================================================

def _build_circle_geometry(
    wall_mask: np.ndarray,
    *,
    r_wall: int = 2,
    channel_half_width_px: int = 28,
    entry_xy: Optional[Point] = None,
    exit_xy: Optional[Point] = None,
    lsq_outer_percentile: float = 90.0,
    scan_r_offset: float = -5.0,
    min_gap_deg: float = 10.0,
) -> MazeGeometry:
    """
    圆形迷宫几何构造。

    流程：
      1. LSQ 拟合外环圆 → 圆盘 canvas_mask（避免 floodFill 从缺口漏入）
      2. channel_mask_raw = canvas_mask AND NOT dilate(wall, r_wall)
      3. 过滤最大连通域 → channel_mask（去除中心装饰小圆的伪通道）
      4. 骨架化 + 距离变换
      5. 角度扫描找外环缺口 → 入口/出口（or 手动指定）
      6. Dijkstra 最短路 → 解路径（BFS 天然处理环状拓扑）
      7. 膨胀解路径骨架 → solution_channel_mask

    设计说明：
      - 圆形骨架含 cycle（同心环本身是 cycle），Dijkstra 返回弧长最短路径。
        对本迷宫，入口到出口只有一条最短路径（A4 确认），无歧义。
      - 不做 Douglas-Peucker 简化（A7），保留弧线精度，供 F2 弧长采样使用。
      - scan_r_offset = -5 px，使扫描点处于外环墙壁内（非缺口→False，缺口→True）。
    """
    h, w = wall_mask.shape[:2]

    # --- 1. 外圆拟合 → canvas_mask ---
    cx, cy, r_out = _fit_outer_circle_lsq(wall_mask, outer_percentile=lsq_outer_percentile)
    # +3 px 余量，确保外环墙壁完整包含在 canvas 内
    canvas_mask = _make_disk_canvas_mask(h, w, cx, cy, r_out + 3)

    # --- 2. 原始 channel_mask（含孤立伪通道）---
    channel_mask_raw = _build_channel_mask(canvas_mask, wall_mask, r_wall=r_wall)

    # 记录过滤前的连通域数（meta 诊断用）
    n_cc_raw, _, _, _ = cv2.connectedComponentsWithStats(
        channel_mask_raw, connectivity=8
    )
    n_cc_raw_fg = n_cc_raw - 1  # 减去背景

    # --- 3. 保留最大连通域（去中心伪通道）---
    channel_mask = _filter_largest_cc(channel_mask_raw)

    # --- 4. 骨架 + 距离变换 ---
    channel_skeleton = skeletonize(channel_mask > 0).astype(np.uint8)
    channel_dist     = _channel_distance_transform(channel_mask)

    # inner_bbox 沿用墙壁外接矩形（方形兼容用，同时也适用于圆形可视化）
    inner_bbox = _inner_bbox_from_walls(wall_mask)

    # --- 5. 入口/出口 ---
    entry_angle_deg: Optional[float] = None
    exit_angle_deg:  Optional[float] = None

    if entry_xy is not None and exit_xy is not None:
        entry, exit_ = entry_xy, exit_xy
    else:
        detected = _detect_entry_exit_circle(
            channel_mask, cx, cy, r_out,
            scan_r_offset=scan_r_offset,
            min_gap_deg=min_gap_deg,
        )
        if detected is None:
            raise RuntimeError(
                "圆形迷宫入口/出口自动检测失败（未找到恰好 2 个缺口）。"
                "请手动指定 entry_xy / exit_xy 参数。"
            )
        entry, exit_ = detected

        # 记录检测到的角度（供 meta）
        def _xy_to_angle(p):
            dx = p[0] - cx
            dy = p[1] - cy
            return float(np.degrees(np.arctan2(dy, dx)) % 360.0)

        entry_angle_deg = _xy_to_angle(entry)
        exit_angle_deg  = _xy_to_angle(exit_)

    # --- 6. 最短路径（Dijkstra on skeleton）---
    entry_anchor = _nearest_skeleton_pixel(channel_skeleton, entry)
    exit_anchor  = _nearest_skeleton_pixel(channel_skeleton, exit_)
    path_pixels  = _dijkstra_on_skeleton(channel_skeleton, entry_anchor, exit_anchor)
    solution_polyline = np.array(path_pixels, dtype=np.int32)

    sol_skel = np.zeros((h, w), dtype=np.uint8)
    sol_skel[solution_polyline[:, 1], solution_polyline[:, 0]] = 1

    # --- 7. 解路径通道 ---
    sol_ch = _dilate_to_solution_channel(sol_skel, channel_mask, channel_half_width_px)

    # --- circle_meta ---
    circle_meta: Dict[str, Any] = {
        "outer_ring_cx": round(float(cx), 2),
        "outer_ring_cy": round(float(cy), 2),
        "outer_ring_r":  round(float(r_out), 2),
        "entry_angle_deg": round(entry_angle_deg, 2) if entry_angle_deg is not None else None,
        "exit_angle_deg":  round(exit_angle_deg,  2) if exit_angle_deg  is not None else None,
        "n_channel_cc_before_filter": int(n_cc_raw_fg),
        "lsq_outer_percentile": float(lsq_outer_percentile),
        "scan_r_offset": float(scan_r_offset),
    }

    return MazeGeometry(
        mask_path="",
        wall_mask=wall_mask,
        canvas_mask=canvas_mask,
        channel_mask=channel_mask,
        channel_skeleton=channel_skeleton,
        channel_dist=channel_dist,
        entry_xy=entry,
        exit_xy=exit_,
        solution_polyline=solution_polyline,
        solution_skeleton_mask=sol_skel,
        solution_channel_mask=sol_ch,
        canvas_hw=(h, w),
        inner_bbox=inner_bbox,
        channel_half_width_px=channel_half_width_px,
        r_wall=int(r_wall),
        solution_length_px=int(len(path_pixels)),
        circle_meta=circle_meta,
    )


# =====================================================================
# 统一主入口
# =====================================================================

def build_maze_geometry(
    mask_path: str,
    *,
    game_type: str = "maze",            # 'maze' | 'circle'
    r_wall: int = 2,
    channel_half_width_px: int = 28,
    # 方形迷宫专用
    entry_corner_size: int = 105,
    use_frame_ring_first: bool = False,
    # 圆形迷宫专用
    lsq_outer_percentile: float = 90.0,
    scan_r_offset: float = -5.0,
    min_gap_deg: float = 10.0,
    # 通用：手动指定入口/出口（覆盖自动检测）
    entry_xy: Optional[Point] = None,
    exit_xy: Optional[Point] = None,
) -> MazeGeometry:
    """
    统一构造迷宫几何对象。

    参数：
        mask_path             : maze_mask.png 或 circle_mask.png（前景=墙壁）。
        game_type             : 'maze'（方形）或 'circle'（圆形），默认 'maze'。
        r_wall                : 墙壁膨胀半径，默认 2 px。
        channel_half_width_px : 解路径骨架膨胀半径（≈ 通道半宽 28 px）。
        entry_corner_size     : [方形专用] 入口/出口简化检测框大小，默认 105 px。
        use_frame_ring_first  : [方形专用] True 时优先尝试 frame_ring 检测。
        lsq_outer_percentile  : [圆形专用] LSQ 拟合外圆时取最外 (100-p)% 的像素。
        scan_r_offset         : [圆形专用] 入口扫描半径相对 r_out 的偏移（默认 -5 px）。
        min_gap_deg           : [圆形专用] 最小缺口弧宽（度），默认 10°。
        entry_xy / exit_xy    : 手动指定锚点（两游戏通用），覆盖自动检测。
    """
    wall_mask = _binarize_mask(mask_path)

    if game_type == "circle":
        geom = _build_circle_geometry(
            wall_mask,
            r_wall=r_wall,
            channel_half_width_px=channel_half_width_px,
            entry_xy=entry_xy,
            exit_xy=exit_xy,
            lsq_outer_percentile=lsq_outer_percentile,
            scan_r_offset=scan_r_offset,
            min_gap_deg=min_gap_deg,
        )
    elif game_type == "maze":
        geom = _build_square_geometry(
            wall_mask,
            r_wall=r_wall,
            channel_half_width_px=channel_half_width_px,
            entry_corner_size=entry_corner_size,
            entry_xy=entry_xy,
            exit_xy=exit_xy,
            use_frame_ring_first=use_frame_ring_first,
        )
    else:
        raise ValueError(f"未知 game_type: {game_type!r}（应为 'maze' 或 'circle'）")

    geom.mask_path = str(mask_path)
    return geom


# =====================================================================
# 可视化（方形/圆形通用）
# =====================================================================

def visualize_maze_geometry(geom: MazeGeometry, out_path: str) -> None:
    """
    输出 BGR 叠加图：
      - 深灰  : 全通道 channel_mask
      - 暗黄  : 全骨架 channel_skeleton
      - 浅绿  : 解路径通道 solution_channel_mask
      - 蓝红  : 解路径骨架 solution_skeleton_mask
      - 白色  : 墙壁 wall_mask
      - 绿圈  : entry_xy
      - 红圈  : exit_xy
      - 蓝框  : inner_bbox 墙壁外接矩形（方形有意义；圆形也绘出供参考）
      - 青色十字 : [圆形] 外环圆心
    """
    h, w = geom.canvas_hw
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    vis[geom.channel_mask > 0]          = (60, 60, 60)
    vis[geom.channel_skeleton > 0]      = (0, 100, 100)
    vis[geom.solution_channel_mask > 0] = (60, 200, 80)
    vis[geom.solution_skeleton_mask > 0] = (50, 50, 230)
    vis[geom.wall_mask > 0]             = (255, 255, 255)

    # inner_bbox 框
    x1, y1, x2, y2 = geom.inner_bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)

    # 圆形专属：绘制外环圆 + 圆心
    if geom.circle_meta is not None:
        cx_c = int(round(geom.circle_meta["outer_ring_cx"]))
        cy_c = int(round(geom.circle_meta["outer_ring_cy"]))
        r_c  = int(round(geom.circle_meta["outer_ring_r"]))
        cv2.circle(vis, (cx_c, cy_c), r_c, (200, 200, 0), 1)   # 拟合外圆（黄色）
        cv2.drawMarker(vis, (cx_c, cy_c), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # entry / exit
    cv2.circle(vis, tuple(int(v) for v in geom.entry_xy), 22, (0, 255, 80), 3)
    cv2.putText(vis, "ENTRY",
                (geom.entry_xy[0] - 28, geom.entry_xy[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)
    cv2.circle(vis, tuple(int(v) for v in geom.exit_xy), 22, (50, 50, 255), 3)
    cv2.putText(vis, "EXIT",
                (geom.exit_xy[0] - 20, geom.exit_xy[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


# =====================================================================
# 命令行
# =====================================================================

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="迷宫通道几何 + 解路径（支持方形/圆形）")
    parser.add_argument("--mask",  required=True, help="maze_mask.png 或 circle_mask.png")
    parser.add_argument("--game",  default="maze", choices=["maze", "circle"])
    parser.add_argument("--vis",   default=None,   help="可视化叠加图输出路径")
    parser.add_argument("--r_wall",    type=int,   default=2)
    parser.add_argument("--r_channel", type=int,   default=28)
    parser.add_argument("--corner",    type=int,   default=105, help="[方形] 入口矩形大小")
    args = parser.parse_args()

    geom = build_maze_geometry(
        args.mask,
        game_type=args.game,
        r_wall=args.r_wall,
        channel_half_width_px=args.r_channel,
        entry_corner_size=args.corner,
    )

    info = {
        "game_type": args.game,
        "canvas_hw": geom.canvas_hw,
        "inner_bbox": geom.inner_bbox,
        "channel_area": int(geom.channel_mask.sum()),
        "channel_skeleton_length": int(geom.channel_skeleton.sum()),
        "solution_path_length_px": geom.solution_length_px,
        "solution_channel_area": int(geom.solution_channel_mask.sum()),
        "entry_xy": list(geom.entry_xy),
        "exit_xy": list(geom.exit_xy),
        "channel_half_width_px": geom.channel_half_width_px,
        "r_wall": geom.r_wall,
    }
    if geom.circle_meta:
        info["circle_meta"] = geom.circle_meta

    print(json.dumps(info, ensure_ascii=False, indent=2))

    if args.vis:
        visualize_maze_geometry(geom, args.vis)
        print(f"[maze_geom] 可视化已保存: {args.vis}")
