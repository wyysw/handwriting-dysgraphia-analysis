"""
maze_geometry_circle_additions.py
===================================
本文件包含为支持圆形迷宫（阶段3）需要添加到 features/maze_geometry.py 的全部内容。

集成步骤：
1. 在 MazeGeometry dataclass 末尾新增三个可选字段（见 [PATCH-1]）
2. 将本文件中的新函数（[NEW FUNCTIONS]）插入 maze_geometry.py 的
   "# 骨架最短路径" 代码块之前（约第 236 行）
3. 替换 build_maze_geometry 函数（见 [PATCH-2]）
4. 更新 visualize_maze_geometry 以适应圆形（见 [PATCH-3]，可选）
5. 更新命令行 __main__ 块（见 [PATCH-4]）
"""

# ======================================================================
# [PATCH-1] MazeGeometry dataclass —— 在现有字段末尾新增以下三个字段
# （插入位置：solution_length_px 字段之后）
# ======================================================================
"""
    # 圆形迷宫专用（方形迷宫调用时均为 None，对现有代码无影响）
    circle_center_xy: Optional[Tuple[int, int]] = None
    outer_ring_radius: Optional[float] = None
    num_channel_components_before_filter: Optional[int] = None
"""


# ======================================================================
# [NEW FUNCTIONS] 圆形迷宫几何分支
# 插入位置：maze_geometry.py 中
#   "# =====================================================================
#   # 骨架最短路径" 代码块之前（约第 236 行）
# ======================================================================

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np

# （以下函数直接粘贴进 maze_geometry.py，无需 import —— maze_geometry.py
#  已在顶部导入了 cv2 / numpy / skimage.morphology.skeletonize 等）


# ——————————————————————————————————————————————
# 圆形外环拟合
# ——————————————————————————————————————————————

def _fit_circle_lsq(
    xs: np.ndarray, ys: np.ndarray
) -> Tuple[float, float, float]:
    """
    最小二乘拟合圆，返回 (cx, cy, radius)。

    线性化：x² + y² + Dx + Ey + F = 0
      A = [x, y, 1]，b = -(x² + y²)，最小二乘解 z = [D, E, F]
      cx = -D/2，cy = -E/2，r = √(cx² + cy² - F)
    """
    A = np.column_stack([xs.astype(float), ys.astype(float), np.ones(len(xs))])
    b = -(xs.astype(float) ** 2 + ys.astype(float) ** 2)
    z, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F_ = float(z[0]), float(z[1]), float(z[2])
    cx = -D / 2.0
    cy = -E / 2.0
    r = float(np.sqrt(max(cx ** 2 + cy ** 2 - F_, 0.0)))
    return cx, cy, r


def _fit_outer_circle(
    wall_mask: np.ndarray,
    outer_percentile: float = 90.0,
    fallback_center: Optional[Tuple[float, float]] = None,
    fallback_radius: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    从墙壁掩码中拟合外边界圆，返回 (cx, cy, r_out)。

    取径向距离 >= outer_percentile 的像素参与最小二乘拟合。
    若拟合失败（r_out 异常），使用 fallback 参数或 bbox 中心估计。

    fallback_center / fallback_radius：可提供硬编码默认值作为 fallback。
    """
    _HARD_CX, _HARD_CY, _HARD_R = 598.0, 812.0, 413.0  # 针对 circle_mask.png 的硬编码

    ys, xs = np.where(wall_mask > 0)
    if len(xs) == 0:
        h, w = wall_mask.shape[:2]
        return (
            fallback_center[0] if fallback_center else _HARD_CX,
            fallback_center[1] if fallback_center else _HARD_CY,
            fallback_radius if fallback_radius else _HARD_R,
        )

    cx0 = (float(xs.min()) + float(xs.max())) / 2.0
    cy0 = (float(ys.min()) + float(ys.max())) / 2.0
    r0 = np.sqrt((xs - cx0) ** 2 + (ys - cy0) ** 2)
    outer = r0 >= np.percentile(r0, outer_percentile)
    xs_out, ys_out = xs[outer], ys[outer]

    try:
        cx, cy, r_out = _fit_circle_lsq(xs_out, ys_out)
        if r_out < 10 or r_out > max(wall_mask.shape):
            raise ValueError(f"r_out={r_out:.1f} 异常，使用 fallback")
    except Exception:
        cx = fallback_center[0] if fallback_center else _HARD_CX
        cy = fallback_center[1] if fallback_center else _HARD_CY
        r_out = fallback_radius if fallback_radius else _HARD_R

    return cx, cy, r_out


# ——————————————————————————————————————————————
# 圆形 canvas_mask
# ——————————————————————————————————————————————

def _make_disk_canvas(
    h: int, w: int, cx: float, cy: float, r: float, extra: int = 2
) -> np.ndarray:
    """
    生成圆盘 canvas_mask（uint8, 0/1）。
    disk at (cx, cy) with radius r + extra，+extra 确保含外墙像素。
    """
    Y, X = np.ogrid[:h, :w]
    return ((X - cx) ** 2 + (Y - cy) ** 2 <= (r + extra) ** 2).astype(np.uint8)


# ——————————————————————————————————————————————
# 连通域过滤（去除中心装饰小圆的孤立 blob）
# ——————————————————————————————————————————————

def _filter_largest_channel_component(
    channel_mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    保留 channel_mask 中面积最大的连通域，返回 (filtered_mask, n_components_before)。

    圆形迷宫的中心装饰小圆（r≈40 px）会在 channel_mask 中形成一个封闭的孤立 blob，
    其面积远小于主环通道系统；保留最大连通域即可自动去除。

    返回：
        filtered_mask         : 过滤后的 channel_mask（uint8, 0/1）
        n_components_before   : 过滤前的连通域数量（不含背景；诊断用，正常应为 2）
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        channel_mask.astype(np.uint8), connectivity=8
    )
    n_fg = n - 1  # 减去背景 label=0
    if n_fg <= 0:
        return channel_mask.copy(), 0
    if n_fg == 1:
        return (labels == 1).astype(np.uint8), 1
    areas = stats[1:, cv2.CC_STAT_AREA]          # (n_fg,) 各前景分量面积
    largest_label = 1 + int(np.argmax(areas))
    filtered = (labels == largest_label).astype(np.uint8)
    return filtered, int(n_fg)


# ——————————————————————————————————————————————
# 圆形入口/出口检测
# ——————————————————————————————————————————————

def _detect_entry_exit_circle(
    channel_mask: np.ndarray,
    cx: float,
    cy: float,
    r_out: float,
    entry_xy: Optional[Tuple[int, int]] = None,
    exit_xy: Optional[Tuple[int, int]] = None,
    r_scan_offset: int = 5,
    angle_resolution: int = 720,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    圆形迷宫入口/出口检测，返回 (entry_xy, exit_xy)。

    优先级：
    1. 若 entry_xy 和 exit_xy 均已提供，直接返回（最高优先）
    2. 在 r = r_out - r_scan_offset 处以 angle_resolution 步的角度扫描，
       找 channel_mask 连续为 True 的两个最大角度段（即外环上的两个缺口），
       取各段角度质心对应的像素作为锚点
    3. 若扫描找不到恰好 2 个缺口，fallback 到硬编码默认值

    硬编码默认值（针对 circle_mask.png / 36circle.png）：
      entry_xy_default = (315, 522)   # 左上缺口，θ ≈ −135°
      exit_xy_default  = (892, 1082)  # 右下缺口，θ ≈ +42.5°

    说明：_nearest_skeleton_pixel 会把锚点纠正到最近骨架像素，
          故硬编码误差几十像素无影响。
    """
    _ENTRY_DEFAULT: Tuple[int, int] = (315, 522)
    _EXIT_DEFAULT: Tuple[int, int] = (892, 1082)

    # 若两个都已提供，直接使用
    if entry_xy is not None and exit_xy is not None:
        return entry_xy, exit_xy

    # ---- 角度扫描 ----
    h, w = channel_mask.shape[:2]
    r_scan = r_out - r_scan_offset
    angles = np.linspace(-np.pi, np.pi, angle_resolution, endpoint=False)
    in_ch = np.zeros(angle_resolution, dtype=bool)
    for i, theta in enumerate(angles):
        px = int(round(cx + r_scan * np.cos(theta)))
        py = int(round(cy + r_scan * np.sin(theta)))
        if 0 <= px < w and 0 <= py < h:
            in_ch[i] = channel_mask[py, px] > 0

    # 处理首尾连接（圆形扫描的环绕情况）
    # 找连续 True 段
    segments: List[List[int]] = []
    cur_seg: Optional[List[int]] = None
    for i, v in enumerate(in_ch):
        if v:
            if cur_seg is None:
                cur_seg = []
            cur_seg.append(i)
        else:
            if cur_seg is not None:
                segments.append(cur_seg)
                cur_seg = None
    if cur_seg is not None:
        segments.append(cur_seg)

    # 若首尾都是 True，首尾连接为同一段
    if len(segments) >= 2 and in_ch[0] and in_ch[-1]:
        segments[0] = segments[-1] + segments[0]
        segments.pop(-1)

    if len(segments) < 2:
        # 扫描未找到 2 个缺口，使用硬编码
        return (
            entry_xy if entry_xy is not None else _ENTRY_DEFAULT,
            exit_xy if exit_xy is not None else _EXIT_DEFAULT,
        )

    # 取面积最大的两个段
    top2 = sorted(segments, key=len, reverse=True)[:2]
    results: List[Tuple[int, int]] = []
    for seg in top2:
        mid_angle = angles[int(round(np.mean(seg)))]
        px = int(round(cx + r_scan * np.cos(mid_angle)))
        py = int(round(cy + r_scan * np.sin(mid_angle)))
        # 限制在图像范围内
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        results.append((px, py))

    # 区分入口（左上，y-x 小）和出口（右下，y-x 大）
    results_sorted = sorted(results, key=lambda p: p[1] - p[0])
    return results_sorted[0], results_sorted[-1]


# ——————————————————————————————————————————————
# 圆形几何总入口
# ——————————————————————————————————————————————

def _build_circle_geometry(
    wall_mask: np.ndarray,
    mask_path: str = "",
    r_wall: int = 2,
    channel_half_width_px: int = 28,
    circle_center: Optional[Tuple[float, float]] = None,
    outer_ring_radius: Optional[float] = None,
    entry_xy: Optional[Tuple[int, int]] = None,
    exit_xy: Optional[Tuple[int, int]] = None,
    circle_scan_entry: bool = False,
) -> "MazeGeometry":
    """
    圆形迷宫几何构造（7 步流程，详见设计文档 4.3.3）。
    返回与方形迷宫结构完全一致的 MazeGeometry 对象。
    """
    # 此函数在 maze_geometry.py 中使用，skeletonize / MazeGeometry 等
    # 已在该文件内定义，无需额外 import。

    h, w = wall_mask.shape[:2]

    # ── 步骤① 拟合外环圆 ──────────────────────────────────────────
    if circle_center is not None and outer_ring_radius is not None:
        cx, cy = float(circle_center[0]), float(circle_center[1])
        r_out = float(outer_ring_radius)
    else:
        cx, cy, r_out = _fit_outer_circle(
            wall_mask,
            fallback_center=circle_center,
            fallback_radius=outer_ring_radius,
        )

    # ── 步骤② 圆盘 canvas_mask ───────────────────────────────────
    canvas_mask = _make_disk_canvas(h, w, cx, cy, r_out, extra=2)

    # ── 步骤③ channel_mask_raw（公式与方形完全一致）─────────────
    channel_mask_raw = _build_channel_mask(canvas_mask, wall_mask, r_wall=r_wall)

    # ── 步骤④ 连通域过滤（圆形特有）────────────────────────────
    channel_mask, n_components_before = _filter_largest_channel_component(channel_mask_raw)

    # ── 步骤⑤ 骨架 + 距离变换（与方形完全一致）─────────────────
    channel_skeleton = skeletonize(channel_mask > 0).astype(np.uint8)
    channel_dist = _channel_distance_transform(channel_mask)

    # ── 步骤⑥ 入口/出口 ──────────────────────────────────────────
    if circle_scan_entry:
        _entry, _exit = _detect_entry_exit_circle(
            channel_mask, cx, cy, r_out,
            entry_xy=entry_xy, exit_xy=exit_xy,
        )
    else:
        # 默认：硬编码（entry_xy/exit_xy 若为 None 则使用内置硬编码）
        _ENTRY_DEFAULT: Tuple[int, int] = (315, 522)
        _EXIT_DEFAULT: Tuple[int, int] = (892, 1082)
        _entry = entry_xy if entry_xy is not None else _ENTRY_DEFAULT
        _exit = exit_xy if exit_xy is not None else _EXIT_DEFAULT

    # ── 步骤⑦ Dijkstra → 解路径（不做 DP 简化）─────────────────
    entry_anchor = _nearest_skeleton_pixel(channel_skeleton, _entry)
    exit_anchor = _nearest_skeleton_pixel(channel_skeleton, _exit)
    path_pixels = _dijkstra_on_skeleton(channel_skeleton, entry_anchor, exit_anchor)
    solution_polyline = np.array(path_pixels, dtype=np.int32)  # (N, 2)

    solution_skeleton_mask = np.zeros((h, w), dtype=np.uint8)
    solution_skeleton_mask[solution_polyline[:, 1], solution_polyline[:, 0]] = 1

    r = max(1, int(channel_half_width_px))
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sol_ch = cv2.dilate(solution_skeleton_mask * 255, kernel)
    solution_channel_mask = ((sol_ch > 0) & (channel_mask > 0)).astype(np.uint8)

    # inner_bbox：用外接矩形近似（复用该字段，坐标对齐时的 fallback 会用到）
    inner_bbox = (
        int(round(cx - r_out)), int(round(cy - r_out)),
        int(round(cx + r_out)), int(round(cy + r_out)),
    )

    return MazeGeometry(
        mask_path=mask_path,
        wall_mask=wall_mask,
        canvas_mask=canvas_mask,
        channel_mask=channel_mask,
        channel_skeleton=channel_skeleton,
        channel_dist=channel_dist,
        entry_xy=_entry,
        exit_xy=_exit,
        solution_polyline=solution_polyline,
        solution_skeleton_mask=solution_skeleton_mask,
        solution_channel_mask=solution_channel_mask,
        canvas_hw=(h, w),
        inner_bbox=inner_bbox,
        channel_half_width_px=r,
        r_wall=int(r_wall),
        solution_length_px=int(len(path_pixels)),
        # 圆形专属字段（dataclass 新增，见 [PATCH-1]）
        circle_center_xy=(int(round(cx)), int(round(cy))),
        outer_ring_radius=float(r_out),
        num_channel_components_before_filter=n_components_before,
    )


# ======================================================================
# [PATCH-2] 替换 build_maze_geometry 函数
# （完整替换原函数，新增 game_type 参数和圆形参数组）
# ======================================================================

def build_maze_geometry(
    mask_path: str,
    *,
    game_type: str = "maze",                  # ← 新增：'maze' | 'circle'
    r_wall: int = 2,
    channel_half_width_px: int = 28,
    # 方形迷宫参数（game_type='circle' 时忽略）
    entry_corner_size: int = 105,
    use_frame_ring_first: bool = False,
    entry_xy: Optional[Tuple[int, int]] = None,
    exit_xy: Optional[Tuple[int, int]] = None,
    # 圆形迷宫参数（game_type='maze' 时忽略）
    circle_center: Optional[Tuple[float, float]] = None,
    outer_ring_radius: Optional[float] = None,
    circle_scan_entry: bool = False,
) -> "MazeGeometry":
    """
    构造迷宫几何对象（方形 + 圆形统一入口）。

    参数：
        mask_path             : maze_mask.png 或 circle_mask.png 路径（前景=墙壁）。
        game_type             : 'maze'（默认）→ 方形迷宫；'circle' → 圆形迷宫。
        r_wall                : 墙壁膨胀半径，默认 2 px。
        channel_half_width_px : solution_skeleton 膨胀半径（≈ 通道半宽 28 px）。

        【方形专用参数】
        entry_corner_size     : 简化版入口/出口检测用的方框大小，默认 105 px。
        use_frame_ring_first  : True 时优先尝试通用 frame_ring 检测。
        entry_xy / exit_xy    : 手动指定锚点（覆盖自动检测）。

        【圆形专用参数】
        circle_center         : (cx, cy) 手动指定圆心，None → 自动最小二乘拟合。
        outer_ring_radius     : 外环半径，None → 自动拟合。
        entry_xy / exit_xy    : 手动指定入口/出口（None → 硬编码默认值或角度扫描）。
        circle_scan_entry     : True → 用角度扫描检测入口/出口，覆盖硬编码默认。
    """
    wall_mask = _binarize_mask(mask_path)

    if game_type == "circle":
        return _build_circle_geometry(
            wall_mask,
            mask_path=str(mask_path),
            r_wall=r_wall,
            channel_half_width_px=channel_half_width_px,
            circle_center=circle_center,
            outer_ring_radius=outer_ring_radius,
            entry_xy=entry_xy,
            exit_xy=exit_xy,
            circle_scan_entry=circle_scan_entry,
        )

    # ============================================================
    # 以下为原方形迷宫逻辑（保持不变）
    # ============================================================
    canvas_mask = _floodfill_canvas(wall_mask)
    channel_mask = _build_channel_mask(canvas_mask, wall_mask, r_wall=r_wall)

    h, w = wall_mask.shape[:2]
    inner_bbox = _inner_bbox_from_walls(wall_mask)

    channel_skeleton = skeletonize(channel_mask > 0).astype(np.uint8)
    channel_dist = _channel_distance_transform(channel_mask)

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

    entry_anchor = _nearest_skeleton_pixel(channel_skeleton, entry)
    exit_anchor = _nearest_skeleton_pixel(channel_skeleton, exit_)
    path_pixels = _dijkstra_on_skeleton(channel_skeleton, entry_anchor, exit_anchor)
    solution_polyline = np.array(path_pixels, dtype=np.int32)

    solution_skeleton_mask = np.zeros((h, w), dtype=np.uint8)
    solution_skeleton_mask[solution_polyline[:, 1], solution_polyline[:, 0]] = 1

    r = max(1, int(channel_half_width_px))
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    solution_channel_mask = cv2.dilate(solution_skeleton_mask * 255, kernel)
    solution_channel_mask = (solution_channel_mask > 0).astype(np.uint8)
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
        # 圆形专属字段：方形时均为 None
        circle_center_xy=None,
        outer_ring_radius=None,
        num_channel_components_before_filter=None,
    )


# ======================================================================
# [PATCH-3] 更新 visualize_maze_geometry（可选）
# 在现有函数末尾追加入口/出口的角度标注（仅圆形时显示）
# ======================================================================
"""
在 visualize_maze_geometry 函数的 entry/exit 绘制代码之后追加：

    # 圆形：在可视化中标注圆心和外环
    if geom.circle_center_xy is not None:
        ccx, ccy = geom.circle_center_xy
        r_out = int(round(geom.outer_ring_radius)) if geom.outer_ring_radius else 413
        cv2.circle(vis, (ccx, ccy), r_out, (80, 80, 200), 1)  # 外环轮廓（蓝）
        cv2.circle(vis, (ccx, ccy), 5, (200, 80, 200), -1)    # 圆心（紫）
"""


# ======================================================================
# [PATCH-4] 更新命令行 __main__ 块（追加 --game 参数）
# ======================================================================
"""
在 parser 中追加：
    parser.add_argument("--game", default="maze", choices=["maze", "circle"],
                        help="游戏类型：maze（默认）| circle")
    parser.add_argument("--circle_scan", action="store_true",
                        help="圆形迷宫：用角度扫描检测入口出口（默认使用硬编码）")

并将 build_maze_geometry 调用改为：
    geom = build_maze_geometry(
        args.mask,
        game_type=args.game,
        r_wall=args.r_wall,
        channel_half_width_px=args.r_channel,
        entry_corner_size=args.corner,
        circle_scan_entry=getattr(args, 'circle_scan', False),
    )
"""
