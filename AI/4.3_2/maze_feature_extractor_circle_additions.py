"""
maze_feature_extractor_circle_additions.py
============================================
本文件包含为支持圆形迷宫（阶段3）需要修改 features/maze_feature_extractor.py 的全部内容。

集成步骤：
1. 新增函数 compute_C1_jitter_ratio_skeleton（见 [NEW FUNCTION]）
   插入位置：compute_F4_offchannel_ratio 函数之后，extract_maze_features 之前。

2. 新增可视化函数 _visualize_C1_skeleton（见 [NEW VIS FUNCTION]）
   插入位置：_visualize_C1_hough 函数之后。

3. 修改 extract_maze_features 函数签名（见 [PATCH-1]）：新增圆形专用参数。

4. 修改 extract_maze_features 函数体（见 [PATCH-2]）：
   4a. 步骤1：build_maze_geometry 调用增加 game_type 和圆形参数
   4b. 步骤6：C1 计算增加 game_type 分派
   4c. 步骤8：JSON meta 增加圆形专属字段
   4d. 步骤10：可视化增加圆形 C1 图

5. 在文件顶部 import 区新增 scipy.spatial（见 [PATCH-0]）

─────────────────────────────────────────────────────────────────
方案 A → 方案 C 的修改说明见文件末尾 [SCHEME A→C]
─────────────────────────────────────────────────────────────────
"""

# ======================================================================
# [PATCH-0] 在 maze_feature_extractor.py 顶部 import 区追加
# ======================================================================
"""
# 在 "import cv2" 或 "import numpy" 附近追加：
try:
    from scipy.spatial import KDTree as _KDTree
except ImportError as _e:
    raise ImportError(
        "骨架距离 C1 需要 scipy：pip install scipy"
    ) from _e
"""


# ======================================================================
# [NEW FUNCTION] 新增骨架距离残差 C1 计算函数
# 插入位置：compute_F4_offchannel_ratio 之后，extract_maze_features 之前
# ======================================================================

from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np


def compute_C1_jitter_ratio_skeleton(
    mapped_strokes: Sequence[np.ndarray],
    channel_skeleton: np.ndarray,
    jitter_tol: float = 3.0,
    channel_half_width: float = 28.0,
) -> Tuple[float, int, int]:
    """
    C1 骨架距离残差法 —— 用于圆形迷宫（方案 A），或三游戏统一（方案 C）。

    对所有用户笔迹点，计算到 channel_skeleton 最近骨架像素的距离 d；
    在 d <= channel_half_width（在通道内）的有效点中，
    d > jitter_tol 的比例即为 C1。

    与 Hough 方案的核心区别：
    - Hough 方案：在用户笔迹上检测直线段，适用于"理想路径是直线"的方形迷宫。
    - 本方案  ：以通道骨架为参考，适用于含弧线的圆形迷宫，也可统一应用于方形。

    实现细节：
    - 使用 scipy.spatial.KDTree 加速最近邻查询。
      骨架点数约 3000–5000，笔迹点数约 8000，KDTree 查询在毫秒级完成。
    - mapped_strokes 中每个元素是 (N_i, 2) float 数组（画布坐标，x 在前 y 在后）。

    返回：
        C1           : 抖动比例（[0, 1]，↓越好）
        bad_count    : 被判为抖动的有效点数（d > jitter_tol 且 d <= channel_half_width）
        total_valid  : 在通道半宽内的有效投影点总数
    """
    from scipy.spatial import KDTree

    # 提取骨架像素坐标（列向量：[x, y]）
    skel_ys, skel_xs = np.where(channel_skeleton > 0)
    if len(skel_xs) == 0:
        return 0.0, 0, 0

    skel_pts = np.stack([skel_xs, skel_ys], axis=1).astype(np.float64)  # (N_skel, 2)

    # 拼接所有笔迹点
    valid_strokes = [np.asarray(s, dtype=np.float64) for s in mapped_strokes if len(s) >= 1]
    if not valid_strokes:
        return 0.0, 0, 0
    all_pts = np.concatenate(valid_strokes, axis=0)  # (M, 2)

    # KDTree 查询最近邻距离
    tree = KDTree(skel_pts)
    dists, _ = tree.query(all_pts, k=1)  # (M,) float64

    # 统计：仅在通道半宽内的点计入
    valid_mask = dists <= float(channel_half_width)
    total_valid = int(valid_mask.sum())
    if total_valid == 0:
        return 0.0, 0, 0

    bad_count = int((dists[valid_mask] > float(jitter_tol)).sum())
    C1 = float(bad_count) / float(total_valid)
    return C1, bad_count, total_valid


# ======================================================================
# [NEW VIS FUNCTION] 圆形迷宫 C1 可视化（骨架距离法）
# 插入位置：_visualize_C1_hough 函数之后
# ======================================================================

def _visualize_C1_skeleton(
    user_mask: np.ndarray,
    channel_skeleton: np.ndarray,
    mapped_strokes: Sequence[np.ndarray],
    jitter_tol: float,
    channel_half_width: float,
    out_path: str,
) -> None:
    """
    C1 骨架距离叠加图（替代圆形迷宫的 C1_hough.png）：
      - 灰   : 用户笔迹 user_mask
      - 红   : channel_skeleton（通道中线骨架）
      - 黄   : 被判为抖动的有效点（d > jitter_tol 且 d <= channel_half_width）
      - 绿   : 在骨架附近但未抖动的有效点（d <= jitter_tol，随机采样展示）
    """
    import cv2
    from scipy.spatial import KDTree
    import os
    from pathlib import Path

    h, w = user_mask.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[user_mask > 0] = (100, 100, 100)
    vis[channel_skeleton > 0] = (40, 40, 200)  # 红（BGR）

    skel_ys, skel_xs = np.where(channel_skeleton > 0)
    if len(skel_xs) == 0 or not mapped_strokes:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, vis)
        return

    skel_pts = np.stack([skel_xs, skel_ys], axis=1).astype(np.float64)
    valid_strokes = [np.asarray(s, dtype=np.float64) for s in mapped_strokes if len(s) >= 1]
    if not valid_strokes:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, vis)
        return

    all_pts = np.concatenate(valid_strokes, axis=0)
    tree = KDTree(skel_pts)
    dists, _ = tree.query(all_pts, k=1)

    valid_mask = dists <= float(channel_half_width)
    bad_mask = valid_mask & (dists > float(jitter_tol))
    good_mask = valid_mask & (dists <= float(jitter_tol))

    # 绘制良好点（绿，最多采样 2000 个避免过密）
    good_idx = np.where(good_mask)[0]
    if len(good_idx) > 2000:
        good_idx = np.random.choice(good_idx, 2000, replace=False)
    for i in good_idx:
        x, y = int(round(all_pts[i, 0])), int(round(all_pts[i, 1]))
        if 0 <= x < w and 0 <= y < h:
            vis[y, x] = (40, 200, 40)  # 绿

    # 绘制抖动点（黄）
    for i in np.where(bad_mask)[0]:
        x, y = int(round(all_pts[i, 0])), int(round(all_pts[i, 1]))
        if 0 <= x < w and 0 <= y < h:
            vis[y, x] = (40, 230, 230)  # 黄（BGR）

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, vis)


# ======================================================================
# [PATCH-1] 修改 extract_maze_features 函数签名
# 在现有参数之后追加以下圆形专用参数（均有默认值，方形调用时完全忽略）
# ======================================================================
"""
在 extract_maze_features 参数列表中，在 "# 迷宫几何" 参数组之后追加：

    # 圆形迷宫专用参数（game_type='maze' 时忽略）
    circle_center: Optional[Tuple[int, int]] = None,      # None → 自动最小二乘拟合
    outer_ring_radius: Optional[float] = None,             # None → 自动拟合
    entry_xy_circle: Optional[Point] = (315, 522),         # 硬编码入口（左上缺口）
    exit_xy_circle: Optional[Point] = (892, 1082),         # 硬编码出口（右下缺口）
    circle_scan_entry: bool = False,                        # True → 角度扫描检测

同时，将 channel_half_width_C1 的默认值说明改为：
    channel_half_width_C1: float = 20.0,    # 方形：20；圆形建议传入 28.0
"""


# ======================================================================
# [PATCH-2a] 修改 extract_maze_features 步骤1：build_maze_geometry 调用
# 将原调用替换为以下代码（方形分支原逻辑保留，仅增加 game_type 和圆形参数）
# ======================================================================
"""
    # ---------- 1. 构造迷宫几何 ----------
    _is_circle = (game_type == "circle")
    geom: MazeGeometry = build_maze_geometry(
        maze_mask_path,
        game_type=game_type,                            # ← 新增
        r_wall=r_wall,
        channel_half_width_px=r_solution_channel,
        # 方形参数（圆形时忽略）
        entry_corner_size=entry_corner_size,
        entry_xy=entry_xy if not _is_circle else (entry_xy_circle if entry_xy is None else entry_xy),
        exit_xy=exit_xy if not _is_circle else (exit_xy_circle if exit_xy is None else exit_xy),
        # 圆形参数（方形时忽略）
        circle_center=circle_center if _is_circle else None,
        outer_ring_radius=outer_ring_radius if _is_circle else None,
        circle_scan_entry=circle_scan_entry if _is_circle else False,
    )
"""


# ======================================================================
# [PATCH-2b] 修改 extract_maze_features 步骤6：C1 计算（方案 A）
# 完整替换原 "# ---------- 6. C1（用户笔迹自身的 Hough 线段） ----------" 块
# ======================================================================

def _build_C1_block_scheme_A() -> str:
    """返回方案 A 的 C1 计算代码片段（文档用，直接粘贴进 extract_maze_features）"""
    return '''
    # ---------- 6. C1 ----------
    if game_type == "circle":
        # ── 方案 A：圆形迷宫使用骨架距离残差法 ──────────────────────────
        # （同时也是方案 C 的圆形实现；方案 C 中此分支扩展为 else 也执行本方法）
        C1, c1_bad, c1_total = compute_C1_jitter_ratio_skeleton(
            mapped_strokes, geom.channel_skeleton,
            jitter_tol=jitter_tol,
            channel_half_width=channel_half_width_C1,
        )
        user_hough_segments = []   # 骨架法不用 Hough，保留占位（可视化判断用）
        c1_method = "skeleton_dist"
    else:
        # ── 方案 A：方形迷宫维持 Hough-on-user（现有实现）───────────────
        user_hough_segments = extract_segments_from_hough(
            user_mask, hough_params=C1_hough_params
        )
        C1, c1_bad, c1_total = compute_C1_jitter_ratio(
            mapped_strokes, user_hough_segments,
            jitter_tol=jitter_tol, channel_half_width=channel_half_width_C1,
        )
        c1_method = "hough_user"

    # 辅助诊断：有效投影点 / 笔迹总点数
    n_user_pts = int(sum(len(s) for s in mapped_strokes))
    c1_projected_fraction = (
        float(c1_total) / float(n_user_pts) if n_user_pts > 0 else 0.0
    )
'''


# ======================================================================
# [PATCH-2c] 修改 extract_maze_features 步骤8：JSON meta 新增圆形字段
# 在现有 meta dict 的 "params" 键之前插入以下内容
# ======================================================================
"""
            # 圆形迷宫专属元信息（方形时对应字段为 None，json.dumps 正常处理）
            "circle_center_xy": (
                list(geom.circle_center_xy) if geom.circle_center_xy else None
            ),
            "outer_ring_radius": (
                round(float(geom.outer_ring_radius), 2)
                if geom.outer_ring_radius is not None else None
            ),
            "num_channel_components_before_filter": (
                geom.num_channel_components_before_filter
            ),
            "C1_method": c1_method,     # "skeleton_dist" 或 "hough_user"

并在 "params" 内追加：
            "game_type": game_type,
            "circle_scan_entry": circle_scan_entry if game_type == "circle" else None,
"""


# ======================================================================
# [PATCH-2d] 修改 extract_maze_features 步骤10：可视化（圆形新增 C1_skeleton.png）
# 将原 "# 3) C1 的 Hough 叠加" 块替换为：
# ======================================================================
"""
        # 3) C1 叠加图
        if game_type == "circle":
            _visualize_C1_skeleton(
                user_mask, geom.channel_skeleton, mapped_strokes,
                jitter_tol=jitter_tol, channel_half_width=channel_half_width_C1,
                out_path=os.path.join(out_vis_dir, f"{sample_id}_C1_skeleton.png"),
            )
        else:
            _visualize_C1_hough(
                user_mask, user_hough_segments, mapped_strokes,
                jitter_tol=jitter_tol, channel_half_width=channel_half_width_C1,
                out_path=os.path.join(out_vis_dir, f"{sample_id}_C1_hough.png"),
            )
"""


# ======================================================================
# [SCHEME A→C] 方案 A 改为方案 C 的完整修改说明
# ======================================================================
"""
方案 C = 方形迷宫也改用骨架距离残差法，三游戏 C1 完全统一。
只需在 [PATCH-2b] 的 C1 计算块中，将 if/else 分支合并为一个统一调用：

【方案 C 替换 [PATCH-2b] 的全部内容为：】

    # ---------- 6. C1（方案 C：三游戏统一使用骨架距离残差法）----------
    C1, c1_bad, c1_total = compute_C1_jitter_ratio_skeleton(
        mapped_strokes, geom.channel_skeleton,
        jitter_tol=jitter_tol,
        channel_half_width=channel_half_width_C1,
    )
    user_hough_segments = []   # 方案 C 不使用 Hough，保留占位
    c1_method = "skeleton_dist"

    n_user_pts = int(sum(len(s) for s in mapped_strokes))
    c1_projected_fraction = (
        float(c1_total) / float(n_user_pts) if n_user_pts > 0 else 0.0
    )

同时将 [PATCH-2d] 的可视化也改为总是输出 C1_skeleton.png：

    # 3) C1 叠加图（方案 C：统一使用骨架可视化）
    _visualize_C1_skeleton(
        user_mask, geom.channel_skeleton, mapped_strokes,
        jitter_tol=jitter_tol, channel_half_width=channel_half_width_C1,
        out_path=os.path.join(out_vis_dir, f"{sample_id}_C1_skeleton.png"),
    )

其余部分（函数签名、步骤 1、步骤 8、JSON meta）无需改动。

注意：方案 C 下，方形迷宫历史样本的 C1 需要重新计算。
建议在特征矩阵 CSV 中记录 "C1_method" 字段以便追溯。
"""
