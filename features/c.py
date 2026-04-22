"""
features/maze_feature_extractor.py

阶段 2 — 迷宫游戏特征提取主程序（4.3 + 4.3.3）。

对一个迷宫样本（方形 / 圆形）提取 7 个标准化特征（F1-F4 + C1-C3）。
本模块只输出"原始指标"（0-1 比例或连续量），归一化由阶段 3 处理。

依赖：
    - features.maze_geometry   ：通道几何 + 解路径
    - features.stroke_utils    ：C1/C2/C3 + 轨迹 IO + 渲染 + Hough（与 sym 共用）

特征定义（方形迷宫）：
    F1 = |user ∩ solution_channel| / |solution_channel|         ↑
    F2 = 沿 solution_polyline 等弧长采样点的命中率              ↑
    F3 = Σ channel_dist[user & ¬channel] / |channel|             ↓  （Q2 已确认：分母用全通道）
    F4 = |user ∩ ¬channel| / |user|                              ↓
    C1, C2, C3：调用 stroke_utils（C1 的 Hough 来源是用户笔迹本身）

JSON 输出结构与对称游戏对齐，meta 中额外记录迷宫几何相关诊断字段。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from scipy.spatial import KDTree as _KDTree
except ImportError as _e:
    raise ImportError(
        "骨架距离 C1 需要 scipy：pip install scipy"
    ) from _e

# ---------------------------------------------------------------
# 允许从 features/ 子目录直接运行时找到项目根
# ---------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from features.maze_geometry import (
    MazeGeometry,
    build_maze_geometry,
    visualize_maze_geometry,
)
from features.stroke_utils import (
    compute_C1_jitter_ratio,
    compute_C2_short_stroke_ratio,
    compute_C3_pressure_cv,
    extract_segments_from_hough,
    load_strokes_with_pressure,
    map_strokes_to_canvas,
    render_strokes_to_mask,
    strokes_to_xy_arrays,
)

Point = Tuple[int, int]


# =====================================================================
# F1：通道覆盖率（解路径），↑越好
# =====================================================================

def compute_F1_solution_coverage(
    user_mask: np.ndarray,
    solution_channel_mask: np.ndarray,
) -> float:
    """
    F1 = |user_mask ∩ solution_channel_mask| / |solution_channel_mask|
    走错岔路不会提升 F1（岔道不在 solution_channel 里），
    沿解路径走得越远 F1 越大。
    """
    sol = (solution_channel_mask > 0)
    denom = int(sol.sum())
    if denom == 0:
        return 0.0
    inside = (user_mask > 0) & sol
    return float(int(inside.sum())) / float(denom)


# =====================================================================
# F2：解路径采样点命中率，↑越好
# =====================================================================

def _sample_polyline_by_arc(
    polyline: np.ndarray, step: float
) -> np.ndarray:
    """
    沿有序 (x, y) polyline 按累积弧长 step 间隔采样。
    返回 (K, 2) int32 数组，永远包含起点；终点若与最后一个采样点距离 ≥ step/2 则也包含。
    """
    if len(polyline) == 0:
        return np.empty((0, 2), dtype=np.int32)
    if len(polyline) == 1:
        return polyline.astype(np.int32)

    pts = polyline.astype(np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)  # (N-1,)
    cum = np.concatenate(([0.0], np.cumsum(seg)))       # (N,)
    total = float(cum[-1])
    if total < step:
        return np.array([polyline[0], polyline[-1]], dtype=np.int32)

    # 目标弧长
    targets = np.arange(0.0, total, float(step))
    # 每个 target 在 cum 中的插入位置
    idx = np.searchsorted(cum, targets, side="right") - 1
    idx = np.clip(idx, 0, len(pts) - 2)
    # 在该段内做线性插值
    t_local = (targets - cum[idx]) / np.maximum(seg[idx], 1e-9)
    sampled = pts[idx] + t_local[:, None] * (pts[idx + 1] - pts[idx])

    # 末端补一个
    if total - targets[-1] >= step / 2.0:
        sampled = np.vstack([sampled, pts[-1]])
    return np.round(sampled).astype(np.int32)


def compute_F2_keypoint_hit_rate(
    user_mask: np.ndarray,
    solution_polyline: np.ndarray,
    sample_step: float = 40.0,
    hit_radius: float = 12.0,
) -> Tuple[float, int, int, np.ndarray]:
    """
    沿 solution_polyline 按 sample_step 像素采样 K 个点；
    用 distanceTransform(~user_mask) 取每个采样点到用户笔迹的距离 d；
    命中：d <= hit_radius。
    返回 (F2, hit_count, total_K, sample_pts (K, 2))。
    """
    sample_pts = _sample_polyline_by_arc(solution_polyline, sample_step)
    K = len(sample_pts)
    if K == 0:
        return 0.0, 0, 0, sample_pts
    user_bin = (user_mask > 0).astype(np.uint8)
    if int(user_bin.sum()) == 0:
        return 0.0, 0, K, sample_pts
    # distanceTransform: 输入前景=非0，输出每个非 user 像素到 user 的距离；
    # 我们要每个采样点到最近 user 像素的距离，所以输入是 (1 - user)*255。
    dist_user = cv2.distanceTransform((1 - user_bin) * 255, cv2.DIST_L2, 3)
    h, w = dist_user.shape[:2]
    hit = 0
    for x, y in sample_pts:
        if 0 <= x < w and 0 <= y < h and dist_user[y, x] <= hit_radius:
            hit += 1
    return float(hit) / float(K), int(hit), int(K), sample_pts


# =====================================================================
# F3：无效书写加权距离，↓越好
# =====================================================================

def compute_F3_invalid_drawing(
    user_mask: np.ndarray,
    channel_mask: np.ndarray,
    channel_dist: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    """
    F3 = Σ channel_dist[illegal] / |channel_mask|

    其中 illegal = user_mask & ¬channel_mask（落在通道外的用户像素）。
    channel_dist 已经是"该像素到最近通道像素的 L2 距离"，
    所以 channel_dist 在 illegal 上的求和自然体现了
    "错得多远就罚多重"——穿墙一格和画到纸外的惩罚力度自动拉开。

    Q2 已确认：分母用全通道 channel_mask 面积，更稳定。

    迷宫无对称轴，故无 cross_penalty。
    """
    user_pts = (user_mask > 0)
    channel_pts = (channel_mask > 0)
    illegal = user_pts & (~channel_pts)
    illegal_dist_sum = float(channel_dist[illegal].sum())
    illegal_pixel_count = int(illegal.sum())

    channel_area = int(channel_pts.sum())
    if channel_area == 0:
        F3 = 0.0
    else:
        F3 = illegal_dist_sum / float(channel_area)

    meta = {
        "illegal_dist_sum": illegal_dist_sum,
        "illegal_pixel_count": illegal_pixel_count,
        "n_cross_axis_pixels": 0,             # 与 sym 对齐占位，迷宫恒为 0
        "cross_penalty_coef": 0.0,            # 同上
        "F3_main_contribution": float(F3),
        "F3_cross_contribution": 0.0,
        "channel_area": channel_area,
    }
    return float(F3), meta


# =====================================================================
# F4：通道外笔迹比，↓越好
# =====================================================================

def compute_F4_offchannel_ratio(
    user_mask: np.ndarray,
    channel_mask: np.ndarray,
) -> float:
    """F4 = |user ∩ ¬channel| / |user|。"""
    user_pts = (user_mask > 0)
    n_user = int(user_pts.sum())
    if n_user == 0:
        return 0.0
    outside = user_pts & (channel_mask == 0)
    return float(int(outside.sum())) / float(n_user)



# ======================================================================
# [NEW FUNCTION] 新增骨架距离残差 C1 计算函数
# 插入位置：compute_F4_offchannel_ratio 之后，extract_maze_features 之前
# ======================================================================

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



# =====================================================================
# 主入口
# =====================================================================

def extract_maze_features(
    txt_path: str,
    png_path: Optional[str],
    maze_mask_path: str,
    *,
    game_type: str = "maze",
    out_json_path: Optional[str] = None,
    sample_id: Optional[str] = None,
    # 入口/出口（可手动指定）
    entry_xy: Optional[Point] = None,
    exit_xy: Optional[Point] = None,
    # F1-F4 参数
    sample_step: float = 40.0,
    hit_radius: float = 12.0,
    # C1-C3 参数
    jitter_tol: float = 3.0,
    channel_half_width_C1: float = 20.0,    # 方形：20；圆形建议传入 28.0
    C1_hough_params: Optional[Dict[str, Any]] = None,
    C2_threshold_ratio: float = 0.02,
    C3_trim_ends: int = 3,
    # 通用
    skip_rows: int = 3,
    line_thickness: int = 3,
    # 迷宫几何
    r_wall: int = 2,
    r_solution_channel: int = 28,            # 实测通道半宽 ≈ 28 px
    entry_corner_size: int = 105,
    # 圆形迷宫专用参数（game_type='maze' 时忽略）
    circle_center: Optional[Tuple[int, int]] = None,      # None → 自动最小二乘拟合
    outer_ring_radius: Optional[float] = None,             # None → 自动拟合
    entry_xy_circle: Optional[Point] = (315, 522),         # 硬编码入口（左上缺口）
    exit_xy_circle: Optional[Point] = (892, 1082),         # 硬编码出口（右下缺口）
    circle_scan_entry: bool = False,                        # True → 角度扫描检测
    # 坐标对齐 fallback：当 png_path=None 时用迷宫内框 bbox
    inner_bbox_fallback: bool = True,
    # 可视化（可选）
    out_vis_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    主入口。对一个迷宫游戏样本提取 7 个特征。

    参数：
        txt_path             : 用户轨迹文件（x y pressure，前 skip_rows 行为文件头）
        png_path             : 用户绘制 PNG（仅用于 bbox 坐标参照）；可为 None
        maze_mask_path       : maze_mask.png（前景=墙壁）
        sample_id            : 样本 id；None 时用 txt 文件名 stem
        entry_xy / exit_xy   : 手动指定入口/出口，None 时自动检测
        其他参数             : 见各特征函数说明
        inner_bbox_fallback  : True 时若 png_path=None 则用迷宫内框 bbox 对齐
                               False 时强制要求 png_path
        out_vis_dir          : 若提供，会输出三张诊断图到该目录

    返回 dict（与 sym 对齐的 JSON 结构）。
    """
    if sample_id is None:
        sample_id = Path(txt_path).stem

    # ---------- 1. 构造迷宫几何 ----------
    # 将原调用替换为以下代码（方形分支原逻辑保留，仅增加 game_type 和圆形参数）
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
    canvas_hw = geom.canvas_hw
    h_canvas, w_canvas = canvas_hw

    # ---------- 2. 读轨迹（含 pressure） ----------
    strokes_wp = load_strokes_with_pressure(txt_path, skip_rows=skip_rows)
    num_strokes = len(strokes_wp)
    total_points = int(sum(len(s["x"]) for s in strokes_wp))
    strokes_xy_raw = strokes_to_xy_arrays(strokes_wp)

    # ---------- 3. 坐标映射 ----------
    if png_path is not None and os.path.exists(png_path):
        mapped_strokes = map_strokes_to_canvas(
            strokes_xy_raw, canvas_hw, reference_png_path=png_path
        )
        align_mode = "png_bbox"
    else:
        if not inner_bbox_fallback:
            raise FileNotFoundError(
                f"png_path 不存在或未提供：{png_path}（且 inner_bbox_fallback=False）"
            )
        mapped_strokes = map_strokes_to_canvas(
            strokes_xy_raw, canvas_hw, target_bbox=geom.inner_bbox
        )
        align_mode = "inner_bbox_fallback"

    # ---------- 4. 渲染用户 mask ----------
    user_mask = render_strokes_to_mask(
        mapped_strokes, canvas_hw, line_thickness=line_thickness
    )

    # ---------- 5. F1-F4 ----------
    F1 = compute_F1_solution_coverage(user_mask, geom.solution_channel_mask)
    F2, kp_hit, kp_total, sample_pts = compute_F2_keypoint_hit_rate(
        user_mask, geom.solution_polyline,
        sample_step=sample_step, hit_radius=hit_radius,
    )
    F3, F3_meta = compute_F3_invalid_drawing(
        user_mask, geom.channel_mask, geom.channel_dist
    )
    F4 = compute_F4_offchannel_ratio(user_mask, geom.channel_mask)

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

    # ---------- 7. C2 / C3 ----------
    C2, c2_thr, c2_nshort = compute_C2_short_stroke_ratio(
        mapped_strokes, canvas_hw=canvas_hw, threshold_ratio=C2_threshold_ratio,
    )
    C3, c3_n = compute_C3_pressure_cv(strokes_wp, trim_ends=C3_trim_ends)

    # ---------- 8. 组装 JSON ----------
    result: Dict[str, Any] = {
        "sample_id": sample_id,
        "game": game_type,
        "F1": round(float(F1), 6),
        "F2": round(float(F2), 6),
        "F3": round(float(F3), 6),
        "F4": round(float(F4), 6),
        "C1": round(float(C1), 6),
        "C2": round(float(C2), 6),
        "C3": round(float(C3), 6),
        "meta": {
            "num_strokes": int(num_strokes),
            "total_points": int(total_points),
            "canvas_hw": [int(h_canvas), int(w_canvas)],
            "inner_bbox": list(geom.inner_bbox),
            "align_mode": align_mode,
            # 几何
            "channel_area": int(geom.channel_mask.sum()),
            "channel_skeleton_length": int(geom.channel_skeleton.sum()),
            "solution_path_length_px": int(geom.solution_length_px),
            "solution_channel_area": int(geom.solution_channel_mask.sum()),
            "entry_xy": list(geom.entry_xy),
            "exit_xy": list(geom.exit_xy),
            # F2
            "num_skeleton_sample_pts": int(kp_total),
            "keypoints_hit": int(kp_hit),
            # C1
            "num_user_hough_segments": int(len(user_hough_segments)),
            "C1_projected_points": int(c1_total),
            "C1_bad_points": int(c1_bad),
            "C1_projected_fraction": round(c1_projected_fraction, 4),
            # C2
            "C2_threshold": round(float(c2_thr), 4),
            "C2_n_short_strokes": int(c2_nshort),
            # C3
            "C3_n_pressure_points": int(c3_n),
            # F3 详情（与 sym 对齐）
            "F3_detail": {
                "illegal_dist_sum": round(float(F3_meta["illegal_dist_sum"]), 4),
                "illegal_pixel_count": int(F3_meta["illegal_pixel_count"]),
                "n_cross_axis_pixels": int(F3_meta["n_cross_axis_pixels"]),
                "cross_penalty_coef": float(F3_meta["cross_penalty_coef"]),
                "F3_main_contribution": round(float(F3_meta["F3_main_contribution"]), 6),
                "F3_cross_contribution": round(float(F3_meta["F3_cross_contribution"]), 6),
                "channel_area": int(F3_meta["channel_area"]),
            },
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
            "params": {
                "sample_step": float(sample_step),
                "hit_radius": float(hit_radius),
                "jitter_tol": float(jitter_tol),
                "channel_half_width_C1": float(channel_half_width_C1),
                "C2_threshold_ratio": float(C2_threshold_ratio),
                "C3_trim_ends": int(C3_trim_ends),
                "r_wall": int(r_wall),
                "r_solution_channel": int(r_solution_channel),
                "entry_corner_size": int(entry_corner_size),
                "line_thickness": int(line_thickness),
                "skip_rows": int(skip_rows),
                "game_type": game_type,
                "circle_scan_entry": circle_scan_entry if game_type == "circle" else None,
            },
        },
    }

    # ---------- 9. 写盘 ----------
    if out_json_path:
        out_dir = os.path.dirname(os.path.abspath(out_json_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # ---------- 10. 可视化（可选） ----------
    if out_vis_dir is not None:
        os.makedirs(out_vis_dir, exist_ok=True)
        # 1) 通道几何 + 解路径
        visualize_maze_geometry(
            geom, os.path.join(out_vis_dir, f"{sample_id}_channel_geometry.png")
        )
        # 2) 特征叠加图（用户笔迹 + 全通道 + 解通道 + 采样点）
        _visualize_feature_overlay(
            geom, user_mask, sample_pts,
            os.path.join(out_vis_dir, f"{sample_id}_feature_overlay.png"),
            hit_radius=hit_radius,
        )
    # 3) C1 叠加图（方案 C：统一使用骨架可视化）
    _visualize_C1_skeleton(
        user_mask, geom.channel_skeleton, mapped_strokes,
        jitter_tol=jitter_tol, channel_half_width=channel_half_width_C1,
        out_path=os.path.join(out_vis_dir, f"{sample_id}_C1_skeleton.png"),
    )

    return result


# =====================================================================
# 可视化
# =====================================================================

def _visualize_feature_overlay(
    geom: MazeGeometry,
    user_mask: np.ndarray,
    sample_pts: np.ndarray,
    out_path: str,
    hit_radius: float = 12.0,
) -> None:
    """
    特征叠加图（feature_overlay.png）：
      - 灰   : 全通道 channel_mask
      - 浅绿 : 解路径通道 solution_channel_mask
      - 蓝   : 用户笔迹 user_mask
      - 绿/红圆点 : 命中 / 未命中的解路径采样点
      - 白   : 墙壁
    """
    h, w = geom.canvas_hw
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[geom.channel_mask > 0] = (50, 50, 50)
    vis[geom.solution_channel_mask > 0] = (40, 130, 60)
    vis[geom.wall_mask > 0] = (255, 255, 255)
    vis[user_mask > 0] = (255, 140, 30)  # 蓝 (BGR)

    # 采样点命中判定
    user_bin = (user_mask > 0).astype(np.uint8)
    if int(user_bin.sum()) > 0:
        dist_user = cv2.distanceTransform((1 - user_bin) * 255, cv2.DIST_L2, 3)
    else:
        dist_user = np.full((h, w), 1e9, dtype=np.float32)
    for x, y in sample_pts:
        if not (0 <= x < w and 0 <= y < h):
            continue
        ok = dist_user[y, x] <= hit_radius
        color = (40, 230, 40) if ok else (40, 40, 230)
        cv2.circle(vis, (int(x), int(y)), 5, color, -1)

    # entry / exit
    cv2.circle(vis, tuple(int(v) for v in geom.entry_xy), 18, (0, 255, 80), 3)
    cv2.circle(vis, tuple(int(v) for v in geom.exit_xy), 18, (50, 50, 255), 3)

    cv2.imwrite(out_path, vis)


def _visualize_C1_hough(
    user_mask: np.ndarray,
    hough_segments: Sequence[Tuple[np.ndarray, np.ndarray]],
    mapped_strokes: Sequence[np.ndarray],
    jitter_tol: float,
    channel_half_width: float,
    out_path: str,
) -> None:
    """
    C1 Hough 叠加图：
      - 灰   : 用户笔迹 user_mask
      - 红线 : Hough 检测到的线段
      - 黄点 : C1 中被判为"抖动"的点（投影距离 > jitter_tol）
    """
    h, w = user_mask.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[user_mask > 0] = (160, 160, 160)

    for a, b in hough_segments:
        cv2.line(vis,
                 (int(round(a[0])), int(round(a[1]))),
                 (int(round(b[0])), int(round(b[1]))),
                 (40, 40, 230), 2)

    # 找抖动点
    if mapped_strokes and hough_segments:
        all_pts = np.concatenate(
            [np.asarray(s, dtype=np.float64) for s in mapped_strokes if len(s) >= 1],
            axis=0,
        )
        N = len(all_pts)
        best_d = np.full(N, np.inf, dtype=np.float64)
        for a, b in hough_segments:
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom < 1e-9:
                continue
            ap = all_pts - a
            t = (ap @ ab) / denom
            t_clip = np.clip(t, 0.0, 1.0)
            proj = a + t_clip[:, None] * ab
            d = np.linalg.norm(all_pts - proj, axis=1)
            ok = (t >= -0.05) & (t <= 1.05) & (d <= channel_half_width)
            update = ok & (d < best_d)
            best_d = np.where(update, d, best_d)
        bad_idx = np.where((best_d > jitter_tol) & (best_d < np.inf))[0]
        for i in bad_idx:
            x, y = all_pts[i]
            if 0 <= x < w and 0 <= y < h:
                vis[int(y), int(x)] = (40, 230, 230)  # 黄

    cv2.imwrite(out_path, vis)


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



# =====================================================================
# 命令行
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="迷宫游戏特征提取器")
    parser.add_argument("--txt", required=True, help="用户轨迹 txt")
    parser.add_argument("--png", default=None, help="用户绘制 PNG（可选）")
    parser.add_argument("--mask", required=True, help="maze_mask.png 路径")
    parser.add_argument("--out", default=None, help="JSON 输出路径")
    parser.add_argument("--vis_dir", default=None, help="可视化目录（输出三张图）")
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--game", default="maze")
    args = parser.parse_args()

    res = extract_maze_features(
        txt_path=args.txt,
        png_path=args.png,
        maze_mask_path=args.mask,
        game_type=args.game,
        out_json_path=args.out,
        sample_id=args.sample_id,
        out_vis_dir=args.vis_dir,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))