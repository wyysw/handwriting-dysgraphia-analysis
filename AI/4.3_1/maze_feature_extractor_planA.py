"""
features/maze_feature_extractor.py  ── 方案 A 版本

阶段 2/3 — 迷宫游戏特征提取主程序，支持方形（maze）和圆形（circle）迷宫。

方案 A（本文件采用）：
  - game_type='circle'：C1 使用骨架距离法（compute_C1_skeleton_distance_ratio）
                        因圆形迷宫理想路径含弧线，Hough 直线检测无法覆盖。
  - game_type='maze'  ：C1 沿用 Hough-on-user-mask 方案（与阶段 2 一致）。

方案 C（见 maze_feature_extractor_planC.py）：
  - 两种 game_type 均使用骨架距离法 C1，与特征定义总表"三游戏统一"完全对齐。

特征定义（方形/圆形共用）：
    F1 = |user ∩ solution_channel| / |solution_channel|         ↑
    F2 = 沿 solution_polyline 等弧长采样点的命中率              ↑
    F3 = Σ channel_dist[user & ¬channel] / |channel|             ↓
    F4 = |user ∩ ¬channel| / |user|                              ↓
    C1 = 抖动比例（方案A：圆形用骨架距离，方形用Hough）
    C2 = 短笔段比例（三游戏统一）
    C3 = 压力变异系数（三游戏统一）

依赖：
    - features.maze_geometry   ：通道几何 + 解路径（v2.1，支持 game_type）
    - features.stroke_utils    ：C1/C2/C3 + 轨迹 IO + 渲染 + Hough（公共库）
    注：stroke_utils 需已添加 compute_C1_skeleton_distance_ratio（见 stroke_utils_addition.py）
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

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
    compute_C1_jitter_ratio,                 # Hough 方案（方形迷宫 C1，方案A）
    compute_C1_skeleton_distance_ratio,      # 骨架距离方案（圆形迷宫 C1，方案A/C）
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
    """F1 = |user_mask ∩ solution_channel_mask| / |solution_channel_mask|"""
    sol   = solution_channel_mask > 0
    denom = int(sol.sum())
    if denom == 0:
        return 0.0
    return float(int(((user_mask > 0) & sol).sum())) / float(denom)


# =====================================================================
# F2：解路径采样点命中率，↑越好
# =====================================================================

def _sample_polyline_by_arc(
    polyline: np.ndarray, step: float
) -> np.ndarray:
    """
    沿有序 (x, y) polyline 按累积弧长 step 间隔采样。
    对直线和曲线均适用（圆形迷宫的曲线骨架同样有效）。
    返回 (K, 2) int32，永远包含起点；终点若与最后一个采样点距离 ≥ step/2 则也包含。
    """
    if len(polyline) == 0:
        return np.empty((0, 2), dtype=np.int32)
    if len(polyline) == 1:
        return polyline.astype(np.int32)

    pts = polyline.astype(np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if total < step:
        return np.array([polyline[0], polyline[-1]], dtype=np.int32)

    targets = np.arange(0.0, total, float(step))
    idx = np.searchsorted(cum, targets, side="right") - 1
    idx = np.clip(idx, 0, len(pts) - 2)
    t_local = (targets - cum[idx]) / np.maximum(seg[idx], 1e-9)
    sampled  = pts[idx] + t_local[:, None] * (pts[idx + 1] - pts[idx])

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
    hit：该点到用户笔迹的距离 <= hit_radius。
    返回 (F2, hit_count, total_K, sample_pts (K, 2))。
    """
    sample_pts = _sample_polyline_by_arc(solution_polyline, sample_step)
    K = len(sample_pts)
    if K == 0:
        return 0.0, 0, 0, sample_pts
    user_bin = (user_mask > 0).astype(np.uint8)
    if int(user_bin.sum()) == 0:
        return 0.0, 0, K, sample_pts

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
    illegal = user_mask & ¬channel_mask（通道外的用户像素）。
    """
    user_pts    = user_mask > 0
    channel_pts = channel_mask > 0
    illegal     = user_pts & (~channel_pts)

    illegal_dist_sum   = float(channel_dist[illegal].sum())
    illegal_pixel_count = int(illegal.sum())
    channel_area        = int(channel_pts.sum())

    F3 = illegal_dist_sum / float(channel_area) if channel_area > 0 else 0.0

    meta = {
        "illegal_dist_sum":      illegal_dist_sum,
        "illegal_pixel_count":   illegal_pixel_count,
        "n_cross_axis_pixels":   0,      # 与 sym 对齐占位，迷宫恒为 0
        "cross_penalty_coef":    0.0,
        "F3_main_contribution":  float(F3),
        "F3_cross_contribution": 0.0,
        "channel_area":          channel_area,
    }
    return float(F3), meta


# =====================================================================
# F4：通道外笔迹比，↓越好
# =====================================================================

def compute_F4_offchannel_ratio(
    user_mask: np.ndarray,
    channel_mask: np.ndarray,
) -> float:
    """F4 = |user ∩ ¬channel| / |user|"""
    user_pts = user_mask > 0
    n_user   = int(user_pts.sum())
    if n_user == 0:
        return 0.0
    outside = user_pts & (channel_mask == 0)
    return float(int(outside.sum())) / float(n_user)


# =====================================================================
# 主入口
# =====================================================================

def extract_maze_features(
    txt_path: str,
    png_path: Optional[str],
    maze_mask_path: str,
    *,
    game_type: str = "maze",             # 'maze' | 'circle'
    out_json_path: Optional[str] = None,
    sample_id: Optional[str] = None,
    out_vis_dir: Optional[str] = None,
    # 入口/出口（可手动指定）
    entry_xy: Optional[Point] = None,
    exit_xy:  Optional[Point] = None,
    # F2 参数
    sample_step: float = 40.0,
    hit_radius:  float = 12.0,
    # C1 参数
    jitter_tol: float = 3.0,
    channel_half_width_C1: float = 28.0,    # 骨架距离法（圆形A/方案C）和 Hough 方案（方形A）共用
    C1_hough_params: Optional[Dict[str, Any]] = None,   # 仅 Hough 方案（方形，方案A）使用
    # C2 / C3 参数
    C2_threshold_ratio: float = 0.02,
    C3_trim_ends: int = 3,
    # 通用
    skip_rows: int = 3,
    line_thickness: int = 3,
    # 迷宫几何
    r_wall: int = 2,
    r_solution_channel: int = 28,
    entry_corner_size: int = 105,           # 仅 game_type='maze' 使用
    # 坐标对齐 fallback：png_path=None 时用迷宫内框 bbox
    inner_bbox_fallback: bool = True,
) -> Dict[str, Any]:
    """
    主入口。对一个迷宫样本提取 7 个特征（F1-F4 + C1-C3）。

    方案 A 的 C1 分派规则：
        game_type='circle' → compute_C1_skeleton_distance_ratio（骨架距离法）
        game_type='maze'   → compute_C1_jitter_ratio（Hough-on-user 方案，与阶段 2 一致）

    参数说明：
        txt_path             : 用户轨迹（x y pressure，前 skip_rows 行为文件头）
        png_path             : 用户绘制 PNG（仅用于 bbox 对齐）；可为 None
        maze_mask_path       : maze_mask.png 或 circle_mask.png
        game_type            : 'maze' 或 'circle'
        inner_bbox_fallback  : png_path 不可用时，用迷宫内框 bbox 做坐标对齐 fallback
        out_vis_dir          : 若提供，输出三张诊断图（详见可视化节）
    """
    if sample_id is None:
        sample_id = Path(txt_path).stem

    # ---------- 1. 迷宫几何（方形/圆形由 game_type 分派）----------
    geom: MazeGeometry = build_maze_geometry(
        maze_mask_path,
        game_type=game_type,
        r_wall=r_wall,
        channel_half_width_px=r_solution_channel,
        entry_corner_size=entry_corner_size,
        entry_xy=entry_xy,
        exit_xy=exit_xy,
    )
    canvas_hw = geom.canvas_hw
    h_canvas, w_canvas = canvas_hw

    # ---------- 2. 读轨迹（含 pressure）----------
    strokes_wp   = load_strokes_with_pressure(txt_path, skip_rows=skip_rows)
    num_strokes  = len(strokes_wp)
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
                f"png_path 不存在或未提供（且 inner_bbox_fallback=False）: {png_path}"
            )
        mapped_strokes = map_strokes_to_canvas(
            strokes_xy_raw, canvas_hw, target_bbox=geom.inner_bbox
        )
        align_mode = "inner_bbox_fallback"

    # ---------- 4. 渲染用户 mask ----------
    user_mask = render_strokes_to_mask(
        mapped_strokes, canvas_hw, line_thickness=line_thickness
    )

    # ---------- 5. F1–F4 ----------
    F1 = compute_F1_solution_coverage(user_mask, geom.solution_channel_mask)
    F2, kp_hit, kp_total, sample_pts = compute_F2_keypoint_hit_rate(
        user_mask, geom.solution_polyline,
        sample_step=sample_step, hit_radius=hit_radius,
    )
    F3, F3_meta = compute_F3_invalid_drawing(
        user_mask, geom.channel_mask, geom.channel_dist
    )
    F4 = compute_F4_offchannel_ratio(user_mask, geom.channel_mask)

    # ---------- 6. C1（方案 A：按 game_type 分派）----------
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │ 方案 A                                                                  │
    # │   game_type='circle' → 骨架距离法（曲线通用）                          │
    # │   game_type='maze'   → Hough-on-user-mask 方案（方形直线路径专用）     │
    # └─────────────────────────────────────────────────────────────────────────┘
    if game_type == "circle":
        # 方案 A / 方案 C 共同用于圆形
        C1, c1_bad, c1_total = compute_C1_skeleton_distance_ratio(
            mapped_strokes,
            geom.channel_skeleton,
            jitter_tol=jitter_tol,
            channel_half_width=channel_half_width_C1,
        )
        user_hough_segments = []   # 圆形不走 Hough 路径，置空供可视化判断
        c1_method_used = "skeleton_distance"
    else:
        # 方案 A：方形保留 Hough-on-user 方案
        user_hough_segments = extract_segments_from_hough(
            user_mask, hough_params=C1_hough_params
        )
        C1, c1_bad, c1_total = compute_C1_jitter_ratio(
            mapped_strokes, user_hough_segments,
            jitter_tol=jitter_tol, channel_half_width=channel_half_width_C1,
        )
        c1_method_used = "hough_on_user"

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
        "game":      game_type,
        "F1": round(float(F1), 6),
        "F2": round(float(F2), 6),
        "F3": round(float(F3), 6),
        "F4": round(float(F4), 6),
        "C1": round(float(C1), 6),
        "C2": round(float(C2), 6),
        "C3": round(float(C3), 6),
        "meta": {
            "num_strokes":   int(num_strokes),
            "total_points":  int(total_points),
            "canvas_hw":     [int(h_canvas), int(w_canvas)],
            "inner_bbox":    list(geom.inner_bbox),
            "align_mode":    align_mode,
            # 几何
            "channel_area":             int(geom.channel_mask.sum()),
            "channel_skeleton_length":  int(geom.channel_skeleton.sum()),
            "solution_path_length_px":  int(geom.solution_length_px),
            "solution_channel_area":    int(geom.solution_channel_mask.sum()),
            "entry_xy":  list(geom.entry_xy),
            "exit_xy":   list(geom.exit_xy),
            # F2
            "num_skeleton_sample_pts": int(kp_total),
            "keypoints_hit":           int(kp_hit),
            # C1
            "C1_method":               c1_method_used,
            "C1_projected_points":     int(c1_total),
            "C1_bad_points":           int(c1_bad),
            "C1_projected_fraction":   round(c1_projected_fraction, 4),
            "num_user_hough_segments": int(len(user_hough_segments)),
            # C2
            "C2_threshold":       round(float(c2_thr), 4),
            "C2_n_short_strokes": int(c2_nshort),
            # C3
            "C3_n_pressure_points": int(c3_n),
            # F3
            "F3_detail": {
                "illegal_dist_sum":      round(float(F3_meta["illegal_dist_sum"]), 4),
                "illegal_pixel_count":   int(F3_meta["illegal_pixel_count"]),
                "n_cross_axis_pixels":   int(F3_meta["n_cross_axis_pixels"]),
                "cross_penalty_coef":    float(F3_meta["cross_penalty_coef"]),
                "F3_main_contribution":  round(float(F3_meta["F3_main_contribution"]), 6),
                "F3_cross_contribution": round(float(F3_meta["F3_cross_contribution"]), 6),
                "channel_area":          int(F3_meta["channel_area"]),
            },
            "params": {
                "sample_step":           float(sample_step),
                "hit_radius":            float(hit_radius),
                "jitter_tol":            float(jitter_tol),
                "channel_half_width_C1": float(channel_half_width_C1),
                "C2_threshold_ratio":    float(C2_threshold_ratio),
                "C3_trim_ends":          int(C3_trim_ends),
                "r_wall":                int(r_wall),
                "r_solution_channel":    int(r_solution_channel),
                "entry_corner_size":     int(entry_corner_size),
                "line_thickness":        int(line_thickness),
                "skip_rows":             int(skip_rows),
            },
        },
    }

    # 圆形迷宫追加专属 meta
    if geom.circle_meta is not None:
        result["meta"]["circle_geometry"] = geom.circle_meta

    # ---------- 9. 写盘 ----------
    if out_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_json_path)) or ".", exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # ---------- 10. 可视化 ----------
    if out_vis_dir is not None:
        os.makedirs(out_vis_dir, exist_ok=True)
        # 图 1：通道几何 + 解路径
        visualize_maze_geometry(
            geom, os.path.join(out_vis_dir, f"{sample_id}_channel_geometry.png")
        )
        # 图 2：特征叠加（通道 + 解通道 + 用户笔迹 + F2 采样点）
        _vis_feature_overlay(
            geom, user_mask, sample_pts,
            os.path.join(out_vis_dir, f"{sample_id}_feature_overlay.png"),
            hit_radius=hit_radius,
        )
        # 图 3：C1 叠加（方案A：圆形用骨架距离图，方形用Hough图）
        if game_type == "circle":
            _vis_C1_skeleton(
                user_mask, geom.channel_skeleton, mapped_strokes,
                jitter_tol=jitter_tol, channel_half_width=channel_half_width_C1,
                out_path=os.path.join(out_vis_dir, f"{sample_id}_C1_skeleton.png"),
            )
        else:
            _vis_C1_hough(
                user_mask, user_hough_segments, mapped_strokes,
                jitter_tol=jitter_tol, channel_half_width=channel_half_width_C1,
                out_path=os.path.join(out_vis_dir, f"{sample_id}_C1_hough.png"),
            )

    return result


# =====================================================================
# 可视化辅助函数
# =====================================================================

def _vis_feature_overlay(
    geom: MazeGeometry,
    user_mask: np.ndarray,
    sample_pts: np.ndarray,
    out_path: str,
    hit_radius: float = 12.0,
) -> None:
    """特征叠加图：通道(灰)+解通道(绿)+墙(白)+用户笔迹(蓝橙)+F2 采样点(绿/红)。"""
    h, w = geom.canvas_hw
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[geom.channel_mask > 0]          = (50, 50, 50)
    vis[geom.solution_channel_mask > 0] = (40, 130, 60)
    vis[geom.wall_mask > 0]             = (255, 255, 255)
    vis[user_mask > 0]                  = (255, 140, 30)

    user_bin = (user_mask > 0).astype(np.uint8)
    if int(user_bin.sum()) > 0:
        dist_user = cv2.distanceTransform((1 - user_bin) * 255, cv2.DIST_L2, 3)
    else:
        dist_user = np.full((h, w), 1e9, dtype=np.float32)

    for x, y in sample_pts:
        if not (0 <= x < w and 0 <= y < h):
            continue
        ok = dist_user[y, x] <= hit_radius
        cv2.circle(vis, (int(x), int(y)), 5, (40, 230, 40) if ok else (40, 40, 230), -1)

    cv2.circle(vis, tuple(int(v) for v in geom.entry_xy), 18, (0, 255, 80), 3)
    cv2.circle(vis, tuple(int(v) for v in geom.exit_xy),  18, (50, 50, 255), 3)
    cv2.imwrite(out_path, vis)


def _vis_C1_skeleton(
    user_mask: np.ndarray,
    channel_skeleton: np.ndarray,
    mapped_strokes: Sequence[np.ndarray],
    jitter_tol: float,
    channel_half_width: float,
    out_path: str,
) -> None:
    """
    C1 骨架距离叠加图（圆形迷宫/方案C）：
      - 灰  : 用户笔迹
      - 青  : 通道骨架（参考中心线）
      - 黄  : 被判为抖动的笔迹点（d <= channel_half_width 且 d > jitter_tol）
    """
    h, w = user_mask.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[user_mask > 0]          = (160, 160, 160)   # 灰：用户笔迹
    vis[channel_skeleton > 0]   = (200, 180, 0)     # 青：骨架

    # 计算每个笔迹点到骨架的距离，标出抖动点
    ys_sk, xs_sk = np.where(channel_skeleton > 0)
    if len(xs_sk) > 0 and mapped_strokes:
        skel_pts = np.column_stack([xs_sk, ys_sk]).astype(np.float64)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(skel_pts)
            for stroke in mapped_strokes:
                if len(stroke) == 0:
                    continue
                pts = np.asarray(stroke, dtype=np.float64)
                dists, _ = tree.query(pts, k=1)
                for i, (x, y) in enumerate(pts.astype(int)):
                    d = dists[i]
                    if d <= channel_half_width and d > jitter_tol:
                        if 0 <= x < w and 0 <= y < h:
                            vis[y, x] = (40, 230, 230)   # 黄（BGR）
        except ImportError:
            pass  # 无 scipy 时跳过标黄

    cv2.imwrite(out_path, vis)


def _vis_C1_hough(
    user_mask: np.ndarray,
    hough_segments: Sequence[Tuple[np.ndarray, np.ndarray]],
    mapped_strokes: Sequence[np.ndarray],
    jitter_tol: float,
    channel_half_width: float,
    out_path: str,
) -> None:
    """
    C1 Hough 叠加图（方形迷宫，方案 A）：
      - 灰  : 用户笔迹
      - 红线: Hough 检测到的线段
      - 黄  : 被判为抖动的笔迹点
    """
    h, w = user_mask.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[user_mask > 0] = (160, 160, 160)

    for a, b in hough_segments:
        cv2.line(vis,
                 (int(round(a[0])), int(round(a[1]))),
                 (int(round(b[0])), int(round(b[1]))),
                 (40, 40, 230), 2)

    if mapped_strokes and hough_segments:
        all_pts = np.concatenate(
            [np.asarray(s, dtype=np.float64) for s in mapped_strokes if len(s) >= 1], axis=0
        )
        N = len(all_pts)
        best_d = np.full(N, np.inf, dtype=np.float64)
        for a, b in hough_segments:
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            ab   = b - a
            denom = float(np.dot(ab, ab))
            if denom < 1e-9:
                continue
            ap = all_pts - a
            t  = (ap @ ab) / denom
            t_clip = np.clip(t, 0.0, 1.0)
            proj = a + t_clip[:, None] * ab
            d = np.linalg.norm(all_pts - proj, axis=1)
            ok = (t >= -0.05) & (t <= 1.05) & (d <= channel_half_width)
            best_d = np.where(ok & (d < best_d), d, best_d)
        for i in np.where((best_d > jitter_tol) & (best_d < np.inf))[0]:
            x, y = all_pts[i].astype(int)
            if 0 <= x < w and 0 <= y < h:
                vis[y, x] = (40, 230, 230)

    cv2.imwrite(out_path, vis)


# =====================================================================
# 命令行
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="迷宫游戏特征提取器（方案A：圆形用骨架C1）")
    parser.add_argument("--txt",       required=True)
    parser.add_argument("--png",       default=None)
    parser.add_argument("--mask",      required=True)
    parser.add_argument("--out",       default=None)
    parser.add_argument("--vis_dir",   default=None)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--game",      default="maze", choices=["maze", "circle"])
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
