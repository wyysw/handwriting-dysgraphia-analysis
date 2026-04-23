"""
features/sym_feature_extractor.py

对称游戏特征提取器（阶段 1）。
对一个对称游戏样本，提取 7 个标准化特征（F1–F4, C1–C3），输出为 dict / JSON。

本模块只输出"原始指标"（0-1 比例或连续量），不做加权/不做打分。
加权与归一化统一放到阶段 3 的 build_feature_matrix.py 中。

可复用的基础设施从 sym_core.sym_analyze3 导入（不重复实现）。
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
# 允许从 features/ 子目录直接运行时找到项目根
# ---------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sym_core.sym_analyze3 import (
    HelperGeometry,
    _extract_target_segments_from_hough,
    bbox_from_mask,
    detect_helper_geometry,
    distance_transform_to_mask,
    extract_keypoints_from_target,
    map_trajectory_strokes_using_reference_bbox,
    read_binary_mask,
    reflect_mask_across_horizontal_axis,
    reflect_strokes_across_horizontal_axis,
    render_trajectory_using_reference_bbox,
)
from pen import analyze as _pen_analyze

Point = Tuple[int, int]


# =====================================================================
# 轨迹读入（保留 pressure，C3 需要）
# =====================================================================

def _load_strokes_with_pressure(
    txt_path: str, skip_rows: int = 3
) -> List[Dict[str, np.ndarray]]:
    """
    返回 list of dict: 每个 dict 含键 'x', 'y', 'pressure'（均为 np.ndarray）。
    坐标是原始设备坐标（未映射到画布）。
    """
    data = _pen_analyze.load_trajectory_data(txt_path, skip_rows=skip_rows)
    if data is None:
        raise RuntimeError(f"[sym_feat] 无法读取轨迹文件: {txt_path}")
    return _pen_analyze.split_into_strokes_simple(data)


# =====================================================================
# F1：翻转容差 F1（IoU-like），↑越好
# =====================================================================

def compute_F1_completion(
    user_reflected_mask: np.ndarray,
    target_mask: np.ndarray,
    dilation_radius: int = 5,
) -> float:
    """
    形状完成度：将 target 和 user（已翻转到 target 同侧）各自膨胀 dilation_radius 后，
    计算 P = mean(tgt_d[user>0]), R = mean(user_d[tgt>0]), F1 = 2PR/(P+R)。
    与画布原尺寸对齐（不做 normalize_mask 缩放），保留大小差异信息。
    """
    user_bin = (user_reflected_mask > 0).astype(np.uint8)
    tgt_bin = (target_mask > 0).astype(np.uint8)
    if int(user_bin.sum()) == 0 or int(tgt_bin.sum()) == 0:
        return 0.0

    k_size = max(3, int(dilation_radius) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    user_d = cv2.dilate(user_bin * 255, kernel)
    tgt_d = cv2.dilate(tgt_bin * 255, kernel)

    user_pts = user_bin > 0
    tgt_pts = tgt_bin > 0
    precision = float((tgt_d[user_pts] > 0).mean())  # 用户点落入 tgt 容差域的比例
    recall = float((user_d[tgt_pts] > 0).mean())      # tgt 点被用户容差域覆盖的比例
    denom = precision + recall
    if denom < 1e-9:
        return 0.0
    return float(2.0 * precision * recall / denom)


# =====================================================================
# F2：网格关键点命中率，↑越好
# =====================================================================

def compute_F2_keypoint_coverage(
    user_reflected_mask: np.ndarray,
    keypoints: Sequence[Point],
    hit_radius: float = 6.0,
) -> Tuple[float, int, int]:
    """
    命中定义：用户翻转笔迹到关键点的最近距离 <= hit_radius（像素）。
    返回 (覆盖率, 命中数, 总点数)。
    """
    total = len(keypoints)
    if total == 0:
        return 0.0, 0, 0
    user_bin = (user_reflected_mask > 0).astype(np.uint8)
    if int(user_bin.sum()) == 0:
        return 0.0, 0, total
    dist = distance_transform_to_mask(user_bin)
    hit = 0
    for x, y in keypoints:
        if 0 <= y < dist.shape[0] and 0 <= x < dist.shape[1] and dist[y, x] <= hit_radius:
            hit += 1
    return float(hit) / float(total), int(hit), int(total)


# =====================================================================
# F3：无效书写加权距离 + 越轴惩罚，↓越好
# =====================================================================

def compute_F3_invalid_drawing(
    user_mask_raw: np.ndarray,      # 用户 mask，未翻转（画布原坐标）
    target_mask: np.ndarray,
    valid_zone_mask: np.ndarray,    # target 膨胀 tol_valid 得到
    axis_y: int,
    cross_penalty_coef: float = 10.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    F3 = [ Σ_{p in illegal_in_reflected} d(p)  +  cross_penalty_coef * N_cross ] / |target|

    步骤：
      1. 将用户 mask 按 axis_y 切成 "下半（应画区）" 和 "上半（越轴区）"
      2. 只把"下半"翻转到上半（target 所在区），得到 reflected_valid
      3. illegal = reflected_valid AND NOT valid_zone_mask
         F3_main  = Σ dist_outside[illegal] / |target|
         其中 dist_outside[p] = p 到最近 valid_zone 像素的距离
      4. cross_pixels = 用户原 mask 中 y <= axis_y 的像素数
         F3_cross = cross_penalty_coef * cross_pixels / |target|
      5. F3 = F3_main + F3_cross

    同时返回 detail meta 供诊断。
    """
    h, w = user_mask_raw.shape[:2]
    user_bin = (user_mask_raw > 0).astype(np.uint8)

    # --- 切分上下半 ---
    lower_mask = np.zeros_like(user_bin)
    if axis_y + 1 < h:
        lower_mask[axis_y + 1 :, :] = 1
    upper_mask = 1 - lower_mask

    user_valid_half = user_bin & lower_mask           # 正确半区（下半）
    user_cross_axis = user_bin & upper_mask           # 越轴部分（上半）
    n_cross = int(user_cross_axis.sum())

    # --- 正确半区翻转到 target 同侧 ---
    reflected_valid = reflect_mask_across_horizontal_axis(user_valid_half, int(axis_y))
    reflected_valid = (reflected_valid > 0).astype(np.uint8)

    # --- valid_zone 外的距离变换 ---
    valid_u8 = (valid_zone_mask > 0).astype(np.uint8)
    # distanceTransform 需要前景=0、背景=255 的输入；这里将 "非 valid" 设为 255，
    # 然后返回的是每个非 valid 像素到最近 valid 像素的距离；valid 像素本身距离=0。
    dist_outside = cv2.distanceTransform((1 - valid_u8) * 255, cv2.DIST_L2, 3)

    illegal = (reflected_valid > 0) & (valid_u8 == 0)
    illegal_dist_sum = float(dist_outside[illegal].sum())
    illegal_pixel_count = int(illegal.sum())

    target_area = int((target_mask > 0).sum())
    if target_area == 0:
        return 0.0, {
            "illegal_dist_sum": 0.0,
            "illegal_pixel_count": 0,
            "n_cross_axis_pixels": n_cross,
            "cross_penalty_coef": float(cross_penalty_coef),
            "F3_main_contribution": 0.0,
            "F3_cross_contribution": 0.0,
            "target_area": 0,
        }

    F3_main = illegal_dist_sum / float(target_area)
    F3_cross = cross_penalty_coef * float(n_cross) / float(target_area)
    F3 = F3_main + F3_cross

    meta = {
        "illegal_dist_sum": illegal_dist_sum,
        "illegal_pixel_count": illegal_pixel_count,
        "n_cross_axis_pixels": n_cross,
        "cross_penalty_coef": float(cross_penalty_coef),
        "F3_main_contribution": float(F3_main),
        "F3_cross_contribution": float(F3_cross),
        "target_area": target_area,
    }
    return float(F3), meta


# =====================================================================
# F4：路径偏离比，↓越好
# =====================================================================

def compute_F4_offpath_ratio(
    user_reflected_mask: np.ndarray,
    valid_zone_mask: np.ndarray,
) -> float:
    """
    F4 = |reflected_user AND NOT valid_zone| / |reflected_user|
    说明：此处使用"整张翻转后的 user_mask"（含越轴部分翻转后的像素）。
          越轴像素翻转后落在下半区，自然与 valid_zone(在上半) 不相交，
          从而也会对 F4 产生惩罚——这是期望行为。
    """
    user_pts = (user_reflected_mask > 0)
    n_user = int(user_pts.sum())
    if n_user == 0:
        return 0.0
    inside = user_pts & (valid_zone_mask > 0)
    return 1.0 - float(inside.sum()) / float(n_user)


# =====================================================================
# C1：抖动比例（点级）,↓越好
# =====================================================================

def compute_C1_jitter_ratio(
    reflected_strokes: Sequence[np.ndarray],
    hough_segments: Sequence[Tuple[np.ndarray, np.ndarray]],
    jitter_tol: float = 3.0,
    channel_half_width: float = 10.0,
) -> Tuple[float, int, int]:
    """
    对每个翻转后的用户点：
      1. 遍历所有霍夫线段，计算点到线段的投影参数 t 和垂直距离 d
      2. 只保留 t ∈ [-0.05, 1.05] 且 d <= channel_half_width 的点
         —— 称为"命中某条目标线段附近的投影点"
      3. 在所有命中线段中，取 d 最小的为该点的归属
    然后：
      total = 被纳入投影的点数
      bad   = 其中 d > jitter_tol 的点数
      C1    = bad / total  （点级平均，不按段加权）

    返回 (C1, bad, total)。
    """
    if not reflected_strokes or not hough_segments:
        return 0.0, 0, 0

    # 把所有点合并成一个大数组 (N, 2)
    pts_list = [np.asarray(s, dtype=np.float64) for s in reflected_strokes if len(s) >= 1]
    if not pts_list:
        return 0.0, 0, 0
    all_pts = np.concatenate(pts_list, axis=0)
    N = len(all_pts)
    if N == 0:
        return 0.0, 0, 0

    best_d = np.full(N, np.inf, dtype=np.float64)
    for a, b in hough_segments:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-9:
            continue
        ap = all_pts - a                                         # (N, 2)
        t = (ap @ ab) / denom                                    # (N,)
        t_clip = np.clip(t, 0.0, 1.0)
        proj = a + t_clip[:, None] * ab                          # (N, 2)
        d = np.linalg.norm(all_pts - proj, axis=1)               # (N,)
        ok = (t >= -0.05) & (t <= 1.05) & (d <= channel_half_width)
        update = ok & (d < best_d)
        best_d = np.where(update, d, best_d)

    projected = best_d < np.inf
    total = int(projected.sum())
    if total == 0:
        return 0.0, 0, 0
    bad = int(((best_d > jitter_tol) & projected).sum())
    return float(bad) / float(total), bad, total


# =====================================================================
# C2：短笔段比例，↓越好
# =====================================================================

def compute_C2_short_stroke_ratio(
    mapped_strokes_xy: Sequence[np.ndarray],
    canvas_shape_hw: Tuple[int, int],
    threshold: Optional[float] = None,
    threshold_ratio: float = 0.02,
) -> Tuple[float, float, int]:
    """
    笔段长度 = 相邻点欧氏距离之和（弧长）
    默认阈值 thr = 画布对角线 × threshold_ratio (= 0.02)
    C2 = sum(len_i for len_i < thr) / sum(all len_i)

    返回 (C2, used_threshold, n_short_strokes)
    """
    h, w = canvas_shape_hw
    diag = float(np.hypot(h, w))
    thr = float(threshold) if threshold is not None else diag * float(threshold_ratio)

    lens: List[float] = []
    for s in mapped_strokes_xy:
        arr = np.asarray(s, dtype=np.float64)
        if len(arr) < 2:
            lens.append(0.0)
            continue
        d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        lens.append(float(d.sum()))

    total_len = float(sum(lens))
    if total_len < 1e-9:
        return 0.0, thr, 0
    short_len = sum(l for l in lens if l < thr)
    n_short = sum(1 for l in lens if l < thr and l > 0)
    return float(short_len) / total_len, thr, int(n_short)


# =====================================================================
# C3：压力变异系数，↓越好
# =====================================================================

def compute_C3_pressure_cv(
    strokes_with_pressure: Sequence[Dict[str, np.ndarray]],
    trim_ends: int = 3,
) -> Tuple[float, int]:
    """
    对每条笔段，去掉首尾 trim_ends 个点后，保留 pressure > 0 的点；
    然后在所有点上计算 σ(p) / μ(p)。

    返回 (C3, n_points_used)。
    """
    collected: List[np.ndarray] = []
    for s in strokes_with_pressure:
        p = np.asarray(s.get("pressure", []), dtype=np.float64)
        if len(p) <= 2 * trim_ends:
            continue
        p_trim = p[trim_ends : len(p) - trim_ends]
        p_trim = p_trim[p_trim > 0]
        if len(p_trim) > 0:
            collected.append(p_trim)
    if not collected:
        return 0.0, 0
    all_p = np.concatenate(collected)
    n = int(len(all_p))
    mu = float(all_p.mean())
    if mu < 1e-9:
        return 0.0, n
    sigma = float(all_p.std(ddof=0))
    return sigma / mu, n


# =====================================================================
# 主入口
# =====================================================================

def extract_sym_features(
    txt_path: str,
    png_path: str,
    blue_mask_path: str,
    helper_mask_path: str,
    out_json_path: Optional[str] = None,
    *,
    sample_id: Optional[str] = None,
    # 参数（均可调）
    dilation_F1: int = 5,
    tol_valid: int = 9,
    hit_radius: float = 6.0,
    jitter_tol: float = 3.0,
    C1_channel_half_width: Optional[float] = None,  # None => 自适应 max(10, min(step)*0.20)
    C2_threshold_ratio: float = 0.02,
    C3_trim_ends: int = 3,
    F3_cross_penalty_coef: float = 10.0,
    skip_rows: int = 3,
) -> Dict[str, Any]:
    """
    主入口。对一个对称游戏样本提取 7 个特征。

    参数：
        txt_path          用户轨迹文件（x y pressure，前 skip_rows 行为文件头）
        png_path          用户绘制 PNG（仅用于 bbox 坐标参照）
        blue_mask_path    对称标准答案掩码（target）
        helper_mask_path  辅助线掩码（用于定位对称轴和网格）
        out_json_path     可选，若提供则自动写盘
        sample_id         样本 id；None 时自动用 txt 文件名 stem
    """
    if sample_id is None:
        sample_id = Path(txt_path).stem

    # ---------- 1. 读取 masks ----------
    target_mask = read_binary_mask(blue_mask_path).astype(np.uint8)
    helper_mask = read_binary_mask(helper_mask_path).astype(np.uint8)
    canvas_shape: Tuple[int, int] = target_mask.shape[:2]
    h_canvas, w_canvas = canvas_shape

    helper: HelperGeometry = detect_helper_geometry(helper_mask)
    axis_y = int(helper.axis_y)

    # ---------- 2. 轨迹读入（保留 pressure） ----------
    strokes_wp = _load_strokes_with_pressure(txt_path, skip_rows=skip_rows)
    num_strokes = len(strokes_wp)
    total_points = int(sum(len(s["x"]) for s in strokes_wp))

    # ---------- 3. 坐标映射（用 png bbox 参照） ----------
    mapped_strokes = map_trajectory_strokes_using_reference_bbox(
        txt_path=txt_path,
        reference_png_path=png_path,
        canvas_shape_hw=canvas_shape,
        skip_rows=skip_rows,
    )

    # ---------- 4. 渲染用户 mask（未翻转） ----------
    user_mask_raw = render_trajectory_using_reference_bbox(
        txt_path=txt_path,
        reference_png_path=png_path,
        canvas_shape_hw=canvas_shape,
        skip_rows=skip_rows,
        line_thickness=3,
    ).astype(np.uint8)

    # ---------- 5. 翻转 ----------
    reflected_user_mask = reflect_mask_across_horizontal_axis(user_mask_raw, axis_y)
    reflected_user_mask = (reflected_user_mask > 0).astype(np.uint8)
    reflected_strokes = reflect_strokes_across_horizontal_axis(
        mapped_strokes, axis_y, canvas_shape
    )

    # ---------- 6. 构造 valid_zone（target 膨胀 tol_valid） ----------
    k_tol = max(3, int(tol_valid) * 2 + 1)
    kernel_tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_tol, k_tol))
    valid_zone_mask = cv2.dilate((target_mask > 0).astype(np.uint8) * 255, kernel_tol)
    valid_zone_mask = (valid_zone_mask > 0).astype(np.uint8)

    # ---------- 7. 关键点 & 霍夫线段 ----------
    keypoints = extract_keypoints_from_target(target_mask, helper, include_midpoints=False)
    hough_segments = _extract_target_segments_from_hough(target_mask)

    # C1 的通道宽度：自适应默认 = max(10, min(step_x, step_y) * 0.20)
    if C1_channel_half_width is None:
        step_vals = [v for v in (helper.step_x, helper.step_y) if v is not None]
        C1_channel_half_width_used = (
            max(10.0, float(min(step_vals)) * 0.20) if step_vals else 10.0
        )
    else:
        C1_channel_half_width_used = float(C1_channel_half_width)

    # ---------- 8. 计算各特征 ----------
    F1 = compute_F1_completion(reflected_user_mask, target_mask, dilation_radius=dilation_F1)
    F2, kp_hit, kp_total = compute_F2_keypoint_coverage(
        reflected_user_mask, keypoints, hit_radius=hit_radius
    )
    F3, F3_meta = compute_F3_invalid_drawing(
        user_mask_raw, target_mask, valid_zone_mask,
        axis_y=axis_y, cross_penalty_coef=F3_cross_penalty_coef,
    )
    F4 = compute_F4_offpath_ratio(reflected_user_mask, valid_zone_mask)

    C1, c1_bad, c1_total = compute_C1_jitter_ratio(
        reflected_strokes, hough_segments,
        jitter_tol=jitter_tol, channel_half_width=C1_channel_half_width_used,
    )
    C2, c2_thr, c2_nshort = compute_C2_short_stroke_ratio(
        mapped_strokes, canvas_shape_hw=canvas_shape,
        threshold_ratio=C2_threshold_ratio,
    )
    C3, c3_n = compute_C3_pressure_cv(strokes_wp, trim_ends=C3_trim_ends)

    # ---------- 9. 组装输出 ----------
    result: Dict[str, Any] = {
        "sample_id": sample_id,
        "game": "sym",
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
            "axis_y": int(axis_y),
            "step_x": int(helper.step_x) if helper.step_x is not None else None,
            "step_y": int(helper.step_y) if helper.step_y is not None else None,
            "num_keypoints": int(kp_total),
            "keypoints_hit": int(kp_hit),
            "num_hough_segments": int(len(hough_segments)),
            "C1_projected_points": int(c1_total),
            "C1_bad_points": int(c1_bad),
            "C2_threshold": round(float(c2_thr), 4),
            "C2_n_short_strokes": int(c2_nshort),
            "C3_n_pressure_points": int(c3_n),
            "F3_detail": {
                "illegal_dist_sum": round(float(F3_meta["illegal_dist_sum"]), 4),
                "illegal_pixel_count": int(F3_meta["illegal_pixel_count"]),
                "n_cross_axis_pixels": int(F3_meta["n_cross_axis_pixels"]),
                "cross_penalty_coef": float(F3_meta["cross_penalty_coef"]),
                "F3_main_contribution": round(float(F3_meta["F3_main_contribution"]), 6),
                "F3_cross_contribution": round(float(F3_meta["F3_cross_contribution"]), 6),
                "target_area": int(F3_meta["target_area"]),
            },
            "canvas_hw": [int(h_canvas), int(w_canvas)],
            "params": {
                "dilation_F1": int(dilation_F1),
                "tol_valid": int(tol_valid),
                "hit_radius": float(hit_radius),
                "jitter_tol": float(jitter_tol),
                "C1_channel_half_width": float(C1_channel_half_width_used),
                "C2_threshold_ratio": float(C2_threshold_ratio),
                "C3_trim_ends": int(C3_trim_ends),
                "F3_cross_penalty_coef": float(F3_cross_penalty_coef),
                "skip_rows": int(skip_rows),
            },
        },
    }

    if out_json_path:
        out_dir = os.path.dirname(os.path.abspath(out_json_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# =====================================================================
# 可视化诊断叠加图（目视验证坐标对齐）
# =====================================================================

def visualize_sym_extraction(
    txt_path: str,
    png_path: str,
    blue_mask_path: str,
    helper_mask_path: str,
    out_png_path: str,
    *,
    tol_valid: int = 9,
    hit_radius: float = 6.0,
    skip_rows: int = 3,
) -> None:
    """
    生成一张 BGR 叠加图：
        - 蓝色 = target_mask（蓝图）
        - 浅灰 = valid_zone_mask - target_mask（容差边带）
        - 白色 = reflected_user_mask（用户翻转笔迹）
        - 粉色 = 原始用户 mask 中"越轴"部分（上半画布）
        - 绿圈 = 命中的关键点；红圈 = 未命中
    """
    target_mask = read_binary_mask(blue_mask_path).astype(np.uint8)
    helper_mask = read_binary_mask(helper_mask_path).astype(np.uint8)
    canvas_shape = target_mask.shape[:2]
    helper = detect_helper_geometry(helper_mask)
    axis_y = int(helper.axis_y)

    user_mask_raw = render_trajectory_using_reference_bbox(
        txt_path=txt_path, reference_png_path=png_path,
        canvas_shape_hw=canvas_shape, skip_rows=skip_rows, line_thickness=3,
    ).astype(np.uint8)
    reflected_user_mask = (reflect_mask_across_horizontal_axis(user_mask_raw, axis_y) > 0).astype(np.uint8)

    # 越轴部分（上半画布）
    upper_mask = np.zeros_like(user_mask_raw)
    upper_mask[: axis_y + 1, :] = 1
    user_cross = (user_mask_raw > 0) & (upper_mask > 0)

    k_tol = max(3, int(tol_valid) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_tol, k_tol))
    valid_zone = (cv2.dilate((target_mask > 0).astype(np.uint8) * 255, kernel) > 0).astype(np.uint8)

    keypoints = extract_keypoints_from_target(target_mask, helper, include_midpoints=False)
    dist_user = distance_transform_to_mask(reflected_user_mask)
    hit_flags = [
        bool(
            0 <= y < dist_user.shape[0]
            and 0 <= x < dist_user.shape[1]
            and dist_user[y, x] <= hit_radius
        )
        for x, y in keypoints
    ]

    h, w = canvas_shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[(valid_zone > 0) & (target_mask == 0)] = (60, 60, 60)       # 容差边带
    vis[target_mask > 0] = (255, 80, 0)                              # 蓝图 (BGR 蓝)
    vis[reflected_user_mask > 0] = (255, 255, 255)                   # 翻转用户
    vis[user_cross] = (180, 100, 255)                                # 越轴部分 (粉)

    # 画对称轴
    cv2.line(vis, (0, axis_y), (w - 1, axis_y), (0, 200, 255), 1)

    for (x, y), ok in zip(keypoints, hit_flags):
        color = (0, 220, 0) if ok else (0, 0, 255)
        cv2.circle(vis, (int(x), int(y)), 5, color, 2)

    os.makedirs(os.path.dirname(os.path.abspath(out_png_path)) or ".", exist_ok=True)
    cv2.imwrite(out_png_path, vis)


# =====================================================================
# 命令行入口
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="对称游戏特征提取器")
    parser.add_argument("--txt", required=True, help="用户轨迹文件 (x y pressure)")
    parser.add_argument("--png", required=True, help="用户绘制 PNG（仅用于 bbox 参照）")
    parser.add_argument("--blue", required=True, help="sym_blue_mask.png")
    parser.add_argument("--helper", required=True, help="sym_helper_mask.png")
    parser.add_argument("--out", default=None, help="输出 JSON 路径（可选）")
    parser.add_argument("--sample_id", default=None, help="样本 id（默认用 txt 文件名）")
    parser.add_argument("--vis", default=None, help="可视化叠加图输出路径（可选）")
    args = parser.parse_args()

    res = extract_sym_features(
        txt_path=args.txt,
        png_path=args.png,
        blue_mask_path=args.blue,
        helper_mask_path=args.helper,
        out_json_path=args.out,
        sample_id=args.sample_id,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))

    if args.vis:
        visualize_sym_extraction(
            txt_path=args.txt,
            png_path=args.png,
            blue_mask_path=args.blue,
            helper_mask_path=args.helper,
            out_png_path=args.vis,
        )
        print(f"[sym_feat] 诊断叠加图已保存到: {args.vis}")