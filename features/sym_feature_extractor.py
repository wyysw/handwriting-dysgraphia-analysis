"""
features/sym_feature_extractor.py

对称游戏特征提取器（阶段 1）。
对一个对称游戏样本，提取 7 个标准化特征（F1–F4, C1–C3），输出为 dict / JSON。

本模块只输出"原始指标"（0-1 比例或连续量），不做加权/不做打分。
加权与归一化统一放到阶段 4 的 build_feature_matrix.py 中。

"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
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


# ---------------------------------------------------------------
# 共用子模块
# ---------------------------------------------------------------
from features.trajectory_io import (
    load_trajectory_data,
    split_into_strokes as split_into_strokes_simple,
    load_strokes_with_pressure as _load_strokes_with_pressure,
)
from features.mask_utils import (
    read_user_drawing_mask as _read_user_mask_png,
    pad_mask_to_shape as _pad_to_shape,
    bbox_from_mask as _bbox_from_mask,
)
from features.stroke_metrics import (
    extract_segments_from_hough as _extract_target_segments_from_hough,
    compute_C1_jitter_ratio,
    compute_C2_short_stroke_ratio,
    compute_C3_pressure_cv,
)


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


# =====================================================================
# 
# =====================================================================

@dataclass
class HelperGeometry:
    all_vertical_lines: List[int]
    all_horizontal_lines: List[int]
    inner_vertical_lines: List[int]
    inner_horizontal_lines: List[int]
    axis_y: int
    outer_box: BBox
    step_x: Optional[int]
    step_y: Optional[int]


def read_binary_mask(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        gray = img
    elif img.shape[2] == 4:
        alpha = img[:, :, 3]
        if int(alpha.max()) > 0:
            return (alpha > 0).astype(np.uint8)
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        return (gray > 0).astype(np.uint8)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (gray > 0).astype(np.uint8)


def _cluster_positions(positions: Sequence[int], max_gap: int = 3) -> List[int]:
    positions = sorted(int(p) for p in positions)
    if not positions:
        return []
    groups: List[List[int]] = [[positions[0]]]
    for p in positions[1:]:
        if p - groups[-1][-1] <= max_gap:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(round(float(np.mean(g)))) for g in groups]


def _median_step(values: Sequence[int]) -> Optional[int]:
    values = sorted(int(v) for v in values)
    if len(values) < 2:
        return None
    gaps = np.diff(values)
    if len(gaps) == 0:
        return None
    return int(round(float(np.median(gaps))))


def detect_helper_geometry(helper_mask: np.ndarray) -> HelperGeometry:
    h, w = helper_mask.shape
    row_sum = np.sum(helper_mask > 0, axis=1)
    col_sum = np.sum(helper_mask > 0, axis=0)
    horizontal_candidates = np.where(row_sum >= int(w * 0.30))[0]
    vertical_candidates = np.where(col_sum >= int(h * 0.30))[0]
    all_horizontal = _cluster_positions(horizontal_candidates.tolist(), max_gap=3)
    all_vertical = _cluster_positions(vertical_candidates.tolist(), max_gap=3)
    if len(all_horizontal) < 3 or len(all_vertical) < 3:
        raise RuntimeError("无法从 helper mask 中稳定检测到外框和网格线")
    inner_horizontal = all_horizontal[1:-1]
    inner_vertical = all_vertical[1:-1]
    center_y = (inner_horizontal[0] + inner_horizontal[-1]) / 2.0
    axis_y = min(inner_horizontal, key=lambda y: abs(y - center_y))
    outer_box = (all_vertical[0], all_horizontal[0], all_vertical[-1], all_horizontal[-1])
    return HelperGeometry(
        all_vertical_lines=all_vertical,
        all_horizontal_lines=all_horizontal,
        inner_vertical_lines=inner_vertical,
        inner_horizontal_lines=inner_horizontal,
        axis_y=int(axis_y),
        outer_box=outer_box,
        step_x=_median_step(inner_vertical),
        step_y=_median_step(inner_horizontal),
    )


def distance_transform_to_mask(mask: np.ndarray) -> np.ndarray:
    inv = (1 - (mask > 0).astype(np.uint8)) * 255
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def extract_keypoints_from_target(
    target_mask: np.ndarray,
    helper: HelperGeometry,
    anchor_dist_thresh: float = 3.0,
    include_midpoints: bool = False,
) -> List[Point]:
    dist = distance_transform_to_mask(target_mask)
    base_points: List[Point] = []
    for x in helper.inner_vertical_lines:
        for y in helper.inner_horizontal_lines:
            if dist[y, x] <= anchor_dist_thresh:
                base_points.append((x, y))
    points = list(dict.fromkeys(base_points))
    point_set = set(points)
    if (not include_midpoints) or helper.step_x is None or helper.step_y is None:
        return points
    offsets = [
        (helper.step_x, 0),
        (0, helper.step_y),
        (helper.step_x, helper.step_y),
        (helper.step_x, -helper.step_y),
    ]
    midpoint_points: List[Point] = []
    for x, y in points:
        for dx, dy in offsets:
            x2, y2 = x + dx, y + dy
            if (x2, y2) not in point_set:
                continue
            mx = int(round((x + x2) / 2.0))
            my = int(round((y + y2) / 2.0))
            if 0 <= my < dist.shape[0] and 0 <= mx < dist.shape[1] and dist[my, mx] <= anchor_dist_thresh:
                midpoint_points.append((mx, my))
    all_points = points + midpoint_points
    all_points = list(dict.fromkeys(all_points))
    return all_points


def _load_trajectory_strokes(txt_path: str, skip_rows: int = 3) -> List[np.ndarray]:
    data = load_trajectory_data(txt_path, skip_rows=skip_rows)
    if data is None:
        raise RuntimeError(f"无法读取轨迹文件: {txt_path}")
    raw_strokes = split_into_strokes_simple(data)
    strokes: List[np.ndarray] = []
    for stroke in raw_strokes:
        pts = np.column_stack([stroke["x"], stroke["y"]]).astype(np.float32)
        if len(pts) >= 2:
            strokes.append(pts)
    return strokes


def _bbox_from_points(strokes: Sequence[np.ndarray]) -> BBox:
    pts = np.concatenate(strokes, axis=0)
    xs = pts[:, 0]
    ys = pts[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def map_trajectory_strokes_using_reference_bbox(
    txt_path: str,
    reference_png_path: str,
    canvas_shape_hw: Tuple[int, int],
    skip_rows: int = 3,
) -> List[np.ndarray]:
    strokes = _load_trajectory_strokes(txt_path, skip_rows=skip_rows)
    if not strokes:
        return []
    ref_mask = _pad_to_shape(_read_user_mask_png(reference_png_path), canvas_shape_hw)
    ref_box = _bbox_from_mask(ref_mask)
    src_box = _bbox_from_points(strokes)
    sx1, sy1, sx2, sy2 = src_box
    rx1, ry1, rx2, ry2 = ref_box
    src_w = max(1.0, float(sx2 - sx1))
    src_h = max(1.0, float(sy2 - sy1))
    dst_w = max(1.0, float(rx2 - rx1))
    dst_h = max(1.0, float(ry2 - ry1))
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    mapped: List[np.ndarray] = []
    for stroke in strokes:
        pts = np.empty_like(stroke, dtype=np.float32)
        pts[:, 0] = rx1 + (stroke[:, 0] - sx1) * scale_x
        pts[:, 1] = ry1 + (stroke[:, 1] - sy1) * scale_y
        mapped.append(pts)
    return mapped


def render_trajectory_using_reference_bbox(
    txt_path: str,
    reference_png_path: str,
    canvas_shape_hw: Tuple[int, int],
    skip_rows: int = 3,
    line_thickness: int = 3,
) -> np.ndarray:
    strokes = map_trajectory_strokes_using_reference_bbox(
        txt_path=txt_path,
        reference_png_path=reference_png_path,
        canvas_shape_hw=canvas_shape_hw,
        skip_rows=skip_rows,
    )
    if not strokes:
        return np.zeros(canvas_shape_hw, dtype=np.uint8)
    canvas = np.zeros(canvas_shape_hw, dtype=np.uint8)
    for stroke in strokes:
        pts = np.round(stroke).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=line_thickness)
    return (canvas > 0).astype(np.uint8)


def reflect_strokes_across_horizontal_axis(
    strokes: Sequence[np.ndarray],
    axis_y: int,
    canvas_shape_hw: Tuple[int, int],
) -> List[np.ndarray]:
    h = canvas_shape_hw[0]
    out: List[np.ndarray] = []
    for stroke in strokes:
        pts = stroke.copy().astype(np.float32)
        pts[:, 1] = 2.0 * float(axis_y) - pts[:, 1]
        valid = (pts[:, 1] >= 0) & (pts[:, 1] < h)
        pts = pts[valid]
        if len(pts) >= 2:
            out.append(pts)
    return out


def reflect_mask_across_horizontal_axis(mask: np.ndarray, axis_y: int) -> np.ndarray:
    out = np.zeros_like(mask)
    ys, xs = np.where(mask > 0)
    new_ys = 2 * axis_y - ys
    valid = (new_ys >= 0) & (new_ys < mask.shape[0])
    out[new_ys[valid], xs[valid]] = 1
    return out




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
    precision = float((tgt_d[user_pts] > 0).mean())
    recall = float((user_d[tgt_pts] > 0).mean())
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
    user_mask_raw: np.ndarray,
    target_mask: np.ndarray,
    valid_zone_mask: np.ndarray,
    axis_y: int,
    cross_penalty_coef: float = 10.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    F3 = [ Σ_{p in illegal_in_reflected} d(p)  +  cross_penalty_coef * N_cross ] / |target|
    """
    h, w = user_mask_raw.shape[:2]
    user_bin = (user_mask_raw > 0).astype(np.uint8)

    lower_mask = np.zeros_like(user_bin)
    if axis_y + 1 < h:
        lower_mask[axis_y + 1 :, :] = 1
    upper_mask = 1 - lower_mask

    user_valid_half = user_bin & lower_mask
    user_cross_axis = user_bin & upper_mask
    n_cross = int(user_cross_axis.sum())

    reflected_valid = reflect_mask_across_horizontal_axis(user_valid_half, int(axis_y))
    reflected_valid = (reflected_valid > 0).astype(np.uint8)

    valid_u8 = (valid_zone_mask > 0).astype(np.uint8)
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
    """
    user_pts = (user_reflected_mask > 0)
    n_user = int(user_pts.sum())
    if n_user == 0:
        return 0.0
    inside = user_pts & (valid_zone_mask > 0)
    return 1.0 - float(inside.sum()) / float(n_user)



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
    dilation_F1: int = 5,
    tol_valid: int = 9,
    hit_radius: float = 6.0,
    jitter_tol: float = 3.0,
    C1_channel_half_width: Optional[float] = None,
    C2_threshold_ratio: float = 0.02,
    C3_trim_ends: int = 3,
    F3_cross_penalty_coef: float = 10.0,
    skip_rows: int = 3,
) -> Dict[str, Any]:
    """
    主入口。对一个对称游戏样本提取 7 个特征。
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
        mapped_strokes, canvas_shape,
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
    vis[(valid_zone > 0) & (target_mask == 0)] = (60, 60, 60)
    vis[target_mask > 0] = (255, 80, 0)
    vis[reflected_user_mask > 0] = (255, 255, 255)
    vis[user_cross] = (180, 100, 255)

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