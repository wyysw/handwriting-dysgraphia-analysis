"""
features/stroke_utils.py

阶段 2 公共库 —— 三游戏共用的笔段处理逻辑（4.3.4）。

本模块把已在阶段 1 经过 sym 验证的"与具体游戏无关"的逻辑抽出来，
供对称游戏 / 方形迷宫 / 圆形迷宫共用。**不引入新算法。**

提供：
    - 轨迹 IO（含 pressure）：load_strokes_with_pressure
    - 坐标映射：map_strokes_to_canvas（bbox 对齐，支持 png_path 或显式 target_bbox）
    - 渲染：render_strokes_to_mask
    - Hough 线段提取：extract_segments_from_hough（支持自定义参数）
    - C1：compute_C1_jitter_ratio
    - C2：compute_C2_short_stroke_ratio
    - C3：compute_C3_pressure_cv

实现来源：
    - load_strokes_with_pressure  ← features.sym_feature_extractor._load_strokes_with_pressure
    - map_strokes_to_canvas       ← sym_core.sym_analyze3.map_trajectory_strokes_using_reference_bbox
    - render_strokes_to_mask      ← sym_core.sym_analyze3.render_trajectory_using_reference_bbox 的内部渲染部分
    - extract_segments_from_hough ← sym_core.sym_analyze3._extract_target_segments_from_hough
    - compute_C1/C2/C3            ← features.sym_feature_extractor.compute_C1/C2/C3
"""

from __future__ import annotations

import math
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

from pen import analyze as _pen_analyze
from sym_core.sym_analyze3 import (
    bbox_from_mask,
    pad_to_shape,
    read_user_mask_png,
)

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]
Segment = Tuple[np.ndarray, np.ndarray]


# =====================================================================
# 1. 轨迹 IO
# =====================================================================

def load_strokes_with_pressure(
    txt_path: str, skip_rows: int = 3
) -> List[Dict[str, np.ndarray]]:
    """
    读 txt → 笔段 dict 列表（每段含 'x', 'y', 'pressure' np.ndarray）。
    坐标是原始设备坐标（未映射到画布）。
    """
    data = _pen_analyze.load_trajectory_data(txt_path, skip_rows=skip_rows)
    if data is None:
        raise RuntimeError(f"[stroke_utils] 无法读取轨迹文件: {txt_path}")
    return _pen_analyze.split_into_strokes_simple(data)


def strokes_to_xy_arrays(
    strokes_wp: Sequence[Dict[str, np.ndarray]],
) -> List[np.ndarray]:
    """从含 pressure 的笔段 dict 提取 (N, 2) xy 数组列表（去除 <2 点的笔段）。"""
    out: List[np.ndarray] = []
    for s in strokes_wp:
        pts = np.column_stack([s["x"], s["y"]]).astype(np.float32)
        if len(pts) >= 2:
            out.append(pts)
    return out


def _bbox_from_stroke_points(strokes: Sequence[np.ndarray]) -> BBox:
    pts = np.concatenate(strokes, axis=0)
    return (int(pts[:, 0].min()), int(pts[:, 1].min()),
            int(pts[:, 0].max()), int(pts[:, 1].max()))


# =====================================================================
# 2. 坐标映射（bbox 对齐）
# =====================================================================

def map_strokes_to_canvas(
    strokes_xy: Sequence[np.ndarray],
    canvas_hw: Tuple[int, int],
    *,
    reference_png_path: Optional[str] = None,
    target_bbox: Optional[BBox] = None,
) -> List[np.ndarray]:
    """
    将原始设备坐标的笔段映射到画布坐标系，按 bbox 对齐。

    参数（二选一）：
        reference_png_path : 用户绘制 PNG（含原图绘制区域），bbox = mask>0 的外接矩形。
                             与 sym_analyze3.map_trajectory_strokes_using_reference_bbox 等价。
        target_bbox        : 直接给定目标 bbox (x1, y1, x2, y2)。
                             用于无 PNG 时的 fallback（如把笔迹 bbox 对齐到迷宫内框）。

    若两者都给，优先用 reference_png_path（与 sym 行为一致）；都不给则报错。
    """
    if not strokes_xy:
        return []

    if reference_png_path is not None:
        ref_mask = pad_to_shape(read_user_mask_png(reference_png_path), canvas_hw)
        rx1, ry1, rx2, ry2 = bbox_from_mask(ref_mask)
    elif target_bbox is not None:
        rx1, ry1, rx2, ry2 = target_bbox
    else:
        raise ValueError("必须提供 reference_png_path 或 target_bbox 之一。")

    sx1, sy1, sx2, sy2 = _bbox_from_stroke_points(strokes_xy)
    src_w = max(1.0, float(sx2 - sx1))
    src_h = max(1.0, float(sy2 - sy1))
    dst_w = max(1.0, float(rx2 - rx1))
    dst_h = max(1.0, float(ry2 - ry1))
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h

    mapped: List[np.ndarray] = []
    for s in strokes_xy:
        out = np.empty_like(s, dtype=np.float32)
        out[:, 0] = rx1 + (s[:, 0] - sx1) * scale_x
        out[:, 1] = ry1 + (s[:, 1] - sy1) * scale_y
        mapped.append(out)
    return mapped


# =====================================================================
# 3. 渲染笔段为二值 mask
# =====================================================================

def render_strokes_to_mask(
    mapped_strokes: Sequence[np.ndarray],
    canvas_hw: Tuple[int, int],
    line_thickness: int = 3,
) -> np.ndarray:
    """
    把已映射到画布坐标的笔段渲染为二值 mask（uint8, 0/1）。
    与 sym_analyze3.render_trajectory_using_reference_bbox 的内部渲染部分等价。
    """
    canvas = np.zeros(canvas_hw, dtype=np.uint8)
    if not mapped_strokes:
        return canvas
    for s in mapped_strokes:
        if len(s) < 2:
            continue
        pts = np.round(s).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=255,
                      thickness=line_thickness)
    return (canvas > 0).astype(np.uint8)


# =====================================================================
# 4. Hough 线段提取（可参数化）
# =====================================================================

# 默认 Hough 参数（与 sym_analyze3._extract_target_segments_from_hough 完全一致）
_DEFAULT_HOUGH_PARAMS: Dict[str, Any] = {
    "rho": 1,
    "theta": np.pi / 180.0,
    "threshold": 25,
    "minLineLength": 25,
    "maxLineGap": 6,
    # 后处理
    "key_tol": 5.0,           # 同向线段聚合的"关键参数"容差
    "min_segment_len": 20.0,  # 合并后最短保留段长
}


def _classify_line_angle(angle_deg: float) -> Optional[str]:
    a = float(angle_deg)
    if abs(a) <= 15.0 or abs(abs(a) - 180.0) <= 15.0:
        return "h"
    if abs(abs(a) - 90.0) <= 15.0:
        return "v"
    if abs(a - 45.0) <= 15.0 or abs(a + 135.0) <= 15.0:
        return "diag_pos"
    if abs(a + 45.0) <= 15.0 or abs(a - 135.0) <= 15.0:
        return "diag_neg"
    return None


def extract_segments_from_hough(
    mask: np.ndarray,
    hough_params: Optional[Dict[str, Any]] = None,
) -> List[Segment]:
    """
    与 sym_analyze3._extract_target_segments_from_hough 等价，
    但 Hough 参数可通过 hough_params 字典覆盖。

    返回 List[(a, b)]：每条合并后的线段两端点（np.float32, shape=(2,)）。
    """
    params = dict(_DEFAULT_HOUGH_PARAMS)
    if hough_params:
        params.update(hough_params)

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(
        mask_u8,
        params["rho"], params["theta"],
        threshold=int(params["threshold"]),
        minLineLength=int(params["minLineLength"]),
        maxLineGap=int(params["maxLineGap"]),
    )
    if lines is None:
        return []

    groups: Dict[str, List[Dict[str, float]]] = {
        "h": [], "v": [], "diag_pos": [], "diag_neg": []
    }
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in raw]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        cls = _classify_line_angle(angle)
        if cls is None:
            continue
        if cls == "h":
            key = (y1 + y2) / 2.0
            span1, span2 = sorted([x1, x2])
        elif cls == "v":
            key = (x1 + x2) / 2.0
            span1, span2 = sorted([y1, y2])
        elif cls == "diag_pos":
            key = ((y1 - x1) + (y2 - x2)) / 2.0
            span1, span2 = sorted([x1, x2])
        else:
            key = ((y1 + x1) + (y2 + x2)) / 2.0
            span1, span2 = sorted([x1, x2])
        groups[cls].append({"key": key, "s1": span1, "s2": span2})

    key_tol = float(params["key_tol"])
    min_len = float(params["min_segment_len"])
    merged: List[Segment] = []
    for cls, entries in groups.items():
        entries = sorted(entries, key=lambda d: d["key"])
        clusters: List[List[Dict[str, float]]] = []
        for e in entries:
            if not clusters or abs(e["key"] - clusters[-1][-1]["key"]) > key_tol:
                clusters.append([e])
            else:
                clusters[-1].append(e)

        for cluster in clusters:
            key = float(np.mean([c["key"] for c in cluster]))
            s1 = float(min(c["s1"] for c in cluster))
            s2 = float(max(c["s2"] for c in cluster))
            if cls == "h":
                a = np.array([s1, key], dtype=np.float32)
                b = np.array([s2, key], dtype=np.float32)
            elif cls == "v":
                a = np.array([key, s1], dtype=np.float32)
                b = np.array([key, s2], dtype=np.float32)
            elif cls == "diag_pos":
                a = np.array([s1, s1 + key], dtype=np.float32)
                b = np.array([s2, s2 + key], dtype=np.float32)
            else:
                a = np.array([s1, key - s1], dtype=np.float32)
                b = np.array([s2, key - s2], dtype=np.float32)
            if float(np.linalg.norm(b - a)) >= min_len:
                merged.append((a, b))

    merged.sort(key=lambda seg: (min(seg[0][0], seg[1][0]),
                                 min(seg[0][1], seg[1][1])))
    return merged


# =====================================================================
# 5. C1：抖动比例（点级），↓越好
# =====================================================================

def compute_C1_jitter_ratio(
    strokes_xy: Sequence[np.ndarray],
    hough_segments: Sequence[Segment],
    jitter_tol: float = 3.0,
    channel_half_width: float = 10.0,
) -> Tuple[float, int, int]:
    """
    对每个用户笔迹点：
      1. 遍历所有 hough 线段，计算到线段的投影参数 t 和垂直距离 d
      2. 只保留 t ∈ [-0.05, 1.05] 且 d <= channel_half_width 的点
         —— 称为"命中某条参考线段附近的投影点"
      3. 在所有命中线段中，取 d 最小的为该点的归属
    然后：
      total = 被纳入投影的点数
      bad   = 其中 d > jitter_tol 的点数
      C1    = bad / total

    与 sym_feature_extractor.compute_C1_jitter_ratio 完全一致。
    返回 (C1, bad, total)。
    """
    if not strokes_xy or not hough_segments:
        return 0.0, 0, 0

    pts_list = [np.asarray(s, dtype=np.float64) for s in strokes_xy if len(s) >= 1]
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
        ap = all_pts - a
        t = (ap @ ab) / denom
        t_clip = np.clip(t, 0.0, 1.0)
        proj = a + t_clip[:, None] * ab
        d = np.linalg.norm(all_pts - proj, axis=1)
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
# 6. C2：短笔段比例，↓越好
# =====================================================================

def compute_C2_short_stroke_ratio(
    mapped_strokes_xy: Sequence[np.ndarray],
    canvas_hw: Tuple[int, int],
    threshold: Optional[float] = None,
    threshold_ratio: float = 0.02,
) -> Tuple[float, float, int]:
    """
    笔段长度 = 相邻点欧氏距离之和（弧长）
    默认阈值 thr = 画布对角线 × threshold_ratio
    C2 = sum(len_i for len_i < thr) / sum(all len_i)

    返回 (C2, used_threshold, n_short_strokes)。
    与 sym_feature_extractor.compute_C2_short_stroke_ratio 完全一致。
    """
    h, w = canvas_hw
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
# 7. C3：压力变异系数，↓越好
# =====================================================================

def compute_C3_pressure_cv(
    strokes_with_pressure: Sequence[Dict[str, np.ndarray]],
    trim_ends: int = 3,
) -> Tuple[float, int]:
    """
    对每条笔段，去掉首尾 trim_ends 点后保留 pressure > 0 的点；
    然后在所有保留点上计算 σ(p) / μ(p)。

    返回 (C3, n_points_used)。
    与 sym_feature_extractor.compute_C3_pressure_cv 完全一致。
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