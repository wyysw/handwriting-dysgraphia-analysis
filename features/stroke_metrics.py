"""
features/stroke_metrics.py

共用子模块：Hough 线段提取 + C1/C2/C3 质量指标。
供 sym_feature_extractor / stroke_utils 及其他游戏模块调用。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Segment = Tuple[np.ndarray, np.ndarray]


# =====================================================================
# Hough 线段提取
# =====================================================================

DEFAULT_HOUGH_PARAMS: Dict[str, Any] = {
    "rho": 1,
    "theta": np.pi / 180.0,
    "threshold": 25,
    "minLineLength": 25,
    "maxLineGap": 6,
    "key_tol": 5.0,
    "min_segment_len": 20.0,
}


def _classify_hough_angle(angle_deg: float) -> Optional[str]:
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
    对二值 mask 做 HoughLinesP，合并同向近邻线段后返回端点列表。

    参数:
        mask        : 二值 mask（uint8 或 bool）
        hough_params: 可选覆盖项，key 参考 DEFAULT_HOUGH_PARAMS

    返回:
        List[(a, b)]：每条合并后线段的两端点（np.float32, shape=(2,)）
    """
    params = dict(DEFAULT_HOUGH_PARAMS)
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
        cls = _classify_hough_angle(angle)
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

    merged.sort(key=lambda seg: (min(seg[0][0], seg[1][0]), min(seg[0][1], seg[1][1])))
    return merged


# =====================================================================
# C1：抖动比例（点级），↓越好
# =====================================================================

def compute_C1_jitter_ratio(
    strokes_xy: Sequence[np.ndarray],
    hough_segments: Sequence[Segment],
    jitter_tol: float = 3.0,
    channel_half_width: float = 10.0,
) -> Tuple[float, int, int]:
    """
    对每个笔迹点计算到最近 Hough 线段的投影距离，统计"抖动"比例。

    返回:
        (C1, bad_count, total_projected)
        C1 = bad / total，其中 bad 为投影距离 > jitter_tol 的点数。
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
# C2：短笔段比例，↓越好
# =====================================================================

def compute_C2_short_stroke_ratio(
    mapped_strokes_xy: Sequence[np.ndarray],
    canvas_hw: Tuple[int, int],
    threshold: Optional[float] = None,
    threshold_ratio: float = 0.02,
) -> Tuple[float, float, int]:
    """
    笔段长度 = 相邻点欧氏距离之和（弧长）。
    默认阈值 = 画布对角线 × threshold_ratio。

    返回:
        (C2, used_threshold, n_short_strokes)
        C2 = 短笔段总长 / 全部笔段总长
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
# C3：压力变异系数，↓越好
# =====================================================================

def compute_C3_pressure_cv(
    strokes_with_pressure: Sequence[Dict],
    trim_ends: int = 3,
) -> Tuple[float, int]:
    """
    对每条笔段去首尾 trim_ends 点后，计算全局压力变异系数 σ/μ。

    返回:
        (C3, n_points_used)
    """
    collected: List[np.ndarray] = []
    for s in strokes_with_pressure:
        p = np.asarray(s.get("pressure", []), dtype=np.float64)
        if len(p) <= 2 * trim_ends:
            continue
        p_trim = p[trim_ends: len(p) - trim_ends]
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