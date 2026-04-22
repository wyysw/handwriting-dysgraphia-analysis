import json
import math
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from pen import analyze  # user provided module
except Exception:
    try:
        from pen import analyze  # local fallback
    except Exception:
        analyze = None

# ======
# 完成模块1，2：形状大小，关键点
# 模块3：线控能力
# 模块4：线段闭合
# ======

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


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


@dataclass
class Stage1Result:
    axis_y: int
    shape_similarity: float
    size_similarity: float
    module1_pass: bool
    keypoint_total: int
    keypoint_hit: int
    keypoint_coverage: float
    keypoint_score: float
    outer_box: BBox
    user_bbox_reflected: BBox
    target_bbox: BBox

    def to_dict(self) -> Dict:
        return {
            "axis_y": self.axis_y,
            "shape_similarity": round(self.shape_similarity, 4),
            "size_similarity": round(self.size_similarity, 4),
            "module1_pass": self.module1_pass,
            "keypoint_total": self.keypoint_total,
            "keypoint_hit": self.keypoint_hit,
            "keypoint_coverage": round(self.keypoint_coverage, 4),
            "keypoint_score": round(self.keypoint_score, 2),
            "outer_box": list(self.outer_box),
            "user_bbox_reflected": list(self.user_bbox_reflected),
            "target_bbox": list(self.target_bbox),
        }


@dataclass
class LineControlSegmentResult:
    segment_id: int
    start: Tuple[float, float]
    end: Tuple[float, float]
    length: float
    analyzed_length: float
    jitter_length: float
    major_jitter_length: float
    jitter_ratio: float
    major_jitter_ratio: float
    jitter_episode_count: int
    unstable: bool

    def to_dict(self) -> Dict:
        return {
            "segment_id": self.segment_id,
            "start": [round(self.start[0], 2), round(self.start[1], 2)],
            "end": [round(self.end[0], 2), round(self.end[1], 2)],
            "length": round(self.length, 2),
            "analyzed_length": round(self.analyzed_length, 2),
            "jitter_length": round(self.jitter_length, 2),
            "major_jitter_length": round(self.major_jitter_length, 2),
            "jitter_ratio": round(self.jitter_ratio, 4),
            "major_jitter_ratio": round(self.major_jitter_ratio, 4),
            "jitter_episode_count": self.jitter_episode_count,
            "unstable": self.unstable,
        }


@dataclass
class LineControlResult:
    segment_count: int
    analyzed_length: float
    jitter_length: float
    major_jitter_length: float
    jitter_ratio: float
    major_jitter_ratio: float
    jitter_episode_count: int
    unstable_segment_count: int
    unstable_segment_ratio: float
    line_control_score: float

    def to_dict(self) -> Dict:
        return {
            "segment_count": self.segment_count,
            "analyzed_length": round(self.analyzed_length, 2),
            "jitter_length": round(self.jitter_length, 2),
            "major_jitter_length": round(self.major_jitter_length, 2),
            "jitter_ratio": round(self.jitter_ratio, 4),
            "major_jitter_ratio": round(self.major_jitter_ratio, 4),
            "jitter_episode_count": self.jitter_episode_count,
            "unstable_segment_count": self.unstable_segment_count,
            "unstable_segment_ratio": round(self.unstable_segment_ratio, 4),
            "line_control_score": round(self.line_control_score, 2),
        }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clear_directory(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


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

    # For white-on-black masks and black-on-transparent drawings, >0 is enough.
    return (gray > 0).astype(np.uint8)


def read_user_mask_png(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)

    if img.ndim == 2:
        return (img < 250).astype(np.uint8)

    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        if int(alpha.max()) > 0:
            return (alpha > 0).astype(np.uint8)
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        return (gray < 250).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (gray < 250).astype(np.uint8)


def pad_to_shape(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape_hw, dtype=np.uint8)
    h = min(shape_hw[0], mask.shape[0])
    w = min(shape_hw[1], mask.shape[1])
    out[:h, :w] = mask[:h, :w]
    return out


def bbox_from_mask(mask: np.ndarray) -> BBox:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_wh(box: BBox) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return x2 - x1 + 1, y2 - y1 + 1


def crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox_from_mask(mask)
    return mask[y1 : y2 + 1, x1 : x2 + 1]


def cluster_positions(positions: Sequence[int], max_gap: int = 3) -> List[int]:
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


def median_step(values: Sequence[int]) -> Optional[int]:
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

    all_horizontal = cluster_positions(horizontal_candidates.tolist(), max_gap=3)
    all_vertical = cluster_positions(vertical_candidates.tolist(), max_gap=3)

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
        step_x=median_step(inner_vertical),
        step_y=median_step(inner_horizontal),
    )


def load_trajectory_strokes(txt_path: str, skip_rows: int = 3) -> List[np.ndarray]:
    if analyze is None:
        raise RuntimeError("analyze.py 未成功导入，无法解析轨迹文本")
    data = analyze.load_trajectory_data(txt_path, skip_rows=skip_rows)
    if data is None:
        raise RuntimeError(f"无法读取轨迹文件: {txt_path}")
    raw_strokes = analyze.split_into_strokes_simple(data)
    strokes: List[np.ndarray] = []
    for stroke in raw_strokes:
        pts = np.column_stack([stroke["x"], stroke["y"]]).astype(np.float32)
        if len(pts) >= 2:
            strokes.append(pts)
    return strokes


def bbox_from_points(strokes: Sequence[np.ndarray]) -> BBox:
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
    strokes = load_trajectory_strokes(txt_path, skip_rows=skip_rows)
    if not strokes:
        return []

    ref_mask = pad_to_shape(read_user_mask_png(reference_png_path), canvas_shape_hw)
    ref_box = bbox_from_mask(ref_mask)
    src_box = bbox_from_points(strokes)

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


def size_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    box_a = bbox_from_mask(mask_a)
    box_b = bbox_from_mask(mask_b)
    wa, ha = bbox_wh(box_a)
    wb, hb = bbox_wh(box_b)
    width_ratio = min(wa, wb) / max(wa, wb)
    height_ratio = min(ha, hb) / max(ha, hb)
    return float((width_ratio + height_ratio) / 2.0)


def normalize_mask(mask: np.ndarray, size: int = 256, pad: int = 16) -> np.ndarray:
    crop = crop_to_bbox(mask).astype(np.uint8)
    h, w = crop.shape
    scale = min((size - 2 * pad) / max(1, w), (size - 2 * pad) / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop * 255, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    ox = (size - new_w) // 2
    oy = (size - new_h) // 2
    canvas[oy : oy + new_h, ox : ox + new_w] = resized
    return (canvas > 0).astype(np.uint8)


def tolerant_f1(mask_pred: np.ndarray, mask_gt: np.ndarray, tolerance_px: int = 9) -> float:
    pred = normalize_mask(mask_pred)
    gt = normalize_mask(mask_gt)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance_px, tolerance_px))
    gt_d = cv2.dilate(gt * 255, k)
    pred_d = cv2.dilate(pred * 255, k)

    pred_points = pred > 0
    gt_points = gt > 0
    if pred_points.sum() == 0 or gt_points.sum() == 0:
        return 0.0

    precision = float((gt_d[pred_points] > 0).mean())
    recall = float((pred_d[gt_points] > 0).mean())
    return float((2 * precision * recall) / (precision + recall + 1e-9))


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

    # 可选：加入“相邻关键交点的中点”。这一步保持启发式和易验证。
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


def keypoint_coverage_score(
    reflected_user_mask: np.ndarray,
    keypoints: Sequence[Point],
    hit_radius_px: float = 6.0,
) -> Tuple[int, int, float]:
    dist = distance_transform_to_mask(reflected_user_mask)
    hit = 0
    for x, y in keypoints:
        if 0 <= y < dist.shape[0] and 0 <= x < dist.shape[1] and dist[y, x] <= hit_radius_px:
            hit += 1
    total = len(keypoints)
    coverage = float(hit / total) if total > 0 else 0.0
    return hit, total, coverage




# =========================
# sym_line_control：理想情况下关键点之间由线段组成，检测连续抖动/波浪线
# =========================

def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values.copy()
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window <= 1 or len(values) < window:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


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


def _extract_target_segments_from_hough(target_mask: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    mask_u8 = (target_mask > 0).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(mask_u8, 1, np.pi / 180.0, threshold=25, minLineLength=25, maxLineGap=6)
    if lines is None:
        return []

    groups: Dict[str, List[Dict[str, float]]] = {"h": [], "v": [], "diag_pos": [], "diag_neg": []}
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in raw]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        cls = _classify_line_angle(angle)
        if cls is None:
            continue
        if cls == "h":
            key = (y1 + y2) / 2.0
            span1, span2 = sorted([x1, x2])
            groups[cls].append({"key": key, "s1": span1, "s2": span2})
        elif cls == "v":
            key = (x1 + x2) / 2.0
            span1, span2 = sorted([y1, y2])
            groups[cls].append({"key": key, "s1": span1, "s2": span2})
        elif cls == "diag_pos":
            key = ((y1 - x1) + (y2 - x2)) / 2.0
            span1, span2 = sorted([x1, x2])
            groups[cls].append({"key": key, "s1": span1, "s2": span2})
        else:
            key = ((y1 + x1) + (y2 + x2)) / 2.0
            span1, span2 = sorted([x1, x2])
            groups[cls].append({"key": key, "s1": span1, "s2": span2})

    merged_segments: List[Tuple[np.ndarray, np.ndarray]] = []
    key_tol = 5.0
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
            if float(np.linalg.norm(b - a)) >= 20.0:
                merged_segments.append((a, b))

    merged_segments.sort(key=lambda seg: (min(seg[0][0], seg[1][0]), min(seg[0][1], seg[1][1])))
    return merged_segments


def _cluster_junctions(segments: Sequence[Tuple[np.ndarray, np.ndarray]], tol: float = 8.0) -> List[np.ndarray]:
    pts = [a.copy() for a, _ in segments] + [b.copy() for _, b in segments]
    clusters: List[List[np.ndarray]] = []
    for p in pts:
        placed = False
        for cluster in clusters:
            center = np.mean(cluster, axis=0)
            if float(np.linalg.norm(p - center)) <= tol:
                cluster.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    out: List[np.ndarray] = []
    for cluster in clusters:
        if len(cluster) >= 2:
            out.append(np.mean(cluster, axis=0).astype(np.float32))
    return out


def _project_to_segment(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-6:
        return 0.0, float(np.linalg.norm(pt - a)), 0.0
    t = float(np.dot(pt - a, ab) / denom)
    proj = a + t * ab
    dist = float(np.linalg.norm(pt - proj))
    seg_len = float(np.linalg.norm(ab))
    s = max(0.0, min(seg_len, t * seg_len))
    cross = float(ab[0] * (pt[1] - a[1]) - ab[1] * (pt[0] - a[0]))
    signed = dist if cross >= 0 else -dist
    return t, dist, signed


def _bin_profile(s_vals: np.ndarray, d_vals: np.ndarray, bin_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(s_vals) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    order = np.argsort(s_vals)
    s_vals = s_vals[order]
    d_vals = d_vals[order]
    bins = np.floor(s_vals / bin_size).astype(np.int32)
    uniq = np.unique(bins)
    out_s, out_d = [], []
    for b in uniq:
        idx = np.where(bins == b)[0]
        out_s.append(float(np.median(s_vals[idx])))
        out_d.append(float(np.median(d_vals[idx])))
    return np.asarray(out_s, dtype=np.float32), np.asarray(out_d, dtype=np.float32)


def _runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and bool(mask[j + 1]):
            j += 1
        if j - i + 1 >= min_len:
            out.append((i, j))
        i = j + 1
    return out


def analyze_line_control(
    target_mask: np.ndarray,
    reflected_strokes: Sequence[np.ndarray],
    helper: HelperGeometry,
) -> Tuple[LineControlResult, List[LineControlSegmentResult], np.ndarray]:
    segments = _extract_target_segments_from_hough(target_mask)
    if not segments:
        box = bbox_from_mask(target_mask)
        segments = [(np.array([box[0], box[1]], dtype=np.float32), np.array([box[2], box[3]], dtype=np.float32))]

    min_step = min([v for v in [helper.step_x, helper.step_y] if v is not None] or [96])
    channel_half_width = max(6.0, float(min_step) * 0.10)
    corner_radius = max(8.0, channel_half_width * 1.2)
    jitter_tol = max(1.8, channel_half_width * 0.22)
    major_tol = max(3.5, channel_half_width * 0.40)
    junctions = _cluster_junctions(segments, tol=8.0)

    seq_map: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {i: [] for i in range(len(segments))}
    for stroke in reflected_strokes:
        cur_sid = None
        cur_pts: List[np.ndarray] = []
        cur_s: List[float] = []
        cur_d: List[float] = []

        def flush() -> None:
            nonlocal cur_sid, cur_pts, cur_s, cur_d
            if cur_sid is not None and len(cur_pts) >= 6:
                seq_map[cur_sid].append((
                    np.asarray(cur_pts, dtype=np.float32),
                    np.asarray(cur_s, dtype=np.float32),
                    np.asarray(cur_d, dtype=np.float32),
                ))
            cur_sid, cur_pts, cur_s, cur_d = None, [], [], []

        for pt in stroke:
            if junctions and any(float(np.linalg.norm(pt - j)) <= corner_radius for j in junctions):
                flush()
                continue
            best = None
            for sid, (a, b) in enumerate(segments):
                t, dist, signed = _project_to_segment(pt, a, b)
                if dist > channel_half_width or t < -0.10 or t > 1.10:
                    continue
                if best is None or dist < best[2]:
                    seg_len = float(np.linalg.norm(b - a))
                    best = (sid, max(0.0, min(seg_len, t * seg_len)), dist, signed)
            if best is None:
                flush()
                continue
            sid, s_val, _, d_val = best
            if cur_sid is None:
                cur_sid = sid
            elif sid != cur_sid:
                flush()
                cur_sid = sid
            cur_pts.append(pt)
            cur_s.append(float(s_val))
            cur_d.append(float(d_val))
        flush()

    point_severity: Dict[Point, int] = {}
    seg_results: List[LineControlSegmentResult] = []
    total_analyzed = 0.0
    total_jitter = 0.0
    total_major = 0.0
    total_episodes = 0

    for sid, (a, b) in enumerate(segments):
        seg_len = float(np.linalg.norm(b - a))
        analyzed = 0.0
        jitter_len = 0.0
        major_len = 0.0
        episodes = 0
        bin_size = max(1.5, min(3.0, seg_len / 40.0 if seg_len > 0 else 2.0))
        min_run = max(3, int(math.ceil(max(8.0, seg_len * 0.06) / bin_size)))
        trend_window = max(5, int(round(seg_len / 14.0)))

        for pts, s_vals, d_vals in seq_map.get(sid, []):
            span = float(np.max(s_vals) - np.min(s_vals)) if len(s_vals) > 0 else 0.0
            if span < 6.0:
                continue
            analyzed += span
            prof_s, prof_d = _bin_profile(s_vals, d_vals, bin_size=bin_size)
            if len(prof_s) < max(5, min_run + 1):
                continue

            path_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) >= 2 else 0.0
            chord_len = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) >= 2 else 0.0
            tortuosity = max(0.0, path_len / max(chord_len, 1e-6) - 1.0) if chord_len > 0 else 0.0

            trend = _moving_average(prof_d, trend_window)
            residual = prof_d - trend
            abs_res = np.abs(residual)
            mean_abs_res = float(abs_res.mean()) if len(abs_res) > 0 else 0.0
            jitter_mask = abs_res > jitter_tol
            runs = _runs(jitter_mask, min_run)
            sign_changes = int(np.sum(np.sign(residual[1:]) * np.sign(residual[:-1]) < 0)) if len(residual) > 1 else 0

            for i0, i1 in runs:
                run_len = float(prof_s[i1] - prof_s[i0] + bin_size)
                is_major = bool(abs_res[i0:i1 + 1].mean() >= major_tol or abs_res[i0:i1 + 1].max() >= major_tol)
                jitter_len += run_len
                if is_major:
                    major_len += run_len
                episodes += 1
                s_lo = float(prof_s[i0] - bin_size)
                s_hi = float(prof_s[i1] + bin_size)
                sev = 2 if is_major else 1
                for p, sval in zip(pts, s_vals):
                    if s_lo <= float(sval) <= s_hi:
                        key = (int(round(p[0])), int(round(p[1])))
                        point_severity[key] = max(point_severity.get(key, 0), sev)

            oscillatory = bool(
                span >= max(20.0, seg_len * 0.12)
                and sign_changes >= 5
                and (mean_abs_res >= 0.35 or tortuosity >= 0.25)
            )
            if oscillatory:
                osc_ratio = min(
                    0.35,
                    0.03 * float(sign_changes)
                    + 0.30 * max(0.0, mean_abs_res - 0.30)
                    + 0.45 * max(0.0, tortuosity - 0.18),
                )
                osc_len = span * osc_ratio
                jitter_len += osc_len
                is_major_osc = bool(mean_abs_res >= 0.65 or tortuosity >= 0.40)
                if is_major_osc:
                    major_len += osc_len * 0.70
                if len(runs) == 0:
                    episodes += 1
                sev = 2 if is_major_osc else 1
                for p in pts:
                    key = (int(round(p[0])), int(round(p[1])))
                    point_severity[key] = max(point_severity.get(key, 0), sev)

        analyzed = min(analyzed, seg_len * 1.35) if seg_len > 0 else analyzed
        jitter_len = min(jitter_len, analyzed)
        major_len = min(major_len, jitter_len)
        jitter_ratio = float(jitter_len / analyzed) if analyzed > 1e-6 else 0.0
        major_ratio = float(major_len / analyzed) if analyzed > 1e-6 else 0.0
        unstable = bool(jitter_ratio >= 0.20 or major_ratio >= 0.08 or episodes >= 2)
        seg_results.append(LineControlSegmentResult(
            segment_id=sid,
            start=(float(a[0]), float(a[1])),
            end=(float(b[0]), float(b[1])),
            length=seg_len,
            analyzed_length=analyzed,
            jitter_length=jitter_len,
            major_jitter_length=major_len,
            jitter_ratio=jitter_ratio,
            major_jitter_ratio=major_ratio,
            jitter_episode_count=episodes,
            unstable=unstable,
        ))
        total_analyzed += analyzed
        total_jitter += jitter_len
        total_major += major_len
        total_episodes += episodes

    jitter_ratio = float(total_jitter / total_analyzed) if total_analyzed > 1e-6 else 0.0
    major_ratio = float(total_major / total_analyzed) if total_analyzed > 1e-6 else 0.0
    unstable_count = int(sum(1 for r in seg_results if r.unstable))
    unstable_ratio = float(unstable_count / max(1, len(seg_results)))
    weighted_ratio = min(1.0, jitter_ratio + 0.60 * major_ratio)
    score = float(max(0.0, 100.0 * (1.0 - weighted_ratio)))

    vis = np.zeros((target_mask.shape[0], target_mask.shape[1], 3), dtype=np.uint8)
    vis[target_mask > 0] = (80, 80, 80)
    for res, (a, b) in zip(seg_results, segments):
        color = (0, 180, 255) if res.unstable else (0, 220, 120)
        cv2.line(vis, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), color, 1)
    for stroke in reflected_strokes:
        for pt in stroke:
            key = (int(round(pt[0])), int(round(pt[1])))
            sev = point_severity.get(key, 0)
            color = (220, 220, 220) if sev == 0 else ((0, 165, 255) if sev == 1 else (0, 0, 255))
            cv2.circle(vis, key, 1, color, -1)

    result = LineControlResult(
        segment_count=len(seg_results),
        analyzed_length=total_analyzed,
        jitter_length=total_jitter,
        major_jitter_length=total_major,
        jitter_ratio=jitter_ratio,
        major_jitter_ratio=major_ratio,
        jitter_episode_count=total_episodes,
        unstable_segment_count=unstable_count,
        unstable_segment_ratio=unstable_ratio,
        line_control_score=score,
    )
    return result, seg_results, vis


def draw_overlay(
    base_shape_hw: Tuple[int, int],
    target_mask: np.ndarray,
    reflected_user_mask: np.ndarray,
    keypoints: Sequence[Point],
    hit_mask: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    h, w = base_shape_hw
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[target_mask > 0] = (255, 0, 0)       # blue-like in BGR
    canvas[reflected_user_mask > 0] = (255, 255, 255)

    if hit_mask is None:
        hit_mask = [True] * len(keypoints)

    for (x, y), ok in zip(keypoints, hit_mask):
        color = (0, 200, 0) if ok else (0, 0, 255)
        cv2.circle(canvas, (int(x), int(y)), 4, color, -1)
    return canvas


def run_stage1(
    blue_mask_path: str,
    helper_mask_path: str,
    user_png_path: str,
    user_txt_path: str,
    out_dir: str,
    skip_rows: int = 3,
    shape_threshold: float = 0.70,
    size_threshold: float = 0.70,
) -> Stage1Result:
    clear_directory(out_dir)

    target_mask = read_binary_mask(blue_mask_path).astype(np.uint8)
    helper_mask = read_binary_mask(helper_mask_path).astype(np.uint8)
    helper = detect_helper_geometry(helper_mask)

    canvas_shape = target_mask.shape

    mapped_strokes = map_trajectory_strokes_using_reference_bbox(
        txt_path=user_txt_path,
        reference_png_path=user_png_path,
        canvas_shape_hw=canvas_shape,
        skip_rows=skip_rows,
    )
    user_mask_from_txt = render_trajectory_using_reference_bbox(
        txt_path=user_txt_path,
        reference_png_path=user_png_path,
        canvas_shape_hw=canvas_shape,
        skip_rows=skip_rows,
        line_thickness=3,
    )

    reflected_user_mask = reflect_mask_across_horizontal_axis(user_mask_from_txt, helper.axis_y)
    reflected_strokes = reflect_strokes_across_horizontal_axis(mapped_strokes, helper.axis_y, canvas_shape)

    shape_score = tolerant_f1(reflected_user_mask, target_mask, tolerance_px=9)
    size_score = size_similarity(reflected_user_mask, target_mask)
    module1_pass = bool(shape_score >= shape_threshold and size_score >= size_threshold)

    keypoints = extract_keypoints_from_target(
        target_mask,
        helper,
        anchor_dist_thresh=3.0,
        include_midpoints=False,
    )
    dist_user = distance_transform_to_mask(reflected_user_mask)
    hit_flags = [dist_user[y, x] <= 6.0 for x, y in keypoints]
    hit, total, coverage = keypoint_coverage_score(reflected_user_mask, keypoints, hit_radius_px=6.0)

    line_control_result, line_control_segments, line_control_vis = analyze_line_control(
        target_mask=target_mask,
        reflected_strokes=reflected_strokes,
        helper=helper,
    )

    result = Stage1Result(
        axis_y=helper.axis_y,
        shape_similarity=shape_score,
        size_similarity=size_score,
        module1_pass=module1_pass,
        keypoint_total=total,
        keypoint_hit=hit,
        keypoint_coverage=coverage,
        keypoint_score=coverage * 100.0,
        outer_box=helper.outer_box,
        user_bbox_reflected=bbox_from_mask(reflected_user_mask),
        target_bbox=bbox_from_mask(target_mask),
    )

    overlay = draw_overlay(canvas_shape, target_mask, reflected_user_mask, keypoints, hit_flags)
    cv2.imwrite(os.path.join(out_dir, "stage1_overlay.png"), overlay)
    cv2.imwrite(os.path.join(out_dir, "stage1_reflected_user_mask.png"), reflected_user_mask * 255)
    cv2.imwrite(os.path.join(out_dir, "stage1_target_mask.png"), target_mask * 255)
    cv2.imwrite(os.path.join(out_dir, "stage1_line_control.png"), line_control_vis)

    key_vis = cv2.cvtColor(target_mask * 255, cv2.COLOR_GRAY2BGR)
    for (x, y), ok in zip(keypoints, hit_flags):
        color = (0, 200, 0) if ok else (0, 0, 255)
        cv2.circle(key_vis, (int(x), int(y)), 4, color, -1)
    cv2.imwrite(os.path.join(out_dir, "stage1_keypoints.png"), key_vis)

    module_scores = {
        "shape_score": round(shape_score * 100.0, 2),
        "size_score": round(size_score * 100.0, 2),
        "module1_score": round((shape_score + size_score) * 50.0, 2),
        "keypoint_score": round(coverage * 100.0, 2),
        "line_control_score": round(line_control_result.line_control_score, 2),
    }

    meta = {
        "axis_y": helper.axis_y,
        "all_vertical_lines": helper.all_vertical_lines,
        "all_horizontal_lines": helper.all_horizontal_lines,
        "inner_vertical_lines": helper.inner_vertical_lines,
        "inner_horizontal_lines": helper.inner_horizontal_lines,
        "step_x": helper.step_x,
        "step_y": helper.step_y,
        "module_scores": module_scores,
        "result": result.to_dict(),
        "sym_line_control": {
            "result": line_control_result.to_dict(),
            "segments": [seg.to_dict() for seg in line_control_segments],
        },
    }
    with open(os.path.join(out_dir, "stage1_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    base = "data/sym"
    out_dir = "output_sym/stage1_2_3_4_output"
    res = run_stage1(
        blue_mask_path=os.path.join(base, "sym_blue_mask.png"),
        helper_mask_path=os.path.join(base, "sym_helper_mask_completed.png"),
        user_png_path=os.path.join(base, "3.png"),
        user_txt_path=os.path.join(base, "3.txt"),
        out_dir=out_dir,
        skip_rows=3,
    )
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
