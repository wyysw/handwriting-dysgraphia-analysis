import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from pen import analyze  # user provided module
except Exception:
    analyze = None

# ======
# 完成模块1，2：形状大小，关键点
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


def ensure_dir(path: str) -> None:
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


def render_trajectory_using_reference_bbox(
    txt_path: str,
    reference_png_path: str,
    canvas_shape_hw: Tuple[int, int],
    skip_rows: int = 3,
    line_thickness: int = 3,
) -> np.ndarray:
    strokes = load_trajectory_strokes(txt_path, skip_rows=skip_rows)
    if not strokes:
        return np.zeros(canvas_shape_hw, dtype=np.uint8)

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

    canvas = np.zeros(canvas_shape_hw, dtype=np.uint8)
    for stroke in strokes:
        pts = np.empty((len(stroke), 1, 2), dtype=np.int32)
        pts[:, 0, 0] = np.round(rx1 + (stroke[:, 0] - sx1) * scale_x).astype(np.int32)
        pts[:, 0, 1] = np.round(ry1 + (stroke[:, 1] - sy1) * scale_y).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=line_thickness)

    return (canvas > 0).astype(np.uint8)


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
    ensure_dir(out_dir)

    target_mask = read_binary_mask(blue_mask_path).astype(np.uint8)
    helper_mask = read_binary_mask(helper_mask_path).astype(np.uint8)
    helper = detect_helper_geometry(helper_mask)

    canvas_shape = target_mask.shape

    user_mask_from_txt = render_trajectory_using_reference_bbox(
        txt_path=user_txt_path,
        reference_png_path=user_png_path,
        canvas_shape_hw=canvas_shape,
        skip_rows=skip_rows,
        line_thickness=3,
    )

    reflected_user_mask = reflect_mask_across_horizontal_axis(user_mask_from_txt, helper.axis_y)

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

    key_vis = cv2.cvtColor(target_mask * 255, cv2.COLOR_GRAY2BGR)
    for (x, y), ok in zip(keypoints, hit_flags):
        color = (0, 200, 0) if ok else (0, 0, 255)
        cv2.circle(key_vis, (int(x), int(y)), 4, color, -1)
    cv2.imwrite(os.path.join(out_dir, "stage1_keypoints.png"), key_vis)

    meta = {
        "axis_y": helper.axis_y,
        "all_vertical_lines": helper.all_vertical_lines,
        "all_horizontal_lines": helper.all_horizontal_lines,
        "inner_vertical_lines": helper.inner_vertical_lines,
        "inner_horizontal_lines": helper.inner_horizontal_lines,
        "step_x": helper.step_x,
        "step_y": helper.step_y,
        "result": result.to_dict(),
    }
    with open(os.path.join(out_dir, "stage1_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    base = "data/sym"
    out_dir = "output_sym/stage1_2_output"
    res = run_stage1(
        blue_mask_path=os.path.join(base, "sym_blue_mask.png"),
        helper_mask_path=os.path.join(base, "sym_helper_mask_completed.png"),
        user_png_path=os.path.join(base, "1.png"),
        user_txt_path=os.path.join(base, "1.txt"),
        out_dir=out_dir,
        skip_rows=3,
    )
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
