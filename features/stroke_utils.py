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

"""

from __future__ import annotations

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

# ---------------------------------------------------------------
# 共用子模块
# ---------------------------------------------------------------
from features.trajectory_io import load_strokes_with_pressure  # noqa: E402
from features.mask_utils import (  # noqa: E402
    read_user_drawing_mask as _read_user_mask_png,
    pad_mask_to_shape as _pad_to_shape,
    bbox_from_mask as _bbox_from_mask,
)
from features.stroke_metrics import (  # noqa: E402
    extract_segments_from_hough,
    compute_C1_jitter_ratio,
    compute_C2_short_stroke_ratio,
    compute_C3_pressure_cv,
)

__all__ = [
    "load_strokes_with_pressure",
    "strokes_to_xy_arrays",
    "map_strokes_to_canvas",
    "render_strokes_to_mask",
    "extract_segments_from_hough",
    "compute_C1_jitter_ratio",
    "compute_C2_short_stroke_ratio",
    "compute_C3_pressure_cv",
]

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]
Segment = Tuple[np.ndarray, np.ndarray]


# =====================================================================
# 1. 轨迹 IO 辅助
# =====================================================================

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
        reference_png_path : 用户绘制 PNG，bbox = mask>0 的外接矩形。
        target_bbox        : 直接给定目标 bbox (x1, y1, x2, y2)。

    若两者都给，优先用 reference_png_path；都不给则报错。
    """
    if not strokes_xy:
        return []

    if reference_png_path is not None:
        ref_mask = _pad_to_shape(_read_user_mask_png(reference_png_path), canvas_hw)
        rx1, ry1, rx2, ry2 = _bbox_from_mask(ref_mask)
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
    """把已映射到画布坐标的笔段渲染为二值 mask（uint8, 0/1）。"""
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
# 4–7. Hough / C1 / C2 / C3 —— 由 stroke_metrics re-export，接口不变
# =====================================================================
