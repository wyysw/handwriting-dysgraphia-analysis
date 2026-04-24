"""
features/mask_utils.py

共用子模块：图像 mask 读取、填充、bbox 提取。
供 sym_feature_extractor / stroke_utils 及其他游戏模块调用。
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


def read_binary_mask(path: str) -> np.ndarray:
    """
    读取 PNG 为二值 mask（uint8, 0/1）。
    支持灰度图、RGBA（优先用 alpha 通道）、BGR。
    """
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


def read_user_drawing_mask(path: str) -> np.ndarray:
    """
    读取用户绘制 PNG 为二值 mask（暗色/非白像素 = 1）。
    与 sym_feature_extractor._read_user_mask_png 等价。
    """
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


def pad_mask_to_shape(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    """将 mask 零填充（或裁剪）到目标 (H, W)。"""
    out = np.zeros(shape_hw, dtype=np.uint8)
    h = min(shape_hw[0], mask.shape[0])
    w = min(shape_hw[1], mask.shape[1])
    out[:h, :w] = mask[:h, :w]
    return out


def bbox_from_mask(mask: np.ndarray) -> BBox:
    """返回 mask 非零区域的 bbox (x1, y1, x2, y2)。"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())