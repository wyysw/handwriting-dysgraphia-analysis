"""
stroke_utils_addition.py
——————————————————————————————————————————————
将以下函数追加到 features/stroke_utils.py。

新增内容：
  compute_C1_skeleton_distance_ratio()
    骨架距离法 C1（适用于圆形迷宫；方案 C 下也用于方形迷宫）。

注意事项：
  - 依赖 scipy.spatial.cKDTree（标准库，默认已安装）。
  - 若 scipy 不可用，回退到 numpy 逐点计算（较慢但正确）。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# 尝试导入 scipy（O(N log M) 最近邻查询）
try:
    from scipy.spatial import cKDTree as _cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def compute_C1_skeleton_distance_ratio(
    mapped_strokes: Sequence[np.ndarray],
    channel_skeleton: np.ndarray,
    jitter_tol: float = 3.0,
    channel_half_width: float = 28.0,
) -> Tuple[float, int, int]:
    """
    骨架距离法 C1（抖动比例）。

    方案 A：圆形迷宫专用（不适用 Hough，曲线笔迹无法被直线检测器捕获）。
    方案 C：三游戏统一，方形迷宫同样使用此函数（替换 Hough-on-user-mask 方案）。

    算法：
        对每个用户笔迹点 p：
          1. 在 channel_skeleton 上找最近骨架像素 q，得到残差 d = dist(p, q)。
          2. 若 d <= channel_half_width，该点"在通道内"，纳入统计（total += 1）。
          3. 若同时 d > jitter_tol，判为"抖动点"（bad += 1）。
        C1 = bad / total

    对比 Hough-on-user-mask 方案（方形迷宫 C1 原实现）：
        - Hough 方案：检测用户笔迹中的直线段，测点到这些直线段的残差。
          优势：对方形迷宫（理想路径全是直线段）测量"局部线性程度"很准确。
        - 骨架距离法：直接测量笔迹点到参考中心线的距离残差。
          优势：对直线和曲线通用；与特征总表"三游戏统一"的定义完全对齐。

    参数：
        mapped_strokes    : 映射到画布坐标的笔段列表，每个元素形如 (N_i, 2) 的 ndarray。
        channel_skeleton  : 通道骨架（uint8, 0/1），与 MazeGeometry.channel_skeleton 一致。
        jitter_tol        : 抖动容限（px），点到骨架距离 > jitter_tol 视为抖动。默认 3.0。
        channel_half_width: 通道半宽（px），点到骨架距离 > channel_half_width 视为通道外，
                            不纳入 C1 统计（防止通道外误走干扰 C1）。默认 28.0。

    返回：
        (C1, bad_count, total_count)
        C1 = bad_count / total_count（total=0 时返回 0.0）
    """
    ys, xs = np.where(channel_skeleton > 0)
    if len(xs) == 0:
        return 0.0, 0, 0

    skel_pts = np.column_stack([xs, ys]).astype(np.float64)  # (M, 2)

    bad   = 0
    total = 0

    if _HAS_SCIPY:
        # O(N log M) — 用 KD-Tree 查询每个笔迹点的最近骨架像素距离
        tree = _cKDTree(skel_pts)
        for stroke in mapped_strokes:
            if len(stroke) == 0:
                continue
            pts = np.asarray(stroke, dtype=np.float64)  # (N, 2)
            min_dists, _ = tree.query(pts, k=1)         # (N,)
            in_ch  = min_dists <= channel_half_width
            total += int(in_ch.sum())
            bad   += int((in_ch & (min_dists > jitter_tol)).sum())
    else:
        # O(N × M) fallback — numpy 广播，速度较慢但无外部依赖
        for stroke in mapped_strokes:
            if len(stroke) == 0:
                continue
            pts = np.asarray(stroke, dtype=np.float64)  # (N, 2)
            # dists: (N, M)
            dists = np.linalg.norm(
                pts[:, None, :] - skel_pts[None, :, :], axis=2
            )
            min_dists = dists.min(axis=1)  # (N,)
            in_ch  = min_dists <= channel_half_width
            total += int(in_ch.sum())
            bad   += int((in_ch & (min_dists > jitter_tol)).sum())

    C1 = float(bad) / float(total) if total > 0 else 0.0
    return C1, bad, total
