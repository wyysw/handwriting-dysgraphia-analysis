"""
4.5.2  M2：纯先验加权线性评分（消融对照）
==========================================
先验权重固定，不依赖任何训练标签。
fit 仅用训练集计算 sigmoid 缩放参数，使输出概率不饱和。

设计说明
--------
- W_PRIOR 按设计文档给定，归一化使 Σw = 7（与特征数一致，便于直觉理解"平均每特征权重=1"）。
- sigmoid_scale_ = std(X_train @ w) + ε
  让分数在训练集上的标准差接近 1，避免 sigmoid 输出集中在 0.5 附近或两端饱和。
- 若训练集极小（N<3），std 可能不稳定，此时 sigmoid_scale_ 退化为 1.0。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import BaseClassifier, FEATURE_NAMES


# ─────────────────────────────────────────────────────────
# 先验权重（设计文档 4.5.2）
# F 轴特征权重 > C 轴特征权重（功能轴更直接反映任务完成质量）
# ─────────────────────────────────────────────────────────
_W_PRIOR_RAW: Dict[str, float] = {
    'F1': 1.1,
    'F2': 1.2,
    'F3': 1.0,
    'F4': 1.1,
    'C1': 0.8,
    'C2': 0.8,
    'C3': 0.8,
}

# 归一化使 Σw = len(FEATURE_NAMES) = 7
_total = sum(_W_PRIOR_RAW.values())
W_PRIOR: Dict[str, float] = {
    k: v * len(FEATURE_NAMES) / _total
    for k, v in _W_PRIOR_RAW.items()
}


class PurePriorScorer(BaseClassifier):
    """
    M2：纯先验加权线性评分器。

    Parameters
    ----------
    （无超参数，权重完全由先验决定）

    Attributes
    ----------
    w_ : np.ndarray, shape=(7,)
        归一化后的先验权重向量（fit 后可用）。
    sigmoid_scale_ : float
        sigmoid 缩放参数，由训练集 score 标准差决定（fit 后可用）。
    """

    # 分母最小值，防止 std≈0 时除零
    _SCALE_EPS: float = 1e-3

    def __init__(self) -> None:
        self.w_: np.ndarray | None = None
        self.sigmoid_scale_: float = 1.0

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_z: np.ndarray,
        y: np.ndarray,
        games: List[str],
    ) -> 'PurePriorScorer':
        """
        固定先验权重；用训练集计算 sigmoid 缩放参数。

        Parameters
        ----------
        X_z   : shape=(N, 7)，z-score 特征矩阵
        y     : shape=(N,)，标签（本模型不使用，仅为接口统一）
        games : 每个样本的游戏名（本模型不使用）

        Returns
        -------
        self
        """
        # 固定权重
        self.w_ = np.array([W_PRIOR[f] for f in FEATURE_NAMES], dtype=float)

        # sigmoid 缩放：让训练集 score 的 std ≈ 1
        scores = X_z @ self.w_                        # shape=(N,)
        self.sigmoid_scale_ = float(np.std(scores)) + self._SCALE_EPS

        return self

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X_z: np.ndarray) -> np.ndarray:
        """
        计算异常概率。

        score = X_z @ w
        prob  = sigmoid(score / sigmoid_scale_)

        Parameters
        ----------
        X_z : shape=(N, 7)

        Returns
        -------
        np.ndarray, shape=(N,)，值域 [0, 1]
        """
        self._check_fitted()
        scores = X_z @ self.w_
        return self._sigmoid(scores / self.sigmoid_scale_)

    # ------------------------------------------------------------------
    # get_feature_importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Dict[str, float]:
        """返回先验权重字典。"""
        self._check_fitted()
        return dict(zip(FEATURE_NAMES, self.w_))

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """数值稳定的 sigmoid。"""
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )

    def _check_fitted(self) -> None:
        if self.w_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted yet. "
                "Call fit() before predict_proba()."
            )

    def __repr__(self) -> str:
        if self.w_ is not None:
            return (
                f"PurePriorScorer(sigmoid_scale_={self.sigmoid_scale_:.4f})"
            )
        return "PurePriorScorer(unfitted)"
