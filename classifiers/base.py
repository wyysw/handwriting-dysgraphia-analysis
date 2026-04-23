"""
4.5.1  四个模型的统一接口
============================
BaseClassifier 定义了所有分类器必须实现的三个方法：
  - fit        训练
  - predict_proba  输出异常概率
  - get_feature_importance  返回特征权重（便于跨模型对比）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


# ─────────────────────────────────────────────────────────
# 全局常量：7 个特征的有序名称（与 feature_matrix.csv 列顺序一致）
# ─────────────────────────────────────────────────────────
FEATURE_NAMES: List[str] = ['F1', 'F2', 'F3', 'F4', 'C1', 'C2', 'C3']


class BaseClassifier(ABC):
    """
    所有分类器的抽象基类。

    子类必须实现：
        fit(X_z, y, games) -> self
        predict_proba(X_z) -> np.ndarray  shape=(N,)，值域 [0,1]
        get_feature_importance()          -> Dict[str, float]
    """

    # ------------------------------------------------------------------
    # 抽象方法
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_z: np.ndarray,   # shape=(N, 7)，已完成 z-score 归一化且方向统一为"越大越异常"
        y: np.ndarray,     # shape=(N,)，0=正常，1=书写障碍
        games: List[str],  # 每个样本的游戏标签，如 ['sym','maze','circle',...]
    ) -> 'BaseClassifier':
        """训练分类器，返回 self（支持链式调用）。"""
        ...

    @abstractmethod
    def predict_proba(self, X_z: np.ndarray) -> np.ndarray:
        """
        返回每个样本为"书写障碍（异常）"的概率。

        Parameters
        ----------
        X_z : np.ndarray, shape=(N, 7)
            与 fit 时格式相同的 z-score 特征矩阵。

        Returns
        -------
        np.ndarray, shape=(N,)
            每个样本的异常概率 p ∈ [0, 1]。
        """
        ...

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        返回特征权重字典，便于跨模型对比与报告。

        Returns
        -------
        Dict[str, float]
            键为特征名（F1–C3），值为对应权重/重要性分数。
        """
        ...

    # ------------------------------------------------------------------
    # 公共辅助方法（子类可直接使用，无需重写）
    # ------------------------------------------------------------------

    def predict(self, X_z: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        将 predict_proba 结果按阈值转换为 0/1 标签。

        Parameters
        ----------
        X_z       : np.ndarray, shape=(N, 7)
        threshold : float，默认 0.5

        Returns
        -------
        np.ndarray, shape=(N,)，int
        """
        return (self.predict_proba(X_z) >= threshold).astype(int)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
