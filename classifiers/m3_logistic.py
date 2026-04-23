"""
4.5.4  M3：L2 Logistic Regression（主力分类器）
================================================
sklearn LogisticRegression 的薄封装，符合 BaseClassifier 接口。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseClassifier, FEATURE_NAMES


class L2LogisticClassifier(BaseClassifier):
    """
    M3：L2 正则化 Logistic Regression。

    Parameters
    ----------
    C : float, 默认 1.0
        正则化强度的倒数（越小越正则）。
        阶段6的 LOSO 循环中建议嵌套网格搜索 C ∈ {0.1, 0.3, 1.0, 3.0}。
    """

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self.model_: LogisticRegression | None = None

    def fit(self, X_z: np.ndarray, y: np.ndarray, games: List[str]) -> 'L2LogisticClassifier':
        self.model_ = LogisticRegression(
            penalty='l2',
            C=self.C,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
        )
        self.model_.fit(X_z, y)
        return self

    def predict_proba(self, X_z: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("L2LogisticClassifier has not been fitted yet.")
        return self.model_.predict_proba(X_z)[:, 1]

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model_ is None:
            raise RuntimeError("L2LogisticClassifier has not been fitted yet.")
        return dict(zip(FEATURE_NAMES, self.model_.coef_[0]))

    def __repr__(self) -> str:
        return f"L2LogisticClassifier(C={self.C})"
