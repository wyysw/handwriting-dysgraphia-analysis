"""
4.5.5  M4：Random Forest（对照基线）
=====================================
sklearn RandomForestClassifier 的薄封装。
作用：证明非线性模型未显著提升，支持"简单模型已够"的方法论主张。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseClassifier, FEATURE_NAMES


class RandomForestClassifier_wrap(BaseClassifier):
    """
    M4：Random Forest 对照分类器。

    Parameters
    ----------
    n_estimators : int, 默认 200
    max_depth    : int, 默认 4（浅树，抑制过拟合）
    min_samples_leaf : int, 默认 3
    random_state : int, 默认 42
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        min_samples_leaf: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model_: RandomForestClassifier | None = None

    def fit(self, X_z: np.ndarray, y: np.ndarray, games: List[str]) -> 'RandomForestClassifier_wrap':
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight='balanced',
            random_state=self.random_state,
        )
        self.model_.fit(X_z, y)
        return self

    def predict_proba(self, X_z: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("RandomForestClassifier_wrap has not been fitted yet.")
        return self.model_.predict_proba(X_z)[:, 1]

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model_ is None:
            raise RuntimeError("RandomForestClassifier_wrap has not been fitted yet.")
        return dict(zip(FEATURE_NAMES, self.model_.feature_importances_))

    def __repr__(self) -> str:
        return (
            f"RandomForestClassifier_wrap("
            f"n_estimators={self.n_estimators}, max_depth={self.max_depth})"
        )
