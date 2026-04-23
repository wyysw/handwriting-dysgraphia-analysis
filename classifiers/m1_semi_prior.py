"""
4.5.3  M1：半先验加权线性评分（主推方法）
==========================================
在先验权重基础上，通过约束优化（盒约束 L-BFGS-B）微调权重，
目标：最大化训练集 AUROC（近似为 Wilcoxon-Mann-Whitney 统计量）。

设计说明
--------
bounds=0.3 表示每个特征权重在先验值的 ±30% 内浮动：
  lo_i = w_prior_i * (1 - bounds)
  hi_i = w_prior_i * (1 + bounds)

若优化结果贴到某维边界，说明数据强烈支持调高/调低该特征权重——
这本身是可报告的发现（哪个特征判别力更强/更弱）。

AUROC 近似
----------
  AUROC ≈ mean(score[y==1][:, None] > score[y==0][None, :])
即正例分数高于负例分数的比例，是 Wilcoxon-Mann-Whitney U 统计量的归一化形式。
梯度由 L-BFGS-B 以数值差分估计（无需解析梯度），对 N≤200 的小数据集速度足够。

继承 PurePriorScorer
--------------------
fit 后 self.w_ / self.sigmoid_scale_ 与父类语义完全一致，
predict_proba / get_feature_importance 直接复用父类实现。
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import minimize

from .base import FEATURE_NAMES
from .m2_pure_prior import PurePriorScorer, W_PRIOR


class SemiPriorScorer(PurePriorScorer):
    """
    M1：半先验加权线性评分器。

    在先验权重 ±bounds 的盒约束内，用训练数据微调权重以最大化 AUROC。

    Parameters
    ----------
    bounds : float, 默认 0.3
        每个特征权重相对于先验值的最大偏移比例（±30%）。
    tol : float, 默认 1e-4
        scipy.optimize.minimize 的收敛容差。
    max_iter : int, 默认 500
        L-BFGS-B 最大迭代次数。

    Attributes
    ----------
    w_ : np.ndarray, shape=(7,)
        优化后的权重向量（fit 后可用）。
    sigmoid_scale_ : float
        sigmoid 缩放参数（fit 后可用，与父类一致）。
    opt_result_ : OptimizeResult
        scipy 优化结果，含收敛信息（fit 后可用，供调试）。
    """

    def __init__(
        self,
        bounds: float = 0.3,
        tol: float = 1e-4,
        max_iter: int = 500,
    ) -> None:
        super().__init__()
        self.bounds = bounds
        self.tol = tol
        self.max_iter = max_iter
        self.opt_result_ = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_z: np.ndarray,
        y: np.ndarray,
        games: List[str],
    ) -> 'SemiPriorScorer':
        """
        在先验权重盒约束内最大化 AUROC，再标定 sigmoid 缩放参数。

        Parameters
        ----------
        X_z   : shape=(N, 7)，z-score 特征矩阵
        y     : shape=(N,)，标签（0=正常，1=障碍）
        games : 每个样本的游戏名（本模型暂不使用）

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=int)
        X_z = np.asarray(X_z, dtype=float)

        # ── 先验权重向量 ──────────────────────────────────────────────
        w_init = np.array([W_PRIOR[f] for f in FEATURE_NAMES], dtype=float)

        # ── 盒约束 ────────────────────────────────────────────────────
        lo = w_init * (1.0 - self.bounds)
        hi = w_init * (1.0 + self.bounds)
        box_bounds = list(zip(lo, hi))

        # ── 目标函数：负 AUROC（最小化 = 最大化 AUROC）────────────────
        def neg_auroc(w: np.ndarray) -> float:
            score = X_z @ w                    # shape=(N,)
            pos = score[y == 1]
            neg = score[y == 0]
            if len(pos) == 0 or len(neg) == 0:
                # 退化情况（单一类别），返回 0（即 AUROC=0，优化无意义）
                return 0.0
            # Wilcoxon-Mann-Whitney：正例 score > 负例 score 的比例
            # 矩阵广播：pos[:, None] > neg[None, :]，shape=(P, Q)
            auroc_approx = float(np.mean(pos[:, None] > neg[None, :]))
            return -auroc_approx

        # ── L-BFGS-B 优化 ─────────────────────────────────────────────
        self.opt_result_ = minimize(
            neg_auroc,
            x0=w_init,
            method='L-BFGS-B',
            bounds=box_bounds,
            options={
                'maxiter': self.max_iter,
                'ftol': self.tol,
                'gtol': self.tol * 0.1,
            },
        )
        self.w_ = self.opt_result_.x  # 优化后的权重

        # ── sigmoid 缩放（与父类逻辑相同）────────────────────────────
        scores = X_z @ self.w_
        self.sigmoid_scale_ = float(np.std(scores)) + self._SCALE_EPS

        return self

    # ------------------------------------------------------------------
    # 额外诊断：展示哪些特征权重贴到边界
    # ------------------------------------------------------------------

    def boundary_report(self) -> dict:
        """
        返回各特征权重是否贴到盒约束边界的诊断信息。

        Returns
        -------
        dict, 键为特征名，值为 {'w_opt', 'w_prior', 'lo', 'hi', 'at_boundary'}
        """
        self._check_fitted()
        w_init = np.array([W_PRIOR[f] for f in FEATURE_NAMES], dtype=float)
        lo = w_init * (1.0 - self.bounds)
        hi = w_init * (1.0 + self.bounds)
        report = {}
        for i, f in enumerate(FEATURE_NAMES):
            w_opt = float(self.w_[i])
            at_lo = abs(w_opt - lo[i]) < 1e-6
            at_hi = abs(w_opt - hi[i]) < 1e-6
            report[f] = {
                'w_opt':       round(w_opt, 6),
                'w_prior':     round(float(w_init[i]), 6),
                'lo':          round(float(lo[i]), 6),
                'hi':          round(float(hi[i]), 6),
                'at_boundary': 'lo' if at_lo else ('hi' if at_hi else 'none'),
            }
        return report

    def __repr__(self) -> str:
        if self.w_ is not None:
            converged = getattr(self.opt_result_, 'success', None)
            return (
                f"SemiPriorScorer("
                f"bounds={self.bounds}, "
                f"converged={converged}, "
                f"sigmoid_scale_={self.sigmoid_scale_:.4f})"
            )
        return f"SemiPriorScorer(bounds={self.bounds}, unfitted)"
