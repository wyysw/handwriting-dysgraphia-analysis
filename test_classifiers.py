"""
测试脚本：验证 classifiers 包的正确性
用法：python test_classifiers.py
"""

import sys
import numpy as np

sys.path.insert(0, '/home/claude')
from classifiers import (
    BaseClassifier, FEATURE_NAMES,
    PurePriorScorer, SemiPriorScorer,
    L2LogisticClassifier, RandomForestClassifier_wrap,
)
from classifiers.m2_pure_prior import W_PRIOR


def make_synthetic_data(n=60, seed=42):
    """生成合成数据：label=1 样本在 F1/F2 上异常偏高（方向已统一为越大越异常）"""
    rng = np.random.RandomState(seed)
    n0, n1 = n // 2, n - n // 2
    games = ['sym'] * (n // 3) + ['maze'] * (n // 3) + ['circle'] * (n - 2 * (n // 3))

    X0 = rng.randn(n0, 7) * 0.8          # 正常样本：z 值在 0 附近
    X1 = rng.randn(n1, 7) * 0.8 + 2.0   # 障碍样本：z 值偏高

    X = np.vstack([X0, X1]).astype(float)
    y = np.array([0] * n0 + [1] * n1, dtype=int)

    # 打乱
    idx = rng.permutation(n)
    return X[idx], y[idx], [games[i] for i in idx]


def auroc(proba, y):
    pos = proba[y == 1]
    neg = proba[y == 0]
    return float(np.mean(pos[:, None] > neg[None, :]))


def run_tests():
    print("=" * 60)
    print("  classifiers 包测试")
    print("=" * 60)

    # ── 基础检查 ──────────────────────────────────────────────
    print("\n[1] FEATURE_NAMES:", FEATURE_NAMES)
    assert len(FEATURE_NAMES) == 7, "特征数量应为 7"

    print("[2] W_PRIOR（归一化后）:")
    total_w = sum(W_PRIOR.values())
    for f, w in W_PRIOR.items():
        print(f"    {f}: {w:.4f}")
    print(f"    Σw = {total_w:.4f}  (期望=7.0)")
    assert abs(total_w - 7.0) < 1e-9, "W_PRIOR 归一化后 Σw 应=7"

    # ── 合成数据 ──────────────────────────────────────────────
    X, y, games = make_synthetic_data(n=60)
    X_train, y_train, g_train = X[:50], y[:50], games[:50]
    X_test,  y_test,  g_test  = X[50:], y[50:], games[50:]

    print(f"\n[3] 合成数据: N={len(X)}, pos={y.sum()}, neg={(y==0).sum()}")

    # ── M2 PurePriorScorer ────────────────────────────────────
    print("\n── M2 PurePriorScorer ──")
    m2 = PurePriorScorer()
    assert isinstance(m2, BaseClassifier), "应继承 BaseClassifier"
    m2.fit(X_train, y_train, g_train)
    p2 = m2.predict_proba(X_test)
    assert p2.shape == (len(X_test),), "输出 shape 错误"
    assert 0.0 <= p2.min() and p2.max() <= 1.0, "概率应在 [0,1]"
    auc2 = auroc(m2.predict_proba(X), y)
    imp2 = m2.get_feature_importance()
    assert set(imp2.keys()) == set(FEATURE_NAMES), "importance 键应为 FEATURE_NAMES"
    print(f"  sigmoid_scale_ = {m2.sigmoid_scale_:.4f}")
    print(f"  AUROC(全集)    = {auc2:.4f}")
    print(f"  importance     = { {k: round(v,3) for k,v in imp2.items()} }")
    print(f"  repr           = {m2}")

    # ── M1 SemiPriorScorer ────────────────────────────────────
    print("\n── M1 SemiPriorScorer ──")
    m1 = SemiPriorScorer(bounds=0.3)
    assert isinstance(m1, BaseClassifier), "应继承 BaseClassifier"
    m1.fit(X_train, y_train, g_train)
    p1 = m1.predict_proba(X_test)
    assert p1.shape == (len(X_test),), "输出 shape 错误"
    assert 0.0 <= p1.min() and p1.max() <= 1.0, "概率应在 [0,1]"
    auc1 = auroc(m1.predict_proba(X), y)
    imp1 = m1.get_feature_importance()
    print(f"  优化收敛: {m1.opt_result_.success}")
    print(f"  sigmoid_scale_ = {m1.sigmoid_scale_:.4f}")
    print(f"  AUROC(全集)    = {auc1:.4f}")
    print(f"  importance     = { {k: round(v,3) for k,v in imp1.items()} }")
    print(f"  repr           = {m1}")

    print("\n  边界贴靠诊断:")
    for f, info in m1.boundary_report().items():
        boundary_str = f"[贴边界:{info['at_boundary']}]" if info['at_boundary'] != 'none' else ""
        print(f"    {f}: prior={info['w_prior']:.3f} → opt={info['w_opt']:.3f}  "
              f"[{info['lo']:.3f}, {info['hi']:.3f}] {boundary_str}")

    # M1 AUROC 应 ≥ M2 AUROC（在训练集上）
    auc1_train = auroc(m1.predict_proba(X_train), y_train)
    auc2_train = auroc(m2.predict_proba(X_train), y_train)
    print(f"\n  训练集 AUROC: M1={auc1_train:.4f}  M2={auc2_train:.4f}")
    assert auc1_train >= auc2_train - 0.01, \
        "M1 在训练集上的 AUROC 应不低于 M2（盒约束优化不应更差）"

    # ── M3 L2 Logistic ────────────────────────────────────────
    print("\n── M3 L2LogisticClassifier ──")
    m3 = L2LogisticClassifier(C=1.0)
    m3.fit(X_train, y_train, g_train)
    p3 = m3.predict_proba(X_test)
    auc3 = auroc(m3.predict_proba(X), y)
    print(f"  AUROC(全集) = {auc3:.4f}")
    print(f"  importance  = { {k: round(v,3) for k,v in m3.get_feature_importance().items()} }")

    # ── M4 Random Forest ──────────────────────────────────────
    print("\n── M4 RandomForestClassifier_wrap ──")
    m4 = RandomForestClassifier_wrap()
    m4.fit(X_train, y_train, g_train)
    p4 = m4.predict_proba(X_test)
    auc4 = auroc(m4.predict_proba(X), y)
    print(f"  AUROC(全集) = {auc4:.4f}")
    print(f"  importance  = { {k: round(v,3) for k,v in m4.get_feature_importance().items()} }")

    # ── predict（阈值分类）──────────────────────────────────────
    print("\n── predict 方法 ──")
    for name, clf in [('M2', m2), ('M1', m1), ('M3', m3), ('M4', m4)]:
        preds = clf.predict(X_test)
        assert set(preds).issubset({0, 1}), "predict 应返回 0/1"
        print(f"  {name} predict 示例: {preds[:10]}")

    # ── 未 fit 时应抛出异常 ────────────────────────────────────
    print("\n── 未 fit 时异常检测 ──")
    for cls in [PurePriorScorer, SemiPriorScorer, L2LogisticClassifier, RandomForestClassifier_wrap]:
        try:
            cls().predict_proba(X_test)
            assert False, f"{cls.__name__} 未 fit 时应抛出异常"
        except RuntimeError:
            print(f"  {cls.__name__}: 正确抛出 RuntimeError ✓")

    print("\n" + "=" * 60)
    print("  所有测试通过 ✓")
    print("=" * 60)


if __name__ == '__main__':
    run_tests()
