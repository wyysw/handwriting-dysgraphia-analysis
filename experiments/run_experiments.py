"""
experiments/run_experiments.py
阶段6：LOSO 评估主流程

python experiments/run_experiments.py --feature_matrix output/feature_matrix.csv --gate_decisions output/gate_decisions.csv --out_dir results/

"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_recall_curve, roc_curve
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 特征名（与 feature_matrix.csv 列名一致）
# ─────────────────────────────────────────────
FEATURE_COLS = None   # 运行时自动从 CSV 推断（排除元数据列）
META_COLS = {"sample_id", "label", "game", "is_unanalyzable"}


# ══════════════════════════════════════════════════════════════════
# 4.6.1 / 4.6.2  模型定义
# ══════════════════════════════════════════════════════════════════

class PurePriorScorer:
    """
    M2：纯先验打分器。
    权重固定为先验（全1或手工设定），不从数据学习。
    得分 = sigmoid(sum(w_prior * x_z))
    """
    def __init__(self, prior_weights=None):
        self.prior_weights = prior_weights  # None → 等权
        self.w_ = None

    def fit(self, X, y, games=None):
        n_feat = X.shape[1]
        if self.prior_weights is not None:
            self.w_ = np.array(self.prior_weights, dtype=float)
        else:
            self.w_ = np.ones(n_feat, dtype=float)
        return self

    def predict_proba(self, X):
        scores = X @ self.w_
        return _sigmoid(scores)


class SemiPriorScorer:
    """
    M1：半先验打分器。
    先验权重作为初始值，用加权梯度下降在 train 上微调（带盒约束：权重 ≥ 0）。
    学习率和迭代次数是超参数，可通过构造器传入。
    """
    def __init__(self, prior_weights=None, lr=0.05, n_iter=200, clip_min=0.0, clip_max=5.0):
        self.prior_weights = prior_weights
        self.lr = lr
        self.n_iter = n_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.w_ = None

    def fit(self, X, y, games=None):
        n_feat = X.shape[1]
        if self.prior_weights is not None:
            w = np.array(self.prior_weights, dtype=float).copy()
        else:
            w = np.ones(n_feat, dtype=float)

        y = y.astype(float)
        for _ in range(self.n_iter):
            p = _sigmoid(X @ w)
            grad = X.T @ (p - y) / len(y)
            w -= self.lr * grad
            # 盒约束：权重非负，且不超过上界
            w = np.clip(w, self.clip_min, self.clip_max)

        self.w_ = w
        return self

    def predict_proba(self, X):
        scores = X @ self.w_
        return _sigmoid(scores)


class L2LogisticClassifier:
    """
    M3：L2 正则化逻辑回归（数据驱动软约束）。
    封装 sklearn LogisticRegression，统一接口。
    """
    def __init__(self, C=1.0, max_iter=500):
        self.C = C
        self.max_iter = max_iter
        self._clf = None

    def fit(self, X, y, games=None):
        self._clf = LogisticRegression(
            C=self.C, penalty="l2", solver="lbfgs",
            max_iter=self.max_iter, random_state=42
        )
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._clf.predict_proba(X)[:, 1]

    @property
    def coef_(self):
        return self._clf.coef_[0] if self._clf is not None else None


class RandomForestClassifierWrap:
    """
    M4：随机森林（非线性基线）。
    """
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._clf = None

    def fit(self, X, y, games=None):
        self._clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._clf.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self):
        return self._clf.feature_importances_ if self._clf is not None else None


# ══════════════════════════════════════════════════════════════════
# 归一化工具（per-game z-score，train-only 拟合）
# ══════════════════════════════════════════════════════════════════

def fit_normalize_stats(X_raw, games, labels):
    """
    在 train 集上（只用 label=0 样本）拟合每个 game 的归一化统计量。
    返回 dict: {game: {feat_idx: (mean, std)}}

    MAD 下限保护：std 最小值 = max(mad_based_std, mad_floor)
    """
    games = np.array(games)
    labels = np.array(labels)
    X_raw = np.array(X_raw)
    n_feat = X_raw.shape[1]

    stats = {}
    for g in np.unique(games):
        mask = (games == g) & (labels == 0)
        X_ref = X_raw[mask]
        if len(X_ref) == 0:
            # fallback: 用全 game 样本
            X_ref = X_raw[games == g]
        feat_stats = {}
        for f in range(n_feat):
            vals = X_ref[:, f]
            mean = np.mean(vals)
            # MAD 估计 std，下限保护（4.B 机制）
            mad = np.median(np.abs(vals - np.median(vals)))
            std_mad = 1.4826 * mad
            std_std = np.std(vals, ddof=1) if len(vals) > 1 else 1.0
            std = max(std_mad, std_std * 0.1, 1e-6)   # MAD下限保护
            feat_stats[f] = (mean, std)
        stats[g] = feat_stats
    return stats


def apply_normalize(X_raw, games, stats):
    """将原始特征矩阵按 per-game 统计量 z-score 化。"""
    X_raw = np.array(X_raw, dtype=float)
    games = np.array(games)
    X_z = np.zeros_like(X_raw)
    n_feat = X_raw.shape[1]

    for i, g in enumerate(games):
        if g not in stats:
            # 未见 game：使用均值0/方差1（不做归一化）
            X_z[i] = X_raw[i]
            continue
        for f in range(n_feat):
            mean, std = stats[g][f]
            X_z[i, f] = (X_raw[i, f] - mean) / std
    return X_z


# ══════════════════════════════════════════════════════════════════
# 评估指标工具
# ══════════════════════════════════════════════════════════════════

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def compute_metrics(y_true, y_prob):
    """计算 AUROC、AUPRC、最优阈值下的 F1 及阈值。"""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if len(np.unique(y_true)) < 2:
        return dict(auroc=float("nan"), auprc=float("nan"),
                    f1_opt=float("nan"), threshold_opt=float("nan"))

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1_vals = np.where((prec + rec) > 0,
                       2 * prec * rec / (prec + rec + 1e-12), 0)
    best_idx = np.argmax(f1_vals)
    f1_opt = f1_vals[best_idx]
    thr_opt = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    return dict(auroc=auroc, auprc=auprc, f1_opt=f1_opt, threshold_opt=thr_opt)


# ══════════════════════════════════════════════════════════════════
# 4.6.1 / 4.6.2  LOSO 主流程
# ══════════════════════════════════════════════════════════════════

def run_loso(df_analyzable, feature_cols, models_dict, renormalize=True):
    """
    执行严格防泄漏的 LOSO（Leave-One-Sample-Out）评估。

    参数
    ----
    df_analyzable : DataFrame（仅可分析样本）
    feature_cols  : 特征列名列表
    models_dict   : {name: model_instance}
    renormalize   : True → 每折重新归一化（防泄漏）
                    False → 使用已归一化特征（快速但泄漏）

    返回
    ----
    probs : {model_name: np.ndarray shape (N,)}
    """
    X_raw = df_analyzable[feature_cols].values.astype(float)
    y = df_analyzable["label"].values.astype(int)
    games = df_analyzable["game"].values
    N = len(df_analyzable)

    probs = {mname: np.zeros(N) for mname in models_dict}

    print(f"\n[LOSO] 共 {N} 个可分析样本，{'严格防泄漏（每折重归一化）' if renormalize else '快速模式（使用预归一化特征）'}")

    for i in range(N):
        if (i + 1) % max(1, N // 10) == 0:
            print(f"  fold {i+1}/{N} ...")

        train_idx = [j for j in range(N) if j != i]
        test_idx = [i]

        if renormalize:
            # ── 4.6.2 严格防泄漏：每折重新归一化 ──
            stats = fit_normalize_stats(
                X_raw[train_idx],
                games[train_idx],
                y[train_idx]
            )
            X_train_z = apply_normalize(X_raw[train_idx], games[train_idx], stats)
            X_test_z = apply_normalize(X_raw[test_idx], games[test_idx], stats)
        else:
            X_train_z = X_raw[train_idx]
            X_test_z = X_raw[test_idx]

        for mname, model in models_dict.items():
            model.fit(X_train_z, y[train_idx], games[train_idx])
            probs[mname][i] = model.predict_proba(X_test_z)[0]

    print("[LOSO] 完成。")
    return probs


# ══════════════════════════════════════════════════════════════════
# 4.6.3  乱画样本并入终判
# ══════════════════════════════════════════════════════════════════

def merge_unanalyzable(df_gate, df_analyzable, probs_loso, model_names):
    """
    将 LOSO 结果与门控判定的乱画样本合并，
    构成全样本 probs_full / y_full / games_full。
    """
    probs_full = {m: [] for m in model_names}
    y_full = []
    games_full = []
    sample_ids_full = []

    # 建立 sample_id → LOSO prob 索引
    loso_idx = {sid: idx for idx, sid in
                enumerate(df_analyzable["sample_id"].values)}

    for _, row in df_gate.iterrows():
        sid = row["sample_id"]
        y_full.append(int(row["label"]))
        games_full.append(row["game"])
        sample_ids_full.append(sid)

        if bool(row.get("is_unanalyzable", False)):
            # 门控直接判为异常
            for m in model_names:
                probs_full[m].append(1.0)
        else:
            if sid in loso_idx:
                idx = loso_idx[sid]
                for m in model_names:
                    probs_full[m].append(probs_loso[m][idx])
            else:
                # 不在 analyzable 中（数据对齐问题），赋 0.5
                for m in model_names:
                    probs_full[m].append(0.5)

    return (
        {m: np.array(v) for m, v in probs_full.items()},
        np.array(y_full),
        np.array(games_full),
        np.array(sample_ids_full)
    )


# ══════════════════════════════════════════════════════════════════
# 4.6.4  评估指标与报告
# ══════════════════════════════════════════════════════════════════

def evaluate_and_report(probs_full, y_full, games_full,
                        df_gate, model_names, out_dir):
    """生成三层指标表并保存 CSV。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 主任务性能 ──
    rows = []
    for m in model_names:
        met = compute_metrics(y_full, probs_full[m])
        met["model"] = m
        rows.append(met)
    df_main = pd.DataFrame(rows).set_index("model")[
        ["auroc", "auprc", "f1_opt", "threshold_opt"]
    ]

    # ── 门控诊断 ──
    unanalyzable_mask = df_gate["is_unanalyzable"].astype(bool)
    known_unanalyzable = unanalyzable_mask.sum()
    gate_recall = unanalyzable_mask[unanalyzable_mask].sum()  # = known_unanalyzable
    # 误判：正常样本被门控（此处即 is_unanalyzable=True 且 label=0）
    false_gate = ((unanalyzable_mask) & (df_gate["label"] == 0)).sum()
    # 门控未识别（乱画但未被门控）
    undetected_scribble = ((~unanalyzable_mask) & (df_gate["label"] == 1)
                           & df_gate.get("is_scribble", pd.Series(False,
                               index=df_gate.index)).astype(bool)).sum()

    gate_diag = {
        "known_unanalyzable": int(known_unanalyzable),
        "gate_recall": int(gate_recall),
        "false_gate_normal": int(false_gate),
        "undetected_scribble": int(undetected_scribble),
    }

    # ── per-game AUROC（M3 主力模型） ──
    m3_name = [m for m in model_names if "L2" in m or "M3" in m]
    m3_name = m3_name[0] if m3_name else model_names[2] if len(model_names) > 2 else model_names[0]

    per_game_rows = []
    for g in np.unique(games_full):
        mask = games_full == g
        if len(np.unique(y_full[mask])) < 2:
            auroc_g = float("nan")
        else:
            auroc_g = roc_auc_score(y_full[mask], probs_full[m3_name][mask])
        per_game_rows.append({"game": g, "auroc": auroc_g,
                               "n_total": int(mask.sum()),
                               "n_pos": int(y_full[mask].sum())})
    df_per_game = pd.DataFrame(per_game_rows).set_index("game")

    # ── 打印报告 ──
    sep = "═" * 65
    print(f"\n{sep}")
    print("  【主任务性能】（全部样本）")
    print(sep)
    print(df_main.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\n{sep}")
    print("  【门控诊断】")
    print(sep)
    for k, v in gate_diag.items():
        print(f"  {k:40s}: {v}")

    print(f"\n{sep}")
    print(f"  【per-game AUROC】（模型: {m3_name}）")
    print(sep)
    print(df_per_game.to_string(float_format=lambda x: f"{x:.4f}"))
    print(sep)

    # 叙事对比
    model_names_list = list(model_names)
    if len(model_names_list) >= 4:
        m1, m2, m3, m4 = model_names_list[:4]
        print("\n【跨模型对比叙事】")
        print(f"  M2→M1 Δ AUROC = {df_main.loc[m1,'auroc']-df_main.loc[m2,'auroc']:+.4f}  （先验微调边际价值）")
        print(f"  M1→M3 Δ AUROC = {df_main.loc[m3,'auroc']-df_main.loc[m1,'auroc']:+.4f}  （硬正则 vs 盒约束）")
        print(f"  M3→M4 Δ AUROC = {df_main.loc[m4,'auroc']-df_main.loc[m3,'auroc']:+.4f}  （非线性增益）")
        if abs(df_main.loc[m4,'auroc'] - df_main.loc[m3,'auroc']) < 0.02:
            print("  → M3 ≈ M4：本任务接近线性可分，简单模型已足够 ✓")

    # ── 保存 ──
    df_main.to_csv(out_dir / "metrics_main.csv")
    pd.DataFrame([gate_diag]).to_csv(out_dir / "gate_diagnostics.csv", index=False)
    df_per_game.to_csv(out_dir / "per_game_auroc.csv")
    print(f"\n[保存] metrics_main.csv / gate_diagnostics.csv / per_game_auroc.csv → {out_dir}")

    return df_main, gate_diag, df_per_game


# ══════════════════════════════════════════════════════════════════
# 4.6.5  可解释性图表
# ══════════════════════════════════════════════════════════════════

def plot_feature_weights(models_dict, feature_cols, out_dir, model_names_order=None):
    """
    4.6.5 特征权重对比条形图 + Spearman 秩相关系数分析。
    """
    out_dir = Path(out_dir)
    n_feat = len(feature_cols)
    order = model_names_order or list(models_dict.keys())

    weight_matrix = {}
    for mname in order:
        m = models_dict[mname]
        if hasattr(m, "w_"):
            w = m.w_
        elif hasattr(m, "coef_") and m.coef_ is not None:
            w = np.abs(m.coef_)
        elif hasattr(m, "feature_importances_") and m.feature_importances_ is not None:
            w = m.feature_importances_
        else:
            w = np.zeros(n_feat)
        # 对齐长度
        if len(w) != n_feat:
            w = np.zeros(n_feat)
        weight_matrix[mname] = w

    # ── 条形图 ──
    fig, axes = plt.subplots(len(order), 1, figsize=(max(10, n_feat * 0.8), 2.5 * len(order)),
                              sharex=True)
    if len(order) == 1:
        axes = [axes]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    x = np.arange(n_feat)

    for ax, mname, color in zip(axes, order, colors):
        w = weight_matrix.get(mname, np.zeros(n_feat))
        ax.bar(x, w, color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(mname, fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Feature Weights / Importances by Model", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "feature_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] feature_weights.png → {out_dir}")

    # ── Spearman 秩相关 ──
    wmat = np.stack([weight_matrix[m] for m in order])   # (n_models, n_feat)
    n_models = len(order)
    spearman_mat = np.eye(n_models)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            r, _ = spearmanr(wmat[i], wmat[j])
            spearman_mat[i, j] = r
            spearman_mat[j, i] = r

    df_sp = pd.DataFrame(spearman_mat, index=order, columns=order)
    print("\n【特征重要性 Spearman 秩相关系数】")
    print(df_sp.to_string(float_format=lambda x: f"{x:.3f}"))
    overall_mean = spearman_mat[np.triu_indices(n_models, k=1)].mean()
    print(f"  均值 = {overall_mean:.3f}", "✓ 跨模型一致" if overall_mean >= 0.7 else "⚠ 一致性较低")
    df_sp.to_csv(out_dir / "spearman_feature_rank.csv")

    # ── Spearman 热图 ──
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(spearman_mat, vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n_models)); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_models)); ax.set_yticklabels(order, fontsize=8)
    for i in range(n_models):
        for j in range(n_models):
            ax.text(j, i, f"{spearman_mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Feature Rank Correlation across Models")
    plt.tight_layout()
    fig.savefig(out_dir / "spearman_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] spearman_heatmap.png → {out_dir}")

    return df_sp


def plot_roc_pr_curves(probs_full, y_full, model_names, out_dir):
    """绘制所有模型的 ROC 和 PR 曲线。"""
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#9467BD", "#8C564B", "#E377C2"]

    for mname, color in zip(model_names, colors):
        y_prob = probs_full[mname]
        # ROC
        fpr, tpr, _ = roc_curve(y_full, y_prob)
        auc = roc_auc_score(y_full, y_prob)
        axes[0].plot(fpr, tpr, color=color, lw=1.8, label=f"{mname} (AUC={auc:.3f})")
        # PR
        prec, rec, _ = precision_recall_curve(y_full, y_prob)
        ap = average_precision_score(y_full, y_prob)
        axes[1].plot(rec, prec, color=color, lw=1.8, label=f"{mname} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve"); axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("PR Curve"); axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] roc_pr_curves.png → {out_dir}")


# ══════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════

def build_models():
    """实例化四个模型，返回 OrderedDict。"""
    from collections import OrderedDict
    return OrderedDict([
        ("M2_PurePrior",    PurePriorScorer()),
        ("M1_SemiPrior",    SemiPriorScorer()),
        ("M3_L2LR",         L2LogisticClassifier(C=1.0)),
        ("M4_RF",           RandomForestClassifierWrap(n_estimators=100)),
    ])


def main():
    parser = argparse.ArgumentParser(description="阶段6 LOSO 评估主流程")
    parser.add_argument("--feature_matrix", required=True,
                        help="output/feature_matrix.csv（可分析样本的特征矩阵）")
    parser.add_argument("--gate_decisions", required=True,
                        help="output/gate_decisions.csv（含乱画标记的门控决定）")
    parser.add_argument("--out_dir", default="results/",
                        help="输出目录（默认 results/）")
    parser.add_argument("--no_renormalize", action="store_true",
                        help="跳过每折重归一化（快速但有数据泄漏，仅用于调试）")
    parser.add_argument("--save_probs", action="store_true",
                        help="是否将全样本预测概率保存为 CSV")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 读数据 ──
    print(f"[读取] feature_matrix: {args.feature_matrix}")
    df_analyzable = pd.read_csv(args.feature_matrix)
    print(f"[读取] gate_decisions: {args.gate_decisions}")
    df_gate = pd.read_csv(args.gate_decisions)

    # 推断特征列
    global FEATURE_COLS
    FEATURE_COLS = [c for c in df_analyzable.columns if c not in META_COLS]
    print(f"[特征] {len(FEATURE_COLS)} 个特征列: {FEATURE_COLS}")

    # 强制 sample_id 列存在
    if "sample_id" not in df_analyzable.columns:
        df_analyzable = df_analyzable.reset_index().rename(columns={"index": "sample_id"})
    if "sample_id" not in df_gate.columns:
        df_gate = df_gate.reset_index().rename(columns={"index": "sample_id"})

    # ── 4.6.1 / 4.6.2  LOSO ──
    models_dict = build_models()
    renormalize = not args.no_renormalize
    probs_loso = run_loso(df_analyzable, FEATURE_COLS, models_dict,
                          renormalize=renormalize)

    # ── 4.6.3  合并乱画 ──
    model_names = list(models_dict.keys())
    probs_full, y_full, games_full, sample_ids_full = merge_unanalyzable(
        df_gate, df_analyzable, probs_loso, model_names
    )

    # 可选：保存全样本概率
    if args.save_probs:
        df_probs = pd.DataFrame({"sample_id": sample_ids_full,
                                  "label": y_full, "game": games_full})
        for m in model_names:
            df_probs[f"prob_{m}"] = probs_full[m]
        df_probs.to_csv(out_dir / "full_probs.csv", index=False)
        print(f"[保存] full_probs.csv → {out_dir}")

    # ── 4.6.4  评估指标 ──
    df_main, gate_diag, df_per_game = evaluate_and_report(
        probs_full, y_full, games_full, df_gate, model_names, out_dir
    )

    # ── 4.6.5  可解释性图表 ──
    # 用全量数据（全部 analyzable 样本）重新拟合一次，提取权重用于可视化
    print("\n[可解释性] 在全量数据上重新拟合以提取特征权重 ...")
    X_all = df_analyzable[FEATURE_COLS].values.astype(float)
    y_all = df_analyzable["label"].values.astype(int)
    games_all = df_analyzable["game"].values

    stats_all = fit_normalize_stats(X_all, games_all, y_all)
    X_all_z = apply_normalize(X_all, games_all, stats_all)

    viz_models = build_models()  # 新实例，避免污染 LOSO 结果
    for m in viz_models.values():
        m.fit(X_all_z, y_all, games_all)

    plot_feature_weights(viz_models, FEATURE_COLS, out_dir,
                         model_names_order=list(viz_models.keys()))
    plot_roc_pr_curves(probs_full, y_full, model_names, out_dir)

    print(f"\n{'─'*65}")
    print(f"[完成] 所有结果已保存至 {out_dir.resolve()}")
    print(f"{'─'*65}")


if __name__ == "__main__":
    main()
