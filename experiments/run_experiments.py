"""
4.6  阶段6：实验评估
======================

对阶段5实现的四个模型（M1, M2, M3, M4）进行严格的 LOSO 交叉验证，
并产出三层指标报告与可解释性分析。

关键设计
--------
1. **严格防泄漏的 LOSO**（4.6.2）：
   每折在 train 集上重新拟合 normalize stats，再 transform 到 test。
   绝不使用全样本预先计算的 feature_matrix.csv。

2. **乱画样本旁路**（4.6.3）：
   被门控判为乱画的样本不参与 LOSO，直接赋 prob=1.0；
   最终的 probs_full / y_full 包含全部样本，反映真实筛查系统性能。

3. **M3 嵌套网格搜索**（4.5.4 / 4.6.1）：
   外层 LOSO 的每折内部，对 C ∈ {0.1, 0.3, 1.0, 3.0} 做内层 LOO 选 C。

python experiments/run_experiments.py --feature_csv data/feature/all.csv --out_dir results/

输出
----
    results/
      ├── per_sample_predictions.csv     每个样本、每个模型的 predicted prob
      ├── main_metrics.csv               三层指标表的核心层（4 个模型 × 4 项指标）
      ├── per_game_auroc.csv             M3 的 per-game AUROC 拆分
      ├── feature_importance.csv         四个模型的特征权重 / coef / importance
      ├── feature_importance_spearman.csv  四组权重的 Spearman 相关矩阵
      ├── gating_diagnostics.json        门控召回/误判等
      └── report.txt                     人类可读的汇总报告
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Tuple

# 抑制 sklearn 1.8+ 关于 LogisticRegression(penalty=...) 的 FutureWarning。
# stage5 的 m3_logistic.py 显式传入 penalty='l2'，未来版本将改用 l1_ratio；
# 这与本阶段的实验结果无关，过滤掉以保持报告输出整洁。
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='sklearn.linear_model._logistic',
)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ── 路径处理：让脚本既可作为模块被调用，也可直接 python 运行 ─────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from classifiers.base import FEATURE_NAMES                       # noqa: E402
from classifiers.m1_semi_prior import SemiPriorScorer            # noqa: E402
from classifiers.m2_pure_prior import PurePriorScorer            # noqa: E402
from classifiers.m3_logistic import L2LogisticClassifier         # noqa: E402
from classifiers.m4_random_forest import RandomForestClassifier_wrap  # noqa: E402
from features.gate_unanalyzable import apply_gate               # noqa: E402
from features.normalize import (                                # noqa: E402
    apply_normalize,
    fit_normalize_stats,
)


def gating_diagnostics(
    df_gated: pd.DataFrame,
    known_unanalyzable_ids: List[str] | None = None,
) -> dict:
    """
    生成阶段6报告所需的门控诊断信息。

    Parameters
    ----------
    df_gated : pd.DataFrame
        apply_gate() 的输出（含 is_unanalyzable, label 列）。
    known_unanalyzable_ids : list[str] or None
        已知乱画样本 ID（用于召回率计算）。
        若为 None，则把"被门控判为乱画"的样本视为真值（输出召回 = 1.0）。

    Returns
    -------
    dict 含: known_unanalyzable / gate_recall / undetected_scribble / false_gate_normal
    """
    flagged = df_gated['is_unanalyzable']

    if known_unanalyzable_ids is None:
        known_set = set(df_gated.loc[flagged, 'sample_id'].tolist())
    else:
        known_set = set(known_unanalyzable_ids)

    is_known = df_gated['sample_id'].isin(known_set)

    diag = {
        'known_unanalyzable':  int(is_known.sum()),
        'gate_recall':         int((is_known & flagged).sum()),
        'undetected_scribble': int((is_known & ~flagged).sum()),
    }
    if 'label' in df_gated.columns:
        diag['false_gate_normal'] = int(
            ((df_gated['label'] == 0) & flagged).sum()
        )
    else:
        diag['false_gate_normal'] = 0
    return diag


# ─────────────────────────────────────────────────────────
# 模型工厂：保持每折独立实例
# ─────────────────────────────────────────────────────────
M3_C_GRID: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0)


def build_models() -> Dict[str, object]:
    """每次调用返回一组全新的、未训练的模型实例。"""
    return {
        'M2_PurePrior': PurePriorScorer(),
        'M1_SemiPrior': SemiPriorScorer(bounds=0.3),
        # M3_L2LR 的 C 在 LOSO 每折内部用嵌套 LOO 选；这里占位
        'M3_L2LR':      L2LogisticClassifier(C=1.0),
        'M4_RF':        RandomForestClassifier_wrap(),
    }


MODEL_NAMES: Tuple[str, ...] = (
    'M2_PurePrior',
    'M1_SemiPrior',
    'M3_L2LR',
    'M4_RF',
)


# ─────────────────────────────────────────────────────────
# 嵌套 CV：为 M3 选 C
# ─────────────────────────────────────────────────────────
def select_best_C(
    feat_dicts_train: List[dict],
    games_train: List[str],
    y_train: np.ndarray,
    C_grid: Tuple[float, ...] = M3_C_GRID,
) -> float:
    """
    在外层 train 集上做内层 LOO，挑选使 mean AUROC 最高的 C。
    每个内层 fold 同样要重新拟合 normalize stats（保持嵌套防泄漏）。

    若内层 train 中某类样本不足 2 个（无法计算 AUROC），则跳过该 fold。
    """
    n = len(feat_dicts_train)
    best_C = 1.0
    best_auc = -np.inf

    for C in C_grid:
        # 内层 LOO 收集 oof 概率
        oof_proba = np.full(n, np.nan, dtype=float)

        for i in range(n):
            inner_train_idx = [j for j in range(n) if j != i]
            inner_test_idx = [i]

            # 内层 train 上重新拟合归一化：
            # 仅用 label=0 行作为参考池（外层已剔除乱画样本，
            # fit_normalize_stats 内部会按 label==0 进一步筛选）
            inner_train_dicts = [feat_dicts_train[j] for j in inner_train_idx]
            inner_train_games = [games_train[j] for j in inner_train_idx]
            inner_train_labels = [int(y_train[j]) for j in inner_train_idx]

            inner_stats = fit_normalize_stats(
                feature_dicts=inner_train_dicts,
                games=inner_train_games,
                labels=inner_train_labels,
            )
            X_tr = apply_normalize(
                feature_dicts=inner_train_dicts,
                games=inner_train_games,
                stats=inner_stats,
            )
            X_te = apply_normalize(
                feature_dicts=[feat_dicts_train[j] for j in inner_test_idx],
                games=[games_train[j] for j in inner_test_idx],
                stats=inner_stats,
            )

            y_tr = y_train[inner_train_idx]
            if len(np.unique(y_tr)) < 2:
                continue

            mdl = L2LogisticClassifier(C=C)
            mdl.fit(
                X_tr,
                y_tr,
                games=inner_train_games,
            )
            oof_proba[i] = float(mdl.predict_proba(X_te)[0])

        # 计算内层 AUROC（仅对有 oof 概率的样本）
        valid = ~np.isnan(oof_proba)
        if valid.sum() < 2 or len(np.unique(y_train[valid])) < 2:
            continue
        auc_C = roc_auc_score(y_train[valid], oof_proba[valid])
        if auc_C > best_auc:
            best_auc = auc_C
            best_C = C

    return best_C


# ─────────────────────────────────────────────────────────
# LOSO 主循环
# ─────────────────────────────────────────────────────────
def run_loso(
    df_analyzable: pd.DataFrame,
    do_nested_cv_for_m3: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    """
    对可分析样本运行严格 LOSO，每折独立归一化、独立训练、独立预测。

    Parameters
    ----------
    df_analyzable : pd.DataFrame
        包含列 sample_id, game, label, F1..C3，且 is_unanalyzable=False 的样本
        （或上游已剔除乱画样本）。
    do_nested_cv_for_m3 : bool
        True 时为 M3 在每折内做嵌套 LOO 选 C；False 时固定 C=1.0（节省时间）。
    verbose : bool
        是否打印每折进度。

    Returns
    -------
    probs_oof : Dict[str, np.ndarray]
        每个模型在 LOSO 下的 out-of-fold 概率，shape=(N_analyzable,)
    chosen_C_per_fold : Dict[str, List[float]]
        每折为 M3 选定的 C（dict 只含 'M3_L2LR' 一个键）
    """
    n = len(df_analyzable)
    feat_dicts = df_analyzable[list(FEATURE_NAMES)].to_dict('records')
    games = df_analyzable['game'].tolist()
    y = df_analyzable['label'].to_numpy().astype(int)
    # 进入 LOSO 的样本均已可分析；fit_normalize_stats 内部会按 label==0 筛参考池

    probs_oof = {m: np.full(n, np.nan, dtype=float) for m in MODEL_NAMES}
    chosen_C_per_fold = {'M3_L2LR': []}

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        test_idx = [i]

        train_dicts = [feat_dicts[j] for j in train_idx]
        train_games = [games[j] for j in train_idx]
        train_labels = [int(y[j]) for j in train_idx]

        # ── 1. 在 train 上重新拟合归一化 ─────────────────────────────
        stats = fit_normalize_stats(
            feature_dicts=train_dicts,
            games=train_games,
            labels=train_labels,
        )
        X_tr = apply_normalize(
            feature_dicts=train_dicts,
            games=train_games,
            stats=stats,
        )
        X_te = apply_normalize(
            feature_dicts=[feat_dicts[j] for j in test_idx],
            games=[games[j] for j in test_idx],
            stats=stats,
        )
        y_tr = y[train_idx]

        # ── 2. 训练四个模型并预测 test 样本 ───────────────────────────
        models = build_models()

        # M3 的 C 选择（嵌套 LOO）
        if do_nested_cv_for_m3:
            best_C = select_best_C(
                feat_dicts_train=train_dicts,
                games_train=train_games,
                y_train=y_tr,
            )
        else:
            best_C = 1.0
        chosen_C_per_fold['M3_L2LR'].append(best_C)
        models['M3_L2LR'] = L2LogisticClassifier(C=best_C)

        for name, mdl in models.items():
            try:
                mdl.fit(X_tr, y_tr, games=train_games)
                probs_oof[name][i] = float(mdl.predict_proba(X_te)[0])
            except Exception as e:
                if verbose:
                    print(
                        f"  [WARN] fold {i} model {name} failed: {e}",
                        flush=True,
                    )
                # 失败时给一个中性概率，避免后续 NaN 传播
                probs_oof[name][i] = 0.5

        if verbose and ((i + 1) % 10 == 0 or (i + 1) == n):
            print(f"  LOSO progress: {i + 1}/{n}", flush=True)

    return probs_oof, chosen_C_per_fold


# ─────────────────────────────────────────────────────────
# 评估指标
# ─────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    """
    返回 auroc / auprc / f1_opt / threshold_opt。

    f1_opt 与 threshold_opt：在 PR 曲线上扫描所有阈值，取使 F1 最大的阈值。
    """
    out = {
        'auroc':         np.nan,
        'auprc':         np.nan,
        'f1_opt':        np.nan,
        'threshold_opt': np.nan,
    }
    if len(np.unique(y_true)) < 2:
        return out

    out['auroc'] = float(roc_auc_score(y_true, prob))
    out['auprc'] = float(average_precision_score(y_true, prob))

    # 扫描阈值找最佳 F1
    precisions, recalls, thresholds = precision_recall_curve(y_true, prob)
    # precision_recall_curve 的 thresholds 长度比 precisions 少 1
    # 我们把每个阈值对应的 F1 算出来，取最大
    best_f1 = -1.0
    best_thr = 0.5
    for k, thr in enumerate(thresholds):
        p = precisions[k]
        r = recalls[k]
        if (p + r) <= 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    out['f1_opt'] = float(best_f1)
    out['threshold_opt'] = float(best_thr)
    return out


def per_game_auroc(
    games: np.ndarray,
    y_true: np.ndarray,
    prob: np.ndarray,
) -> pd.DataFrame:
    """对指定模型的 prob 按 game 拆分计算 AUROC。"""
    rows = []
    for g in sorted(np.unique(games)):
        mask = games == g
        n_total = int(mask.sum())
        n_pos = int((y_true[mask] == 1).sum())
        if len(np.unique(y_true[mask])) < 2:
            auroc = np.nan
        else:
            auroc = float(roc_auc_score(y_true[mask], prob[mask]))
        rows.append({
            'game':    g,
            'auroc':   auroc,
            'n_total': n_total,
            'n_pos':   n_pos,
        })
    return pd.DataFrame(rows).set_index('game')


# ─────────────────────────────────────────────────────────
# 特征重要性
# ─────────────────────────────────────────────────────────
def fit_full_models_for_importance(
    df_analyzable: pd.DataFrame,
    chosen_C_per_fold: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """
    在全部可分析样本上拟合一次，仅用于报告特征重要性。
    （注意：与 LOSO 评估完全分离——LOSO 提供性能数字，这里提供"最终模型"的解释。）

    M3 的 C 取嵌套 CV 在各折所选 C 的众数，作为"最常被选中的 C"。
    """
    feat_dicts = df_analyzable[list(FEATURE_NAMES)].to_dict('records')
    games = df_analyzable['game'].tolist()
    y = df_analyzable['label'].to_numpy().astype(int)

    stats = fit_normalize_stats(feat_dicts, games, [int(v) for v in y])
    X = apply_normalize(feat_dicts, games, stats)

    # M3 用众数 C
    cs = chosen_C_per_fold.get('M3_L2LR', [1.0])
    if len(cs) > 0:
        # mode：出现频率最高的 C；若并列取较小者（更正则化）
        unique, counts = np.unique(cs, return_counts=True)
        mode_C = float(unique[np.argmax(counts)])
    else:
        mode_C = 1.0

    models = {
        'M2_PurePrior': PurePriorScorer(),
        'M1_SemiPrior': SemiPriorScorer(bounds=0.3),
        'M3_L2LR':      L2LogisticClassifier(C=mode_C),
        'M4_RF':        RandomForestClassifier_wrap(),
    }
    importance: Dict[str, Dict[str, float]] = {}
    for name, mdl in models.items():
        mdl.fit(X, y, games=games)
        imp = mdl.get_feature_importance()
        importance[name] = {f: float(imp[f]) for f in FEATURE_NAMES}
    return importance


def importance_to_table(
    importance: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    把 importance dict 转成"对外展示的"权重表：
      - M2_PurePrior, M1_SemiPrior：直接是线性权重
      - M3_L2LR：取 |coef|（用于幅值比较；正负号已可在 raw 表中查）
      - M4_RF：直接是 feature_importances_

    返回行=模型、列=特征 的 DataFrame。
    """
    rows = []
    for m in MODEL_NAMES:
        d = importance[m]
        if m == 'M3_L2LR':
            row = {f: abs(d[f]) for f in FEATURE_NAMES}
        else:
            row = {f: d[f] for f in FEATURE_NAMES}
        row['__model__'] = m
        rows.append(row)
    return pd.DataFrame(rows).set_index('__model__')[list(FEATURE_NAMES)]


def importance_spearman_matrix(
    importance_table: pd.DataFrame,
) -> pd.DataFrame:
    """计算四组权重间的 Spearman 秩相关矩阵。"""
    n_models = len(importance_table)
    mat = np.zeros((n_models, n_models), dtype=float)
    names = importance_table.index.tolist()
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                mat[i, j] = 1.0
            else:
                rho, _ = spearmanr(
                    importance_table.loc[ni].to_numpy(),
                    importance_table.loc[nj].to_numpy(),
                )
                mat[i, j] = float(rho) if rho is not None else np.nan
    return pd.DataFrame(mat, index=names, columns=names)


# ─────────────────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────────────────
def format_report(
    main_metrics: pd.DataFrame,
    gate_diag: dict,
    per_game_table: pd.DataFrame,
    importance_table: pd.DataFrame,
    spearman_matrix: pd.DataFrame,
    chosen_C_per_fold: Dict[str, List[float]],
) -> str:
    """生成人类可读的汇总报告字符串（对应文档 4.6.4）。"""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("阶段6  实验评估报告")
    lines.append("=" * 72)
    lines.append("")

    # 主任务性能
    lines.append("【主任务性能】（全部样本，包含旁路的乱画样本 prob=1.0）")
    lines.append(main_metrics.round(4).to_string())
    lines.append("")

    # 模型对比叙事
    if 'auroc' in main_metrics.columns:
        a = main_metrics['auroc']
        if all(m in a.index for m in MODEL_NAMES):
            d_m2_m1 = a['M1_SemiPrior'] - a['M2_PurePrior']
            d_m1_m3 = a['M3_L2LR']      - a['M1_SemiPrior']
            d_m3_m4 = a['M4_RF']        - a['M3_L2LR']
            lines.append("【关键对比】")
            lines.append(f"  M2 → M1  : Δ AUROC = {d_m2_m1:+.4f}")
            lines.append(f"  M1 → M3  : Δ AUROC = {d_m1_m3:+.4f}")
            lines.append(f"  M3 → M4  : Δ AUROC = {d_m3_m4:+.4f}")
            if abs(d_m2_m1) < 1e-6:
                lines.append(
                    "  注：M1 与 M2 的 AUROC 完全相同——"
                    "M1 的近似 AUROC 目标 (Wilcoxon-Mann-Whitney 指示函数) "
                    "是分段常数，L-BFGS-B 的数值梯度几乎处处为零, "
                    "优化器在初始权重处即收敛。可考虑将目标平滑化"
                    "（如 log-loss 或 ranking surrogate）以发挥 M1 的设计意图。"
                )
            if abs(d_m3_m4) < 0.02:
                lines.append(
                    "  → M3 ≈ M4：本任务接近线性可分，简单模型已足够。"
                )
            elif d_m3_m4 > 0.05:
                lines.append(
                    "  → M4 显著优于 M3：决策边界存在非线性，"
                    "应反思线性 LR 的容量是否够。"
                )
            lines.append("")

    # 门控诊断
    lines.append("【门控诊断】")
    for k, v in gate_diag.items():
        lines.append(f"  {k:<32s}: {v}")
    lines.append("")

    # per-game AUROC
    lines.append("【per-game AUROC 拆分】（模型: M3_L2LR）")
    lines.append(per_game_table.round(4).to_string())
    lines.append("")

    # 特征权重对比
    lines.append("【特征权重对比】")
    lines.append("（M2/M1=线性权重; M3=|coef|; M4=feature_importances_）")
    lines.append(importance_table.round(4).to_string())
    lines.append("")

    # Spearman 相关矩阵
    lines.append("【特征重要性的跨模型 Spearman 秩相关】")
    lines.append(spearman_matrix.round(3).to_string())
    rho_off = spearman_matrix.values[
        ~np.eye(spearman_matrix.shape[0], dtype=bool)
    ]
    rho_off = rho_off[~np.isnan(rho_off)]
    if len(rho_off) > 0:
        lines.append(
            f"  非对角元均值 ρ = {np.mean(rho_off):.3f}; "
            f"中位数 ρ = {np.median(rho_off):.3f}"
        )
        if np.median(rho_off) >= 0.7:
            lines.append(
                "  → ρ ≥ 0.7：特征重要性排序在不同模型间高度一致, "
                "方法论稳健性获支持。"
            )
    lines.append("")

    # M3 选 C 的统计
    cs = chosen_C_per_fold.get('M3_L2LR', [])
    if len(cs) > 0:
        unique, counts = np.unique(cs, return_counts=True)
        c_freq = ', '.join(
            f"C={u}:{c}" for u, c in sorted(zip(unique, counts))
        )
        lines.append("【M3 嵌套 CV 选 C 的频次】")
        lines.append(f"  {c_freq}")
        lines.append("")

    lines.append("=" * 72)
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────
def run(
    feature_csv: str,
    out_dir: str,
    do_nested_cv_for_m3: bool = True,
    verbose: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print(f"[1/6] 读取样本表: {feature_csv}", flush=True)
    df = pd.read_csv(feature_csv)
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]  # 处理 BOM
    required = {'sample_id', 'game', 'label'} | set(FEATURE_NAMES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"all.csv 缺列: {missing}")

    # ── 门控 ────────────────────────────────────────────────────────
    if verbose:
        print("[2/6] 应用乱画门控", flush=True)
    df_gated = apply_gate(df)
    gate_diag = gating_diagnostics(df_gated)
    if verbose:
        print(f"      → {gate_diag}", flush=True)

    df_analyzable = (
        df_gated[~df_gated['is_unanalyzable']]
        .reset_index(drop=True)
    )
    df_unanalyzable = (
        df_gated[df_gated['is_unanalyzable']]
        .reset_index(drop=True)
    )

    # ── LOSO ────────────────────────────────────────────────────────
    if verbose:
        print(
            f"[3/6] LOSO（{len(df_analyzable)} 个可分析样本，"
            f"M3 嵌套 CV={do_nested_cv_for_m3}）",
            flush=True,
        )
    probs_oof, chosen_C = run_loso(
        df_analyzable,
        do_nested_cv_for_m3=do_nested_cv_for_m3,
        verbose=verbose,
    )

    # ── 合并乱画样本（prob=1.0）──────────────────────────────────
    if verbose:
        print("[4/6] 合并乱画样本到全样本预测", flush=True)
    n_full = len(df_gated)
    probs_full: Dict[str, np.ndarray] = {}
    for m in MODEL_NAMES:
        p = np.full(n_full, np.nan, dtype=float)
        # 可分析样本：写入 LOSO 概率
        ana_mask = ~df_gated['is_unanalyzable'].to_numpy()
        p[ana_mask] = probs_oof[m]
        # 乱画样本：直接 1.0
        p[~ana_mask] = 1.0
        probs_full[m] = p

    y_full = df_gated['label'].to_numpy().astype(int)
    games_full = df_gated['game'].to_numpy()
    sample_ids_full = df_gated['sample_id'].to_numpy()

    # 每样本预测明细
    pred_df = pd.DataFrame({
        'sample_id':       sample_ids_full,
        'game':            games_full,
        'label':           y_full,
        'is_unanalyzable': df_gated['is_unanalyzable'].to_numpy(),
        'triggered_rules': df_gated['triggered_rules'].to_numpy(),
    })
    for m in MODEL_NAMES:
        pred_df[f'prob_{m}'] = probs_full[m]
    pred_df.to_csv(
        os.path.join(out_dir, 'per_sample_predictions.csv'),
        index=False,
        encoding='utf-8',
    )

    # ── 计算主指标 ─────────────────────────────────────────────────
    if verbose:
        print("[5/6] 计算主指标", flush=True)
    main_rows = []
    for m in MODEL_NAMES:
        metrics = compute_metrics(y_full, probs_full[m])
        metrics['model'] = m
        main_rows.append(metrics)
    main_metrics = (
        pd.DataFrame(main_rows)
        .set_index('model')[['auroc', 'auprc', 'f1_opt', 'threshold_opt']]
    )
    main_metrics.to_csv(
        os.path.join(out_dir, 'main_metrics.csv'),
        encoding='utf-8',
    )

    # per-game AUROC（针对 M3）
    per_game_table = per_game_auroc(games_full, y_full, probs_full['M3_L2LR'])
    per_game_table.to_csv(
        os.path.join(out_dir, 'per_game_auroc.csv'),
        encoding='utf-8',
    )

    # ── 特征重要性 ─────────────────────────────────────────────────
    if verbose:
        print("[6/6] 计算特征重要性 + Spearman 一致性", flush=True)
    importance = fit_full_models_for_importance(df_analyzable, chosen_C)
    importance_table = importance_to_table(importance)
    importance_table.to_csv(
        os.path.join(out_dir, 'feature_importance.csv'),
        encoding='utf-8',
    )

    spearman_matrix = importance_spearman_matrix(importance_table)
    spearman_matrix.to_csv(
        os.path.join(out_dir, 'feature_importance_spearman.csv'),
        encoding='utf-8',
    )

    # 门控诊断写盘
    with open(os.path.join(out_dir, 'gating_diagnostics.json'), 'w', encoding='utf-8') as f:
        json.dump(gate_diag, f, indent=2, ensure_ascii=False)

    # ── 报告 ──────────────────────────────────────────────────────
    report = format_report(
        main_metrics=main_metrics,
        gate_diag=gate_diag,
        per_game_table=per_game_table,
        importance_table=importance_table,
        spearman_matrix=spearman_matrix,
        chosen_C_per_fold=chosen_C,
    )
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    if verbose:
        print()
        print(report)
        print()
        print(f"全部输出已保存到: {out_dir}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="阶段6：四个模型的 LOSO 评估与对比"
    )
    p.add_argument(
        '--feature_csv',
        type=str,
        required=True,
        help='原始特征表（all.csv），表头需含 sample_id,game,label,F1..C3',
    )
    p.add_argument(
        '--out_dir',
        type=str,
        default='results/',
        help='结果输出目录',
    )
    p.add_argument(
        '--no-nested-cv',
        action='store_true',
        help='跳过 M3 的嵌套 CV，固定 C=1.0（更快）',
    )
    p.add_argument(
        '--quiet',
        action='store_true',
        help='抑制进度输出',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(
        feature_csv=args.feature_csv,
        out_dir=args.out_dir,
        do_nested_cv_for_m3=not args.no_nested_cv,
        verbose=not args.quiet,
    )