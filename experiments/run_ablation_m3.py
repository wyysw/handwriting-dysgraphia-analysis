"""
M3 L2 Logistic Regression ablation experiments
==============================================

为厘清三段式特征处理管道中关键组件对最终性能的贡献，本脚本以
L2 正则化 Logistic Regression 为唯一基础模型，在完全相同的 LOSO
协议下运行 baseline 与四组消融：

1. no_scribble_gate      : 无乱画门控；全部样本进入 LOSO，不再 prob=1.0 旁路。
2. no_robust_norm        : 无稳健归一化；将 median/MAD 改为 mean/std，其余仍按 game 拟合。
3. no_motor_features     : 无运动控制特征；仅使用 F1..F4，不使用 C1..C3。
4. no_cross_game_norm    : 无跨游戏/按游戏归一化；改为全局 z-score（跨所有 game 拟合一组统计量）。

所有实验均：
- 只使用训练折拟合归一化统计量，避免泄漏；
- 对 C ∈ {0.1, 0.3, 1.0, 3.0} 在外层训练折内做嵌套 LOO 选择；
- 除被消融组件外，保持特征方向统一、clip、class_weight='balanced' 等逻辑一致。

用法：
python experiments/run_ablation_m3.py --feature_csv data/feature/all.csv --out_dir results/ablation_m3

快速调试可加 --no-nested-cv 固定 C=1.0。
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='sklearn.linear_model._logistic',
)

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from classifiers.m3_logistic import L2LogisticClassifier  # noqa: E402
from features.gate_unanalyzable import apply_gate          # noqa: E402
from features.normalize import DIRECTION, DEFAULT_CLIP     # noqa: E402

ALL_FEATURES: List[str] = ["F1", "F2", "F3", "F4", "C1", "C2", "C3"]
SHAPE_FEATURES: List[str] = ["F1", "F2", "F3", "F4"]
C_GRID: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0)
EPS = 1e-3


@dataclass(frozen=True)
class AblationConfig:
    name: str
    description: str
    use_gate: bool = True
    gate_bypass_prob: bool = True
    norm_mode: str = "robust_by_game"  # robust_by_game | meanstd_by_game | robust_global
    feature_names: Tuple[str, ...] = tuple(ALL_FEATURES)


CONFIGS: Tuple[AblationConfig, ...] = (
    AblationConfig(
        name="baseline",
        description="原始三段式管道：乱画门控 + 按游戏 median/MAD 稳健归一化 + F1..C3 + 乱画 prob=1.0 旁路。",
    ),
    AblationConfig(
        name="no_scribble_gate",
        description="无乱画门控：全部样本进入 LOSO；不剔除乱画样本，也不使用 prob=1.0 旁路。",
        use_gate=False,
        gate_bypass_prob=False,
    ),
    AblationConfig(
        name="no_robust_norm",
        description="无稳健归一化：按游戏使用训练折 label=0 参考池的 mean/std 替代 median/MAD。",
        norm_mode="meanstd_by_game",
    ),
    AblationConfig(
        name="no_motor_features",
        description="无运动控制特征：仅使用 F1..F4，不使用 C1..C3。",
        feature_names=tuple(SHAPE_FEATURES),
    ),
    AblationConfig(
        name="no_cross_game_norm",
        description="无按游戏归一化：跨所有 game 使用一组全局 z-score 统计量。",
        norm_mode="robust_global",
    ),
)


def _safe_scale(arr: np.ndarray, center: float, mode: str) -> Tuple[float, float]:
    """返回 (spread_raw, protected_scale)。"""
    if len(arr) == 0:
        return 0.0, EPS
    if mode == "robust":
        spread = float(np.median(np.abs(arr - center)))
        scale = 1.4826 * spread
    elif mode == "meanstd":
        spread = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
        scale = spread
    else:
        raise ValueError(f"Unknown scale mode: {mode}")
    return spread, max(float(scale), 0.05 * abs(float(center)) + EPS)


def fit_stats(
    feature_dicts: List[dict],
    games: List[str],
    labels: List[int],
    feature_names: Tuple[str, ...],
    norm_mode: str,
) -> dict:
    """只在训练折 label=0 样本上拟合归一化统计量。"""
    if norm_mode not in {"robust_by_game", "meanstd_by_game", "robust_global"}:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    groups = sorted(set(games)) if norm_mode != "robust_global" else ["__global__"]
    stats: dict = {g: {} for g in groups}

    for group in groups:
        for f in feature_names:
            vals = []
            for fd, game, lab in zip(feature_dicts, games, labels):
                if lab != 0:
                    continue
                if norm_mode != "robust_global" and game != group:
                    continue
                vals.append(float(fd[f]))

            arr = np.asarray(vals, dtype=float)
            if len(arr) == 0:
                center, spread, scale, ref_n = 0.0, 0.0, EPS, 0
            elif norm_mode == "meanstd_by_game":
                center = float(np.mean(arr))
                spread, scale = _safe_scale(arr, center, "meanstd")
                ref_n = len(arr)
            else:
                center = float(np.median(arr))
                spread, scale = _safe_scale(arr, center, "robust")
                ref_n = len(arr)

            stats[group][f] = {
                "center": center,
                "spread": spread,
                "scale": scale,
                "ref_n": int(ref_n),
            }
    return stats


def apply_stats(
    feature_dicts: List[dict],
    games: List[str],
    stats: dict,
    feature_names: Tuple[str, ...],
    norm_mode: str,
    clip: Tuple[float, float] = DEFAULT_CLIP,
) -> np.ndarray:
    X = np.zeros((len(feature_dicts), len(feature_names)), dtype=float)
    for i, (fd, game) in enumerate(zip(feature_dicts, games)):
        group = "__global__" if norm_mode == "robust_global" else game
        if group not in stats:
            # 外层测试样本的 game 未在训练折出现时，退化为第一组统计量；LOSO 常规数据中通常不会发生。
            group = next(iter(stats.keys()))
        for j, f in enumerate(feature_names):
            s = stats[group][f]
            z_raw = (float(fd[f]) - s["center"]) / s["scale"]
            z = DIRECTION[f] * z_raw
            X[i, j] = float(np.clip(z, clip[0], clip[1]))
    return X


def compute_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    out = {"auroc": np.nan, "auprc": np.nan, "f1_opt": np.nan, "threshold_opt": np.nan}
    if len(np.unique(y_true)) < 2:
        return out
    out["auroc"] = float(roc_auc_score(y_true, prob))
    out["auprc"] = float(average_precision_score(y_true, prob))
    precisions, recalls, thresholds = precision_recall_curve(y_true, prob)
    best = (-1.0, 0.5)
    for k, thr in enumerate(thresholds):
        p, r = float(precisions[k]), float(recalls[k])
        f1 = 0.0 if (p + r) <= 0 else 2 * p * r / (p + r)
        if f1 > best[0]:
            best = (f1, float(thr))
    out["f1_opt"], out["threshold_opt"] = float(best[0]), float(best[1])
    return out


def select_best_C(
    train_dicts: List[dict],
    train_games: List[str],
    y_train: np.ndarray,
    cfg: AblationConfig,
) -> float:
    best_C, best_auc = 1.0, -np.inf
    n = len(train_dicts)
    for C in C_GRID:
        oof = np.full(n, np.nan, dtype=float)
        for i in range(n):
            inner_train_idx = [j for j in range(n) if j != i]
            y_tr = y_train[inner_train_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            inner_dicts = [train_dicts[j] for j in inner_train_idx]
            inner_games = [train_games[j] for j in inner_train_idx]
            inner_labels = [int(y_train[j]) for j in inner_train_idx]
            stats = fit_stats(inner_dicts, inner_games, inner_labels, cfg.feature_names, cfg.norm_mode)
            X_tr = apply_stats(inner_dicts, inner_games, stats, cfg.feature_names, cfg.norm_mode)
            X_te = apply_stats([train_dicts[i]], [train_games[i]], stats, cfg.feature_names, cfg.norm_mode)
            mdl = L2LogisticClassifier(C=C)
            mdl.fit(X_tr, y_tr, games=inner_games)
            oof[i] = float(mdl.predict_proba(X_te)[0])
        valid = ~np.isnan(oof)
        if valid.sum() >= 2 and len(np.unique(y_train[valid])) >= 2:
            auc = float(roc_auc_score(y_train[valid], oof[valid]))
            if auc > best_auc:
                best_auc, best_C = auc, C
    return best_C


def run_loso_lr(df_eval: pd.DataFrame, cfg: AblationConfig, do_nested_cv: bool, verbose: bool) -> Tuple[np.ndarray, List[float]]:
    n = len(df_eval)
    feat_dicts = df_eval[list(cfg.feature_names)].to_dict("records")
    games = df_eval["game"].tolist()
    y = df_eval["label"].to_numpy(dtype=int)
    probs = np.full(n, np.nan, dtype=float)
    chosen_C: List[float] = []

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        y_tr = y[train_idx]
        if len(np.unique(y_tr)) < 2:
            probs[i] = 0.5
            chosen_C.append(1.0)
            continue
        train_dicts = [feat_dicts[j] for j in train_idx]
        train_games = [games[j] for j in train_idx]
        train_labels = [int(y[j]) for j in train_idx]

        stats = fit_stats(train_dicts, train_games, train_labels, cfg.feature_names, cfg.norm_mode)
        X_tr = apply_stats(train_dicts, train_games, stats, cfg.feature_names, cfg.norm_mode)
        X_te = apply_stats([feat_dicts[i]], [games[i]], stats, cfg.feature_names, cfg.norm_mode)

        best_C = select_best_C(train_dicts, train_games, y_tr, cfg) if do_nested_cv else 1.0
        chosen_C.append(float(best_C))
        mdl = L2LogisticClassifier(C=best_C)
        mdl.fit(X_tr, y_tr, games=train_games)
        probs[i] = float(mdl.predict_proba(X_te)[0])

        if verbose and ((i + 1) % 10 == 0 or (i + 1) == n):
            print(f"    LOSO progress: {i + 1}/{n}", flush=True)
    return probs, chosen_C


def prepare_eval_frame(df_raw: pd.DataFrame, cfg: AblationConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.use_gate:
        df_gated = apply_gate(df_raw)
    else:
        df_gated = df_raw.copy()
        df_gated["F3_zscore"] = np.nan
        df_gated["F4_zscore"] = np.nan
        df_gated["is_unanalyzable"] = False
        df_gated["triggered_rules"] = ""
    df_eval = df_gated[~df_gated["is_unanalyzable"].astype(bool)].reset_index(drop=True)
    return df_gated.reset_index(drop=True), df_eval


def run_one(df_raw: pd.DataFrame, cfg: AblationConfig, out_dir: str, do_nested_cv: bool, verbose: bool) -> Dict[str, float]:
    if verbose:
        print(f"\n[{cfg.name}] {cfg.description}", flush=True)
    df_gated, df_eval = prepare_eval_frame(df_raw, cfg)
    probs_eval, chosen_C = run_loso_lr(df_eval, cfg, do_nested_cv, verbose)

    n_full = len(df_gated)
    probs_full = np.full(n_full, np.nan, dtype=float)
    ana_mask = ~df_gated["is_unanalyzable"].to_numpy(dtype=bool)
    probs_full[ana_mask] = probs_eval
    if cfg.gate_bypass_prob:
        probs_full[~ana_mask] = 1.0

    y_full = df_gated["label"].to_numpy(dtype=int)
    metrics = compute_metrics(y_full, probs_full)
    metrics.update({
        "experiment": cfg.name,
        "n_full": int(n_full),
        "n_loso": int(len(df_eval)),
        "n_gated": int((~ana_mask).sum()),
        "features": "+".join(cfg.feature_names),
        "norm_mode": cfg.norm_mode,
        "use_gate": bool(cfg.use_gate),
    })

    pred = pd.DataFrame({
        "sample_id": df_gated["sample_id"],
        "game": df_gated["game"],
        "label": y_full,
        "experiment": cfg.name,
        "is_unanalyzable": df_gated["is_unanalyzable"].to_numpy(dtype=bool),
        "triggered_rules": df_gated["triggered_rules"].fillna(""),
        "prob_M3_L2LR": probs_full,
    })
    pred.to_csv(os.path.join(out_dir, f"predictions_{cfg.name}.csv"), index=False, encoding="utf-8")

    c_df = pd.DataFrame({"fold": np.arange(1, len(chosen_C) + 1), "chosen_C": chosen_C})
    c_df.to_csv(os.path.join(out_dir, f"chosen_C_{cfg.name}.csv"), index=False, encoding="utf-8")
    return metrics


def write_report(metrics_df: pd.DataFrame, out_dir: str) -> None:
    base = metrics_df[metrics_df["experiment"] == "baseline"].iloc[0]
    table = metrics_df.copy()
    for col in ["auroc", "auprc", "f1_opt"]:
        table[f"delta_{col}_vs_baseline"] = table[col] - float(base[col])
    table.to_csv(os.path.join(out_dir, "ablation_metrics_with_delta.csv"), index=False, encoding="utf-8")

    lines = []
    lines.append("M3 L2 Logistic Regression 消融实验报告")
    lines.append("=" * 60)
    lines.append("所有实验均使用相同 LOSO 协议；除消融组件外，其余流程保持一致。")
    lines.append("")
    cols = ["experiment", "auroc", "auprc", "f1_opt", "threshold_opt", "n_loso", "n_gated", "delta_auroc_vs_baseline", "delta_auprc_vs_baseline", "delta_f1_opt_vs_baseline"]
    lines.append(table[cols].round(4).to_string(index=False))
    lines.append("")
    lines.append("实验说明：")
    for cfg in CONFIGS:
        lines.append(f"- {cfg.name}: {cfg.description}")
    with open(os.path.join(out_dir, "ablation_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(feature_csv: str, out_dir: str, do_nested_cv: bool = True, verbose: bool = True) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(feature_csv)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    required = {"sample_id", "game", "label"} | set(ALL_FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"feature_csv 缺少必要列: {sorted(missing)}")

    rows = []
    for cfg in CONFIGS:
        rows.append(run_one(df, cfg, out_dir, do_nested_cv, verbose))
    metrics_df = pd.DataFrame(rows)
    ordered = ["experiment", "auroc", "auprc", "f1_opt", "threshold_opt", "n_full", "n_loso", "n_gated", "features", "norm_mode", "use_gate"]
    metrics_df = metrics_df[ordered]
    metrics_df.to_csv(os.path.join(out_dir, "ablation_metrics.csv"), index=False, encoding="utf-8")
    write_report(metrics_df, out_dir)
    if verbose:
        print("\n消融实验核心指标：")
        print(metrics_df[["experiment", "auroc", "auprc", "f1_opt", "threshold_opt", "n_loso", "n_gated"]].round(4).to_string(index=False))
        print(f"\n输出目录: {out_dir}")
    return metrics_df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3 L2LR 四组消融实验（LOSO）")
    p.add_argument("--feature_csv", required=True, help="原始特征表，需含 sample_id,game,label,F1..C3")
    p.add_argument("--out_dir", default="results/ablation_m3", help="输出目录")
    p.add_argument("--no-nested-cv", action="store_true", help="固定 C=1.0，用于快速调试")
    p.add_argument("--quiet", action="store_true", help="抑制进度输出")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.feature_csv, args.out_dir, do_nested_cv=not args.no_nested_cv, verbose=not args.quiet)
