"""
4.6.6  SHAP：M3 的样本级可解释性
==================================

对在全部可分析样本上训练好的 M3（L2 Logistic Regression），
用 SHAP 解释每个样本"为什么被判为异常 / 正常"。

为什么用 LinearExplainer
------------------------
M3 是线性模型，LinearExplainer 给出的是精确（非近似）的 SHAP 值，
计算开销也最小——非常适合本任务的小样本量。

输出
----
1. 全样本的 SHAP 值矩阵 → results/shap_values_m3.csv
2. 按 |SHAP| 平均的全局特征重要性 → results/shap_global_importance.csv
3. （可选）针对若干样本（如 top-3 假阳性 / top-3 假阴性）的"贡献明细" → results/shap_case_studies.csv

用法
----
    python experiments/shap_m3.py \\
        --feature_csv data/feature/all.csv \\
        --predictions results/per_sample_predictions.csv \\
        --out_dir results/

python experiments/shap_m3.py --feature_csv data/feature/all.csv --predictions results/per_sample_predictions.csv --out_dir results/ --C 0.1

"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from classifiers.base import FEATURE_NAMES                       # noqa: E402
from classifiers.m3_logistic import L2LogisticClassifier         # noqa: E402
from features.gate_unanalyzable import apply_gate               # noqa: E402
from features.normalize import (                                # noqa: E402
    apply_normalize,
    fit_normalize_stats,
)


def _import_shap():
    try:
        import shap  # type: ignore
        return shap
    except ImportError:
        raise ImportError(
            "未安装 shap。请运行：pip install shap"
        )


def compute_shap_for_m3(
    df_analyzable: pd.DataFrame,
    C: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, L2LogisticClassifier]:
    """
    在可分析样本上训练 M3，并计算 SHAP 值。

    Returns
    -------
    X_z         : np.ndarray, shape=(N, 7)  归一化后的特征矩阵
    shap_values : np.ndarray, shape=(N, 7)  每样本每特征的 SHAP 贡献
    model       : 已训练的 L2LogisticClassifier
    """
    shap = _import_shap()

    feat_dicts = df_analyzable[list(FEATURE_NAMES)].to_dict('records')
    games = df_analyzable['game'].tolist()
    y = df_analyzable['label'].to_numpy().astype(int)

    stats = fit_normalize_stats(
        feat_dicts, games, [int(v) for v in y],
    )
    X_z = apply_normalize(feat_dicts, games, stats)

    model = L2LogisticClassifier(C=C)
    model.fit(X_z, y, games=games)

    # LinearExplainer 直接对 sklearn 的 LogisticRegression 工作
    # 背景分布用全部 X_z（小样本下 masker 没必要再下采样）
    explainer = shap.LinearExplainer(model.model_, X_z)
    sv = explainer.shap_values(X_z)
    # sv 在二分类下可能是 ndarray (N,P) 或 list[ndarray]；统一为 ndarray
    if isinstance(sv, list):
        sv = sv[1]  # 取正类的 SHAP

    return X_z, np.asarray(sv), model


def select_case_studies(
    pred_df: pd.DataFrame,
    n_each: int = 3,
) -> List[str]:
    """
    挑选"最值得讲故事的"样本：
      - top n 假阳性（label=0 但 prob 最高）
      - top n 假阴性（label=1 但 prob 最低）
    并把每个游戏的一个典型 TP / TN 也带上。

    返回 sample_id 列表（可能少于 4*n_each 个，因为有些类别样本可能很少）。
    """
    if 'prob_M3_L2LR' not in pred_df.columns:
        return []

    # 仅看可分析样本（乱画样本 prob=1.0 是机制赋值，没有 SHAP 意义）
    if 'is_unanalyzable' in pred_df.columns:
        df = pred_df[~pred_df['is_unanalyzable']].copy()
    else:
        df = pred_df.copy()

    selected: List[str] = []

    fp = df[df['label'] == 0].sort_values('prob_M3_L2LR', ascending=False)
    selected.extend(fp['sample_id'].head(n_each).tolist())

    fn = df[df['label'] == 1].sort_values('prob_M3_L2LR', ascending=True)
    selected.extend(fn['sample_id'].head(n_each).tolist())

    # 每个游戏挑一个最强 TP（label=1 且 prob 最高）
    for g, sub in df[df['label'] == 1].groupby('game'):
        top = sub.sort_values('prob_M3_L2LR', ascending=False).head(1)
        selected.extend(top['sample_id'].tolist())

    # 去重保序
    seen = set()
    out = []
    for s in selected:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def run(
    feature_csv: str,
    predictions_csv: str | None,
    out_dir: str,
    C: float = 1.0,
    verbose: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print(f"[1/3] 读取数据: {feature_csv}", flush=True)
    df = pd.read_csv(feature_csv)
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]

    df_gated = apply_gate(df)
    df_analyzable = (
        df_gated[~df_gated['is_unanalyzable']]
        .reset_index(drop=True)
    )

    if verbose:
        print(
            f"[2/3] 训练 M3 并计算 SHAP（C={C}, "
            f"N={len(df_analyzable)}）",
            flush=True,
        )
    X_z, shap_values, _ = compute_shap_for_m3(df_analyzable, C=C)

    # ── 保存全样本 SHAP 矩阵 ─────────────────────────────────────
    sv_df = pd.DataFrame(shap_values, columns=[f'shap_{f}' for f in FEATURE_NAMES])
    sv_df.insert(0, 'sample_id', df_analyzable['sample_id'].to_numpy())
    sv_df.insert(1, 'game',      df_analyzable['game'].to_numpy())
    sv_df.insert(2, 'label',     df_analyzable['label'].to_numpy())
    # 同时附上归一化后的特征值，便于解读 SHAP 贡献
    for j, f in enumerate(FEATURE_NAMES):
        sv_df[f'{f}_z'] = X_z[:, j]
    sv_path = os.path.join(out_dir, 'shap_values_m3.csv')
    sv_df.to_csv(sv_path, index=False)

    # ── 全局特征重要性 = mean(|SHAP|) ────────────────────────────
    global_imp = pd.DataFrame({
        'feature':      FEATURE_NAMES,
        'mean_abs_shap': np.mean(np.abs(shap_values), axis=0),
        'mean_shap':     np.mean(shap_values, axis=0),  # 带方向
    }).sort_values('mean_abs_shap', ascending=False)
    gi_path = os.path.join(out_dir, 'shap_global_importance.csv')
    global_imp.to_csv(gi_path, index=False)

    # ── 案例分析 ─────────────────────────────────────────────────
    if verbose:
        print("[3/3] 生成案例分析", flush=True)
    case_ids: List[str] = []
    if predictions_csv is not None and os.path.exists(predictions_csv):
        pred_df = pd.read_csv(predictions_csv)
        case_ids = select_case_studies(pred_df, n_each=3)

    if len(case_ids) > 0:
        case_rows = []
        for sid in case_ids:
            mask = df_analyzable['sample_id'] == sid
            if mask.sum() == 0:
                continue
            i = int(np.where(mask.to_numpy())[0][0])
            row = {
                'sample_id': sid,
                'game':      df_analyzable.iloc[i]['game'],
                'label':     int(df_analyzable.iloc[i]['label']),
            }
            for j, f in enumerate(FEATURE_NAMES):
                row[f'{f}_z']    = float(X_z[i, j])
                row[f'{f}_shap'] = float(shap_values[i, j])
            row['shap_sum'] = float(shap_values[i].sum())
            case_rows.append(row)
        cs_path = os.path.join(out_dir, 'shap_case_studies.csv')
        pd.DataFrame(case_rows).to_csv(cs_path, index=False)

    if verbose:
        print()
        print("【SHAP 全局特征重要性（按 mean|SHAP| 降序）】")
        print(global_imp.round(4).to_string(index=False))
        print()
        print(f"全部输出已保存到: {out_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3 的 SHAP 可解释性分析")
    p.add_argument(
        '--feature_csv',
        type=str,
        required=True,
        help='原始特征表 all.csv',
    )
    p.add_argument(
        '--predictions',
        type=str,
        default=None,
        help=(
            'run_experiments.py 输出的 per_sample_predictions.csv，'
            '用于挑选 FP/FN 案例（可选）'
        ),
    )
    p.add_argument(
        '--out_dir',
        type=str,
        default='results/',
    )
    p.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='M3 的 L2 正则化倒数（建议与 run_experiments 选出的众数一致）',
    )
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(
        feature_csv=args.feature_csv,
        predictions_csv=args.predictions,
        out_dir=args.out_dir,
        C=args.C,
        verbose=not args.quiet,
    )
