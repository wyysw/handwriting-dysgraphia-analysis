"""
阶段4.A：乱画门控
功能：根据硬规则判断样本是否为"乱画"（不可分析样本）。
输入：包含 F1..F4 特征的 DataFrame（含 sample_id, game, label）
输出：追加 is_unanalyzable / triggered_rules / F3_zscore / F4_zscore 列

python features/gate_unanalyzable.py --input data/feature/all.csv --output data/feature/gate_decisions.csv

"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


def _compute_game_zscore(series: pd.Series) -> pd.Series:
    """对一个 Series 计算 z-score（mean/std），样本量 <2 时返回 0。"""
    if len(series) < 2:
        return pd.Series(0.0, index=series.index)
    mu = series.mean()
    sd = series.std(ddof=1)
    if sd < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sd


def apply_gate(
    df: pd.DataFrame,
    f1_col: str = "F1",
    f2_col: str = "F2",
    f3_col: str = "F3",
    f4_col: str = "F4",
    game_col: str = "game",
) -> pd.DataFrame:
    """
    对 DataFrame 中每行应用乱画门控规则，返回带以下新列的 DataFrame：
      - F3_zscore       : 按 game 分组的 F3 z-score
      - F4_zscore       : 按 game 分组的 F4 z-score
      - is_unanalyzable : bool
      - triggered_rules : 触发的规则编号字符串（多条用'|'分隔），空字符串表示未触发

    门控规则：
      Rule1: F2 < 0.4 AND F1 < 0.05
      Rule2: F2 < 0.4 AND (F3 > 1 OR F4 > 0.3)
      Rule3: 按 game 分组的 F3_zscore > 2 OR F4_zscore > 2
    """
    df = df.copy()

    # ── 按 game 分组计算 F3/F4 的 z-score ──────────────────────────────
    df["F3_zscore"] = df.groupby(game_col)[f3_col].transform(_compute_game_zscore)
    df["F4_zscore"] = df.groupby(game_col)[f4_col].transform(_compute_game_zscore)

    # ── 逐行评估规则 ────────────────────────────────────────────────────
    triggered_rules = []
    is_unanalyzable = []

    for _, row in df.iterrows():
        rules = []

        # Rule 1
        if row[f2_col] < 0.4 and row[f1_col] < 0.05:
            rules.append("Rule1")

        # Rule 2
        if row[f2_col] < 0.4 and (row[f3_col] > 1 or row[f4_col] > 0.3):
            rules.append("Rule2")

        # Rule 3
        if row["F3_zscore"] > 2 or row["F4_zscore"] > 2:
            rules.append("Rule3")

        triggered_rules.append("|".join(rules))
        is_unanalyzable.append(len(rules) > 0)

    df["is_unanalyzable"] = is_unanalyzable
    df["triggered_rules"] = triggered_rules

    return df


def run_gate(
    input_csv: str,
    output_csv: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    从 CSV 读取特征，应用门控，写出 gate_decisions.csv。

    输出列顺序：
      sample_id, game, label, F1, F2, F3, F4, F3_zscore, F4_zscore,
      is_unanalyzable, triggered_rules
    """
    df = pd.read_csv(input_csv)

    df_out = apply_gate(df)

    # 规范列顺序
    out_cols = [
        "sample_id", "game", "label",
        "F1", "F2", "F3", "F4",
        "F3_zscore", "F4_zscore",
        "is_unanalyzable", "triggered_rules",
    ]
    # 保留原 df 中其余列（如 C1..C3），但 gate_decisions 文档规范只要求上述列
    extra_cols = [c for c in df_out.columns if c not in out_cols]
    df_out = df_out[out_cols + extra_cols]

    df_out.to_csv(output_csv, index=False)

    if verbose:
        n_total = len(df_out)
        n_flag = df_out["is_unanalyzable"].sum()
        n_label1_flagged = df_out[df_out["label"] == 1]["is_unanalyzable"].sum()
        n_label0_flagged = df_out[df_out["label"] == 0]["is_unanalyzable"].sum()
        print(f"[Gate] 总样本: {n_total}  标记乱画: {n_flag}")
        print(f"       label=1 中被标记: {n_label1_flagged}")
        print(f"       label=0 中被标记（误判）: {n_label0_flagged}  ← 期望为 0")
        # 按规则统计
        for rule in ["Rule1", "Rule2", "Rule3"]:
            cnt = df_out["triggered_rules"].str.contains(rule).sum()
            print(f"       {rule} 触发次数: {cnt}")

    return df_out


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="乱画门控：为每个样本打 is_unanalyzable 标记")
    parser.add_argument("--input", required=True, help="输入 CSV（含 F1..F4 等特征）")
    parser.add_argument("--output", required=True, help="输出 gate_decisions.csv 路径")
    parser.add_argument("--quiet", action="store_true", help="不打印统计信息")
    args = parser.parse_args()

    run_gate(args.input, args.output, verbose=not args.quiet)