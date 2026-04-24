"""
阶段4.C：特征汇总主入口 build_feature_matrix.py

功能：
  1. 读取 data/feature/all.csv（或多个 json_dirs）
  2. 调用 gate_unanalyzable → 得到 is_unanalyzable 标记
  3. 筛选 {可分析 且 label=0} 样本 → fit_normalize_stats
  4. 对 {全部可分析样本} 调用 apply_normalize
  5. 输出三个文件：
       output/feature_matrix.csv      # 可分析样本的 z-score 矩阵
       output/gate_decisions.csv      # 全部样本的门控结果
       output/normalize_stats.json    # 归一化统计量

用法：

python features/build_feature_matrix.py --feature_csv data/feature/all.csv --out_dir data/feature


  # 如果特征分散在多个 JSON 目录中（阶段1-3的输出），改用：
  python features/build_feature_matrix.py \
      --json_dirs data/feature/sym data/feature/maze data/feature/circle \
      --labels "data/raw/labels.csv" \
      --out_dir output/
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ── 允许直接运行（不安装包）时找到同目录模块 ────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gate_unanalyzable import apply_gate
from normalize import (
    FEATURE_NAMES,
    fit_normalize_stats,
    apply_normalize,
    save_stats,
)


# ─────────────────────────────────────────────────────────────────────────────
# 数据读取辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def load_from_csv(csv_path: str) -> pd.DataFrame:
    """从单个 CSV（含 F1..C3 + label + game + sample_id）读取特征。"""
    df = pd.read_csv(csv_path)
    required = {"sample_id", "game", "label"} | set(FEATURE_NAMES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少列：{missing}")
    return df.reset_index(drop=True)


def load_from_json_dirs(json_dirs: List[str], labels_csv: str) -> pd.DataFrame:
    """
    从多个 JSON 目录（阶段1-3 输出）读取特征，与 labels.csv 合并。
    JSON 文件名格式：{sample_id}.json，包含 F1..C3 等键。
    labels.csv 格式：sample_id, game, label
    """
    labels_df = pd.read_csv(labels_csv)
    rows = []

    for jdir in json_dirs:
        jdir_path = Path(jdir)
        if not jdir_path.exists():
            warnings.warn(f"JSON 目录不存在，跳过：{jdir}", RuntimeWarning)
            continue
        for jf in sorted(jdir_path.glob("*.json")):
            sample_id = jf.stem
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            row = {"sample_id": sample_id}
            for feat in FEATURE_NAMES:
                row[feat] = data.get(feat, float("nan"))
            rows.append(row)

    if not rows:
        raise RuntimeError("未找到任何 JSON 文件，请检查 --json_dirs 参数")

    feat_df = pd.DataFrame(rows)
    df = feat_df.merge(labels_df, on="sample_id", how="left")
    missing_labels = df["label"].isna().sum()
    if missing_labels > 0:
        warnings.warn(f"{missing_labels} 个样本在 labels.csv 中找不到标注", RuntimeWarning)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df_raw: pd.DataFrame,
    out_dir: str,
    verbose: bool = True,
) -> dict:
    """
    完整的阶段4流程：门控 → 归一化 → 输出三个文件。

    返回一个结果字典，方便外部调用时直接使用数据。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1：门控 ─────────────────────────────────────────────────────
    if verbose:
        print("[Step 1] 运行乱画门控...")
    df_gate = apply_gate(df_raw)

    gate_csv = out_path / "gate_decisions.csv"
    gate_cols = [
        "sample_id", "game", "label",
        "F1", "F2", "F3", "F4",
        "F3_zscore", "F4_zscore",
        "is_unanalyzable", "triggered_rules",
    ]
    # 保留 C1..C3 等额外列
    extra = [c for c in df_gate.columns if c not in gate_cols]
    df_gate[gate_cols + extra].to_csv(gate_csv, index=False)

    n_total = len(df_gate)
    n_flag = int(df_gate["is_unanalyzable"].sum())
    n_analyzable = n_total - n_flag
    if verbose:
        print(f"  总样本: {n_total}，乱画: {n_flag}，可分析: {n_analyzable}")
        n0_flagged = int(df_gate[df_gate["label"] == 0]["is_unanalyzable"].sum())
        print(f"  label=0 中被门控（期望为 0）: {n0_flagged}")
        print(f"  gate_decisions.csv → {gate_csv}")

    # ── Step 2：筛选可分析样本 ────────────────────────────────────────────
    df_analyzable = df_gate[~df_gate["is_unanalyzable"].astype(bool)].reset_index(drop=True)

    # ── Step 3：仅用 label=0 样本拟合归一化统计量 ──────────────────────────
    if verbose:
        print("[Step 2] 拟合归一化统计量（基于 label=0 且可分析样本）...")

    ref_mask = df_analyzable["label"] == 0
    ref_dicts = df_analyzable.loc[ref_mask, FEATURE_NAMES].to_dict(orient="records")
    ref_games = df_analyzable.loc[ref_mask, "game"].tolist()
    ref_labels = [0] * len(ref_dicts)

    stats = fit_normalize_stats(ref_dicts, ref_games, ref_labels)

    stats_json = out_path / "normalize_stats.json"
    save_stats(stats, str(stats_json))
    if verbose:
        for g, gstats in stats.items():
            for f, s in gstats.items():
                print(f"  {g:7s} {f}: median={s['median']:.4f}, "
                      f"scale={s['scale']:.4f}, ref_n={s['ref_n']}")
        print(f"  normalize_stats.json → {stats_json}")

    # ── Step 4：对全部可分析样本做 z-score ───────────────────────────────
    if verbose:
        print("[Step 3] 计算 z-score 矩阵...")

    all_dicts = df_analyzable[FEATURE_NAMES].to_dict(orient="records")
    all_games = df_analyzable["game"].tolist()
    z_matrix = apply_normalize(all_dicts, all_games, stats)

    # ── Step 5：输出 feature_matrix.csv ───────────────────────────────────
    z_cols = [f"{f}_z" for f in FEATURE_NAMES]
    df_z = pd.DataFrame(z_matrix, columns=z_cols)

    df_fm = pd.concat(
        [
            df_analyzable[["sample_id", "game", "label"]].reset_index(drop=True),
            df_z,
        ],
        axis=1,
    )

    fm_csv = out_path / "feature_matrix.csv"
    df_fm.to_csv(fm_csv, index=False)

    if verbose:
        print(f"  feature_matrix.csv → {fm_csv}  ({len(df_fm)} 行)")
        print("\n[Step 4] 初步人工检查提示：")
        label0_z = z_matrix[np.array(df_analyzable["label"].tolist()) == 0]
        label1_z = z_matrix[np.array(df_analyzable["label"].tolist()) == 1]
        print(f"  label=0 样本 z 均值: {label0_z.mean(axis=0).round(2)}")
        print(f"  label=1 样本 z 均值: {label1_z.mean(axis=0).round(2)}")
        n_extreme_0 = int((np.abs(label0_z) > 3).sum())
        n_extreme_1 = int((label1_z > 2).sum())
        print(f"  label=0 中 |z|>3 的单元格数（期望接近 0）: {n_extreme_0}")
        print(f"  label=1 中 z>2  的单元格数（期望 ≥1-2）: {n_extreme_1}")

    return {
        "df_gate": df_gate,
        "df_feature_matrix": df_fm,
        "stats": stats,
        "z_matrix": z_matrix,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="阶段4：特征汇总 + 归一化 → feature_matrix.csv"
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--feature_csv",
        help="直接提供含 F1..C3 + label + game + sample_id 的 CSV（如 data/feature/all.csv）",
    )
    src_group.add_argument(
        "--json_dirs",
        nargs="+",
        help="阶段1-3 JSON 输出目录（需配合 --labels）",
    )
    parser.add_argument(
        "--labels",
        default="data/raw/labels.csv",
        help="labels.csv 路径（仅 --json_dirs 时使用）",
    )
    parser.add_argument(
        "--out_dir",
        default="output",
        help="输出目录（默认 output/）",
    )
    parser.add_argument("--quiet", action="store_true", help="不打印进度信息")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.feature_csv:
        df_raw = load_from_csv(args.feature_csv)
    else:
        df_raw = load_from_json_dirs(args.json_dirs, args.labels)

    build_feature_matrix(df_raw, out_dir=args.out_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()