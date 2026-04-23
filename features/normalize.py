"""
阶段4.B：per-game robust z-score 归一化
功能：基于 label=0 且可分析样本的 median/MAD 进行 robust z-score 归一化，
      并统一方向为"越大越异常"。
"""

from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


FEATURE_NAMES: List[str] = ["F1", "F2", "F3", "F4", "C1", "C2", "C3"]

# 越大越异常：F3/F4/C1/C2/C3 取 +1；F1/F2 越大越正常，取 -1 翻转
DIRECTION: Dict[str, int] = {
    "F1": -1, "F2": -1,
    "F3": +1, "F4": +1,
    "C1": +1, "C2": +1, "C3": +1,
}

DEFAULT_CLIP: Tuple[float, float] = (-3.0, 6.0)


def fit_normalize_stats(
    feature_dicts: List[dict],
    games: List[str],
    labels: List[int],
    feature_names: List[str] = FEATURE_NAMES,
) -> dict:
    """
    仅在 label=0 的样本上（可分析）估计每个 (game, feature) 的 median / MAD / scale。

    返回结构：
    {
      'sym':    {'F1': {'median': 0.72, 'mad': 0.08, 'scale': 0.119, 'ref_n': 4}, ...},
      'maze':   {...},
      'circle': {...},
    }
    """
    games_set = sorted(set(games))
    stats: dict = {g: {} for g in games_set}

    for g in games_set:
        for f in feature_names:
            ref_pool = [
                fd[f]
                for fd, game, lab in zip(feature_dicts, games, labels)
                if game == g and lab == 0
            ]

            if len(ref_pool) == 0:
                warnings.warn(
                    f"[normalize] game={g} feature={f}: 参考池为空，使用 median=0 scale=1e-3",
                    RuntimeWarning,
                )
                stats[g][f] = {"median": 0.0, "mad": 0.0, "scale": 1e-3, "ref_n": 0}
                continue

            arr = np.array(ref_pool, dtype=float)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            scale = 1.4826 * mad
            # 下限保护：防止 MAD≈0 时 z-score 爆炸（sym 只有 4 个正常样本时尤为关键）
            scale = max(scale, 0.05 * abs(med) + 1e-3)

            if len(ref_pool) < 3:
                warnings.warn(
                    f"[normalize] game={g} feature={f}: 参考池仅 {len(ref_pool)} 个样本，"
                    f"MAD 估计不稳定（scale={scale:.4f}）",
                    RuntimeWarning,
                )

            stats[g][f] = {
                "median": med,
                "mad": mad,
                "scale": scale,
                "ref_n": len(ref_pool),
            }

    return stats


def apply_normalize(
    feature_dicts: List[dict],
    games: List[str],
    stats: dict,
    feature_names: List[str] = FEATURE_NAMES,
    direction: Dict[str, int] = DIRECTION,
    clip: Tuple[float, float] = DEFAULT_CLIP,
) -> np.ndarray:
    """
    对每个样本按 (game, feature) 应用 robust z-score，统一方向后裁剪。

    返回：shape (N, len(feature_names)) 的 np.ndarray
    """
    N = len(feature_dicts)
    K = len(feature_names)
    result = np.zeros((N, K), dtype=float)

    for i, (fd, g) in enumerate(zip(feature_dicts, games)):
        for j, f in enumerate(feature_names):
            s = stats[g][f]
            z_raw = (fd[f] - s["median"]) / s["scale"]
            z = direction[f] * z_raw
            z = float(np.clip(z, clip[0], clip[1]))
            result[i, j] = z

    return result


def normalize_dataframe(
    df: pd.DataFrame,
    feature_names: List[str] = FEATURE_NAMES,
    direction: Dict[str, int] = DIRECTION,
    clip: Tuple[float, float] = DEFAULT_CLIP,
    stats: Optional[dict] = None,
) -> Tuple[np.ndarray, dict]:
    """
    便捷函数：从 DataFrame 中直接完成 fit + apply。
    只使用 is_unanalyzable==False 且 label==0 的行来拟合 stats。
    全部可分析行（is_unanalyzable==False）均被 transform。

    返回：
      z_matrix  — shape (N_analyzable, 7) 的 z-score 矩阵
      stats     — 归一化统计字典
    """
    # 可分析行
    mask_analyzable = ~df["is_unanalyzable"].astype(bool) if "is_unanalyzable" in df.columns else pd.Series(True, index=df.index)
    df_analyzable = df[mask_analyzable].reset_index(drop=True)

    feature_dicts = df_analyzable[feature_names].to_dict(orient="records")
    games = df_analyzable["game"].tolist()
    labels = df_analyzable["label"].tolist()

    if stats is None:
        # 只用 label=0 且可分析的行来拟合
        ref_dicts = [fd for fd, lab in zip(feature_dicts, labels) if lab == 0]
        ref_games = [g for g, lab in zip(games, labels) if lab == 0]
        ref_labels = [0] * len(ref_dicts)
        stats = fit_normalize_stats(ref_dicts, ref_games, ref_labels, feature_names)

    z_matrix = apply_normalize(feature_dicts, games, stats, feature_names, direction, clip)
    return z_matrix, stats


def save_stats(stats: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def load_stats(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── CLI（单独使用时调试用） ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="robust z-score 归一化（调试用）")
    parser.add_argument("--input", required=True, help="含 F1..C3 + label + game 的 CSV")
    parser.add_argument("--stats_out", default=None, help="输出 normalize_stats.json 路径")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "is_unanalyzable" not in df.columns:
        df["is_unanalyzable"] = False

    z_matrix, stats = normalize_dataframe(df)

    print(f"z-score 矩阵形状: {z_matrix.shape}")
    print(f"各特征均值（应接近 0）: {z_matrix.mean(axis=0).round(3)}")
    print(f"各特征标准差: {z_matrix.std(axis=0).round(3)}")

    if args.stats_out:
        save_stats(stats, args.stats_out)
        print(f"统计量已保存至: {args.stats_out}")