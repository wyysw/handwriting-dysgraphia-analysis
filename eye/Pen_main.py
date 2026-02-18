# Pen_main.py
"""
主程序入口，用于交互式选择单个文件进行分析和可视化，并对比 refine 前后的效果。
"""

import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

# --- 导入自定义模块 ---
import pen_trajectory_plotter
import analyze

# --- 配置参数 ---
SKIP_ROWS = 3
DISTANCE_THRESHOLD = 1000.0
MAX_STROKES_FOR_REFINE = 2
MERGE_THRESHOLD = 15000.0


def plot_results(fig, ax, characters, xlim, ylim, title_suffix, highlight_pairs=None):
    """
    highlight_pairs: List[Tuple[int, int]]，要高亮的原始字符索引对
    """
    total_chars = len(characters)
    rng = np.random.default_rng(seed=42)
    random_colors = rng.random((total_chars, 3))

    # 如果有 highlight_pairs，构建高亮索引集合
    highlight_indices = set()
    if highlight_pairs:
        for i, j in highlight_pairs:
            highlight_indices.add(i)
            highlight_indices.add(j)

    for idx, character in enumerate(characters):
        if idx in highlight_indices:
            color = 'red'  # 被合并的字用红色
            linewidth = 2.0
        else:
            color = random_colors[idx]
            linewidth = 1.0

        for stroke in character:
            ax.plot(stroke['x'], stroke['y'], color=color, linewidth=linewidth)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()

    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    ax.set_aspect(x_span / y_span, adjustable='box')

    ax.set_title(f"Result ({title_suffix}) — {total_chars} chars")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def main():
    print("[main] 请选择一个电子笔轨迹数据文件 (.txt)...")
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="选择电子笔轨迹数据文件",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()

    if not file_path:
        print("[main] 未选择文件，程序退出。")
        return

    print(f"[main] 已选择文件: {file_path}")

    # --- 加载数据（返回字典）---
    raw_data = analyze.load_trajectory_data(file_path, skip_rows=SKIP_ROWS)
    if raw_data is None:
        print("[main] 数据加载失败，程序退出。")
        return

    # --- 计算坐标范围（从字典中提取）---
    try:
        x_coords = raw_data['x']
        y_coords = raw_data['y']
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        margin = 500
        xlim = (x_min - margin, x_max + margin)
        ylim = (y_min - margin, y_max + margin)
        print(f"[main] 数据范围 (含边距): X={xlim}, Y={ylim}")
    except Exception as e:
        print(f"[main] 计算坐标范围失败: {e}")
        return

    # --- 分割笔画 ---
    strokes_list = analyze.split_into_strokes_simple(raw_data)
    if not strokes_list:
        print("[main] 笔画分割失败，程序退出。")
        return

    # --- 聚类成字（原始结果）---
    characters_original = analyze.cluster_strokes_simple(strokes_list, threshold=1000.00)
    if not characters_original:
        print("[main] 聚类失败，程序退出。")
        return


    # 在 main() 中调用 refine 的地方
    characters_refined, merged_pairs = analyze.refine_characters(
        characters_original,
        max_strokes=MAX_STROKES_FOR_REFINE,
        merge_threshold=MERGE_THRESHOLD
    )

    total_orig = len(characters_original)
    total_refined = len(characters_refined)
    print(f"[main] 原始聚类: {len(characters_original)} 个字 → Refine 后: {len(characters_refined)} 个字")
    if merged_pairs:
        print(f"[main] 被合并的字符对（原始索引）: {merged_pairs}")
    else:
        print("[main] 无字符被合并")

    # total_orig = len(characters_original)
    # total_refined = len(characters_refined)
    # print(f"[main] 原始聚类: {total_orig} 个字 → Refine 后: {total_refined} 个字")

    # 绘制原始结果（带高亮）
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    plot_results(
        fig1, ax1,
        characters_original,  # ← 用原始字符列表
        xlim, ylim,
        "Original (with merged highlighted)",
        highlight_pairs=merged_pairs  # ← 传入合并对
    )

    # 绘制 refine 后结果（普通颜色）
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    plot_results(
        fig2, ax2,
        characters_refined,
        xlim, ylim,
        "Refined"
    )

    plt.show()
    # ✅ 关键：最后统一显示，两张图会同时弹出！
    plt.show()  # 非阻塞？不，仍是阻塞，但会同时显示所有未关闭的图！

    print("\n[main] 分析与可视化完成！")

if __name__ == "__main__":
    main()