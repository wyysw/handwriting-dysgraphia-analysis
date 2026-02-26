# main.py
"""
主程序入口，支持可重复的随机抽样测试。
从指定文件夹中随机抽取 N 个文件进行处理，结果输出到 ../result/ 目录。
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime
import random

import pen_trajectory_plotter  # 导入绘图模块（保留，以防后续需要）
import analyze  # 导入分析模块

# --- 配置参数 ---
# 请根据需要修改这些参数
FOLDER_PATH = r"D:/Works/PyProject/手写/sharp_data_20250816/20250728_data"  # 原始数据文件夹
RESULT_DIR = r"D:/Works/PyProject/手写/sharp_data_20250816/result"         # 结果输出目录
SKIP_ROWS = 3
NUM_SAMPLES = 50            # 每次随机抽取的文件数量
RANDOM_SEED = 42            # 随机种子，确保结果可重复

# 核心算法参数
DISTANCE_THRESHOLD = 1000.0
MAX_STROKES_FOR_REFINE = 2
MERGE_THRESHOLD = 50.0

# --- 批量随机抽样测试 ---
def main():
    """主函数：随机抽取文件进行可重复测试。"""
    print(f"[main] 开始可重复随机抽样测试")
    print(f"[main] 数据源: {FOLDER_PATH}")
    print(f"[main] 抽样数量: {NUM_SAMPLES}, 随机种子: {RANDOM_SEED}")
    print(f"[main] 结果将保存至: {RESULT_DIR}")

    # 设置随机种子，确保可重复性
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)  # 如果使用 np.random.choice

    # 获取所有 .txt 文件
    txt_files = glob.glob(os.path.join(FOLDER_PATH, "*.txt"))
    if not txt_files:
        print(f"[main] 错误：在 '{FOLDER_PATH}' 中未找到任何 .txt 文件。")
        return

    # 随机打乱并抽取样本
    total_files = len(txt_files)
    print(f"[main] 共找到 {total_files} 个 .txt 文件。")

    if NUM_SAMPLES >= total_files:
        print(f"[main] 抽样数量 ({NUM_SAMPLES}) 大于等于总文件数，将处理所有文件。")
        selected_files = txt_files
    else:
        # 使用 random.sample 确保无重复抽样
        selected_files = random.sample(txt_files, NUM_SAMPLES)
        print(f"[main] 已使用 seed={RANDOM_SEED} 随机抽取 {NUM_SAMPLES} 个文件。")

    # 创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"test_seed{RANDOM_SEED}_n{NUM_SAMPLES}_{timestamp}.txt"
    log_filepath = os.path.join(RESULT_DIR, log_filename)

    # 写入日志文件
    with open(log_filepath, 'w', encoding='utf-8') as log_file:
        # 写入元信息
        log_file.write(f"可重复测试日志\n")
        log_file.write(f"随机种子: {RANDOM_SEED}\n")
        log_file.write(f"抽样数量: {NUM_SAMPLES}\n")
        log_file.write(f"数据源: {os.path.basename(FOLDER_PATH)}\n")
        log_file.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("\n")  # 空行

        # 写入核心参数
        log_file.write(f"核心参数:\n")
        log_file.write(f"DISTANCE_THRESHOLD = {DISTANCE_THRESHOLD}\n")
        log_file.write(f"MAX_STROKES_FOR_REFINE = {MAX_STROKES_FOR_REFINE}\n")
        log_file.write(f"MERGE_THRESHOLD = {MERGE_THRESHOLD}\n")
        log_file.write("\n")  # 空行

        # 表头
        log_file.write("文件名\t识别文字个数\t备注\n")

        # 遍历选中的文件
        for file_path in selected_files:
            filename = os.path.basename(file_path)
            print(f"\n[main] 处理: {filename}")

            try:
                # --- 1. 加载数据 ---
                raw_data = analyze.load_trajectory_data(file_path, skip_rows=SKIP_ROWS)
                if raw_data is None:
                    log_file.write(f"{filename}\t加载失败\t\n")
                    continue

                # --- 2. 笔画分割 ---
                strokes_list = analyze.split_into_strokes_simple(raw_data)
                if not strokes_list:
                    log_file.write(f"{filename}\t分割失败\t\n")
                    continue

                # --- 3. 聚类成字 ---
                characters_list = analyze.cluster_strokes_simple(strokes_list, DISTANCE_THRESHOLD)
                if not characters_list:
                    log_file.write(f"{filename}\t聚类失败\t\n")
                    continue

                # --- 4. 后处理校验 (我们重点关注这里！) ---
                # 记录合并前的字数
                before_refine = len(characters_list)
                refined_characters = analyze.refine_characters(
                    characters_list,
                    max_strokes=MAX_STROKES_FOR_REFINE,
                    merge_threshold=MERGE_THRESHOLD
                )
                after_refine = len(refined_characters) if refined_characters else 0

                if not refined_characters:
                    log_file.write(f"{filename}\t后处理失败\t\n")
                    continue

                num_chars = after_refine
                # 在备注中写入合并前的字数，便于分析 refine 效果
                note = f"合并前:{before_refine}" if before_refine != num_chars else ""
                print(f"[main] ✅ {filename} -> {num_chars} 个字 (合并前: {before_refine})")

                log_file.write(f"{filename}\t{num_chars}\t{note}\n")

            except Exception as e:
                print(f"[main] 处理 '{filename}' 时出错: {e}")
                log_file.write(f"{filename}\t处理错误\t{str(e)}\n")

    print(f"\n[main] 测试完成！")
    print(f"[main] 日志已保存至: {log_filepath}")


if __name__ == "__main__":
    main()