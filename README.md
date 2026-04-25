# Handwriting Dysgraphia Analysis

---

This repository contains the implementation for the undergraduate thesis:

**面向书写障碍儿童的手写交互数据评估算法设计与实现**

**Algorithm Design and Implementation for Evaluating Handwriting Interaction Data in Children with Dysgraphia**

**Technologies**: Python

---

## 实验推进路线（按阶段）

```
阶段 1 （已完成）  对称游戏特征提取器（重构/扩展 sym_analyze3.py）
   ↓
阶段 2（已完成）   迷宫游戏特征提取器（方形）
   ↓
阶段 3（已完成）   迷宫游戏特征提取器（圆形，与方形迷宫共用基类）
   ↓
阶段 4（已完成）   特征汇总处理
   ↓
阶段 5（已完成）   分类器设计与实现
   ↓
阶段 6（已完成）   实验评估
```


## 项目架构

```
project_root/
│
├── data/                              # 数据目录
│
├── shape/                             # 阶段0：游戏图层提取模块
│   ├── __init__.py
│   ├── final_shape_migong.py          # 方形迷宫
│   ├── final_shape_sym.py             # 对称图形
│   └── final_shape_circle.py          # 圆形迷宫
│
├── features/                          # 阶段2~4：特征处理模块
│   ├── stroke_utils.py                # C1/C2/C3 公共实现
│   ├── sym_feature_extractor.py       # 对称游戏 F1-F4
│   ├── maze_feature_extractor.py      # 迷宫游戏 F1-F4（方形+圆形）
│   ├── maze_geometry.py               # 迷宫通道几何+解路径
│   ├── mask_utils.py                  # 图像mask读取、填充、bbox提取
│   ├── stroke_metrics.py              # Hough线段提取+C1/C2/C3质量指标
│   ├── build_feature_matrix.py        # 汇总+归一化+乱画检测
│   ├── gate_unanalyzable.py           # 乱画门控
│   ├── normalize.py                   # 非障碍样本的robust z-score归一化
│   └── trajectory_io.py               # 轨迹文件读取与笔段切割
│
├── classifiers/                       # 阶段5：分类器
│   ├── m1_semi_prior.py               # 半先验加权线性评分
│   ├── m2_pure_prior.py               # 纯先验加权线性评分
│   ├── m3_logistic.py                 # L2 Logistic Regression
│   ├── m4_random_forest.py            # Random Forest
│   └── base.py
│
├── experiments/                       # 阶段6：实验测试
│   ├── shape_m3.py                    # SHAP测试
│   └── run_experiments.py             # LOSO 评估
│
├── results/                           # 阶段5输出
│   ├── feature_weights.png
│   ├── gate_diagnostics.csv
│   ├── metrics_main.csv
│   ├── per_game_auroc.csv
│   ├── roc_pr_curves.png
│   ├── spearman_feature_rank.csv
│   └── spearman_heatmap.png
│
├── tools/                             # 非核心代码
│   ├── archive_feature.py             # 整合提取后的特征值
│   └── overlay_img.py
│
├── 1_shape.py                         # 调度shape/中脚本运行游戏图层提取程序
│
├── 2_feature.py                       # 调度features/中与三个游戏特征提取相关的模块
│
├── test_classifiers.py                # 调度分类器运行
│
└── EXP_DOC.md                         # 实验文档
```

## 运行流程

```powershell
# 游戏图层提取
python 1_shape.py

# 对称游戏特征提取
python 2_feature.py --game sym  # 批量处理
python features/sym_feature_extractor.py --txt data/raw/sym/{id}.txt --png data/raw/sym/{id}.png --blue data/shape_out/sym_blue_mask.png --helper data/shape_out/sym_helper_mask.png  --out data/feature/sym/{id}.json --vis data/feature/sym/vis_{id}.png  # 单个样本

# 方形迷宫游戏特征提取
python 2_feature.py --game maze
python features/maze_feature_extractor.py --txt data/raw/maze/{id}.txt --png data/raw/maze/{id}.png --mask data/shape_out/maze_mask.png --out data/feature/maze/{id}.json --vis_dir data/feature/maze/vis_{id} --sample_id {id} --game maze

# 圆形迷宫游戏特征提取
python 2_feature.py --game circle
python features/maze_feature_extractor.py --txt data/raw/circle/{id}.txt --png data/raw/circle/{id}.png --mask data/shape_out/circle_mask.png --game circle --out data/feature/circle/{id}.json --vis_dir data/feature/circle/vis_{id} --sample_id {id}

# 特征归一化
python features/build_feature_matrix.py --feature_csv data/feature/all.csv --out_dir data/feature

# 运行分类器
python test_classifiers.py

# 运行诊断
python run_experiments.py --feature_matrix data/feature/all.csv --gate_decisions data/feature/gate_decisions.csv --out_dir results/


```


## 特征总表

三个游戏（对称、方形迷宫、圆形迷宫）的 7 维特征（F1–F4, C1–C3）说明总表如下：

| 轴 | ID | 名称 | 方向 | 对称游戏 | 方形迷宫 | 圆形迷宫 |
|----|----|------|------|----------|----------|----------|
| F | F1 | 完成度 | ↑（越大越好） | **翻转容差 F1**：膨胀后的用户翻转笔迹与目标图形的 F1 分数（IoU‑like） | **通道覆盖率（解路径）**：`\|user ∩ solution_channel\| / \|solution_channel\|`，反映用户笔迹覆盖解路径的比例 | 同方形迷宫，`solution_channel` 基于圆形解路径生成 |
| F | F2 | 关键结构对齐 | ↑（越大越好） | **关键点命中率**：目标图形网格交点（含辅助线交点）被用户笔迹覆盖的比例（半径 ≤6 像素） | **解路径采样点命中率**：沿解路径按弧长等距采样（步长 40 像素），落在用户笔迹半径 12 像素内的比例 | 同方形迷宫（弧长采样天然适用于圆形路径） |
| F | F3 | 无效书写 | ↓（越小越好） | **无效书写加权距离**：用户笔迹（翻转后）偏离有效区域（目标图形膨胀 9 像素）的加权距离，加上越轴惩罚（上半画布额外 ×10），除以目标面积 | **无效书写加权距离**：`Σ channel_dist[user & ¬channel] / \|channel\|`，`channel_dist` 为通道外像素到最近通道的距离 | 同方形迷宫 |
| F | F4 | 路径偏离比 | ↓（越小越好） | **路径偏离比**：用户笔迹（翻转后）落在有效区域（目标图形膨胀 9 像素）外的像素比例 | **通道外笔迹比**：`\|user ∩ ¬channel\| / \|user\|`，用户笔迹落在全通道外的比例 | 同方形迷宫 |
| C | C1 | 抖动比例 | ↓（越小越好） | **抖动比例**：用户笔迹点到最近霍夫线段（从目标图形提取）的垂直距离 > 3 像素的点数 / 有效投影点数 | **抖动比例**：用户笔迹点到自笔迹提取的霍夫线段（HoughLinesP）的垂直距离 > 3 像素的比例（参数可调） | **抖动比例**：用户笔迹点到骨架（`channel_skeleton`）的最近距离 > 3 像素的比例（距离残差法） |
| C | C2 | 短笔段比例 | ↓（越小越好） | **短笔段比例**：长度 < 画布对角线 × 0.02 的笔段总长度 / 所有笔段总长度 | 同对称游戏 | 同对称游戏 |
| C | C3 | 压力变异系数 | ↓（越小越好） | **压力变异系数**：所有有效压力值（剔除笔段两端各 3 点）的标准差 / 均值 | 同对称游戏 | 同对称游戏 |

> **说明**：
> - 方向“↑”表示数值越大代表书写质量越好（正常儿童倾向高值），“↓”表示数值越小越好（正常儿童倾向低值）。
> - 对称游戏特征详细参数（膨胀半径、命中半径、惩罚系数等）参见文档 §4.1.2。
> - 方形/圆形迷宫特征详细参数（采样步长、命中半径、通道膨胀半径等）参见文档 §4.2.2、§4.3.4–§4.3.5。
> - C1 在圆形迷宫中采用骨架距离法，与方形（霍夫法）实现不同，但二者均衡量“笔迹偏离理想轨迹的比例”，方法论上保持统一。




