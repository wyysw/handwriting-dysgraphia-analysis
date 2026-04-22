# Handwriting Dysgraphia Analysis

---

This repository contains the implementation for the undergraduate thesis:

**面向书写障碍儿童的手写交互数据评估算法设计与实现**

**Algorithm Design and Implementation for Evaluating Handwriting Interaction Data in Children with Dysgraphia**

**Technologies**: Python

---

## 一、游戏页面图片预处理

```
python shape.py
```

```
project_root/
├── data/
│   ├── 34migong.png           # 方形迷宫测评原图（1201×1601）
│   ├── 35duichen.png          # 对称补画测评原图（1201×1601）
│   └── 36circle.png           # 圆形迷宫测评原图（1201×1601）
├── shape/
│   ├── __init__.py
│   ├── final_shape_migong.py  # 方形迷宫提取
│   ├── final_shape_sym.py     # 对称图辅助线提取
│   └── final_shape_circle.py  # 圆形迷宫提取
├── output_maze/shape_maze/    # 方形迷宫输出
├── output_sym/shape_sym/      # 对称图输出
├── output_circle/shape_circle/# 圆形迷宫输出
└── shape.py                   # 统一调度脚本（执行并可视化）
```

| 脚本 | 输入 | 关键输出 |
|---|---|---|
| `shape/final_shape_sym.py` | `data/35duichen.png` | `sym_blue_mask.png`（蓝色半边图形）, `sym_helper_mask_completed.png`（网格+对称轴+外框） |
| `shape/final_shape_migong.py` | `data/34migong.png` | `maze_mask.png`（迷宫墙壁线条） |
| `shape/final_shape_circle.py` | `data/36circle.png` | `circle_mask.png`（圆形迷宫墙壁线条） |
| `shape.py` | — | 统一调度以上三个脚本 |

所有输出掩码与原图保持 **相同的 1201 × 1601 尺寸**，背景像素为 0（黑），前景（被提取的线条）像素为 255（白）。


## 二、Pen模块

**from pen import pen_trajectory_plotter**

```python
def plot_stroke(stroke_data, xlim, ylim, ax=None, fig=None, font_prop=None, stroke_index=None, color=None):
    """
    在给定的坐标轴上绘制单个笔画。

    可重复调用以在同一个图上绘制多个笔画。

    参数:
    stroke_data (dict): 包含单个笔画数据的字典。
                        必须包含 'x', 'y', 'pressure' 键。
    xlim (tuple): (xmin, xmax) 用于设置X轴范围。
    ylim (tuple): (ymin, ymax) 用于设置Y轴范围。
    ax (matplotlib.axes.Axes, optional): 要绘制到的坐标轴对象。
                                         如果为 None, 将创建新的图形和坐标轴。
    fig (matplotlib.figure.Figure, optional): 与 ax 关联的图形对象。
                                              仅在 ax 不为 None 时需要传递。
    font_prop (matplotlib.font_manager.FontProperties, optional):
              用于图表中文显示的字体属性。如果为 None, 将尝试自动查找。
    stroke_index (str, optional): 当前笔画的标识符, 用于标题或调试显示。
    color (tuple or str, optional): 指定笔画的颜色, 例如 (r, g, b) 或 'red'。
                                    若为None, 则使用基于压力值的颜色映射(viridis)，压力高的点颜色较亮。

    返回:
    tuple: (fig, ax) 返回使用的图形和坐标轴对象。
    """
```

**from pen import analyze**

```python
def load_trajectory_data(filepath, skip_rows=0):
    """
    从文件加载电子笔轨迹数据。
    参数:
        filepath (str): 数据文件路径。
        skip_rows (int): 要跳过的文件开头行数。
    返回:
        dict or None: 成功时返回包含 'x', 'y', 'pressure' 的字典，失败时返回 None。
    """
```

```python
def split_into_strokes_simple(data):
    """
    根据压力值是否为0, 将连续点序列分割为多个笔画。
    压力 > 0 表示笔尖落下, 压力 = 0 表示提笔。
    参数： data —— 包含 x, y, pressure 的字典。
    返回：列表，每个元素是一个笔画字典，包含该笔画的 x, y, pressure 数组。
    遍历压力序列，当压力 > 0 时将点加入当前笔画，
    当压力 = 0 且当前笔画非空时，保存当前笔画并开始新笔画。
    最后将最后一个笔画也加入列表。
    """
```

```python
def calculate_adaptive_threshold(strokes, k=2.0, min_threshold=300.0, max_threshold=2500.0):
    """
    基于笔画中心点的平均最近邻距离计算自适应聚类阈值。原理：阈值 = k * (所有笔画到其最近邻笔画中心的平均距离)
    优点：直接反映笔画空间密度，适应不同书写大小和风格。
    Parameters:
        strokes: 笔画列表List of stroke dicts with 'x', 'y'
        k: 缩放系数，建议初始值 1.8~2.5 （可调）
        min_threshold / max_threshold: 安全边界，防止极端值
    Returns:
        float: 计算出的自适应阈值
    """
```


## 三、对称游戏特征提取

```
python features/sym_feature_extractor.py --txt data/samples/sym/{id}.txt --png data/samples/sym/{id}.png --blue output_sym/shape_sym/sym_blue_mask.png --helper output_sym/shape_sym/sym_helper_mask_completed.png  --out output_sym/extract/{id}.json --vis vis_{id}.png
```


## 四、方形迷宫特征提取

```
python features/maze_feature_extractor.py --txt data/samples/maze/{id}.txt --png data/samples/maze/{id}.png --mask output_maze/shape_maze/maze_mask.png --out output_maze/extract/{id}.json --vis_dir output_maze/extract/vis_{id} --sample_id {id}
```