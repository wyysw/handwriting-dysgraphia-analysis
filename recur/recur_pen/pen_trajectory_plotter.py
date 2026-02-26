# pen_trajectory_plotter.py
"""
电子笔轨迹可视化模块。

提供函数用于绘制单个笔画或完整轨迹图。
与数据分析逻辑分离。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.font_manager as fm

# --- 内部辅助函数 ---

def _find_chinese_font():
    """尝试找到系统中一个支持中文的字体。"""
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'STHeiti', 'Songti SC',
        'Arial Unicode MS', 'Noto Sans CJK SC', 'DejaVu Sans'
    ]
    for font_name in chinese_fonts:
        try:
            font_prop = fm.FontProperties(family=font_name, size=12)
            return font_prop
        except:
            continue
    return fm.FontProperties(size=12) # Fallback to default

# 注意：_prepare_plot_data 函数已移除, 因为其功能已整合到 plot_stroke 中
# 或者可以保留, 但需要修改以支持颜色。这里选择直接在 plot_stroke 中处理。

# --- 对外公开的主函数 ---

# ... (plot_full_trajectory 保持不变) ...

# --- 新增/修改：用于绘制单个笔画的函数, 增加颜色支持 ---

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
    x_coords = np.asarray(stroke_data['x'])
    y_coords = np.asarray(stroke_data['y'])
    pressures = np.asarray(stroke_data['pressure'])

    if not (len(x_coords) == len(y_coords) == len(pressures)):
        raise ValueError("输入的 'x', 'y', 'pressure' 数组长度必须相同。")

    # 字体处理
    if font_prop is None:
        font_prop = _find_chinese_font()

    # 如果没有提供 ax, 则创建新的
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        title = '电子笔笔画'
        if stroke_index is not None:
            title += f" ({stroke_index})"
        ax.set_title(title, fontproperties=font_prop)
        ax.set_xlabel('X 坐标', fontproperties=font_prop)
        ax.set_ylabel('Y 坐标', fontproperties=font_prop)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(font_prop)
    # 如果提供了 ax, 确保其属性设置正确（第一次调用时）
    # 注意：这里假设第一次调用时设置了 xlim, ylim 等, 后续调用不会更改
    # 如果需要动态更新范围, 逻辑会更复杂

    # 至少需要两个点才能形成线段
    if len(pressures) < 2:
        print(f"[plotter] 笔画 {stroke_index} 点数不足 (少于2个), 跳过绘制。") # 可选提示
        return fig, ax # 直接返回, 不绘制

    points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if color is not None:
        # 如果指定了颜色, 则使用统一颜色绘制所有线段
        lc = LineCollection(segments, colors=color)
        # 可以选择性地设置线宽等属性
        # lc.set_linewidth(2)
    else:
        # 如果未指定颜色, 则使用原始的基于压力的颜色映射
        # 重新拉伸当前笔画的压力值到 [0, 1] 用于颜色映射
        pressures_remapped = np.array([])
        if len(pressures) > 0:
            p_min = pressures.min()
            p_max = pressures.max()
            if p_max != p_min:
                pressures_remapped = (pressures - p_min) / (p_max - p_min)
            else:
                pressures_remapped = np.zeros_like(pressures)

        lc = LineCollection(segments, cmap='viridis')
        lc.set_array(pressures_remapped)

    # 将 LineCollection 添加到坐标轴
    ax.add_collection(lc)

    # 注意：plt.show() 不应在此函数内调用, 以便于外部控制何时显示。
    # 显示逻辑应由调用者控制。
    return fig, ax
