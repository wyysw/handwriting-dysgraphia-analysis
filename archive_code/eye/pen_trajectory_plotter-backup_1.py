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
    return fm.FontProperties(size=12)  # Fallback to default


def _prepare_plot_data(x_coords, y_coords, pressures_remapped):
    """准备绘图所需的 LineCollection 数据。"""
    # 至少需要两个点才能形成线段
    if len(pressures_remapped) < 2:
        return None, None

    points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis')
    lc.set_array(pressures_remapped)
    return lc, segments


# --- 对外公开的主函数 ---

def plot_full_trajectory(data, pressure_threshold=0, title_suffix="", font_prop=None):
    """
    (保留) 根据给定的数据绘制完整的电子笔轨迹图。

    参数:
    data (dict): 包含轨迹数据的字典。
                 必须包含以下键:
                 - 'x' (numpy.ndarray 或 list): X 坐标数组。
                 - 'y' (numpy.ndarray 或 list): Y 坐标数组。
                 - 'pressure' (numpy.ndarray 或 list): 压力值数组。
                 可选包含以下键:
                 - 'xlim' (tuple, optional): (xmin, xmax) 用于设置X轴范围。
                                             如果未提供，将根据x_coords自动计算。
                 - 'ylim' (tuple, optional): (ymin, ymax) 用于设置Y轴范围。
                                             如果未提供，将根据y_coords自动计算。
    pressure_threshold (float, optional): 压力阈值。低于此值的压力将被视为0。
                                          默认为 0 (不过滤)。
    title_suffix (str, optional): 添加到图表标题的后缀。
    font_prop (matplotlib.font_manager.FontProperties, optional):
              用于图表中文显示的字体属性。如果为 None，将尝试自动查找。
    """
    try:
        x_coords = np.asarray(data['x'])
        y_coords = np.asarray(data['y'])
        pressures = np.asarray(data['pressure'])

        if not (len(x_coords) == len(y_coords) == len(pressures)):
            raise ValueError("输入的 'x', 'y', 'pressure' 数组长度必须相同。")

        # 处理 xlim 和 ylim
        if 'xlim' in data and data['xlim'] is not None:
            xlim = data['xlim']
        else:
            margin_percent = 0.02
            x_min, x_max = x_coords.min(), x_coords.max()
            x_range = x_max - x_min
            x_margin = x_range * margin_percent
            xlim = (x_min - x_margin, x_max + x_margin)

        if 'ylim' in data and data['ylim'] is not None:
            ylim = data['ylim']
        else:
            margin_percent = 0.02
            y_min, y_max = y_coords.min(), y_coords.max()
            y_range = y_max - y_min
            y_margin = y_range * margin_percent
            ylim = (y_min - y_margin, y_max + y_margin)

        # 应用压力阈值
        if pressure_threshold > 0:
            valid_mask = pressures >= pressure_threshold
            x_coords_filtered = x_coords[valid_mask]
            y_coords_filtered = y_coords[valid_mask]
            pressures_filtered = pressures[valid_mask]
        else:
            x_coords_filtered = x_coords
            y_coords_filtered = y_coords
            pressures_filtered = pressures

        # 重新拉伸压力值到 [0, 1] 用于颜色映射
        pressures_remapped = np.array([])
        if len(pressures_filtered) > 0:
            p_min_filtered = pressures_filtered.min()
            p_max_filtered = pressures_filtered.max()
            if p_max_filtered == p_min_filtered:
                pressures_remapped = np.zeros_like(pressures_filtered)
            else:
                pressures_remapped = (pressures_filtered - p_min_filtered) / (p_max_filtered - p_min_filtered)

        # 字体处理
        if font_prop is None:
            font_prop = _find_chinese_font()

        # 准备绘图数据
        lc, _ = _prepare_plot_data(x_coords_filtered, y_coords_filtered, pressures_remapped)

        # 创建并显示图形
        fig, ax = plt.subplots(figsize=(12, 9), dpi=100)

        if lc is not None:
            line = ax.add_collection(lc)
            cbar = fig.colorbar(line, ax=ax)
            cbar.set_label('相对压力 (重映射)', fontproperties=font_prop)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')

        ax.set_title(f'电子笔轨迹 {title_suffix}', fontproperties=font_prop)
        ax.set_xlabel('X 坐标', fontproperties=font_prop)
        ax.set_ylabel('Y 坐标', fontproperties=font_prop)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(font_prop)

        plt.show()

    except Exception as e:
        print(f"绘图时发生错误: {e}")


# --- 新增：用于绘制单个笔画的函数 ---

def plot_stroke(stroke_data, xlim, ylim, ax=None, fig=None, font_prop=None, stroke_index=None):
    """
    在给定的坐标轴上绘制单个笔画。

    此函数设计为可重复调用以在同一个图上绘制多个笔画。

    参数:
    stroke_data (dict): 包含单个笔画数据的字典。
                        必须包含 'x', 'y', 'pressure' 键。
    xlim (tuple): (xmin, xmax) 用于设置X轴范围。
    ylim (tuple): (ymin, ymax) 用于设置Y轴范围。
    ax (matplotlib.axes.Axes, optional): 要绘制到的坐标轴对象。
                                         如果为 None，将创建新的图形和坐标轴。
    fig (matplotlib.figure.Figure, optional): 与 ax 关联的图形对象。
                                              仅在 ax 不为 None 时需要传递。
    font_prop (matplotlib.font_manager.FontProperties, optional):
              用于图表中文显示的字体属性。如果为 None，将尝试自动查找。
    stroke_index (int, optional): 当前笔画的索引，用于标题显示。

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

    # 如果没有提供 ax，则创建新的
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        title = '电子笔笔画'
        if stroke_index is not None:
            title += f" {stroke_index}"
        ax.set_title(title, fontproperties=font_prop)
        ax.set_xlabel('X 坐标', fontproperties=font_prop)
        ax.set_ylabel('Y 坐标', fontproperties=font_prop)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(font_prop)
    # 如果提供了 ax，确保其属性设置正确（第一次调用时）
    # 注意：这里假设第一次调用时设置了 xlim, ylim 等，后续调用不会更改
    # 如果需要动态更新范围，逻辑会更复杂

    # 重新拉伸当前笔画的压力值到 [0, 1] 用于颜色映射
    pressures_remapped = np.array([])
    if len(pressures) > 0:
        p_min = pressures.min()
        p_max = pressures.max()
        if p_max != p_min:
            pressures_remapped = (pressures - p_min) / (p_max - p_min)
        else:
            pressures_remapped = np.zeros_like(pressures)

    # 准备并绘制当前笔画数据
    lc, _ = _prepare_plot_data(x_coords, y_coords, pressures_remapped)
    if lc is not None:
        ax.add_collection(lc)
        # 注意：如果在同一个图上绘制多个笔画且需要 colorbar，
        # 动态更新 colorbar 会比较复杂。
        # 一种简单方法是在第一次绘制时创建一个基于所有笔画压力范围的 colorbar，
        # 或者为每个笔画单独处理颜色（如当前实现）。
        # 这里我们为每个笔画独立映射颜色。

    # 注意：plt.show() 不应在此函数内调用，以便于外部控制何时显示。
    # 显示逻辑应由调用者控制。
    return fig, ax




