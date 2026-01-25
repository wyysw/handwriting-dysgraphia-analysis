# analyze.py
"""
电子笔轨迹数据分析模块。

此模块将包含用于处理和分析轨迹数据的函数，
例如笔画分割、特征提取、笔画聚类（成字）等。
"""

import numpy as np

# --- 笔画分割 ---

def split_into_strokes_simple(data):
    """
    使用简单的基于压力为0的点分割方法，将轨迹数据分割成笔画。

    这是一个基础的分割方法，后续可以被更复杂的基于机器学习的模型替代。

    参数:
        data (dict): 包含完整轨迹数据的字典。
                     必须包含 'x', 'y', 'pressure' 键，
                     其值为 numpy.ndarray 或可转换为 numpy.ndarray 的列表。

    返回:
        list: 一个列表，其中每个元素都是一个字典，
              代表一个分割出的笔画，包含 'x', 'y', 'pressure' 键。
              如果输入数据为空或无效，可能返回空列表。
    """
    try:
        # 确保输入是 numpy 数组
        x_coords = np.asarray(data['x'])
        y_coords = np.asarray(data['y'])
        pressures = np.asarray(data['pressure'])

        # 基本校验
        if not (len(x_coords) == len(y_coords) == len(pressures)):
            raise ValueError("输入数据的 'x', 'y', 'pressure' length 不一致。")

        if len(x_coords) == 0:
            print("警告：输入数据为空。")
            return []

        strokes = []
        current_stroke_x = []
        current_stroke_y = []
        current_stroke_p = []

        # 遍历所有数据点
        for i in range(len(pressures)):
            if pressures[i] > 0:
                # 如果当前点有压力 (> 0)，则属于当前笔画
                current_stroke_x.append(x_coords[i])
                current_stroke_y.append(y_coords[i])
                current_stroke_p.append(pressures[i])
            else:
                # 如果当前点压力为 0 (或 <= 0)
                if current_stroke_x:  # 检查当前笔画缓存是否非空
                    # 如果缓存非空，说明一个笔画结束，保存它
                    strokes.append({
                        'x': np.array(current_stroke_x),
                        'y': np.array(current_stroke_y),
                        'pressure': np.array(current_stroke_p)
                    })
                    # 重置缓存，准备下一个笔画
                    current_stroke_x = []
                    current_stroke_y = []
                    current_stroke_p = []
                # 如果缓存为空（即连续的0压力点），则忽略这些点

        # 循环结束后，检查是否还有未保存的笔画（以非0压力点结尾的情况）
        if current_stroke_x:
            strokes.append({
                'x': np.array(current_stroke_x),
                'y': np.array(current_stroke_y),
                'pressure': np.array(current_stroke_p)
            })

        print(f"[analyze] 轨迹数据已分割为 {len(strokes)} 个笔画。")
        return strokes

    except Exception as e:
        print(f"[analyze] 分割笔画时发生错误: {e}")
        # 根据需要决定是否 re-raise 异常
        # raise # 如果希望调用者处理，可以取消注释
        return [] # 或返回空列表表示失败

# --- 笔画聚类 (成字) ---

def cluster_strokes_simple(strokes, distance_threshold):
    """
    使用简单的基于距离的启发式方法，将笔画聚类成“字”。

    假设字是按顺序书写的。通过计算连续笔画之间的“距离”（当前笔画起点到上一笔画终点）
    来判断是否属于同一字。距离小于阈值则合并。

    这是一个基础的聚类方法，后续可以被更复杂的基于机器学习的模型替代。

    参数:
        strokes (list): 由 split_into_strokes_simple 生成的笔画列表。
                        每个元素是包含 'x', 'y', 'pressure' 的字典。
        distance_threshold (float): 判断两个笔画是否属于同一字的距离阈值。

    返回:
        list: 一个列表，其中每个元素都是一个“字”。
              每个“字”本身也是一个列表，包含构成该字的所有笔画 (字典)。
              例如: [ [stroke1, stroke2], [stroke3], [stroke4, stroke5, stroke6], ... ]
              如果输入笔画列表为空，返回空列表。
    """
    if not strokes:
        print("[analyze] 警告：输入笔画列表为空，无法进行聚类。")
        return []

    characters = []
    current_character = [strokes[0]] # 将第一个笔画作为第一个字符的开始

    # 从第二个笔画开始遍历
    for i in range(1, len(strokes)):
        prev_stroke = strokes[i - 1]
        curr_stroke = strokes[i]

        # 获取前一笔画的终点坐标
        if len(prev_stroke['x']) > 0:
            prev_end_x = prev_stroke['x'][-1]
            prev_end_y = prev_stroke['y'][-1]
        else:
            # 如果前一笔画没有点（理论上不应发生），跳过聚类逻辑
            print(f"[analyze] 警告：笔画 {i-1} 没有数据点，跳过与笔画 {i} 的聚类判断。")
            current_character.append(curr_stroke)
            continue

        # 获取当前笔画的起点坐标
        if len(curr_stroke['x']) > 0:
            curr_start_x = curr_stroke['x'][0]
            curr_start_y = curr_stroke['y'][0]
        else:
            # 如果当前笔画没有点，也跳过
            print(f"[analyze] 警告：笔画 {i} 没有数据点，跳过聚类判断。")
            continue # 不将其加入任何字符

        # 计算欧几里得距离
        distance = np.sqrt((curr_start_x - prev_end_x)**2 + (curr_start_y - prev_end_y)**2)

        # print(f"[Debug] Stroke {i-1} end -> Stroke {i} start distance: {distance:.2f}") # Debug 信息

        if distance < distance_threshold:
            # 距离足够近，认为是同一字的一部分
            current_character.append(curr_stroke)
        else:
            # 距离太远，认为是新字的开始
            characters.append(current_character)
            current_character = [curr_stroke]

    # 循环结束后，将最后一个正在构建的字符加入列表
    if current_character:
        characters.append(current_character)

    print(f"[analyze] {len(strokes)} 个笔画已聚类为 {len(characters)} 个字 (使用距离阈值 {distance_threshold})。")
    return characters

# TODO: 未来可以添加更复杂的聚类函数，例如：
# def cluster_strokes_ml(strokes, model):
#     """
#     使用机器学习模型进行笔画聚类。
#     model: 一个预训练的模型对象，能够根据笔画特征和上下文判断归属。
#     """
#     pass
