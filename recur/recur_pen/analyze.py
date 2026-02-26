# analyze.py
"""
电子笔轨迹数据分析模块。
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

def load_trajectory_data(filepath, skip_rows=0):
    """
    从文件加载电子笔轨迹数据。

    参数:
        filepath (str): 数据文件路径。
        skip_rows (int): 要跳过的文件开头行数。

    返回:
        dict or None: 成功时返回包含 'x', 'y', 'pressure' 的字典，失败时返回 None。
    """
    print(f"[analyze] 正在从 '{filepath}' 加载数据 (跳过前 {skip_rows} 行)...")
    try:
        data_array = np.loadtxt(filepath, skiprows=skip_rows)
        print(f"[analyze] 数据加载成功，共 {len(data_array)} 行。")
        data_dict = {
            'x': data_array[:, 0],
            'y': data_array[:, 1],
            'pressure': data_array[:, 2]
        }
        return data_dict
    except FileNotFoundError:
        print(f"[analyze] 错误：找不到文件 '{filepath}'。")
    except ValueError as e:
        print(f"[analyze] 错误：解析文件 '{filepath}' 时出错。请检查文件格式。详细信息: {e}")
    except Exception as e:
        print(f"[analyze] 加载文件 '{filepath}' 时发生未知错误: {e}")
    return None


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
    try:
        x_coords = np.asarray(data['x'])
        y_coords = np.asarray(data['y'])
        pressures = np.asarray(data['pressure'])

        if not (len(x_coords) == len(y_coords) == len(pressures)):
            raise ValueError("输入数据的 'x', 'y', 'pressure' length 不一致。")

        if len(x_coords) == 0:
            print("警告：输入数据为空。")
            return []

        strokes = []
        current_stroke_x = []
        current_stroke_y = []
        current_stroke_p = []

        for i in range(len(pressures)):
            if pressures[i] > 0:
                current_stroke_x.append(x_coords[i])
                current_stroke_y.append(y_coords[i])
                current_stroke_p.append(pressures[i])
            else:
                if current_stroke_x:
                    strokes.append({
                        'x': np.array(current_stroke_x),
                        'y': np.array(current_stroke_y),
                        'pressure': np.array(current_stroke_p)
                    })
                    current_stroke_x = []
                    current_stroke_y = []
                    current_stroke_p = []

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
        return []


"""计算单个笔画的中心点（所有点的均值）"""
def _calculate_stroke_center(stroke):
    x = stroke['x']
    y = stroke['y']
    if len(x) == 0 or len(y) == 0:
        return 0.0, 0.0
    return np.mean(x), np.mean(y)


"""一个字符(多个笔画)的中心点"""
def _calculate_character_center(character):
    all_x_points = []
    all_y_points = []
    for stroke in character:
        all_x_points.extend(stroke['x'])
        all_y_points.extend(stroke['y'])
    if not all_x_points or not all_y_points:
        return 0.0, 0.0
    return np.mean(all_x_points), np.mean(all_y_points)


# ========================
# ✅ 新增：自适应阈值计算
# ========================
def calculate_adaptive_threshold(strokes, k=2.0, min_threshold=300.0, max_threshold=2500.0):
    """
    基于笔画中心点的平均最近邻距离计算自适应聚类阈值。

    原理：阈值 = k * (所有笔画到其最近邻笔画中心的平均距离)
    优点：直接反映笔画空间密度，适应不同书写大小和风格。

    Parameters:
        strokes: 笔画列表List of stroke dicts with 'x', 'y'
        k: 缩放系数，建议初始值 1.8~2.5 （可调）
        min_threshold / max_threshold: 安全边界，防止极端值

    Returns:
        float: 计算出的自适应阈值
    """
    if len(strokes) < 2:
        # 只有一个或零个笔画，无法计算邻居，返回默认
        return 1000.0

    # 计算所有笔画中心
    centers = []
    for stroke in strokes:
        cx = np.mean(stroke['x'])
        cy = np.mean(stroke['y'])
        centers.append([cx, cy])
    centers = np.array(centers)  # shape: (N, 2)

    # 计算两两距离矩阵(所有中心点之间的欧氏距离矩阵)
    dist_matrix = squareform(pdist(centers, metric='euclidean'))

    # 将对角线设为无穷大，避免自己到自己的距离为0
    np.fill_diagonal(dist_matrix, np.inf)

    # 对每个笔画，找到离它最近的其他笔画的中心距离（已排除自身）
    nearest_distances = np.min(dist_matrix, axis=1)

    # 求这些最近邻距离的平均最近邻距离
    avg_nearest = np.mean(nearest_distances)

    # 计算自适应阈值
    adaptive = k * avg_nearest

    # 安全裁剪到 [min_threshold, max_threshold] 之间，防止极端值
    adaptive = np.clip(adaptive, min_threshold, max_threshold)

    return float(adaptive)


# ========================
# ✅ 修改：支持动态阈值
# ========================
def cluster_strokes_simple(strokes, threshold=None):
    """
    贪心聚类：将 strokes 聚合成 characters。将笔画贪心聚类成字符
    若 threshold=None, 则自动计算自适应阈值。
    """
    if threshold is None:
        threshold = calculate_adaptive_threshold(strokes, k=2.2)  # 可调k
        print(f"[analyze] 使用自适应阈值 (最近邻法): {threshold:.1f}")
    else:
        print(f"[analyze] 使用固定阈值: {threshold}")

    if not strokes:
        print("[analyze] 警告：输入笔画列表为空，无法进行聚类。")
        return []

    characters = []
    # 将第一个笔画作为第一个字符的开始，记录该字符的当前中心
    current_character = [strokes[0]]
    char_center_x, char_center_y = _calculate_stroke_center(strokes[0])

    # 遍历每个笔画
    for i in range(1, len(strokes)):
        # 计算当前笔画的中心
        curr_stroke = strokes[i]
        stroke_center_x, stroke_center_y = _calculate_stroke_center(curr_stroke)
        # 计算当前笔画的中心与当前字符中心的距离
        distance = np.sqrt((stroke_center_x - char_center_x)**2 + (stroke_center_y - char_center_y)**2)

        if distance < threshold:
            """
            若距离 < 阈值，
            则将该笔画加入当前字符，并更新字符中心
            （重新计算所有已加入笔画点的均值）
            """
            current_character.append(curr_stroke)
            # 更新当前字的中心
            all_x = np.concatenate([s['x'] for s in current_character])
            all_y = np.concatenate([s['y'] for s in current_character])
            char_center_x = np.mean(all_x)
            char_center_y = np.mean(all_y)
        else:
            """否则，将当前字符保存，并开始新字符，以当前笔画为新字符的起始。"""
            characters.append(current_character)
            current_character = [curr_stroke]
            char_center_x, char_center_y = stroke_center_x, stroke_center_y

    if current_character:   # 将最后一个字符加入列表
        characters.append(current_character)

    print(f"[analyze] {len(strokes)} 个笔画已聚类为 {len(characters)} 个字 (阈值={threshold:.1f})。")
    return characters


# ========================
# 后处理：尝试合并相邻的、笔画数较少的字符。
# ========================
def refine_characters(characters, max_strokes=2, merge_threshold=50.0):
    """
    返回: (new_characters, merged_pairs)
        merged_pairs: List[Tuple[int, int]]
            记录在原 characters 中被合并的字符索引对(原characters中索引 i 和 j 被合并)
        new_characters: 合并后的新字符列表
    参数：
        characters: 初步聚类得到的字符列表。
        max_strokes: 若字符的笔画数 ≤ 此值，则视为“短字符”，可能参与合并。
        merge_threshold: 允许合并的最大中心距离。
    """
    if len(characters) < 2:
        return characters, []

    short_char_indices = [
        i for i, char in enumerate(characters)
        if len(char) <= max_strokes
    ]

    if not short_char_indices:
        return characters, []

    # 计算中心
    centers = []
    for char in characters:
        all_x = np.concatenate([stroke['x'] for stroke in char])
        all_y = np.concatenate([stroke['y'] for stroke in char])
        centers.append((np.mean(all_x), np.mean(all_y)))

    merged = [False] * len(characters)
    new_characters = []
    merged_pairs = []  # ← 新增：记录合并的索引对

    # 遍历短字符索引
    for i in short_char_indices:
        if merged[i]: # 当前字符未被合并
            continue
        current_char = characters[i]
        cx1, cy1 = centers[i]
        merged_with_next = False

        # 下一个相邻字符
        for j in range(i + 1, len(characters)):
            if merged[j]:   # 字符j未被合并
                continue
            if j != i + 1:  # 只考虑紧邻i的j=i+1
                break
            cx2, cy2 = centers[j]
            dist = np.hypot(cx1 - cx2, cy1 - cy2)
            if dist < merge_threshold:  # 小于阈值则合并
                merged[i] = True
                merged[j] = True
                new_characters.append(characters[i] + characters[j])
                merged_pairs.append((i, j))  # ← 记录合并
                merged_with_next = True
                break

        if not merged_with_next:
            new_characters.append(current_char)

    # 添加未参与 refine 的字符(未被合并的字符直接保留,生成新的字符列表)
    for k in range(len(characters)):
        if not merged[k] and k not in short_char_indices:
            new_characters.append(characters[k])

    return new_characters, merged_pairs  # ← 返回合并信息