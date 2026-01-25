# analyze.py
"""
电子笔轨迹数据分析模块。
"""

import numpy as np

def split_into_strokes_simple(data):
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


def _calculate_stroke_center(stroke):
    x = stroke['x']
    y = stroke['y']
    if len(x) == 0 or len(y) == 0:
        return 0.0, 0.0
    return np.mean(x), np.mean(y)


def _calculate_character_center(character):
    all_x_points = []
    all_y_points = []
    for stroke in character:
        all_x_points.extend(stroke['x'])
        all_y_points.extend(stroke['y'])
    if not all_x_points or not all_y_points:
        return 0.0, 0.0
    return np.mean(all_x_points), np.mean(all_y_points)


def cluster_strokes_simple(strokes, distance_threshold):
    if not strokes:
        print("[analyze] 警告：输入笔画列表为空，无法进行聚类。")
        return []

    characters = []
    current_character = [strokes[0]]
    char_center_x, char_center_y = _calculate_stroke_center(strokes[0])

    for i in range(1, len(strokes)):
        curr_stroke = strokes[i]
        stroke_center_x, stroke_center_y = _calculate_stroke_center(curr_stroke)

        distance = np.sqrt((stroke_center_x - char_center_x)**2 + (stroke_center_y - char_center_y)**2)

        if distance < distance_threshold:
            current_character.append(curr_stroke)
            all_x_points = []
            all_y_points = []
            for s in current_character:
                all_x_points.extend(s['x'])
                all_y_points.extend(s['y'])
            char_center_x = np.mean(all_x_points)
            char_center_y = np.mean(all_y_points)
        else:
            characters.append(current_character)
            current_character = [curr_stroke]
            char_center_x, char_center_y = stroke_center_x, stroke_center_y

    if current_character:
        characters.append(current_character)

    print(f"[analyze] {len(strokes)} 个笔画已聚类为 {len(characters)} 个字 (使用中心点距离阈值 {distance_threshold})。")
    return characters


def refine_characters(characters, max_strokes=2, merge_threshold=30.0):
    if len(characters) < 2:
        return characters

    refined_chars = [characters[0]]
    i = 1
    while i < len(characters):
        current_char = characters[i]
        prev_char = refined_chars[-1]
        num_strokes_current = len(current_char)

        if num_strokes_current > max_strokes:
            refined_chars.append(current_char)
            i += 1
            continue

        center_curr = _calculate_character_center(current_char)
        center_prev = _calculate_character_center(prev_char)
        distance = np.sqrt((center_curr[0] - center_prev[0])**2 + (center_curr[1] - center_prev[1])**2)

        if distance < merge_threshold:
            print(f"[analyze] 后处理：将笔画数为 {num_strokes_current} 的字（中心 {center_curr}）"
                  f" 与前一个字（中心 {center_prev}，距离 {distance:.2f}）合并。")
            prev_char.extend(current_char)
            i += 1
        else:
            refined_chars.append(current_char)
            i += 1

    print(f"[analyze] 后处理完成：{len(characters)} 个字优化为 {len(refined_chars)} 个字。")
    return refined_chars