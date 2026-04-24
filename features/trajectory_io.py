"""
features/trajectory_io.py

共用子模块：轨迹文件读取与笔段切割。
供 sym_feature_extractor / stroke_utils 及其他游戏模块调用。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def load_trajectory_data(filepath: str, skip_rows: int = 0) -> Optional[Dict[str, np.ndarray]]:
    """
    从文本文件读取 x/y/pressure 三列数据。

    返回:
        dict: 包含 'x', 'y', 'pressure' 的 np.ndarray，失败时返回 None。
    """
    print(f"[trajectory_io] 正在从 '{filepath}' 加载数据 (跳过前 {skip_rows} 行)...")
    try:
        data_array = np.loadtxt(filepath, skiprows=skip_rows)
        print(f"[trajectory_io] 数据加载成功，共 {len(data_array)} 行。")
        return {
            "x": data_array[:, 0],
            "y": data_array[:, 1],
            "pressure": data_array[:, 2],
        }
    except FileNotFoundError:
        print(f"[trajectory_io] 错误：找不到文件 '{filepath}'。")
    except ValueError as e:
        print(f"[trajectory_io] 错误：解析文件时出错，请检查格式。详细信息: {e}")
    except Exception as e:
        print(f"[trajectory_io] 加载文件时发生未知错误: {e}")
    return None


def split_into_strokes(data: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """
    按 pressure==0 切割笔段，返回笔段 dict 列表。

    每段 dict 含 'x', 'y', 'pressure'（均为 np.ndarray）。
    """
    x_coords = np.asarray(data["x"])
    y_coords = np.asarray(data["y"])
    pressures = np.asarray(data["pressure"])

    if not (len(x_coords) == len(y_coords) == len(pressures)):
        raise ValueError("'x', 'y', 'pressure' 长度不一致。")
    if len(x_coords) == 0:
        return []

    strokes: List[Dict[str, np.ndarray]] = []
    cur_x: List[float] = []
    cur_y: List[float] = []
    cur_p: List[float] = []

    for i in range(len(pressures)):
        if pressures[i] > 0:
            cur_x.append(x_coords[i])
            cur_y.append(y_coords[i])
            cur_p.append(pressures[i])
        else:
            if cur_x:
                strokes.append({
                    "x": np.array(cur_x),
                    "y": np.array(cur_y),
                    "pressure": np.array(cur_p),
                })
                cur_x, cur_y, cur_p = [], [], []

    if cur_x:
        strokes.append({
            "x": np.array(cur_x),
            "y": np.array(cur_y),
            "pressure": np.array(cur_p),
        })

    print(f"[trajectory_io] 轨迹已分割为 {len(strokes)} 个笔画。")
    return strokes


def load_strokes_with_pressure(
    txt_path: str, skip_rows: int = 3
) -> List[Dict[str, np.ndarray]]:
    """
    读 txt → 笔段 dict 列表（每段含 'x', 'y', 'pressure' np.ndarray）。
    坐标是原始设备坐标（未映射到画布）。
    """
    data = load_trajectory_data(txt_path, skip_rows=skip_rows)
    if data is None:
        raise RuntimeError(f"[trajectory_io] 无法读取轨迹文件: {txt_path}")
    return split_into_strokes(data)