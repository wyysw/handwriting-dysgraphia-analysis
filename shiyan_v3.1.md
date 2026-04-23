# 书写障碍分类实验——完整推进指导文档

> **版本**： v3.1
> 阶段三已完成，已设计后续

> **文档定位**：本文档是跨对话协作的主参考文档，整合实验设计与流程指导。
> 在每次新对话开始时将本文档连同相关代码一起提供给 AI 助手，即可无缝延续工作。

---


## 1. 项目总览

### 1.1 研究目标

通过分析儿童在三款手写游戏中的绘制轨迹，判断该绘制是否由书写障碍（Dysgraphia）儿童产生（**二分类**问题）。强调**方法论贡献**：同时兼顾"游戏结果"（功能轴 F）和"游戏过程"（控制轴 C），设计跨游戏通用的轻量级分类器和特征框架，从特征抽取到落地分类（这个分类器不是只能训练单一游戏，而是对这三个游戏都行之有效；分类器只是为了表征我的工作从一个纯工程项的抽取特征变成了有落地的分类）。


### 1.2 三款游戏

| 游戏 | 任务描述 | 参考图像文件 |
|---|---|---|
| 对称（Symmetry） | 依据对称轴，将左侧/上半区蓝色图形镜像补画到右侧/下半区 | `sym_blue_mask.png`, `sym_helper_mask_completed.png` |
| 方形迷宫（Square Maze） | 在直线迷宫中从起点走到终点 | `maze_mask.png` |
| 圆形迷宫（Circle Maze） | 在环形路径迷宫中从起点走到终点 | `circle_mask.png` |

对于对称游戏，在绘制者看来原始蓝色图形位于图片左侧，对称轴为纵向竖直对称轴；但在程序看来，原始图形位于图片上半区，对称轴为横向水平对称轴。


### 1.3 数据规格

- **轨迹格式**：每行 `x y pressure`，前 3 行为文件头（`eink` / `1` / `0,0`）需跳过
- **压力**：> 0 为落笔，= 0 为抬笔
- **无时间戳**（速度、耗时类特征全部不可用）
- **样本量与标注**：参考labels.csv。
- **从txt到png的映射方法**：参考Pen模块

| 文件 | 格式说明 |
|---|---|
| **所有样本轨迹**（建议每游戏 ≥ 10 份） | `data/samples/{sym,maze,circle}/{sample_id}.txt` |
| **对应用户绘制图片** | `data/samples/{sym,maze,circle}/{sample_id}.png`（仅用于 bbox 参照） |

**`labels.csv` 格式**：
```csv
sample_id,game,label
sym_001,sym,0
sym_002,sym,1
maze_001,maze,0
circle_001,circle,1
...
```
其中 `label=0` 表示正常，`label=1` 表示书写障碍。


### 1.4 核心方法论约束

- 样本粒度为 **per-game**（单次绘制 = 一条样本），不做跨游戏的 subject 级拼接
- 可用信号：**空间坐标 + 压力值 + 笔段结构**（由 pressure 跳变推断）
- 方法论贡献优先于单个特征实现的精度

---

## 2. 预处理部分说明

### 2.1 游戏面板预处理（已完成）

**作用**：从三张原始测评图提取标准参考结构，生成二值掩码。

| 脚本 | 输入 | 关键输出 |
|---|---|---|
| `shape/final_shape_sym.py` | `data/35duichen.png` | `sym_blue_mask.png`（蓝色半边图形）, `sym_helper_mask_completed.png`（网格+对称轴+外框） |
| `shape/final_shape_migong.py` | `data/34migong.png` | `maze_mask.png`（迷宫墙壁线条） |
| `shape/final_shape_circle.py` | `data/36circle.png` | `circle_mask.png`（圆形迷宫墙壁线条） |
| `shape.py` | — | 统一调度以上三个脚本 |

所有掩码均为 1201×1601 单通道 PNG，与原图像素级对齐。


### 2.2 Pen模块（样本数据使用）

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


## 3. 实验推进路线（按阶段）

```
阶段 1 （已完成）  对称游戏特征提取器（重构/扩展 sym_analyze3.py）
   ↓
阶段 2（已完成）   迷宫游戏特征提取器（方形）
   ↓
阶段 3（已完成）   迷宫游戏特征提取器（圆形，与方形迷宫共用基类）
   ↓
阶段 4（待完成）   特征汇总处理
   ↓
阶段 5（待完成）   分类器设计与实现
   ↓
阶段 6（待完成）   实验评估
```

---

## 4. 实验详细设计

### 4.1 阶段1：对称游戏特征提取器

```
python features/sym_feature_extractor.py --txt data/samples/sym/{id}.txt --png data/samples/sym/{id}.png --blue output_sym/shape_sym/sym_blue_mask.png --helper output_sym/shape_sym/sym_helper_mask_completed.png  --out output_sym/extract/{id}.json --vis output_sym/extract/vis_{id}.png
```

#### 4.1.1 程序：`features/sym_feature_extractor.py`

**目的**：对一个对称游戏样本，提取 7 个标准化特征（F1–F4, C1–C3），输出为 JSON。这是对称游戏的**核心特征计算程序**。

**设计说明**：
本程序不沿用 `sym_analyze3.py` 中的打分逻辑（其将成为加权分类器的基础），而是提取**原始指标值**（0–1 的比例或连续量），统一由阶段 3 的管道进行归一化。

从 sym_analyze3.py 直接导入以下通用函数：

| 导入函数 | 用途 |
|---|---|
| `read_binary_mask`, `read_user_mask_png`, `pad_to_shape`, `bbox_from_mask` | 图像/mask IO |
| `detect_helper_geometry` | 定位 axis_y、内网格线、step_x/y |
| `map_trajectory_strokes_using_reference_bbox` | **轨迹坐标映射**（按 png bbox 对齐）|
| `render_trajectory_using_reference_bbox` | 将映射后笔段渲染为二值 mask |
| `reflect_mask_across_horizontal_axis`, `reflect_strokes_across_horizontal_axis` | 关于 axis_y 翻转 |
| `distance_transform_to_mask` | 距离变换 |
| `extract_keypoints_from_target` | 网格交点关键点 |
| `_extract_target_segments_from_hough` | 霍夫线段（用于 C1）|

**不复用**的：`analyze_line_control`、`run_stage1`、`tolerant_f1`、`keypoint_coverage_score`

---


#### 4.1.2 特征定义

**F1：翻转容差 F1（0-1，↑越好）**：使用容差膨胀后的 F1 分数，衡量翻转后的用户笔迹与目标图形的重叠程度。

```python
A = dilate(reflected_user_mask, r)  # 用户笔迹膨胀
B = dilate(target_mask, r)          # 目标图形膨胀
precision = mean(B[reflected_user_mask>0])   # 用户点里有多少落在"target 容差域"
recall    = mean(A[target_mask>0])           # target 点里有多少被"用户容差域"覆盖
F1 = 2PR / (P + R + 1e-9)
```

- `dilation_radius = 5` 像素
- **不做 normalize_mask 归一缩放**，保留"大小差异"信息
- 按**画布原尺寸**计算 IoU 式容差指标



**F2：关键点命中率（0-1，↑越好）**：网格交点关键点被用户笔迹覆盖的比例。

```python
keypoints = extract_keypoints_from_target(target_mask, helper, include_midpoints=False)
dist = distance_transform_to_mask(reflected_user_mask)
hit  = Σ 1[dist[y,x] <= hit_radius]
F2 = hit / len(keypoints)
```

- `hit_radius = 6` 像素

**F3：无效书写加权距离（≥0，↓越好）**：用户笔迹偏离有效区域的加权距离，包含越轴惩罚。

```python
valid_zone_mask = dilate(target_mask, tol_valid)          # 允许误差的区域
dist_outside    = distanceTransform(1 - valid_zone_mask)  # 每个非 valid 像素 → valid 的最近距离
# 主要违规区域（下半画布的越界部分）
illegal_main = reflected_user_mask & (y > axis_y) & ~valid_zone_mask
main_contrib = Σ dist_outside[illegal_main]
# 越轴惩罚（画到上半画布的部分）
cross_axis_pixels = reflected_user_mask & (y <= axis_y)
cross_contrib = cross_penalty * count(cross_axis_pixels)
F3 = (main_contrib + cross_contrib) / Σ target_mask
```

- `tol_valid = 9` 像素
- `cross_penalty = 10`（越轴像素的惩罚系数）
- 对称轴为**水平线** axis_y
- 用户应在下半画布（y > axis_y）书写
- 越界到上半画布的像素会被额外惩罚


**F4：路径偏离比（0-1，↓越好）**：用户笔迹中偏离有效区域的像素比例。

```python
F4 = |reflected_user_mask & ~valid_zone_mask| / |reflected_user_mask|
```

**C1：抖动比例（0-1，↓越好）**：笔段点偏离霍夫线段的比例（点级统计）。

```python
hough_segments = _extract_target_segments_from_hough(target_mask)
for 每条用户笔段中的每个点 p (in reflected_strokes):
    找最近的 hough 线段 (a,b) 且投影距离 <= channel_half_width
    （否则丢弃此点）
    residual = 垂直距离
total_pts = 有效投影点数
bad_pts   = Σ 1[residual > jitter_tol]
C1 = bad_pts / total_pts
```

- `jitter_tol = 3.0` 像素
- `channel_half_width = 20` 像素
- **加权方式**：点级统计（所有霍夫线段上的投影点一起算；不按笔段长度加权）


**C2：短笔段比例（0-1，↓越好）**：短笔段总长度占所有笔段总长度的比例。

```python
stroke_lens = [arc_length(s) for s in strokes_mapped]
thr = diag(canvas) * 0.02  # 阈值：画布对角线 * 0.02
short_total = Σ stroke_lens[i] for i where stroke_lens[i] < thr
C2 = short_total / Σ stroke_lens
```

- `C2_thr_ratio = 0.02`（固定比例）


**C3：压力变异系数（≥0，↓越好）**：笔段压力值的变异系数（标准差/均值）。

```python
all_p = concatenate([
    stroke['pressure'][trim:-trim] 
    for stroke in strokes_with_pressure 
    if len > 2*trim
])
all_p = all_p[all_p > 0]
C3 = std(all_p) / (mean(all_p) + 1e-9)
```

- `C3_trim_ends = 3`（裁剪笔段两端的点数）
- 在 sym_feature_extractor 里写 `load_trajectory_strokes_with_pressure()`保留 pressure 维度
- 基于 `analyze.split_into_strokes_simple`，返回含 pressure 的 dict 列表
- xy 用于空间特征、pressure 用于 C3，两者通过"同一笔段 index"对应


| 轴 | ID | 名称     | 方向      | 对称游戏                  | 方形迷宫     | 圆形迷宫     |
| - | -- | ------ | ------- | --------------------- | -------- | -------- |
| F | F1 | 完成度    | ↑（越大越好） | 翻转容差 F1（IoU-like）     | 通道覆盖率    | 环形通道覆盖率  |
| F | F2 | 关键结构对齐 | ↑       | 网格交点命中率               | 骨架采样点命中率 | 弧度采样点命中率 |
| F | F3 | 无效书写   | ↓（越小越好） | 越界像素加权距离/标准面积         | 同左       | 同左       |
| F | F4 | 路径偏离比  | ↓       | 期望图形外笔迹比              | 通道外笔迹比   | 通道外笔迹比   |
| C | C1 | 抖动比例   | ↓       | 笔迹到参考骨架的残差越界比例（三游戏统一） | ←        | ←        |
| C | C2 | 短笔段比例  | ↓       | 短段总长/所有笔段总长（三游戏统一）    | ←        | ←        |
| C | C3 | 压力变异系数 | ↓       | σ(p)/μ(p)（三游戏统一）      | ←        | ←        |

#### 4.1.3 主入口

```
extract_sym_features(txt_path, png_path, blue_mask_path, helper_mask_path, out_json_path)
  ├─ 读 target_mask（蓝图）、helper_mask
  ├─ detect_helper_geometry → axis_y, inner_v/h_lines, step_x/y
  ├─ 轨迹→笔段（调用 analyze.load_trajectory_data / split_into_strokes_simple）
  │    并保留 pressure 维度（用于 C3 特征）
  ├─ map_trajectory_strokes_using_reference_bbox → 画布坐标下的笔段
  ├─ render → user_mask_on_canvas
  ├─ 翻转：
  │    reflected_user_mask = reflect_mask_across_horizontal_axis(user_mask, axis_y)
  │    reflected_strokes  = reflect_strokes_across_horizontal_axis(strokes_mapped, axis_y, canvas_shape)
  ├─ 构造 valid_zone：valid_zone_mask = dilate(target_mask, radius=tol_F)
  ├─ 计算 F1, F2, F3, F4, C1, C2, C3
  ├─ 组装 JSON（含 meta 诊断字段）
  └─ 返回 dict；若指定 out_json_path 则写盘
```

#### 4.1.4 输入输出

**输入**：
- `txt_path`：用户轨迹文件（`x y pressure`，跳过前 3 行）
- `png_path`：用户绘制图片（仅用于 bbox 坐标参照）
- `blue_mask_path`：`sym_blue_mask.png`（对称标准答案）
- `helper_mask_path`：`sym_helper_mask_completed.png`（含对称轴和网格）
- `out_json_path`（可选）：若提供，自动保存 JSON

**输出**（JSON / dict）：

```json
{
  "sample_id": "s3",
  "game": "sym",
  "F1": 0.xxxx, "F2": 0.xxxx, "F3": 0.xxxx, "F4": 0.xxxx,
  "C1": 0.xxxx, "C2": 0.xxxx, "C3": 0.xxxx,
  "meta": {
    "num_strokes": 3,
    "total_points": 13299,
    "axis_y": 800,
    "step_x": 96, "step_y": 96,
    "num_keypoints": 12,
    "keypoints_hit": 10,
    "cross_axis_pixels": 52,
    "canvas_hw": [1601, 1201],
    "params": {"dilation_F1": 5, "tol_valid": 9, "hit_radius": 6, "jitter_tol": 3.0, "C2_thr_ratio": 0.02, "C3_trim_ends": 3}
  }
}
```

**预期结果**（对正常样本）：
- F1 > 0.6，F2 > 0.7，F3 < 0.05，F4 < 0.25
- C1 < 0.2，C2 < 0.15，C3 < 0.5
- 对障碍样本应出现至少 1–2 个特征的明显偏差


**测试方法**：
1. 对已有样本运行，检查 JSON 是否正常输出，各特征值是否在合理范围
2. 生成一张可视化叠加图（用户翻转笔迹 + 蓝图 + 关键点命中情况），目视确认坐标对齐无误

---


### 4.2 阶段2：方形迷宫游戏特征提取器

```
python features/maze_feature_extractor.py --txt data/samples/maze/{id}.txt --png data/samples/maze/{id}.png --mask output_maze/shape_maze/maze_mask.png --out output_maze/extract/{id}.json --vis_dir output_maze/extract/vis_{id} --sample_id {id}
```

迷宫游戏的 F1–F4 依赖通道几何，与对称游戏算法不同；
C1–C3 的**定义与对称游戏保持一致**，应提炼为公共库复用。
样本txt文件表示的轨迹的坐标对齐与阶段1完全一致。

```
maze_mask.png
  │
  ▼  ① 通道几何构造（与之前相同）
channel_mask, canvas_mask, channel_dist
  │
  ▼  ② 【新增】solve_maze_path 子模块 —— 提取最快通关路径
entry_xy, exit_xy, 
solution_polyline,          # 沿弧长有序的点列
solution_skeleton_mask,     # 1px 宽的解路径骨架
solution_channel_mask       # 解路径周边的"正确通道"(给 F1 用)
  │
  ├──► F1 = user ∩ solution_channel / solution_channel   （分母是解路径通道，不是全通道）
  ├──► F2 = 沿 solution_polyline 等弧长采样的命中率
  ├──► F3 = 用 channel_mask(全通道) 做 valid_zone，加权距离
  └──► F4 = user 落在 channel_mask(全通道) 外的比例
```


#### 4.2.1 通道几何模块 ：`features/maze_geometry.py`

仅依赖 maze_mask.png，与轨迹无关。

```
python features/maze_geometry.py --mask output_maze/shape_maze/maze_mask.png --vis output_maze/extract/_vis_geometry.png
```

**步骤 1 —— 外框填充得到画布内部区域 `canvas_mask`**  
`maze_mask.png` 是一张前景=墙壁线条的二值图。迷宫最外层一圈墙一定是封闭的矩形框。用 `cv2.floodFill` 从画布四角以外的"背景像素"出发，把外部填满；取反即可得到"外框内部区域"`canvas_mask`。


**步骤 2 —— 通道掩码 `channel_mask`**  
```
channel_mask = canvas_mask AND NOT dilate(wall_mask, r_wall)
```
`r_wall` 约 2–3 像素，用于确保墙壁两侧通道**干净分离**（不因墙的细缝连通）。


**步骤 3 —— 通道骨架 `channel_skeleton`**  
`skimage.morphology.skeletonize(channel_mask > 0)`。对方形迷宫，骨架基本是一条由直线段组成的折线；但若迷宫含死胡同/分叉，骨架会有分支。这里**不做分支剪枝**，保留完整骨架作为参考结构。

**步骤 4 —— 通道距离变换 `channel_dist`**  
`channel_dist = distanceTransform(1 - channel_mask, DIST_L2)` —— 每个非通道像素到最近通道的距离，用于 F3 的越界加权。

**下面是通道几何解路径模块 solve_maze_path(channel_mask, wall_mask)**

**步骤 6：自动检测入口 / 出口**

```
canvas_bbox = bbox_from_mask(wall_mask)           # 外框包围矩形
frame_ring = 四条边上的像素集合 (宽 1-2 px)
openings = frame_ring ∩ channel_mask               # 框上"可走"的像素
connected_components(openings) → 应得到 2 个连通域
  → 每个连通域取其质心作为入口/出口锚点
```

对maze_mask.png，入口在右上、出口在左下——两个连通域位置明确，直接返回 2 个 `(x, y)` 锚点。若自动检测不到恰好 2 个，fallback 到"用户在函数参数里手动指定 entry_xy、exit_xy"。

实际上，对于这个方形迷宫，入口、出口区域可以直接视为：从外边框的内顶点出发，向边框内方向延伸的105像素*105像素大小的矩形。具体实现时可直接使用，简化实现。


**步骤 2：基于骨架建图，BFS/Dijkstra 求最短路径**

```
skeleton = skimage.morphology.skeletonize(channel_mask)
entry_anchor = 骨架上距离入口锚点最近的像素
exit_anchor  = 同上
构造图 G：
  节点 = 骨架像素
  边   = 8 邻接，权重 = 欧氏距离 (直=1, 对角=√2)
path_pixels = shortest_path(G, entry_anchor, exit_anchor)
             （用 scipy.sparse.csgraph.dijkstra 或手写 BFS 都行）
```

骨架像素约 10³ 量级，图算法瞬间完成。对这张迷宫我预估解路径长度在 3000–5000 像素。


**步骤 3：生成三种输出表示**

```
solution_polyline       : path_pixels 去抖化后的有序 (x,y) 列表（可选 Douglas–Peucker 简化）
solution_skeleton_mask  : path_pixels 渲染成 1px mask，用于可视化调试
solution_channel_mask   : dilate(solution_skeleton_mask, r = 估算的通道半宽)
                          用于 F1 的分母
```

`r` ：通道宽度在整个迷宫里基本一致，固定为28像素。


#### 4.2.2 特征

**F1**：通道覆盖率(解路径)（↑）
```
F1 = |user_mask ∩ solution_channel_mask| / |solution_channel_mask|
```
解读："用户笔迹覆盖了解路径多大比例"。走错岔路不会提升 F1（因为岔道不在 `solution_channel_mask` 里），走得远但沿着解路径会提升 F1。


**F2**：解路径采样点命中率（↑）
```
沿 solution_polyline 按弧长每 sample_step 像素取一个点 → sample_pts (K 个)
dist_user = distanceTransform(~user_mask)
hit = Σ 1[dist_user[p] <= hit_radius]
F2 = hit / K
```
沿弧长采样一次实现就能迁移到圆形迷宫（圆形时换成按弧长等距采样，本质是按弧度等距）。

参数建议：`sample_step = 40 px`（对 1201×1601 画布约 30 个采样点），`hit_radius = 12 px`（略宽于笔迹线宽，允许轻微偏离）。实际跑出来再调。


**F3**：无效书写加权距离（↓）

F3 里使用 `channel_dist`（违规像素到最近通道的距离）自然体现了"错得多远就罚多重"——穿墙一格和画到纸外的惩罚力度自动拉开。不额外加硬惩罚系数，保持实现简单。

**F4**：通道外笔迹比（↓）

使用 `channel_dist`（违规像素到最近通道的距离），落在 channel_mask(全通道) 外的比例

`F4 = |user_mask ∩ ¬channel_mask| / |user_mask|`

**C1**：运动控制能力

除转角处以外用户的笔迹理想情况下都是**横向或纵向直线段**，Hough 能很稳定地把这些"用户应该画成直线"的部分识别出来。
**实现：把 `sym_feature_extractor.compute_C1_jitter_ratio` 直接复用，只换 `hough_segments` 的来源。使用手写笔迹的 Hough 线段。**

```
user_mask = render_trajectory_using_reference_bbox(txt, png, canvas_hw, line_thickness=3)
user_hough_segments = extract_segments_from_hough(user_mask)   # 复用 sym_analyze3._extract_target_segments_from_hough
C1, bad, total = compute_C1_jitter_ratio(
    mapped_strokes, user_hough_segments,
    jitter_tol=3.0, channel_half_width=channel_half_width,
)
```

`_extract_target_segments_from_hough` 在 sym_core 里现成，它本身已经包含了按角度分类（h/v/diag_pos/diag_neg）→ 聚合共线片段 → 去除过短段的完整流程。**唯一可能需要调整的是 HoughLinesP 的阈值**：

| 参数 | sym 目标图形 | 用户笔迹(方形迷宫) | 理由 |
|---|---|---|---|
| `threshold` | 25 | 20–25 | 用户笔迹有抖动，投票数略低 |
| `minLineLength` | 25 | 40–60 | 方形迷宫的直线段较长，排除短过渡段 |
| `maxLineGap` | 6 | 10 | 用户笔迹有小间隙 |

**建议**：先直接复用 sym 的默认参数跑一次 l3/l4，观察 Hough 输出是否合理，再决定是否需要为"用户笔迹"单独开一组阈值。`extract_segments_from_hough` 可以接一个可选的 `hough_params` 字典，方便调参。

**边界情况 —— Hough 找不到足够线段时**：
- 正常样本：应有几十条合并后的线段；
- 极端抖动的障碍样本：可能只有少数几条或零条。

这种情况下"投影命中数 total"会很小，C1 = bad/total 不稳定。我建议在 meta 里额外记录 `projected_fraction = total / num_user_points`，作为"线段一致性程度"的辅助诊断。**核心公式与 sym 完全一致（`C1 = bad / projected`）**，以满足"三游戏统一"的方法论约束。

**C2**：与对称游戏相同。
尽管迷宫样本的笔画数可能只有 1–2 个，使得C2恒为0、在迷宫上几乎丧失判别力，但此处接受 C2 在迷宫上失效，让阶段 3 的归一化 + 阶段 4 的分类器自己学习其权重。


**C3**：压力变异系数。
对应到迷宫数据，`analyze.py` 的 `split_into_strokes_simple` 会丢弃 pressure=0 的点，笔段内部的压力全 > 0，可以直接套阶段1的公式。




#### 4.2.3 程序（公共库）：`features/stroke_utils.py`

把 C1/C2/C3 + 轨迹映射 + Hough 等通用逻辑从sym_feature_extractor抽出来。

| 公共函数 | 从哪儿搬 | 用途 |
|---|---|---|
| `load_strokes_with_pressure(txt)` | `sym_feature_extractor._load_strokes_with_pressure` | 读 txt，返回含 pressure 的笔段 dict 列表 |
| `map_strokes_to_canvas(strokes, png_path, canvas_hw)` | `sym_analyze3.map_trajectory_strokes_using_reference_bbox` | bbox 映射 |
| `render_strokes_to_mask(strokes_mapped, canvas_hw, thickness)` | `sym_analyze3.render_trajectory_using_reference_bbox` 的内部渲染部分 | 笔段渲染 |
| `extract_segments_from_hough(mask, **hough_params)` | `sym_analyze3._extract_target_segments_from_hough` | Hough→共线合并（对称和迷宫复用） |
| `compute_C1_jitter_ratio(strokes, segments, …)` | `sym_feature_extractor.compute_C1_jitter_ratio` | C1 |
| `compute_C2_short_stroke_ratio(strokes, canvas_hw, …)` | 同上 | C2 |
| `compute_C3_pressure_cv(strokes_wp, …)` | 同上 | C3 |



#### 4.2.4 迷宫特征提取主程序：`features/maze_feature_extractor.py`

只设计实现迷宫特有的 F1–F4。C1/C2/C3 调用公共库`stroke_utils.py`。迷宫的 C1 参考骨架 = `channel_skeleton`。


**主入口**：
```
extract_maze_features(
    txt_path, png_path,
    maze_mask_path,
    game_type='maze',    # 为圆形迷宫预留
    out_json_path=None, out_vis_dir=None,
    *, entry_xy=None, exit_xy=None,       # 自动检测失败时手动指定
    sample_step=40, hit_radius=12.0,
    jitter_tol=3.0, C1_hough_params=None,
    C2_threshold_ratio=0.02, C3_trim_ends=3,
) -> dict
```

方形迷宫和圆形迷宫共用同一个提取器，通过 `game_type` 参数区分。关键预处理差异：

| 步骤 | 方形迷宫 | 圆形迷宫 |
|---|---|---|
| 通道掩码来源 | `maze_mask.png` | `circle_mask.png` |
| 正确路径提取 | 用形态学腐蚀获取"通道内部"（迷宫墙是线条，腐蚀后中间空白即通道） | 同左，但需识别环形结构 |
| F2 关键点定义 | 路径骨架上的岔路口/转角 | 路径骨架上每隔固定弧度的采样点 |
| 坐标对齐 | 同对称游戏 | 同左 |

**通道内部掩码生成**（关键步骤，需在提取器中实现）：
```
wall_mask = maze_mask.png（前景=墙壁线条）
canvas_mask = 外框填充（用 floodFill 填充外框内部区域）
channel_mask = canvas_mask AND NOT wall_mask
    → 前景 = 可行走通道区域
channel_skeleton = skimage.morphology.skeletonize(channel_mask)
    → 通道中线骨架
```

**输入**：


**输出**
JSON 输出中 `meta` 额外记录：
- `entry_xy`, `exit_xy`, `solution_path_length`（像素数）
- `channel_area`, `solution_channel_area`
- `num_skeleton_sample_pts`, `keypoints_hit`
- `num_user_hough_segments`, `C1_projected_fraction`
- `F3_detail`（与 sym 对齐，但 `n_cross_axis_pixels=0`、`cross_penalty_coef=0`）


**可视化：**

1. `channel_geometry.png` — 全通道+骨架+检测到的入口出口+解路径（红色高亮）。肉眼确认解路径提取正确。
2. `feature_overlay.png` — 全通道(灰)+解路径通道(浅绿)+用户笔迹(蓝)+采样点(绿命中/红未中)。
3. `C1_hough_overlay.png` — 用户笔迹(灰)+识别到的 Hough 线段(红)+被判为抖动的点(黄)。用来目视验证 C1 是否合理。


---





### 4.3 阶段3：圆形迷宫游戏特征提取器

```
python features/maze_feature_extractor.py --txt data/samples/circle/{id}.txt --png data/samples/circle/{id}.png --mask output_circle/shape_circle/circle_mask.png --game circle --out output_circle/extract/{id}.json --vis_dir output_circle/extract/vis_{id} --sample_id {id}
```

---

#### 4.3.1 设计总原则

两迷宫复用**同一套提取器与特征定义**，通过 `game_type='circle'` 区分几何分支。所有差异集中在 `maze_geometry.py` 的几何构造步骤；**特征层 F1–F4 代码零改动，C2/C3 代码零改动**。C1 是唯一存在方案分歧的地方（见 4.3.5）。

与方形迷宫相比，圆形迷宫的核心差异如下表：

| 方面 | 方形迷宫 | 圆形迷宫 |
|---|---|---|
| 画布边界 | 矩形外框（封闭） | 圆形外环（有两个缺口） |
| `canvas_mask` 构造 | `floodFill` 从四角灌入 | 最小二乘拟合外环圆 → 填充圆盘 |
| 入口/出口位置 | 右上 / 左下角附近 | 外环上的两个缺口 |
| 额外清洗步骤 | 无 | 连通域过滤（去除中心装饰小圆的孤立 blob） |
| `solution_polyline` | 可 DP 简化 | **不做 DP 简化**（避免圆弧被折线粗化） |
| 通道半宽（实测） | 28 px | 29 px（取 28 复用同一默认值） |

---

#### 4.3.2 圆形迷宫几何概况（实测）


| 参数 | 值 | 备注 |
|---|---|---|
| 近似圆心 `(cx, cy)` | (598, 812) | 最小二乘拟合外环所得 |
| 外环墙壁半径 `r_out` | ≈ 413 px | 径向直方图峰值 r≈405–423 |
| 同心环数量 | 5 条通道环 + 外边界 + 中心装饰小圆 | 共 7 个环状结构 |
| 环中心间距 | ≈ 63 px（等间距）| 墙壁厚约 4 px，净通道宽 ≈ 59 px |
| 通道净宽 / 半宽 | ≈ 59 px / 29 px | 与方形迷宫 28 px 几乎一致 |
| 入口缺口角度 | θ ∈ [−150°, −120°] | 左上方，math 角（+x 轴为 0°）|
| 出口缺口角度 | θ ∈ [+30°, +55°] | 右下方，同上 |
| 中心装饰小圆半径 | ≈ 36–45 px | 无通行意义，需过滤 |

**拓扑特殊性**：同心环在骨架图中形成 cycle，但经验证，此迷宫有且只有一条最短路径（从左上入口到右下出口），Dijkstra 直接得到唯一解，无需额外处理。

---

#### 4.3.3 程序（扩展）：`features/maze_geometry.py` 圆形分支

`build_maze_geometry` 增加 `game_type` 参数，`game_type='circle'` 时调用新增的 `_build_circle_geometry`。`MazeGeometry` 数据类新增三个可选字段记录圆形专属元信息。

```
wall_mask (来自 circle_mask.png)
  │
  ▼  ① 最小二乘拟合外环圆 → (cx, cy, r_out)
     （若显式提供 circle_center / outer_ring_radius 则跳过拟合直接使用）
  │
  ▼  ② 填充圆盘 → canvas_mask
     canvas_mask = disk(center=(cx,cy), radius=r_out+2)
  │
  ▼  ③ channel_mask_raw（与方形公式完全一致）
     channel_mask_raw = canvas_mask AND NOT dilate(wall_mask, r_wall)
  │
  ▼  ④ 【圆形特有】连通域过滤 → channel_mask
     保留最大连通域，去除中心装饰小圆围成的孤立 blob
  │
  ▼  ⑤ 骨架 + 距离变换（与方形完全一致）
     channel_skeleton = skeletonize(channel_mask)
     channel_dist     = distanceTransform(~channel_mask)
  │
  ▼  ⑥ 入口/出口锚点
     默认：硬编码坐标 entry_xy=(315,522), exit_xy=(892,1082)
     备选：角度扫描自动寻找缺口（circle_scan_entry=True）
  │
  ▼  ⑦ Dijkstra on skeleton → solution_polyline（不做 DP 简化）
     solution_skeleton_mask = 1px 渲染
     solution_channel_mask  = dilate(solution_skeleton_mask, r=28) ∩ channel_mask
```


**步骤①：最小二乘拟合外环圆**

取距 bbox 中心最远的 10% 墙壁像素（即外环像素）参与拟合，线性化圆方程求解：

```
方程：x² + y² + Dx + Ey + F = 0
令 A = [x, y, 1]，b = -(x² + y²)，解 A·z = b（最小二乘）
cx = -D/2，cy = -E/2，r_out = √(cx² + cy² - F)
```

若拟合结果异常（r_out < 10 或超出画布尺寸），fallback 到硬编码默认值 `(cx=598, cy=812, r_out=413)`。

**步骤②：填充圆盘 canvas_mask**

```python
Y, X = np.ogrid[:h, :w]
canvas_mask = ((X - cx)**2 + (Y - cy)**2 <= (r_out + 2)**2).astype(np.uint8)
```

`+2` 确保圆盘含外墙像素本身，与方形迷宫"floodFill 结果包含外框墙"对齐。

**步骤③：通道掩码**（调用已有 `_build_channel_mask`，代码零改动）

**步骤④：连通域过滤（圆形专用）**

中心装饰小圆（r≈36–45 px）围出了一个封闭区域，该区域在 `channel_mask_raw` 中形成一个与主环系统不连通的孤立 blob（面积约 3500 px²，远小于主环系统的约 400000 px²）。保留面积最大的连通域即可将其过滤。

```python
n, labels, stats, _ = cv2.connectedComponentsWithStats(channel_mask_raw, connectivity=8)
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
channel_mask = (labels == largest_label).astype(np.uint8)
```

**步骤⑤：骨架与距离变换**（调用已有函数，代码零改动）

骨架含环状拓扑，**不剪枝**。

**步骤⑥：入口/出口**

| 模式 | 触发条件 | 实现 |
|---|---|---|
| 硬编码（默认） | `circle_scan_entry=False`（默认） | 直接使用 `entry_xy=(315,522)`, `exit_xy=(892,1082)` |
| 角度扫描（备用） | `circle_scan_entry=True` | 在 `r = r_out - 5` 处以 0.5° 分辨率扫描 channel_mask，找连续 True 角度段，取两个最大段的角度质心 |

两种模式均支持外部显式传入 `entry_xy` / `exit_xy` 覆盖（与方形一致）。

`_nearest_skeleton_pixel` 将锚点纠正到最近骨架像素，硬编码误差几十像素完全无影响。

**步骤⑦：解路径**

调用已有的 `_dijkstra_on_skeleton`，**不做 Douglas-Peucker 简化**（方形的 DP 简化对直线无损，但对圆弧会将弧线粗化为折线，损害 F2 按弧长采样的均匀性）。之后的 `solution_skeleton_mask` 和 `solution_channel_mask` 生成代码与方形完全一致。

**MazeGeometry 数据类新增字段**

```python
# 在 MazeGeometry dataclass 末尾新增三个可选字段：
circle_center_xy: Optional[Tuple[int, int]] = None    # 圆心坐标（圆形时有值）
outer_ring_radius: Optional[float] = None              # 外环半径（圆形时有值）
num_channel_components_before_filter: Optional[int] = None  # 过滤前连通域数（诊断用）
```

方形迷宫调用时三个字段均为 `None`，对现有代码无影响。

**`build_maze_geometry` 接口修改**

```python
def build_maze_geometry(
    mask_path: str,
    *,
    game_type: str = "maze",          # ← 新增参数
    r_wall: int = 2,
    channel_half_width_px: int = 28,
    # 方形迷宫参数（game_type='circle' 时忽略）
    entry_corner_size: int = 105,
    use_frame_ring_first: bool = False,
    # 圆形迷宫参数（game_type='maze' 时忽略）
    circle_center: Optional[Tuple[float, float]] = None,   # None → 自动拟合
    outer_ring_radius: Optional[float] = None,              # None → 自动拟合
    entry_xy: Optional[Point] = None,
    exit_xy: Optional[Point] = None,
    circle_scan_entry: bool = False,
) -> MazeGeometry:
```

---

#### 4.3.4 特征定义

F1–F4 直接复用 `maze_feature_extractor.py` 中的四个函数。

| 特征 | 公式 | 圆形适用性说明 |
|---|---|---|
| F1 | `\|user ∩ sol_ch\| / \|sol_ch\|` | 无变化，sol_ch 来自圆形解路径 |
| F2 | 沿 solution_polyline 按弧长 40 px 采样，命中率 | 天然适用于曲线 polyline |
| F3 | `Σ channel_dist[user & ¬ch] / \|ch\|` | 无变化 |
| F4 | `\|user ∩ ¬ch\| / \|user\|` | 无变化 |
| C2 | 短笔段总长 / 所有笔段总长 | 无变化，调用 `stroke_utils` |
| C3 | σ(pressure) / μ(pressure) | 无变化，调用 `stroke_utils` |

---

#### 4.3.5 C1：运动控制能力（↓）— 方案说明

圆形使用骨架距离残差法，方形维持 Hough-on-user 不变。

```python
def compute_C1_jitter_ratio_skeleton(
    mapped_strokes, channel_skeleton,
    jitter_tol=3.0, channel_half_width=28.0
):
    """
    对所有用户笔迹点，计算到 channel_skeleton 最近骨架像素的距离 d；
    在 d <= channel_half_width（在通道内）的有效点中，
    d > jitter_tol 的比例即为 C1。
    """
    skel_ys, skel_xs = np.where(channel_skeleton > 0)
    skel_pts = np.stack([skel_xs, skel_ys], axis=1).astype(np.float64)
    
    all_pts = np.concatenate(mapped_strokes, axis=0)  # (M, 2)
    
    # KDTree 加速（骨架点≈3000–5000，笔迹点≈8000，毫秒级完成）
    from scipy.spatial import KDTree
    tree = KDTree(skel_pts)
    dists, _ = tree.query(all_pts, k=1)              # (M,)
    
    valid = dists <= channel_half_width
    total = int(valid.sum())
    if total == 0:
        return 0.0, 0, 0
    bad = int((dists[valid] > jitter_tol).sum())
    return float(bad) / float(total), bad, total
```

---

#### 4.3.6 参数

在 `extract_maze_features` 的参数列表中，方形迷宫原有参数不变，新增以下圆形专用参数（均有默认值，方形迷宫调用时完全忽略）：

```python
# 【圆形迷宫几何参数】（game_type='maze' 时不生效）
circle_center: Optional[Tuple[int, int]] = None,     # None → 自动最小二乘拟合
outer_ring_radius: Optional[float] = None,            # None → 自动最小二乘拟合
entry_xy_circle: Optional[Point] = (315, 522),        # 硬编码入口（左上缺口中心附近）
exit_xy_circle: Optional[Point] = (892, 1082),        # 硬编码出口（右下缺口中心附近）
circle_scan_entry: bool = False,                       # True → 改用角度扫描检测入口出口
```

`channel_half_width_C1` 建议值：方形保持 20.0（原值），圆形改为 28.0（与通道半宽一致）。

| 参数 | 方形值 | 圆形值 | 是否在代码中区分 |
|---|---|---|---|
| `r_wall` | 2 | 2 | 否（统一） |
| `r_solution_channel`（膨胀半宽）| 28 | 28 | 否（统一） |
| `sample_step`（F2）| 40.0 | 40.0 | 否 |
| `hit_radius`（F2）| 12.0 | 12.0 | 否 |
| `jitter_tol`（C1）| 3.0 | 3.0 | 否 |
| `channel_half_width_C1` | 25.0 | 28.0 | **是**（`game_type` 分派时设置默认值）|
| C1 方法 | Hough-on-user（方案A）/ 骨架距离（方案C）| 骨架距离 | **是** |
| DP 简化 | 有 | **无** | **是** |
| `entry_corner_size` | 105 | 不适用 | **是** |
| `circle_center` | 不适用 | (598, 812) | **是** |
| `outer_ring_radius` | 不适用 | ≈ 413 | **是** |
| `entry_xy` / `exit_xy` | 自动检测 | (315, 522) / (892, 1082) | **是** |

---

#### 4.3.7 输入 / 输出

**输入**

与方形迷宫完全一致：
- `txt_path`：轨迹文件（前 3 行为文件头，跳过）
- `png_path`：用户绘制 PNG（仅用于 bbox 坐标参照）；若为纯黑/不存在，自动 fallback 到 `inner_bbox_fallback`（用圆形外接矩形 bbox 对齐）
- `maze_mask_path`：`circle_mask.png`（前景=墙壁）
- `game_type='circle'`

**输出（JSON）**

主字段 `F1–C3` 格式与方形完全一致。`meta` 新增圆形专属字段：

```json
{
  ...
  "meta": {
    ...
    "C1_method": "skeleton_dist",   ← 方形 Hough 时为 "hough_user"
    ...
    "circle_center_xy": [598, 812],          ← 圆形专属
    "outer_ring_radius": 413.5,              ← 圆形专属
    "num_channel_components_before_filter": 2, ← 圆形专属（应为 2：主环+中心伪通道）
    "params": {
      ...
      "channel_dilate_no_dp": true,          ← 圆形专属：标记未做 DP 简化
      ...
    }
  }
}
```

**可视化输出**

| 图文件名 | 内容 | 圆形与方形的差异 |
|---|---|---|
| `{id}_channel_geometry.png` | 全通道+骨架+入口出口+解路径（红色高亮） | 骨架为环状结构，需检查中心 blob 是否已被过滤 |
| `{id}_feature_overlay.png` | 全通道（灰）+解通道（浅绿）+用户笔迹（蓝）+F2采样点 | 无差异，弧线 polyline 的采样点应均匀分布于环上 |
| `{id}_C1_skeleton.png` | 用户笔迹（灰）+ channel_skeleton（红）+ 抖动点（黄） | 替代方形的 `C1_hough.png`；显示骨架而非 Hough 线段 |

---

#### 4.3.8 预期结果（供调参参考）

对正常样本（如 c9，label=0）：
- F1 > 0.55：用户笔迹覆盖超过一半的解路径通道
- F2 > 0.65：解路径上超过 2/3 的采样点被笔迹经过
- F3 < 0.10：越出通道的加权惩罚较小
- F4 < 0.20：超过 80% 的笔迹在通道内
- C1 < 0.25、C2 < 0.15、C3 < 0.5

对障碍样本（如 c7，label=1）应在上述一个或多个特征上出现明显偏差。


---



### 4.4 阶段4：特征汇总 + 归一化管道

#### 4.4.0 模块总览

```
per-game JSON（阶段1-3已有）
      ↓
【4.A】gate_unanalyzable.py  乱画门控
      ↓
      ├─ 可分析样本 → 进入4.B
      └─ 乱画样本 → 旁路到阶段6终判（直接 prob=1.0）
      ↓
【4.B】normalize.py  per-game robust z-score
  （基准：本游戏内 label=0 且可分析 的样本的 median/MAD）
      ↓
【4.C】build_feature_matrix.py  主程序（A→B的编排 + 输出）
      ↓
feature_matrix.csv + gate_decisions.csv + normalize_stats.json
```

**核心原则**：
- 门控采用无监督、可解释的硬规则（不训练门控分类器，避免循环论证）。
- 归一化统计量只在 `label=0 且 可分析` 的样本上估计，让 z-score 的语义明确为"相对正常儿童的偏离量"。
- 所有归一化后的特征方向统一为"越大越异常"，便于后续加权评分。

---

#### 4.4.1 阶段4.A：乱画门控 `features/gate_unanalyzable.py`

**输入**:data/feature/all.csv

**门控规则**：

IF (F2 < 0.4 AND F1 < 0.05)
   → 异常
ELSE IF (F2 < 0.4 AND (F3 > 1 OR F4 > 0.3))
   → 异常
ELSE IF 以game字段分组，计算各组F3、F4的Z分数，Z分数>2
   → 异常
ELSE
   → 正常

**输出 `gate_decisions.csv`**：追加is_unanalyzable / triggered_rules / F3_zscore / F4_zscore 列

**验收标准**：
- 12 个已知乱画样本召回率 = 100%
- label=0 样本误判数 = 0

---

#### 4.4.2 阶段4.B：归一化 `features/normalize.py`

**功能**：per-game robust z-score 归一化。基于 label=0 且可分析样本的 median/MAD 进行 robust z-score 归一化，并统一方向为"越大越异常"。

**函数接口**（设计成 `fit / transform` 分离，便于阶段6的LOSO每折重新拟合）：

```python
def fit_normalize_stats(
    feature_dicts: List[dict],   # 可分析 + label=0 的样本特征dict列表
    games: List[str],
    feature_names: List[str] = ['F1','F2','F3','F4','C1','C2','C3'],
) -> dict:
    """
    对每个(game, feature)估计 median / MAD / scale，返回 stats 字典。
    
    stats = {
      'sym':  {'F1': {'median': 0.72, 'mad': 0.08, 'scale': 0.119, 'ref_n': 4}, ...},
      'maze': {...},
      'circle': {...},
    }
    """

def apply_normalize(
    feature_dicts: List[dict],
    games: List[str],
    stats: dict,
    feature_names: List[str] = ['F1','F2','F3','F4','C1','C2','C3'],
    direction: dict = {'F1':-1,'F2':-1,'F3':+1,'F4':+1,'C1':+1,'C2':+1,'C3':+1},
    clip: Tuple[float,float] = (-3.0, 6.0),
) -> np.ndarray:  # shape (N, 7)
    """
    对每个样本应用 z-score，统一方向为"越大越异常"，裁剪后返回矩阵。
    """
```

**核心公式**（位于 `fit_normalize_stats` 内）：

```python
for g in {'sym', 'maze', 'circle'}:
    for f in feature_names:
        ref_pool = [x[f] for x,game,lab in zip(dicts,games,labels) 
                    if game==g and lab==0 and not is_unanalyzable(x)]
        
        med = np.median(ref_pool)
        mad = np.median(np.abs(np.array(ref_pool) - med))
        scale = 1.4826 * mad                             # 使MAD与正态σ可比
        scale = max(scale, 0.05 * abs(med) + 1e-3)       # 下限，防MAD≈0
        
        stats[g][f] = {'median': med, 'mad': mad, 'scale': scale, 'ref_n': len(ref_pool)}
```

**在 `apply_normalize` 内**：

```python
z_raw = (x - stats[g][f]['median']) / stats[g][f]['scale']
z = direction[f] * z_raw            # F1,F2取负号（正向特征"越大越异常"需反向）
z = np.clip(z, clip[0], clip[1])
```

**关键实现要点**：

1. **MAD下限**：`scale = max(1.4826·MAD, 0.05·|median| + 1e-3)`。sym 游戏只有 4 个正常样本，MAD 可能几乎为 0 导致 z-score 爆炸——这个下限非常关键。
2. **参考池可能很小**：sym 只有 4 个 label=0 样本，circle 只有 5 个，maze 只有 11 个。必要时在 meta 中记录 `ref_n`，若某 `ref_n < 3` 则发出警告（但不终止）。
3. **裁剪范围 `(-3, +6)` 的理由**：左侧 -3 对应"比正常还正常"，继续往左的信息价值已饱和；右侧 +6 保留严重异常样本的强信号。如果发现大量样本被裁到 +6，说明 MAD 估计过小，需要调整下限。
4. **乱画样本不参与归一化的拟合，但也不进入最终的特征矩阵**——它们直接在阶段6被赋予 prob=1.0。

---

#### 4.4.3 阶段4.C：主入口 `features/build_feature_matrix.py`

```bash
python features/build_feature_matrix.py --feature_csv data/feature/all.csv --out_dir output
```

**流程**：
```
1. 读取 data/feature/all.csv（或多个 json_dirs）
2. 调用 gate_unanalyzable → 得到 is_unanalyzable 标记
3. 筛选 {可分析 且 label=0} 样本 → fit_normalize_stats
4. 对 {全部可分析样本} 调用 apply_normalize
5. 输出三个文件：
   - output/feature_matrix.csv       # 可分析样本的z-score矩阵（主分类器输入）
   - output/gate_decisions.csv       # 全部样本的门控结果（阶段6重并入用）
   - output/normalize_stats.json     # 归一化统计量（调试 + 报告用）
```

**输出 `feature_matrix.csv`**：
```csv
sample_id,game,label,F1_z,F2_z,F3_z,F4_z,C1_z,C2_z,C3_z
```

**检查**：
- label=0 样本的 z 值应大致在 [-2, +2] 范围内；

    label=0 样本 z 均值: [ 0.28  0.48  0.5   1.25 -0.06  0.56  0.03]

- label=1 样本应在至少 1-2 个特征上出现 z > 2 的明显偏离；

    label=1 样本 z 均值: [1.27 2.24 2.88 4.07 0.4  1.03 0.49]

- 若正常样本出现大量 |z| > 3，说明 MAD 下限需要调高；

    label=0 中 |z|>3 的单元格数（期望接近 0）: 10

- 若障碍样本 z 值普遍不高，说明特征判别力不足，可能需要回头调整阶段1-3的特征参数。

    label=1 中 z>2  的单元格数（期望 ≥1-2）: 79


---



### 4.5 阶段5：分类器

#### 4.5.1 四个模型的统一接口

```python
class BaseClassifier:
    def fit(self, X_z: np.ndarray, y: np.ndarray, games: List[str]) -> 'BaseClassifier':
        ...
    def predict_proba(self, X_z: np.ndarray) -> np.ndarray:
        """返回异常概率 p ∈ [0,1]（shape=(N,)）"""
        ...
    def get_feature_importance(self) -> Dict[str, float]:
        """返回特征权重（便于跨模型对比）"""
        ...
```

四个实现：`PurePriorScorer` (M2)、`SemiPriorScorer` (M1)、`L2LogisticClassifier` (M3)、`RandomForestClassifier_wrap` (M4)。

---

#### 4.5.2 M2：纯先验加权线性评分（消融对照，先写）

**先写 M2 是因为它最简单，可以作为 M1 的骨架**。

```python
W_PRIOR = {
    'F1': 1.2, 'F2': 1.2, 'F3': 1.0, 'F4': 1.0,
    'C1': 0.8, 'C2': 0.8, 'C3': 0.8,
}
# 归一化使 Σw = 7
total = sum(W_PRIOR.values())
W_PRIOR = {k: v * 7 / total for k, v in W_PRIOR.items()}

class PurePriorScorer:
    def fit(self, X_z, y, games):
        self.w_ = np.array([W_PRIOR[f] for f in FEATURE_NAMES])
        return self
    
    def predict_proba(self, X_z):
        score = X_z @ self.w_           # shape=(N,)
        # 用 sigmoid 把分数映射到 [0,1]，斜率参数用训练集标定
        return 1.0 / (1.0 + np.exp(-score / self.sigmoid_scale_))
    
    def get_feature_importance(self):
        return dict(zip(FEATURE_NAMES, self.w_))
```

**sigmoid 缩放参数**：在 `fit` 时计算 `sigmoid_scale_ = std(X_train @ w) + ε`，让 score 在训练集上的标准差为 1 量级，避免 sigmoid 饱和。

#### 4.5.3 M1：半先验加权线性评分

```python
class SemiPriorScorer(PurePriorScorer):
    def fit(self, X_z, y, games, bounds=0.3):
        # 从先验开始
        w_init = np.array([W_PRIOR[f] for f in FEATURE_NAMES])
        
        # 定义约束优化：在 w_init * (1±bounds) 的盒约束内最大化 AUROC
        from scipy.optimize import minimize
        
        def neg_auroc(w):
            score = X_z @ w
            # 近似AUROC（Wilcoxon-Mann-Whitney统计量）
            pos = score[y==1]; neg = score[y==0]
            if len(pos)==0 or len(neg)==0: return 0.0
            return -np.mean(pos[:,None] > neg[None,:])
        
        lo = w_init * (1 - bounds)
        hi = w_init * (1 + bounds)
        res = minimize(neg_auroc, w_init, method='L-BFGS-B',
                       bounds=list(zip(lo, hi)))
        self.w_ = res.x
        # 后续同 PurePriorScorer
```

**边界值 0.3 的含义**：每个特征权重在先验值的 ±30% 内微调。如果数据强烈主张某特征的权重"应该"更高或更低，优化结果会贴到边界上——这本身就是一个可报告的发现。

#### 4.5.4 M3：L2 Logistic Regression（主力）

```python
from sklearn.linear_model import LogisticRegression

class L2LogisticClassifier:
    def __init__(self, C=1.0):
        self.C = C
    
    def fit(self, X_z, y, games):
        self.model_ = LogisticRegression(
            penalty='l2', C=self.C,
            class_weight='balanced',     # 类别不平衡
            max_iter=1000, solver='lbfgs',
        )
        self.model_.fit(X_z, y)
        return self
    
    def predict_proba(self, X_z):
        return self.model_.predict_proba(X_z)[:, 1]
    
    def get_feature_importance(self):
        return dict(zip(FEATURE_NAMES, self.model_.coef_[0]))
```

**C 参数的选择**：在阶段6的LOSO循环内嵌套一个小的网格搜索 `C ∈ {0.1, 0.3, 1.0, 3.0}`，用内层LOO选最优。考虑到样本量小，嵌套CV会让评估时间变长但结果更诚实。若时间紧张，直接用 `C=1.0` 并在报告中说明。

#### 4.5.5 M4：Random Forest（对照）

```python
from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifier_wrap:
    def fit(self, X_z, y, games):
        self.model_ = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,               # 浅树
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42,
        )
        self.model_.fit(X_z, y)
        return self
    
    def predict_proba(self, X_z):
        return self.model_.predict_proba(X_z)[:, 1]
    
    def get_feature_importance(self):
        return dict(zip(FEATURE_NAMES, self.model_.feature_importances_))
```

**作用**：**不是为了打败 M3，而是为了证明"非线性模型并未显著提升"，从而支持"简单模型已够"的方法论主张**。如果 M4 意外地显著优于 M3，说明决策边界非线性，需要反思 LR 是否足够。

---


## 4.6 阶段6：评估 `experiments/run_experiments.py`

### 4.6.1 LOSO 主流程

```bash
python experiments/run_experiments.py \
    --feature_matrix output/feature_matrix.csv \
    --gate_decisions output/gate_decisions.csv \
    --out_dir results/
```

**核心循环**（伪代码）：

```python
# 读可分析样本
df_analyzable = load(feature_matrix.csv)    # 乱画样本不在此
# 读门控结果（含乱画）
df_gate = load(gate_decisions.csv)

N = len(df_analyzable)
models = {'M1':SemiPriorScorer(), 'M2':PurePriorScorer(),
          'M3':L2LogisticClassifier(), 'M4':RandomForestClassifier_wrap()}

# 每个模型的每个样本一个预测概率
probs = {m: np.zeros(N) for m in models}

for i in range(N):
    train_idx = [j for j in range(N) if j != i]
    test_idx  = [i]
    
    # 【关键】归一化统计量只在 train 上拟合
    # 但feature_matrix.csv已经是归一化后的，这意味着此处有数据泄漏！
    # ↓ 所以正确做法见下一小节
    
    for mname, model in models.items():
        model.fit(X_z[train_idx], y[train_idx], games[train_idx])
        probs[mname][i] = model.predict_proba(X_z[test_idx])[0]
```

### 4.6.2 严格防泄漏的正确实现

上面的简化流程有一个严重问题：`feature_matrix.csv` 是用**全部可分析样本**（含测试样本）的统计量算的 z-score。测试样本参与了自己的归一化——数据泄漏。

**正确的 LOSO 需要保存原始特征，每折重新归一化**：

```python
# 读原始特征（未归一化）
df_raw = load_all_raw_features(json_dirs)
df_raw = df_raw[df_raw['is_unanalyzable'] == False]   # 只留可分析样本

X_raw = df_raw[FEATURE_NAMES].values
y = df_raw['label'].values
games = df_raw['game'].values

for i in range(N):
    train_idx = [j for j in range(N) if j != i]
    test_idx  = [i]
    
    # 每折重新拟合归一化
    stats = fit_normalize_stats(
        feature_dicts=X_raw[train_idx], 
        games=games[train_idx],
        labels=y[train_idx],
    )
    X_train_z = apply_normalize(X_raw[train_idx], games[train_idx], stats)
    X_test_z  = apply_normalize(X_raw[test_idx],  games[test_idx],  stats)
    
    for mname, model in models.items():
        model.fit(X_train_z, y[train_idx], games[train_idx])
        probs[mname][i] = model.predict_proba(X_test_z)[0]
```

**注意**：归一化参考池是 train 中的 label=0 样本。如果 test 样本恰好是某游戏的 label=0 样本之一，train 中该游戏的参考池会少一个样本——这会让 sym 的 label=0 池从 4 降到 3，MAD 估计更不稳定。**阶段4.B 的 MAD下限保护机制此时非常关键**。

### 4.6.3 乱画样本并入终判

```python
# LOSO结束后，对每个乱画样本直接赋异常概率
unanalyzable_ids = df_gate[df_gate['is_unanalyzable']==True]['sample_id']

# 构造全样本的 prob 和 y
probs_full = {m: [] for m in models}
y_full = []
games_full = []

for row in df_gate.iterrows():
    sid = row['sample_id']
    y_full.append(row['label'])
    games_full.append(row['game'])
    if row['is_unanalyzable']:
        for m in models: probs_full[m].append(1.0)     # 门控判异常
    else:
        idx = df_analyzable.index[df_analyzable['sample_id']==sid][0]
        for m in models: probs_full[m].append(probs[m][idx])
```

**所有评估指标都基于 `probs_full` 和 `y_full` 计算**——这反映的是完整筛查系统的性能。

### 4.6.4 评估指标与报告

**三层指标表**：

```
【主任务性能】（在 全部样本 上计算）
                    AUROC    AUPRC    F1@opt    阈值@opt
M2 (纯先验)        0.xxx    0.xxx    0.xxx     0.xx
M1 (半先验)        0.xxx    0.xxx    0.xxx     0.xx
M3 (L2 LR)        0.xxx    0.xxx    0.xxx     0.xx
M4 (RF)           0.xxx    0.xxx    0.xxx     0.xx

【门控诊断】
已知乱画样本数：12
门控召回：xx/12
门控误判（正常样本被门控）：x
门控未识别的乱画（进入主分类器）：x

【per-game AUROC 拆分】（仅主力 M3）
sym:    0.xxx  (n=7+1乱画)
maze:   0.xxx  (n=29+5乱画)
circle: 0.xxx  (n=15+6乱画)
```

**关键对比叙事**：
- M2 → M1：数据微调的边际价值
- M1 → M3：软约束（权重盒约束）vs 硬约束（L2正则）
- M3 → M4：非线性模型的增益
- 若 M3 ≈ M4：**"本任务接近线性可分，简单模型已足够"——这是核心结论**

### 4.6.5 可解释性图表

**特征权重对比条形图**：

```
            F1   F2   F3   F4   C1   C2   C3
M1 w_final  1.3  1.5  1.0  0.8  0.6  0.4  0.9    ← 半先验微调
M2 w_prior  1.2  1.2  1.0  1.0  0.8  0.8  0.8    ← 纯先验（不变）
M3 |coef|   0.9  1.2  0.7  0.6  0.3  0.1  0.8    ← 数据驱动
M4 impor.   0.18 0.21 0.15 0.12 0.08 0.05 0.21   ← 树模型
```

**跨模型一致性分析**：计算四组权重的 **Spearman 秩相关系数**，若 ≥ 0.7 说明特征重要性排序在不同模型间一致，是方法论稳健性的有力证据。

### 4.6.6 暂不做但可选的加分项

1. **LOGO（Leave-One-Game-Out）评估**：根据你的选择，**先不做，等 LOSO 结果出来再决定**。如果 LOSO 上 AUROC ≥ 0.8 且跨模型一致，再加 LOGO 验证跨游戏泛化。
2. **归一化消融**：跑一次"不做归一化 / 不做per-game 归一化"的基线，对比 AUROC 下降量——这是归一化方法论贡献的直接证据。
3. **SHAP 值**：对 M3 做样本级的 SHAP 解释，展示具体某个样本为什么被判为异常。对方法论不是必需的，但放报告里很有说服力。

---


---

