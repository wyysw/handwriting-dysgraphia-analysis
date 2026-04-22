# 书写障碍分类实验——完整推进指导文档

> **版本**： v1.0
> 阶段二开始之前！

> **文档说明**：本文档是跨对话协作的主参考文档，整合实验设计与流程指导。
> 在每次新对话开始时将本文档连同相关代码一起提供给 AI 助手，即可无缝延续工作。特征指标定义在具体实现过程中允许修改，修改后本文档对应章节同步更新并标注。

---

## 目录

1. [项目总览](#1-项目总览)
2. [现有基础盘点](#2-现有基础盘点)
3. [实验推进路线（按阶段）](#3-实验推进路线按阶段)
4. [各程序详细说明](#4-各程序详细说明)
   - 4.1 [阶段 1：对称游戏特征提取器](#42-阶段-1对称游戏特征提取器)
   - 4.3 [阶段 2：迷宫游戏特征提取器](#43-阶段-2迷宫游戏特征提取器)
   - 4.4 [阶段 3：特征汇总与归一化管道](#44-阶段-3特征汇总与归一化管道)
   - 4.5 [阶段 4：三方案分类器](#45-阶段-4三方案分类器)
   - 4.6 [阶段 5：实验评估脚本](#46-阶段-5实验评估脚本)
5. [特征定义参考表](#5-特征定义参考表)
6. [测试所需文件清单](#6-测试所需文件清单)
7. [项目预期架构](#7-项目预期架构)
8. [关键决策记录与待确认事项](#8-关键决策记录与待确认事项)

---

## 1. 项目总览

### 1.1 研究目标

通过分析儿童在三款手写游戏中的绘制轨迹，判断该绘制是否由书写障碍儿童产生（**二分类**问题）。强调**方法论贡献**：同时兼顾"游戏结果"（功能轴 F）和"游戏过程"（控制轴 C），设计跨游戏通用的特征框架，验证双轴联合判别优于单轴打分。

### 1.2 三款游戏

| 游戏 | 任务描述 | 参考图像文件 |
|---|---|---|
| 对称（Symmetry） | 依据对称轴，将左侧/上半区蓝色图形镜像补画到右侧/下半区 | `sym_blue_mask.png`, `sym_helper_mask_completed.png` |
| 方形迷宫（Square Maze） | 在直线迷宫中从起点走到终点 | `maze_mask.png` |
| 圆形迷宫（Circle Maze） | 在环形路径迷宫中从起点走到终点 | `circle_mask.png` |

对于对称游戏，在绘制者看来原始蓝色图形位于图片左侧，对称轴为纵向竖直对称轴；但在程序看来，原始图形位于图片上半区，对称轴为横向水平对称轴。

### 1.3 数据规格

- **轨迹格式**：每行 `x y pressure`，前 3 行为文件头（`eink` / `1` / `0,0`）需跳过
- **压力**：> 0 为落笔，= 0 为抬笔；坐标范围约 0–14000
- **无时间戳**（速度、耗时类特征全部不可用）
- **样本量与标注**：参考labels.csv

### 1.4 核心方法论约束

- 样本粒度为 **per-game**（无法关联同一儿童的不同游戏数据，单次绘制 = 一条样本），不做跨游戏的 subject 级拼接
- 可用信号：**空间坐标 + 压力值 + 笔段结构**（由 pressure 跳变推断）
- 分类器选用小样本适用的轻量模型（Logistic Regression 或深度 ≤ 3 的决策树）
- 方法论贡献优先于单个特征实现的精度

---

## 2. 现有基础盘点

### 2.1 模块 1：图形预处理（已完成）

**作用**：从三张原始测评图提取标准参考结构，生成二值掩码。

| 脚本 | 输入 | 关键输出 |
|---|---|---|
| `shape/final_shape_sym.py` | `data/35duichen.png` | `sym_blue_mask.png`（蓝色半边图形）, `sym_helper_mask_completed.png`（网格+对称轴+外框） |
| `shape/final_shape_migong.py` | `data/34migong.png` | `maze_mask.png`（迷宫墙壁线条） |
| `shape/final_shape_circle.py` | `data/36circle.png` | `circle_mask.png`（圆形迷宫墙壁线条） |
| `shape.py` | — | 统一调度以上三个脚本 |

所有掩码均为 1201×1601 单通道 PNG，与原图像素级对齐。

### 2.2 模块 2：对称游戏评估（已基本完成）

对称游戏评估模块的实现对应后文中定义的阶段1，详细内容在“1_阶段一实现文档”中。

**子模块实现**：根据阶段一代码更新，主要特征提取函数如下：

* **F1：完成度（翻转容差）**：使用膨胀的目标图形和翻转后的用户图形计算**IoU-like**重叠度。公式为：
  [
  F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{P + R + 1e-9}
  ]
* **F2：关键点命中率**：计算用户笔迹与网格交点之间的命中率。公式为：
  [
  F2 = \frac{\text{命中关键点数}}{\text{总关键点数}}
  ]
* **F3：无效书写加权距离**：计算用户笔迹偏离有效区域的加权距离。公式为：
  [
  F3 = \frac{\sum \text{违规像素的距离}}{\text{标准图形像素总数}}
  ]
  其中包含**越轴惩罚**，确保用户在下半画布书写。
* **F4：路径偏离比**：计算不在有效区域的笔迹比例。公式为：
  [
  F4 = \frac{\text{偏离区域的像素数}}{\text{总笔迹像素数}}
  ]
* **C1：抖动比例**：计算每个笔段点到霍夫线段的垂直残差，超出容差的点占比。
* **C2：短笔段比例**：计算短笔段总长度占所有笔段总长度的比例，短笔段按画布对角线的2%设定阈值。
* **C3：压力变异系数**：计算笔段压力的标准差与均值的比值，裁剪笔段两端的压力数据避免异常。


### 2.3 模块 3及以后：尚未开始

迷宫特征提取、分类器均从零开始。其中分类器逻辑集成在 `classify.py` 中，不作为独立模块。

---

## 3. 实验推进路线（按阶段）

```
阶段 1   对称游戏特征提取器（重构/扩展 sym_analyze3.py）
   ↓
阶段 2   迷宫游戏特征提取器（方形 + 圆形，共用基类）
   ↓
阶段 3   特征汇总 + 归一化管道（输出标准化特征矩阵 CSV）
   ↓
阶段 4   三方案分类器（方案 A 加权打分 / 方案 B 双轴偏离 / 方案 C 机器学习）
   ↓
阶段 5   实验评估（LOO-CV + 分游戏拆解 + 跨游戏泛化 + 消融）
```

> **并行建议**：阶段 1 和阶段 2 在逻辑上独立，可在对称游戏特征调通之后，将通用笔段分析逻辑（C1/C2/C3）提炼为公共库，再直接复用于迷宫。

---

## 4. 各程序详细说明

---

### 4.1 阶段 1：对称游戏特征提取器（已完成）

---

#### 程序：`features/sym_feature_extractor.py`

**目的**：对一个对称游戏样本，提取 7 个标准化特征（F1–F4, C1–C3），输出为 JSON。这是对称游戏的**核心特征计算程序**。

**设计说明**：
本程序不沿用 `sym_analyze3.py` 中的打分逻辑（其将成为方案 A 的基础），而是提取**原始指标值**（0–1 的比例或连续量），统一由阶段 3 的管道进行归一化。

从 sym_analyze3.py 直接导入以下通用基础设施（不重复实现）：

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

#### 主要函数与特征定义

**F1：翻转容差 F1（0-1，↑越好）**：使用容差膨胀后的 F1 分数，衡量翻转后的用户笔迹与目标图形的重叠程度。

```python
A = dilate(reflected_user_mask, r)  # 用户笔迹膨胀
B = dilate(target_mask, r)          # 目标图形膨胀
precision = mean(B[reflected_user_mask>0])   # 用户点里有多少落在"target 容差域"
recall    = mean(A[target_mask>0])           # target 点里有多少被"用户容差域"覆盖
F1 = 2PR / (P + R + 1e-9)
```

**关键参数**：
- `dilation_radius = 5` 像素

**与 sym_analyze3.tolerant_f1 的区别**：
- **不做 normalize_mask 归一缩放**，保留"大小差异"信息
- 按**画布原尺寸**计算 IoU 式容差指标



**F2：关键点命中率（0-1，↑越好）**：网格交点关键点被用户笔迹覆盖的比例。

```python
keypoints = extract_keypoints_from_target(target_mask, helper, include_midpoints=False)
dist = distance_transform_to_mask(reflected_user_mask)
hit  = Σ 1[dist[y,x] <= hit_radius]
F2 = hit / len(keypoints)
```

**关键参数**：
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

**关键参数**：
- `tol_valid = 9` 像素
- `cross_penalty = 10`（越轴像素的惩罚系数）

**设计说明**：
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

**关键参数**：
- `jitter_tol = 3.0` 像素
- `channel_half_width = 20` 像素

**加权方式**：点级统计（所有霍夫线段上的投影点一起算；不按笔段长度加权）

**C2：短笔段比例（0-1，↓越好）**：短笔段总长度占所有笔段总长度的比例。

```python
stroke_lens = [arc_length(s) for s in strokes_mapped]
# 阈值：画布对角线 * 0.02（ANS5 选择方案 a）
thr = diag(canvas) * 0.02
short_total = Σ stroke_lens[i] for i where stroke_lens[i] < thr
C2 = short_total / Σ stroke_lens
```

**关键参数**：
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

**关键参数**：
- `C3_trim_ends = 3`（裁剪笔段两端的点数）

**数据获取**：
- 不改 sym_analyze3，在 sym_feature_extractor 里另写 `load_trajectory_strokes_with_pressure()`
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



**主入口**：


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

**输入**：
- `txt_path`：用户轨迹文件（`x y pressure`，跳过前 3 行）
- `png_path`：用户绘制图片（仅用于 bbox 坐标参照，与 sym_analyze3 一致）
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

### 4.3 阶段 2：迷宫游戏特征提取器

迷宫游戏的 F1–F4 依赖通道几何，与对称游戏算法不同；
C1–C3 的**定义与对称游戏一致**，应提炼为公共库复用。如以下公共库设计与此前完成的阶段1实现不同，以阶段1实现的版本为准。

有一个问题：目前没有两个迷宫的标准答案，要如何解决呢？

#### 程序 A（公共库）：`features/stroke_utils.py`

**目的**：提供与具体游戏无关的笔段处理工具，供三个游戏的特征提取器共用。

**主要函数**：
```
parse_trajectory(txt_path) → List[np.ndarray]
    读取 txt 文件，跳过前 3 行，按 pressure=0 切分，
    返回笔段列表，每段为 shape (N, 3) 的数组（x, y, pressure）

strokes_to_mask(strokes, canvas_h, canvas_w, line_width=3) → np.ndarray
    将笔段列表渲染为二值掩码（uint8, 0/255）

map_strokes_to_canvas(strokes, src_bbox, dst_bbox) → List[np.ndarray]
    线性缩放：将轨迹坐标从 src_bbox 映射到 dst_bbox
    src_bbox = (x_min, y_min, x_max, y_max) in trajectory space
    dst_bbox = (x_min, y_min, x_max, y_max) in canvas space

compute_C1_jitter(strokes, reference_skeleton, jitter_tol=3.0) → float
    通用 C1 实现（参考骨架可以是对称期望图形骨架，或迷宫通道骨架）

compute_C2_short_stroke(strokes, threshold_ratio=0.02) → float
    通用 C2 实现（threshold = 画布对角线 × threshold_ratio）

compute_C3_pressure_cv(strokes, trim_ends=3) → float
    通用 C3 实现
```

---

#### 程序 B：`features/maze_feature_extractor.py`

**目的**：对方形迷宫或圆形迷宫样本，提取相同结构的 7 特征。

**设计说明**：
方形迷宫和圆形迷宫共用同一个提取器，通过 `game_type` 参数区分。关键预处理差异：

| 步骤 | 方形迷宫 | 圆形迷宫 |
|---|---|---|
| 通道掩码来源 | `maze_mask.png` | `circle_mask.png` |
| 正确路径提取 | 用形态学腐蚀获取"通道内部"（迷宫墙是线条，腐蚀后中间空白即通道） | 同左，但需识别环形结构 |
| F2 关键点定义 | 路径骨架上的岔路口/转角 | 路径骨架上每隔固定弧度的采样点 |
| 坐标对齐 | 同对称游戏：用用户 png bbox 做线性映射 | 同左 |

**通道内部掩码生成**（关键步骤，需在提取器中实现）：
```
wall_mask = maze_mask.png（前景=墙壁线条）
canvas_mask = 外框填充（用 floodFill 填充外框内部区域）
channel_mask = canvas_mask AND NOT wall_mask
    → 前景 = 可行走通道区域
channel_skeleton = skimage.morphology.skeletonize(channel_mask)
    → 通道中线骨架
```

**主要函数**：
```
compute_F1_channel_coverage(user_mask, channel_mask) → float [0,1]
    用户笔迹落在通道内的像素数 / 通道总像素数

compute_F2_skeleton_coverage(user_mask, channel_skeleton, sample_step=20) → float [0,1]
    方形迷宫：在通道骨架上每隔 sample_step 像素取一个采样点，
              计算用户笔迹对这些点的命中率（距离 ≤ 8px 算命中）
    圆形迷宫：在环形骨架上每隔固定弧度取采样点，计算命中率

compute_F3_invalid_drawing(user_mask, channel_mask, channel_dist_transform) → float
    等同对称游戏 F3：违规像素距通道边界的加权距离之和 / 通道面积

compute_F4_offpath_ratio(user_mask, channel_mask) → float [0,1]
    通道外用户像素 / 用户总像素
```

C1/C2/C3 直接从 `stroke_utils.py` 调用，迷宫的 C1 参考骨架 = `channel_skeleton`。

**主入口**：
```python
def extract_maze_features(txt_path, png_path,
                           maze_mask_path,
                           game_type,          # 'maze' or 'circle'
                           out_json_path=None) -> dict
```

**输入**：
- `txt_path`：用户轨迹文件
- `png_path`：用户绘制图片（bbox 参照）
- `maze_mask_path`：`maze_mask.png` 或 `circle_mask.png`
- `game_type`：字符串标识

**输出**（格式同对称游戏，`"game"` 字段为 `"maze"` 或 `"circle"`）

**预期结果**：与对称游戏类似，正常样本各特征值合理，障碍样本出现偏差。

**需要你提供的文件**：
- 方形迷宫 `.txt` 样本（含正常和障碍各 1 份以上）及对应 `.png`
- 圆形迷宫 `.txt` 样本（含正常和障碍各 1 份以上）及对应 `.png`
- `maze_mask.png` 和 `circle_mask.png`（模块 1 输出）

**测试方法**：
1. 对 `l1.txt`（方形迷宫样本）和 `c1.txt`（圆形迷宫样本）分别运行
2. 生成通道掩码可视化图，确认 `channel_mask` 正确覆盖了可行走区域（而不是墙壁）
3. 叠加用户笔迹与通道，目视确认 F4 的分子（通道外像素）符合直觉

---

### 4.4 阶段 3：特征汇总与归一化管道

#### 程序：`features/build_feature_matrix.py`

**目的**：将所有样本的原始特征 JSON 汇总成一个带标签的 CSV，并为每个游戏估计归一化参数 (μ, σ)，输出归一化后的特征矩阵，供分类器直接读取。

**功能**：
1. 批量读取 `data/samples/{game_type}/` 下所有样本，调用对应特征提取器，生成/读取 JSON
2. 将所有 JSON 合并为 `features_raw.csv`（每行一个样本，列为 sample_id, game, label, F1…C3）
3. **乱画检测（Scribble Gate）**：满足以下全部条件则标记为 `scribble=1`，从后续流程剔除：
   - 用户笔迹 mask 与标准图形 mask 的 IoU < 0.05
   - F3 原始值 > 某经验阈值（初始设为正常样本 F3 均值的 5 倍）
   - 用户笔迹总长 / 标准路径长 > 3
4. 对非乱画样本，**按游戏分组**，仅用训练集中的正常样本估计 (μ, σ)（若使用全样本 LOO-CV，则在每个 fold 内动态计算）
5. 对每个特征做 z-score 归一化：`z = (x − μ) / max(σ, ε)`，输出 `features_normalized.csv`

**关于归一化参数估计**（考虑到正常样本仅 5–10 份）：
- 优先使用 **中位数 + MAD（中位数绝对偏差）** 作为稳健估计：`μ̃ = median`，`σ̃ = 1.4826 × MAD`
- 若某游戏正常样本 < 5，在论文中如实说明，并讨论其对归一化稳定性的影响

**输入**：
- `data/samples/{game_type}/*.txt`（全部样本轨迹）
- `data/labels.csv`
- 模块 1 输出的各掩码文件
- （可选）已有的特征 JSON 缓存，避免重复计算

**输出**：
- `features/features_raw.csv`：原始特征矩阵（含 scribble 标记）
- `features/features_normalized.csv`：归一化后矩阵（已剔除乱画样本）
- `features/normalization_params.json`：每游戏、每特征的 (μ̃, σ̃) 值（部署时仅需此文件）
- `features/scribble_log.csv`：被乱画检测剔除的样本列表

**预期结果**：
- 正常样本归一化后特征值中心在 0 附近，障碍样本应在"差的方向"有明显偏移
- 三个游戏的归一化样本在特征空间中可以直接混合比较

---

### 4.5 阶段 4：三方案分类器

#### 程序：`classifier/classify.py`

**目的**：实现三套分类方案，输入均为归一化特征矩阵，输出均为二分类预测（0=正常，1=障碍）。

---

**方案 A：加权打分法（基线）**

对应 `sym_analyze3.py` 现有打分逻辑的泛化版本。

```
对每个样本：
  F_score_A = w_F1 * norm(F1) + w_F2 * norm(F2) - w_F3 * norm(F3) - w_F4 * norm(F4)
  C_score_A = - w_C1 * norm(C1) - w_C2 * norm(C2) - w_C3 * norm(C3)
  total_score = α * F_score_A + (1-α) * C_score_A
  预测 = 1 if total_score < threshold else 0
```

其中权重向量 `(w_F1,...,w_C3)` 和 `threshold` 需要在训练集上通过**网格搜索最大化 Sensitivity**（或 AUC）确定。

注意：方案 A 的权重是按游戏固定的（三个游戏各一组权重），体现其局限性。

---

**方案 B：双轴偏离度法（主方法）**

```
对每个样本（使用归一化后特征 z_F1...z_C3）：

  # 单边惩罚（ReLU）：只惩罚往"差"的方向偏离
  # F 轴：F1、F2 越小越差（z < 0 为偏差），F3、F4 越大越差（z > 0 为偏差）
  dev_F1 = max(0, -z_F1)     # F1 低于正常 → 惩罚
  dev_F2 = max(0, -z_F2)
  dev_F3 = max(0,  z_F3)     # F3 高于正常 → 惩罚
  dev_F4 = max(0,  z_F4)
  F_score = mean(dev_F1, dev_F2, dev_F3, dev_F4)   # 功能轴偏离度

  # C 轴：C1、C2、C3 越大越差（z > 0 为偏差）
  dev_C1 = max(0, z_C1)
  dev_C2 = max(0, z_C2)
  dev_C3 = max(0, z_C3)
  C_score = mean(dev_C1, dev_C2, dev_C3)           # 控制轴偏离度

  # 四象限判别（在二维平面 (F_score, C_score) 上）
  # 阈值 θ_F, θ_C 在训练集上确定
  if F_score > θ_F or C_score > θ_C:
      预测 = 1 (障碍)
  else:
      预测 = 0 (正常)
```

方案 B 的输出包含可解释的二维坐标，可绘制散点图展示四象限分布：
- 右上（F↓C↓）：功能与控制均差（典型书写障碍）
- 右下（F↓C↑）：功能差但控制尚可（可能任务理解困难）
- 左上（F↑C↓）：功能可以但控制差（运动控制问题）
- 左下（F↑C↑）：正常

---

**方案 C：机器学习分类器**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 输入：7 维归一化特征 + （可选）3 维 one-hot 游戏类型
# 选择：Logistic Regression（首选，系数可解释）
#       DecisionTree(max_depth=3)（备选，分支可直观展示）
# 正则化参数 C 通过内层 CV 确定

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
```

方案 C 的优势在于自动学习特征权重，可直接输出概率，便于计算 AUC。

---

**主入口**：
```python
def train_and_predict(X_train, y_train, X_test, method='B',
                      params=None) -> (np.ndarray, dict)
    # 返回 (y_pred, metadata)
    # metadata 包含方案 B 的 (F_score, C_score) 或方案 C 的概率
```

---

### 4.6 阶段 5：实验评估脚本

暂略

## 5. 特征定义参考表

（此表为当前设计，允许在实现过程中修改，修改后请同步更新本节并标注日期）

| 轴 | ID | 名称 | 方向 | 对称游戏 | 方形迷宫 | 圆形迷宫 |
|---|---|---|---|---|---|---|
| F | F1 | 完成度 | ↑（越大越好） | 翻转容差 F1（IoU-like） | 通道覆盖率 | 环形通道覆盖率 |
| F | F2 | 关键结构对齐 | ↑ | 网格交点命中率 | 骨架采样点命中率 | 弧度采样点命中率 |
| F | F3 | 无效书写 | ↓（越小越好） | 越界像素加权距离/标准面积 | 同左 | 同左 |
| F | F4 | 路径偏离比 | ↓ | 期望图形外笔迹比 | 通道外笔迹比 | 通道外笔迹比 |
| C | C1 | 抖动比例 | ↓ | 笔迹到参考骨架的残差越界比例（三游戏统一） | ← | ← |
| C | C2 | 短笔段比例 | ↓ | 短段总长/所有笔段总长（三游戏统一） | ← | ← |
| C | C3 | 压力变异系数 | ↓ | σ(p)/μ(p)（三游戏统一） | ← | ← |

**方向说明**：
- ↑（越大越好）的特征在方案 B 中：z < 0（低于正常）时惩罚
- ↓（越小越好）的特征在方案 B 中：z > 0（高于正常）时惩罚

---

## 6. 测试所需文件清单

### 6.1 已有文件（项目内）

| 文件 | 路径 | 用途 |
|---|---|---|
| `1.txt` | `/project/1.txt` | 对称游戏轨迹样本（测试特征提取器） |
| `l1.txt` | `/project/l1.txt` | 方形迷宫轨迹样本 |
| `c1.txt` | `/project/c1.txt` | 圆形迷宫轨迹样本 |
| `sym_blue_mask.png` | 上传图片 | 对称标准答案掩码 |
| `sym_helper_mask_completed.png` | 上传图片 | 对称辅助线掩码（含对称轴） |

### 6.2 我提供的文件

| 文件 | 格式说明 |
|---|---|
| **所有样本轨迹**（建议每游戏 ≥ 10 份） | `data/samples/{sym,maze,circle}/{sample_id}.txt` |
| **对应用户绘制图片** | `data/samples/{sym,maze,circle}/{sample_id}.png`（仅用于 bbox 参照） |
| **标签文件** `labels.csv` | 见下方格式 |
| `maze_mask.png` | 模块 1 输出 |
| `circle_mask.png` | 模块 1 输出 |

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

**建议提供（用于质量检查）**：
- 你自己目视判断"画得好"和"画得差"各 1–2 份样本的 `.txt`，用于验证特征值方向是否符合直觉

---

## 7. 项目预期架构

```
project_root/
│
├── data/                              # 数据目录
│   ├── 34migong.png                   # 原始测评图
│   ├── 35duichen.png
│   ├── 36circle.png
│   ├── labels.csv                     # 已整理
│   └── samples/
│       ├── sym/                       # 已整理
│       │   ├── s1.txt
│       │   ├── s1.png
│       │   └── ...
│       ├── maze/
│       │   └── ...
│       └── circle/
│           └── ...  
│
├── shape/                             # 模块1（已完成，不修改）
│   ├── __init__.py
│   ├── final_shape_migong.py
│   ├── final_shape_sym.py
│   └── final_shape_circle.py
│
├── output_maze/shape_maze/            # 模块1输出（已生成，不修改）
│   └── maze_mask.png
├── output_sym/shape_sym/
│   ├── sym_blue_mask.png
│   └── sym_helper_mask_completed.png
├── output_circle/shape_circle/
│   └── circle_mask.png
│
├── pen/                               # 已有工具库（不修改）
│   └── analyze.py
│
├── sym_core/
│   ├── __init__.py
│   └── sym_analyze3.py      # 从项目根移动过来，不修改（作为方案A的基础 + 可复用函数来源）
│
├── features/                          # 阶段1-3（新建）
│   ├── stroke_utils.py                # C1/C2/C3 公共实现
│   ├── sym_feature_extractor.py       # 对称游戏 F1-F4
│   ├── maze_feature_extractor.py      # 迷宫游戏 F1-F4（方形+圆形共用）
│   ├── build_feature_matrix.py        # 汇总+归一化+乱画检测
│   ├── features_raw.csv               # 生成的原始特征矩阵
│   ├── features_normalized.csv        # 生成的归一化特征矩阵
│   └── normalization_params.json      # 归一化参数（μ̃, σ̃）
│
├── classifier/                        # 阶段4（新建）
│   └── classify.py                    # 方案A/B/C实现
│
├── experiments/                       # 阶段5（新建）
│   └── run_experiments.py
│
├── results/                           # 实验结果输出（新建，自动生成）
│   ├── main_results_table.csv
│   ├── dual_axis_scatter.png
│   ├── cross_game_table.csv
│   └── ablation_table.csv
│
├── shape.py
└── README.md                          # （后续更新）
```

---

## 8. 关键决策记录与待确认事项

### 8.1 已确认的设计决策

| 决策 | 说明 |
|---|---|
| 归一化参数估计方法 | 优先使用中位数+MAD（稳健估计），因正常样本仅 5–10 份 |
| 分类器选型 | 方案 C 使用 Logistic Regression，不使用 RF/XGBoost（小样本方差大） |


### 8.2 待确认事项（需要时在对话中确认）

| 编号 | 问题 | 影响 |
|---|---|---|
| Q3 | 方案 B 的阈值 (θ_F, θ_C) 如何确定？ | 建议在训练集上最大化 Sensitivity 后设定 |
| Q4 | 迷宫游戏的"通道内部"掩码生成是否需要人工辅助？ | 若 floodFill 自动提取通道失败（如入口处断开），可能需要手动标注入口像素 |
| Q5 | 对称游戏的线段闭合（原子模块 4）是否纳入正式特征体系？ | 若纳入，补充为 F2 的扩展；若不纳入，则在论文中说明 |
