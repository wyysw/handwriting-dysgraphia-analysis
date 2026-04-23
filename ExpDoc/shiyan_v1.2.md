# 书写障碍分类实验——完整推进指导文档

> **版本**： v1.2
> 阶段二基本完成

> **文档说明**：本文档是跨对话协作的主参考文档，整合实验设计与流程指导。
> 在每次新对话开始时将本文档连同相关代码一起提供给 AI 助手，即可无缝延续工作。特征指标定义在具体实现过程中允许修改，修改后本文档对应章节同步更新并标注。

---

## 目录

1. [项目总览]
2. [现有基础盘点]
3. [实验推进路线（按阶段）]
4. [各程序详细说明]
   - 4.1 [阶段 1：对称游戏特征提取器]
   - 4.2 [阶段 2：迷宫游戏特征提取器]
   - 4.3 [阶段 3：特征汇总与归一化管道]
   - 4.4 [阶段 4：三方案分类器]
   - 4.5 [阶段 5：实验评估脚本]
5. [特征定义参考表]
6. [测试所需文件清单]
7. [项目预期架构]
8. [关键决策记录与待确认事项]

---

## 1. 项目总览

### 1.1 研究目标

通过分析儿童在三款手写游戏中的绘制轨迹，判断该绘制是否由书写障碍儿童产生（**二分类**问题）。强调**方法论贡献**：同时兼顾"游戏结果"（功能轴 F）和"游戏过程"（控制轴 C），设计跨游戏通用的特征框架，验证双轴联合判别优于单轴打分。

### 1.2 三款游戏

| 游戏 | 任务描述 | 参考图像文件 |
|---|---|---|
| 对称（Symmetry） | 依据对称轴，将左侧/上半区蓝色图形镜像补画到右侧/下半区 | `sym_blue_mask.png`, `sym_helper_mask.png` |
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
| `shape/final_shape_sym.py` | `data/raw/35duichen.png` | `sym_blue_mask.png`（蓝色半边图形）, `sym_helper_mask.png`（网格+对称轴+外框） |
| `shape/final_shape_migong.py` | `data/raw/34migong.png` | `maze_mask.png`（迷宫墙壁线条） |
| `shape/final_shape_circle.py` | `data/raw/36circle.png` | `circle_mask.png`（圆形迷宫墙壁线条） |
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


### 2.3 模块 3：方形迷宫游戏评估（已基本完成，待调优）

方形迷宫特征提取已基本完成，详见阶段3的描述。

现在分类器和圆形迷宫特征提取还没有实现。其中分类器逻辑集成在 `classify.py` 中，不作为独立模块。

---

## 3. 实验推进路线（按阶段）

```
阶段 1   对称游戏特征提取器（重构/扩展 sym_analyze3.py）
   ↓
阶段 2   迷宫游戏特征提取器（方形）
   ↓
阶段 3   迷宫游戏特征提取器（圆形，与方形迷宫共用基类）（待完成）
   ↓
阶段 4   特征汇总 + 归一化管道（输出标准化特征矩阵 CSV）（待确认）
   ↓
阶段 5   三方案分类器（方案 A 加权打分 / 方案 B 双轴偏离 / 方案 C 机器学习）（待修改）
   ↓
阶段 6   实验评估（方法待定）
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
- `helper_mask_path`：`sym_helper_mask.png`（含对称轴和网格）
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

### 4.2 阶段 2：方形迷宫游戏特征提取器

迷宫游戏的 F1–F4 依赖通道几何，与对称游戏算法不同；
C1–C3 的**定义与对称游戏保持一致**，应提炼为公共库复用。
 
以下设计主要考虑方形迷宫。

坐标对齐与阶段1完全一致：`map_trajectory_strokes_using_reference_bbox(txt, user_png, canvas_hw)`。这一步在公共库里以 `map_strokes_to_canvas` 的形式存在，但内部实现就是照搬阶段1。

#### 4.3.1 通道几何框架模块（阶段2最核心的新工作）

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

#### 4.3.2 通道几何解路径子模块：`solve_maze_path(channel_mask, wall_mask)`

**步骤 1：自动检测入口 / 出口**

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

#### 4.3.3 主要特征

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



#### 4.3.4 程序 A（公共库）：`features/stroke_utils.py`

**目的**：把sym_feature_extractor里已经成熟、与具体游戏无关的笔段处理逻辑（C1/C2/C3、笔段渲染、bbox 映射）抽出来，供对称/迷宫多游戏共用。不引入新算法。


**主要函数**：
| 公共函数 | 从哪儿搬 | 用途 |
|---|---|---|
| `load_strokes_with_pressure(txt)` | `sym_feature_extractor._load_strokes_with_pressure` | 读 txt，返回含 pressure 的笔段 dict 列表 |
| `map_strokes_to_canvas(strokes, png_path, canvas_hw)` | `sym_analyze3.map_trajectory_strokes_using_reference_bbox` | bbox 映射 |
| `render_strokes_to_mask(strokes_mapped, canvas_hw, thickness)` | `sym_analyze3.render_trajectory_using_reference_bbox` 的内部渲染部分 | 笔段渲染 |
| `extract_segments_from_hough(mask, **hough_params)` | `sym_analyze3._extract_target_segments_from_hough` | Hough→共线合并（对称和迷宫复用） |
| `compute_C1_jitter_ratio(strokes, segments, …)` | `sym_feature_extractor.compute_C1_jitter_ratio` | C1 |
| `compute_C2_short_stroke_ratio(strokes, canvas_hw, …)` | 同上 | C2 |
| `compute_C3_pressure_cv(strokes_wp, …)` | 同上 | C3 |


---

#### 4.3.5 程序 B：`features/maze_feature_extractor.py`

**目的**：只设计实现迷宫特有的 F1–F4。C1/C2/C3 调用公共库。

**设计说明**：
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

**主要函数**：

C1/C2/C3 直接从 `stroke_utils.py` 调用，迷宫的 C1 参考骨架 = `channel_skeleton`。

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

**输入**：
- `txt_path`：用户轨迹文件（l3.txt）
- `png_path`：用户绘制图片（bbox 参照）（l3.png）
- `maze_mask_path`：`maze_mask.png` 或 `circle_mask.png`
- `game_type`：字符串标识
- maze_mask.png外框闭合，入口在框内右上角，出口在框内左下角。


**预期结果**：与对称游戏类似，正常样本各特征值合理，障碍样本出现偏差。
JSON 结构与对称游戏对齐：
```json
{
  "sample_id": "l3",
  "game": "maze",
  "F1": ..., "F2": ..., "F3": ..., "F4": ...,
  "C1": ..., "C2": ..., "C3": ...,
  "meta": {
    "num_strokes": 1,
    "total_points": 13110,
    "canvas_hw": [1601, 1201],
    "channel_area": ...,
    "skeleton_length": ...,
    "num_skeleton_keypoints": ...,
    "keypoints_hit": ...,
    "user_bbox": [...],
    "params": {"sample_step": 20, "hit_radius": 8, "r_wall": 2, "jitter_tol": 3.0, "channel_half_width": 20, "C2_thr_ratio": 0.02, "C3_trim_ends": 3}
  }
}
```
JSON 输出中 `meta` 额外记录：
- `entry_xy`, `exit_xy`, `solution_path_length`（像素数）
- `channel_area`, `solution_channel_area`
- `num_skeleton_sample_pts`, `keypoints_hit`
- `num_user_hough_segments`, `C1_projected_fraction`
- `F3_detail`（与 sym 对齐，但 `n_cross_axis_pixels=0`、`cross_penalty_coef=0`）


**测试方法**：
1. 对 `l1.txt`（方形迷宫样本）运行
2. 生成通道掩码可视化图，确认 `channel_mask` 正确覆盖了可行走区域（而不是墙壁）
3. 叠加用户笔迹与通道，目视确认 F4 的分子（通道外像素）符合直觉

**可视化建议：**

1. `channel_geometry.png` — 全通道+骨架+检测到的入口出口+解路径（红色高亮）。肉眼确认解路径提取正确。
2. `feature_overlay.png` — 全通道(灰)+解路径通道(浅绿)+用户笔迹(蓝)+采样点(绿命中/红未中)。
3. `C1_hough_overlay.png` — 用户笔迹(灰)+识别到的 Hough 线段(红)+被判为抖动的点(黄)。用来目视验证 C1 是否合理。


---

### 4.3 阶段 3：特征汇总与归一化管道

#### 程序：`features/build_feature_matrix.py`

**目的**：将所有样本的原始特征 JSON 汇总成一个带标签的 CSV，并为每个游戏估计归一化参数 (μ, σ)，输出归一化后的特征矩阵，供分类器直接读取。

**功能**：
1. 批量读取 `data/raw/{game_type}/` 下所有样本，调用对应特征提取器，生成/读取 JSON
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
- `data/raw/{game_type}/*.txt`（全部样本轨迹）
- `data/raw/labels.csv`
- 模块 1 输出的各掩码文件
- （可选）已有的特征 JSON 缓存，避免重复计算

**输出**：
- `features/features_raw.csv`：原始特征矩阵（含 scribble 标记）
- `features/features_normalized.csv`：归一化后矩阵（已剔除乱画样本）
- `features/normalization_params.json`：每游戏、每特征的 (μ̃, σ̃) 值（部署时仅需此文件）
- `features/scribble_log.csv`：被乱画检测剔除的样本列表

**预期结果**：
- 正常样本归一化后特征值中心在 0 附近，障碍样本应在"差的方向"有明显偏移
- 各个游戏的归一化样本在特征空间中可以直接混合比较

---

### 4.4 阶段 4：三方案分类器


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

---

### 4.5 阶段 5：实验评估脚本（暂不考虑这一部分，等之前的阶段完成后再修改）

#### 程序：`experiments/run_experiments.py`

**目的**：运行所有对比实验，输出结果表格和可视化图。

**实验设计**：

**实验 1：主结果（Leave-One-Sample-Out CV）**
```
对 features_normalized.csv 中每个样本 i：
  训练集 = 其余所有样本
  重新用训练集的正常样本估计 (μ̃, σ̃)（防止测试集信息泄漏）
  对训练集归一化后，拟合分类器
  对样本 i 用同一 (μ̃, σ̃) 归一化后预测
汇总所有预测 → 计算 Accuracy, Sensitivity, Specificity, AUC
```
三种方案各跑一次，输出对比表。

**实验 2：分游戏拆解**
将主结果按 game 字段拆开，分别统计三游戏的指标，检验统一模型是否在三游戏上表现均衡。

**实验 3：跨游戏泛化**
```
3 组实验（循环 held_out_game ∈ {sym, maze, circle}）：
  训练集 = 另外两个游戏的全部样本（用各自的 (μ̃, σ̃) 归一化）
  测试集 = held_out_game 的全部样本（用该游戏的 (μ̃, σ̃) 归一化）
  分类器训练于合并训练集 → 测试于 held_out 测试集
```
此实验是方法论亮点，验证特征框架的跨游戏可迁移性。

**实验 4：消融实验**（暂不考虑）
```
消融 1：仅使用 F 轴（F1-F4）vs 仅使用 C 轴（C1-C3）vs F+C 联合
消融 2：有/无按游戏归一化（不归一化直接合并 vs 归一化后合并）
（均基于方案 B，使用 LOO-CV）
```

**输出**：
- `results/main_results_table.csv`：三方案在四个实验上的 Accuracy/Sensitivity/Specificity/AUC
- `results/dual_axis_scatter.png`：方案 B 的 (F_score, C_score) 散点图，按游戏和标签着色
- `results/cross_game_table.csv`：跨游戏泛化实验结果
- `results/ablation_table.csv`：消融实验结果
- `results/feature_importance.png`：方案 C 的逻辑回归系数可视化

**评估指标**（重要性排序）：
1. **Sensitivity（召回率，即真阳性率）**：临床场景漏诊代价最高
2. **AUC**：综合评估判别能力
3. **Specificity（真阴性率）**：避免过多误报
4. **Accuracy**：整体准确率

---


## 5. 特征定义参考表



| 轴 | ID | 名称 | 方向 | 对称游戏 | 方形迷宫 | 圆形迷宫 |
|---|---|---|---|---|---|---|
| F | F1 | 完成度 | ↑（越大越好） | 翻转容差 F1（IoU-like） | 通道覆盖率 | 环形通道覆盖率 |
| F | F2 | 关键结构对齐 | ↑ | 网格交点命中率 | 骨架采样点命中率 | 弧度采样点命中率 |
| F | F3 | 无效书写 | ↓（越小越好） | 越界像素加权距离/标准面积 | 同左 | 同左 |
| F | F4 | 路径偏离比 | ↓ | 期望图形外笔迹比 | 通道外笔迹比 | 通道外笔迹比 |
| C | C1 | 抖动比例 | ↓ | 笔迹到参考骨架的残差越界比例（三游戏统一） | ← | ← |
| C | C2 | 短笔段比例 | ↓ | 短段总长/所有笔段总长（三游戏统一） | ← | ← |
| C | C3 | 压力变异系数 | ↓ | σ(p)/μ(p)（三游戏统一） | ← | ← |


---

## 6. 测试所需文件清单

| 文件 | 格式说明 |
|---|---|
| **所有样本轨迹**（建议每游戏 ≥ 10 份） | `data/raw/{sym,maze,circle}/{sample_id}.txt` |
| **对应用户绘制图片** | `data/raw/{sym,maze,circle}/{sample_id}.png`（仅用于 bbox 参照） |
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
│   └── sym_helper_mask.png
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
└── README.md         # （后续更新）
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
