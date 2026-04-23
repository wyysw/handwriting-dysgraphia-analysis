# 书写障碍分类实验——完整推进指导文档

## 1. 项目总览

### 1.1 研究目标

通过分析儿童在三款手写游戏中的绘制轨迹，判断该绘制是否由书写障碍（Dysgraphia）儿童产生（**二分类**问题）。强调**方法论贡献**：同时兼顾"游戏结果"（功能轴 F）和"游戏过程"（控制轴 C），设计跨游戏通用的轻量级分类器和特征框架，从特征抽取到落地分类（这个分类器不是只能训练单一游戏，而是对这三个游戏都行之有效；分类器只是为了表征我的工作从一个纯工程项的抽取特征变成了有落地的分类）。


### 1.2 三款游戏

| 游戏 | 任务描述 | 参考图像文件 |
|---|---|---|
| 对称（Symmetry） | 依据对称轴，将左侧/上半区蓝色图形镜像补画到右侧/下半区 | `sym_blue_mask.png`, `sym_helper_mask_completed.png` |
| 方形迷宫（Square Maze） | 在直线迷宫中从起点走到终点 | `maze_mask.png` |
| 圆形迷宫（Circle Maze） | 在环形路径迷宫中从起点走到终点 | `circle_mask.png` |

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



---

## 2. 预处理部分说明

### 2.1 游戏面板预处理（已完成）

**作用**：从三张原始测评图提取标准参考结构，生成二值掩码。



### 2.2 Pen模块（样本数据使用）

略


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

提取**原始指标值**（0–1 的比例或连续量），统一由阶段 3 的管道进行归一化。

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

形如output_sym/extract/s4.json

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


**输出**
JSON 输出中 `meta` 额外记录：
- `entry_xy`, `exit_xy`, `solution_path_length`（像素数）
- `channel_area`, `solution_channel_area`
- `num_skeleton_sample_pts`, `keypoints_hit`
- `num_user_hough_segments`, `C1_projected_fraction`
- `F3_detail`（与 sym 对齐，但 `n_cross_axis_pixels=0`、`cross_penalty_coef=0`）



### 4.3 阶段3：圆形迷宫游戏特征提取器

```
python features/maze_feature_extractor.py --txt data/samples/circle/{id}.txt --png data/samples/circle/{id}.png --mask output_circle/shape_circle/circle_mask.png --game circle --out output_circle/extract/{id}.json --vis_dir output_circle/extract/vis_{id} --sample_id {id}
```

---

#### 4.3.1 设计总原则

两迷宫复用**同一套提取器与特征定义**，通过 `game_type='circle'` 区分几何分支。所有差异集中在 `maze_geometry.py` 的几何构造步骤；**特征层 F1–F4 代码零改动，C2/C3 代码零改动**。C1 是唯一存在方案分歧的地方（见 4.3.5）。


---

#### 4.3.2 圆形迷宫几何概况（实测）

略

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
```

---

#### 4.3.6 参数

与轨迹无关

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


---


### 4.6 阶段6：实验总结与评估（暂不考虑）

#### 程序：`experiments/run_experiments.py`



---

