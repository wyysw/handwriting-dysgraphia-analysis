# 4.3 阶段3：圆形迷宫游戏特征提取器

命令行调用：
```
python features/maze_feature_extractor.py \
  --txt data/samples/circle/{id}.txt \
  --png data/samples/circle/{id}.png \
  --mask output_circle/shape_circle/circle_mask.png \
  --game circle \
  --out output_circle/extract/{id}.json \
  --vis_dir output_circle/extract/vis_{id} \
  --sample_id {id}
```

---

## 4.3.1 设计总原则

圆形迷宫与方形迷宫复用**同一套提取器与特征定义**，通过 `game_type='circle'` 区分几何分支。所有差异集中在 `maze_geometry.py` 的几何构造步骤；**特征层 F1–F4 代码零改动，C2/C3 代码零改动**。C1 是唯一存在方案分歧的地方（见 4.3.5）。

与方形迷宫相比，圆形迷宫的核心差异如下表：

| 方面 | 方形迷宫 | 圆形迷宫 |
|---|---|---|
| 画布边界 | 矩形外框（封闭） | 圆形外环（有两个缺口） |
| `canvas_mask` 构造 | `floodFill` 从四角灌入 | 最小二乘拟合外环圆 → 填充圆盘 |
| 拓扑 | 基本为树 | 含环状结构（同心环本身是 cycle） |
| 入口/出口位置 | 右上 / 左下角附近 | 外环上的两个缺口 |
| 额外清洗步骤 | 无 | 连通域过滤（去除中心装饰小圆的孤立 blob） |
| `solution_polyline` | 可 DP 简化 | **不做 DP 简化**（避免圆弧被折线粗化） |
| 通道半宽（实测） | 28 px | 29 px（取 28 复用同一默认值） |

---

## 4.3.2 圆形迷宫几何概况（实测）

以下参数由对 `circle_mask.png` 的分析得出，用于指导实现与参数设置：

| 参数 | 值 | 备注 |
|---|---|---|
| 画布尺寸 | 1601 × 1201 px | 与方形迷宫、对称游戏一致 |
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

## 4.3.3 程序（扩展）：`features/maze_geometry.py` 圆形分支

`build_maze_geometry` 增加 `game_type` 参数，`game_type='circle'` 时调用新增的 `_build_circle_geometry`。`MazeGeometry` 数据类新增三个可选字段记录圆形专属元信息。

### 七步流程

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

### 步骤详述

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

### MazeGeometry 数据类新增字段

```python
# 在 MazeGeometry dataclass 末尾新增三个可选字段：
circle_center_xy: Optional[Tuple[int, int]] = None    # 圆心坐标（圆形时有值）
outer_ring_radius: Optional[float] = None              # 外环半径（圆形时有值）
num_channel_components_before_filter: Optional[int] = None  # 过滤前连通域数（诊断用）
```

方形迷宫调用时三个字段均为 `None`，对现有代码无影响。

### `build_maze_geometry` 接口修改

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

## 4.3.4 特征定义

F1–F4 的公式、代码与方形完全相同，直接复用 `maze_feature_extractor.py` 中的四个函数。下表仅列出关键行为：

| 特征 | 公式 | 圆形适用性说明 |
|---|---|---|
| F1 | `\|user ∩ sol_ch\| / \|sol_ch\|` | 无变化，sol_ch 来自圆形解路径 |
| F2 | 沿 solution_polyline 按弧长 40 px 采样，命中率 | 天然适用于曲线 polyline |
| F3 | `Σ channel_dist[user & ¬ch] / \|ch\|` | 无变化 |
| F4 | `\|user ∩ ¬ch\| / \|user\|` | 无变化 |
| C2 | 短笔段总长 / 所有笔段总长 | 无变化，调用 `stroke_utils` |
| C3 | σ(pressure) / μ(pressure) | 无变化，调用 `stroke_utils` |

---

## 4.3.5 C1：运动控制能力（↓）— 方案说明

这是阶段 3 唯一存在方案选择的特征。

### 背景：为何圆形无法沿用方形 Hough 方案

方形迷宫的 C1 实现"Hough on user mask"的前提是：**理想路径是直线段**，Hough 能在用户笔迹上检测出这些直线，再测点到线的残差。圆形迷宫的理想路径包含**大量弧线段**，Hough 无法检测弧线，若直接照搬则 C1 只评估了放射线段的抖动，系统性遗漏占主体的弧线部分。

### 两种方案

**方案 A（代码文件实现的版本）**：圆形使用骨架距离残差法，方形维持 Hough-on-user 不变。

| | 方案 A | 方案 C |
|---|---|---|
| 方形 C1 | Hough-on-user（现有代码，不改动）| 骨架距离残差法（**改动方形**）|
| 圆形 C1 | 骨架距离残差法（新增）| 骨架距离残差法（同方案 A）|
| 对称 C1 | Hough-on-target（阶段1代码不改动）| ← 理论上也应改，但对称游戏骨架不同，暂不讨论 |
| 方法论一致性 | 圆形与阶段1特征表定义一致；方形仍是 Hough | 方形+圆形完全统一（迷宫两种游戏）|
| 代码改动量 | 最小（仅新增函数） | 新增函数 + 修改方形调用处一行 |

### 骨架距离残差法实现

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

### 方案 A → 方案 C 的修改方法

在 `maze_feature_extractor.py` 的 C1 计算段（`# ---------- 6. C1 ----------`），将：
```python
if game_type == "circle":
    C1, c1_bad, c1_total = compute_C1_jitter_ratio_skeleton(...)
    c1_method = "skeleton_dist"
else:
    user_hough_segments = extract_segments_from_hough(...)
    C1, c1_bad, c1_total = compute_C1_jitter_ratio(...)
    c1_method = "hough_user"
```
改为：
```python
# 方案 C：两种迷宫统一使用骨架距离残差法
C1, c1_bad, c1_total = compute_C1_jitter_ratio_skeleton(
    mapped_strokes, geom.channel_skeleton,
    jitter_tol=jitter_tol, channel_half_width=channel_half_width_C1,
)
user_hough_segments = []   # 占位（可视化中仅在 hough 模式下使用）
c1_method = "skeleton_dist"
```

此外将方案 C 对应的可视化从 `C1_hough.png` 改为 `C1_skeleton.png`（见 4.3.7）。

---

## 4.3.6 主入口新增参数

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

---

## 4.3.7 输入 / 输出

### 输入

与方形迷宫完全一致：
- `txt_path`：轨迹文件（前 3 行为文件头，跳过）
- `png_path`：用户绘制 PNG（仅用于 bbox 坐标参照）；若为纯黑/不存在，自动 fallback 到 `inner_bbox_fallback`（用圆形外接矩形 bbox 对齐）
- `maze_mask_path`：`circle_mask.png`（前景=墙壁）
- `game_type='circle'`

### 输出（JSON）

主字段 `F1–C3` 格式与方形完全一致。`meta` 新增圆形专属字段：

```json
{
  "sample_id": "c7",
  "game": "circle",
  "F1": 0.xxxx,  "F2": 0.xxxx,  "F3": 0.xxxx,  "F4": 0.xxxx,
  "C1": 0.xxxx,  "C2": 0.xxxx,  "C3": 0.xxxx,
  "meta": {
    "num_strokes": 1,
    "total_points": 8855,
    "canvas_hw": [1601, 1201],
    "align_mode": "png_bbox",
    "channel_area": ...,
    "solution_path_length_px": ...,
    "entry_xy": [315, 522],
    "exit_xy": [892, 1082],
    "num_skeleton_sample_pts": ...,
    "keypoints_hit": ...,
    "C1_projected_points": ...,
    "C1_bad_points": ...,
    "C1_projected_fraction": ...,
    "C1_method": "skeleton_dist",   ← 方案 A/C 均为此值；方形 Hough 时为 "hough_user"
    "C2_threshold": ...,
    "C3_n_pressure_points": ...,
    "F3_detail": { ... },
    "circle_center_xy": [598, 812],          ← 圆形专属
    "outer_ring_radius": 413.5,              ← 圆形专属
    "num_channel_components_before_filter": 2, ← 圆形专属（应为 2：主环+中心伪通道）
    "params": {
      "sample_step": 40.0,
      "hit_radius": 12.0,
      "jitter_tol": 3.0,
      "channel_half_width_C1": 28.0,
      "C2_threshold_ratio": 0.02,
      "C3_trim_ends": 3,
      "r_wall": 2,
      "r_solution_channel": 28,
      "channel_dilate_no_dp": true,          ← 圆形专属：标记未做 DP 简化
      "circle_scan_entry": false
    }
  }
}
```

### 可视化输出（三张图）

| 图文件名 | 内容 | 圆形与方形的差异 |
|---|---|---|
| `{id}_channel_geometry.png` | 全通道+骨架+入口出口+解路径（红色高亮） | 骨架为环状结构，需检查中心 blob 是否已被过滤 |
| `{id}_feature_overlay.png` | 全通道（灰）+解通道（浅绿）+用户笔迹（蓝）+F2采样点 | 无差异，弧线 polyline 的采样点应均匀分布于环上 |
| `{id}_C1_skeleton.png` | 用户笔迹（灰）+ channel_skeleton（红）+ 抖动点（黄） | 替代方形的 `C1_hough.png`；显示骨架而非 Hough 线段 |

---

## 4.3.8 参数总表（圆形 vs 方形）

| 参数 | 方形值 | 圆形值 | 是否在代码中区分 |
|---|---|---|---|
| `r_wall` | 2 | 2 | 否（统一） |
| `r_solution_channel`（膨胀半宽）| 28 | 28 | 否（统一） |
| `sample_step`（F2）| 40.0 | 40.0 | 否 |
| `hit_radius`（F2）| 12.0 | 12.0 | 否 |
| `jitter_tol`（C1）| 3.0 | 3.0 | 否 |
| `channel_half_width_C1` | 20.0 | 28.0 | **是**（`game_type` 分派时设置默认值）|
| C1 方法 | Hough-on-user（方案A）/ 骨架距离（方案C）| 骨架距离 | **是** |
| DP 简化 | 有 | **无** | **是** |
| `entry_corner_size` | 105 | 不适用 | **是** |
| `circle_center` | 不适用 | (598, 812) | **是** |
| `outer_ring_radius` | 不适用 | ≈ 413 | **是** |
| `entry_xy` / `exit_xy` | 自动检测 | (315, 522) / (892, 1082) | **是** |

---

## 4.3.9 预期结果（供调参参考）

对正常样本（如 c9，label=0）：
- F1 > 0.55：用户笔迹覆盖超过一半的解路径通道
- F2 > 0.65：解路径上超过 2/3 的采样点被笔迹经过
- F3 < 0.10：越出通道的加权惩罚较小
- F4 < 0.20：超过 80% 的笔迹在通道内
- C1 < 0.25、C2 < 0.15、C3 < 0.5

对障碍样本（如 c7，label=1）应在上述一个或多个特征上出现明显偏差。
