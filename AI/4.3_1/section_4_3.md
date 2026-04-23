# 4.3 阶段3：圆形迷宫游戏特征提取器

> **说明**：本节为 v2.0 文档新增内容，填充至 `4.3 阶段3：圆形迷宫游戏特征提取器`。
> 其余各节内容不变。


---

## 4.3.0 圆形迷宫几何分析（设计依据）

通过对 `circle_mask.png` 的墙壁像素进行极坐标分析，得到如下**对实现有直接影响的几何事实**：

**圆心与同心环结构**（近似圆心 (cx=600, cy=811)，由 LSQ 拟合得 cx≈600, cy≈811, r_out≈417）：

| 环 | 墙壁半径 (px) | 用途 |
|---|---|---|
| 中心小圆 | ~36–45 | 装饰，无通行意义，需过滤（见步骤3） |
| 环 1–5 | ~99, 162, 225, 288, 351 | 迷宫走廊边界 |
| 外环 | ~405–423（中心≈417） | 最外侧边界，含两个入口/出口缺口 |

环间距 ≈ 63 px 均匀等距，墙厚 ≈ 4 px，**通道净宽 ≈ 59 px，半宽 ≈ 29 px**（与方形迷宫 28 px 接近，参数直接共用）。

**入口/出口缺口位置**（通过外环内侧角度扫描，r_scan = r_out - 5 检测 True-run）：
- **入口（左上粉色箭头）**：角度 ≈ 225°（math 坐标，y 向下），画布坐标约 (309, 520)
- **出口（右下粉色箭头）**：角度 ≈ 45°，画布坐标约 (891, 1102)
- 两个缺口各约 29° 弧宽，自动检测可靠

---

## 4.3.1 与方形迷宫的核心差异

| 维度 | 方形迷宫 | 圆形迷宫 | 影响哪些步骤 |
|---|---|---|---|
| 画布边界 | 封闭矩形框 | 有缺口的圆形 | canvas_mask 生成方式（步骤1） |
| 骨架拓扑 | 树（基本） | 含环状 cycle（同心环） | Dijkstra 处理方式（仍有效） |
| 理想路径线型 | 横纵直线段 | 弧线 + 放射线段 | C1 计算方案（见 4.3.3 C1 节） |
| 装饰元素 | 无 | 中心小圆（r≈40，形成封闭伪通道） | 需过滤 channel_mask 连通域 |
| 入口/出口检测 | 内框四边连通域 | 外环缺口（角度扫描 True-run） | `_detect_entry_exit_circle` |
| 路径骨架简化 | 可做 DP 简化 | **不做 DP 简化**（保留弧线精度） | F2 弧长采样精度 |

**F1、F2、F3、F4 的计算公式代码零改动**——这些特征都基于抽象的几何对象（`channel_mask`, `solution_channel_mask`, `solution_polyline`, `channel_dist`），只要 `maze_geometry.py` 的圆形分支输出相同结构，特征层完全复用。

---

## 4.3.2 通道几何模块：`maze_geometry.py`（v2.1 更新）

在 v2.0 基础上，`maze_geometry.py` 新增以下函数，并通过 `build_maze_geometry(game_type='circle')` 分派：

```
build_maze_geometry(mask_path, game_type='circle')
  │
  ├─ _fit_outer_circle_lsq(wall_mask)
  │    取最外 10% 的墙壁像素，代数最小二乘拟合圆 → cx, cy, r_out
  │
  ├─ _make_disk_canvas_mask(h, w, cx, cy, r_out+3)
  │    生成填充圆盘，替代方形的 floodFill（因外环有缺口，floodFill 会从缺口漏入）
  │
  ├─ _build_channel_mask(canvas_mask, wall_mask, r_wall=2)    ← 与方形完全相同
  │
  ├─ _filter_largest_cc(channel_mask_raw)
  │    仅保留最大连通域：中心小圆围出 ~4000px 的伪通道孤立区，过滤后只剩主环系统
  │
  ├─ skeletonize(channel_mask)                                ← 与方形完全相同
  ├─ _channel_distance_transform(channel_mask)                ← 与方形完全相同
  │
  ├─ _detect_entry_exit_circle(channel_mask, cx, cy, r_out, scan_r_offset=-5)
  │    在 r_scan = r_out - 5 处角度扫描 360°：
  │      非缺口角度 → 处于外环墙内 → channel_mask=False
  │      缺口角度   → 无外环墙   → channel_mask=True
  │    找 True-run（连续 True 的角度段），取两段中心点 → entry/exit 坐标
  │    按 x+y 升序：左上 = entry，右下 = exit
  │
  ├─ _nearest_skeleton_pixel(channel_skeleton, entry/exit)    ← 与方形完全相同
  ├─ _dijkstra_on_skeleton(skeleton, entry_anchor, exit_anchor) ← 与方形完全相同
  │    注：圆形骨架含 cycle，但 Dijkstra 自然返回弧长最短路径（本迷宫唯一路径）
  │
  └─ _dilate_to_solution_channel(sol_skel, channel_mask, r=28) ← 与方形完全相同
       → solution_channel_mask
```

**`build_maze_geometry` 函数签名新增 `game_type` 参数**（向后兼容，默认 `'maze'`）：

```python
def build_maze_geometry(
    mask_path: str,
    *,
    game_type: str = "maze",          # 新增：'maze' | 'circle'
    r_wall: int = 2,
    channel_half_width_px: int = 28,
    # 方形专用（circle 下忽略）
    entry_corner_size: int = 105,
    use_frame_ring_first: bool = False,
    # 圆形专用（maze 下忽略）
    lsq_outer_percentile: float = 90.0,
    scan_r_offset: float = -5.0,
    min_gap_deg: float = 10.0,
    # 通用（两种迷宫）
    entry_xy: Optional[Point] = None,
    exit_xy:  Optional[Point] = None,
) -> MazeGeometry
```

**`MazeGeometry` dataclass 新增字段**：

```python
circle_meta: Optional[Dict[str, Any]] = None
# 内容（仅 game_type='circle' 时非 None）：
# {
#   "outer_ring_cx": 599.9,
#   "outer_ring_cy": 811.2,
#   "outer_ring_r":  417.0,
#   "entry_angle_deg": 225.0,
#   "exit_angle_deg":  45.0,
#   "n_channel_cc_before_filter": 77,   # 过滤前连通域数，≥2 说明有伪通道
#   "lsq_outer_percentile": 90.0,
#   "scan_r_offset": -5.0
# }
```

---

## 4.3.3 特征定义：圆形与方形完全统一

F1–F4 公式代码**零改动**，仅输入几何对象来源不同（圆形通道 vs 方形通道）。

**F1**：解路径通道覆盖率（↑）
```
F1 = |user_mask ∩ solution_channel_mask| / |solution_channel_mask|
```
用户沿弧形解路径走得越完整，F1 越高；走错环不会提升 F1（岔道不在 solution_channel 内）。

**F2**：解路径采样点命中率（↑）
```
sample_pts = 沿 solution_polyline 按弧长每 40px 取一个点（对曲线同样有效）
F2 = Σ 1[dist_user(p) <= 12px] / len(sample_pts)
```
因 solution_polyline 本身就是弧长有序像素序列（不做 DP 简化），等弧长采样对直线和圆弧完全一致。

**F3**：无效书写加权距离（↓）
```
F3 = Σ channel_dist[user_mask & ¬channel_mask] / |channel_mask|
```
公式与方形完全相同。中心装饰小圆区域已从 channel_mask 中过滤，用户若误画在该区域也会被计入 F3 越界惩罚。

**F4**：通道外笔迹比（↓）
```
F4 = |user_mask ∩ ¬channel_mask| / |user_mask|
```
与方形完全相同。

**C1**：抖动比例（↓）
> ⚠️ **此处是方案 A 和方案 C 唯一的差异点。**

**方案 A（`maze_feature_extractor_planA.py`）**：

| game_type | C1 实现 | 理由 |
|---|---|---|
| `'circle'` | 骨架距离法（下方公式） | 圆弧路径无法被 Hough 直线检测器捕获 |
| `'maze'`   | Hough-on-user-mask（阶段2原方案）| 保持阶段2结果可比性 |

**方案 C（`maze_feature_extractor_planC.py`）**：

| game_type | C1 实现 | 理由 |
|---|---|---|
| `'circle'` | 骨架距离法 | 同上 |
| `'maze'`   | 骨架距离法（**与方案A不同**） | 三游戏方法论完全统一，与特征定义总表对齐 |

代价：选择方案 C 后，方形迷宫的 C1 需重新运行（与阶段 2 产出的 Hough-based 结果不兼容）。

**骨架距离法 C1 公式**（方案 A 的圆形分支 / 方案 C 的两种迷宫）：
```python
for 每个用户笔迹点 p：
    d = dist(p, 最近的 channel_skeleton 像素)
    if d <= channel_half_width (28 px)：  # 点在通道内，纳入统计
        total += 1
        if d > jitter_tol (3.0 px)：
            bad += 1
C1 = bad / total   （total=0 时返回 0.0）
```

实现：调用 `stroke_utils.compute_C1_skeleton_distance_ratio`（`stroke_utils_addition.py` 追加）。
使用 `scipy.spatial.cKDTree` 实现 O(N log M) 最近邻查询（N≈8000点，M≈8000骨架像素，每样本约0.1秒）。

**C2**：短笔段比例（↓）——与方形完全相同（直接复用 `compute_C2_short_stroke_ratio`）。

**C3**：压力变异系数（↓）——与方形完全相同（直接复用 `compute_C3_pressure_cv`）。

---

## 4.3.4 主入口：`maze_feature_extractor.py`（扩展 game_type 分派）

命令行调用格式（与方形迷宫完全对称）：

```bash
python features/maze_feature_extractor.py \
    --txt  data/samples/circle/{id}.txt \
    --png  data/samples/circle/{id}.png \
    --mask output_circle/shape_circle/circle_mask.png \
    --game circle \
    --out  output_circle/extract/{id}.json \
    --vis_dir output_circle/extract/vis_{id} \
    --sample_id {id}
```

Python API：

```python
from features.maze_feature_extractor import extract_maze_features

result = extract_maze_features(
    txt_path   = "data/samples/circle/c7.txt",
    png_path   = "data/samples/circle/c7.png",   # 全黑时自动 fallback 到 inner_bbox
    maze_mask_path = "output_circle/shape_circle/circle_mask.png",
    game_type  = "circle",                         # 触发圆形几何分支
    sample_id  = "c7",
    out_json_path = "output_circle/extract/c7.json",
    out_vis_dir   = "output_circle/extract/vis_c7",
)
```

**关于 `png_path` 全黑问题**：阶段 2 确认用户 PNG 在方形迷宫中有内容。
若圆形样本 PNG 同样可用，使用 `png_bbox` 模式；若 PNG 全黑（如 c7.png, c9.png），
自动 fallback 到 `inner_bbox_fallback` 模式（用 `circle_mask.png` 的墙壁外接矩形做 bbox 参照）。

---

## 4.3.5 JSON 输出格式

与方形迷宫 JSON 对齐，新增 `circle_geometry` 字段：

```json
{
  "sample_id": "c7",
  "game":      "circle",
  "F1": 0.xxxx, "F2": 0.xxxx, "F3": 0.xxxx, "F4": 0.xxxx,
  "C1": 0.xxxx, "C2": 0.xxxx, "C3": 0.xxxx,
  "meta": {
    "num_strokes": 1,
    "total_points": 8664,
    "canvas_hw": [1601, 1201],
    "inner_bbox": [185, 395, 1017, 1226],
    "align_mode": "inner_bbox_fallback",
    "channel_area": 488720,
    "channel_skeleton_length": 8240,
    "solution_path_length_px": ...,
    "solution_channel_area": ...,
    "entry_xy": [309, 520],
    "exit_xy":  [891, 1102],
    "num_skeleton_sample_pts": ...,
    "keypoints_hit": ...,
    "C1_method": "skeleton_distance",
    "C1_projected_points": ...,
    "C1_bad_points": ...,
    "C1_projected_fraction": ...,
    "num_user_hough_segments": 0,
    "C2_threshold": ...,
    "C2_n_short_strokes": ...,
    "C3_n_pressure_points": ...,
    "F3_detail": { "..." },
    "params": { "..." },
    "circle_geometry": {
      "outer_ring_cx": 599.9,
      "outer_ring_cy": 811.2,
      "outer_ring_r":  417.0,
      "entry_angle_deg": 225.0,
      "exit_angle_deg":  45.0,
      "n_channel_cc_before_filter": 77,
      "lsq_outer_percentile": 90.0,
      "scan_r_offset": -5.0
    }
  }
}
```

---

## 4.3.6 可视化输出

输出至 `output_circle/extract/vis_{id}/`：

1. **`{id}_channel_geometry.png`**
   通道（灰）+ 骨架（暗黄）+ 解路径通道（绿）+ 解路径骨架（蓝）+ 墙壁（白）+ 外环圆拟合（黄线）+ 圆心（青十字）+ 入口（绿圈）+ 出口（红圈）。
   → 肉眼验证 LSQ 拟合、入口/出口检测、解路径提取是否正确。

2. **`{id}_feature_overlay.png`**
   通道（灰）+ 解路径通道（绿）+ 用户笔迹（橙）+ F2 采样点（命中绿/未命中红）。
   → 验证坐标对齐和 F1/F2 计算是否合理。

3. **`{id}_C1_skeleton.png`**（两种方案均输出此图，方形迷宫仅方案C输出）
   用户笔迹（灰）+ 通道骨架（青，参考中心线）+ 抖动点（黄，d > jitter_tol 且在通道内）。
   → 验证 C1 骨架距离法是否合理识别抖动区域。

---

## 4.3.7 参数总表

| 参数 | 默认值 | 方形 | 圆形 | 说明 |
|---|---|---|---|---|
| `r_wall` | 2 | ✓ | ✓ | 墙壁膨胀半径 |
| `channel_half_width_C1` | 28 | ✓ | ✓ | C1 通道半宽（骨架距离法） |
| `jitter_tol` | 3.0 | ✓ | ✓ | C1 抖动容限（px） |
| `r_solution_channel` | 28 | ✓ | ✓ | 解路径骨架膨胀半径 |
| `sample_step` | 40 | ✓ | ✓ | F2 弧长采样间距（px） |
| `hit_radius` | 12 | ✓ | ✓ | F2 命中半径（px） |
| `C2_threshold_ratio` | 0.02 | ✓ | ✓ | C2 短笔段阈值比例 |
| `C3_trim_ends` | 3 | ✓ | ✓ | C3 裁剪笔段两端点数 |
| `entry_corner_size` | 105 | ✓ | — | 方形入口矩形大小 |
| `lsq_outer_percentile` | 90.0 | — | ✓ | LSQ 外圆拟合百分位 |
| `scan_r_offset` | -5.0 | — | ✓ | 入口扫描半径偏移（px） |
| `min_gap_deg` | 10.0 | — | ✓ | 最小缺口弧宽（度） |

---

## 4.3.8 文件修改清单

| 文件 | 动作 | 说明 |
|---|---|---|
| `features/maze_geometry.py` | **替换** | v2.1，新增 `_build_circle_geometry` 及相关函数，`build_maze_geometry` 加 `game_type` 参数 |
| `features/stroke_utils.py` | **追加** | 将 `stroke_utils_addition.py` 的内容添加到文件末尾（`compute_C1_skeleton_distance_ratio`） |
| `features/maze_feature_extractor.py` | **替换** | 选择方案A或方案C版本（见下方） |

**方案选择**：
- 选方案A → 使用 `maze_feature_extractor_planA.py` 替换
- 选方案C → 使用 `maze_feature_extractor_planC.py` 替换，并重跑所有方形迷宫样本的 C1

**同步修改（若选方案C）**：
- 4.2 节 C1 描述中 "Hough-on-user-mask" 改为"骨架距离法"
- 特征定义总表（第 5 节）C1 行的"方形迷宫"列更新为"骨架距离法（与圆形统一）"
