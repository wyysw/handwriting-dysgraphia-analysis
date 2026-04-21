# Handwriting Dysgraphia Analysis

This repository contains the implementation for the undergraduate thesis:

**面向书写障碍儿童的手写交互数据评估算法设计与实现**

**Algorithm Design and Implementation for Evaluating Handwriting Interaction Data in Children with Dysgraphia**

**Technologies**: Python

---


# 模块1：游戏图形预处理模块（已完成）

## 一、概述

项目使用若干张标准化测评图纸（尺寸统一为 **1201 × 1601 像素**）作为儿童书写/绘图能力的评估素材，儿童在这些图纸上进行描摹、对称补画、走迷宫等任务，算法根据儿童笔迹与标准图形的偏差进行量化评估。

**"图形提取模块"** 完成了项目的 **第一阶段工作**：
将三张原始测评图中的 **标准答案参考结构（迷宫线条 / 对称辅助线）** 自动分割出来，生成干净的二值掩码，供后续的笔迹对齐、偏差测量、路径判定等算法直接使用。


## 二、目录结构

```
project_root/
├── data/
│   ├── 34migong.png           # 方形迷宫测评原图（1201×1601）
│   ├── 35duichen.png          # 对称补画测评原图（1201×1601）
│   └── 36circle.png           # 圆形迷宫测评原图（1201×1601）
├── shape/
│   ├── __init__.py
│   ├── final_shape_migong.py  # 方形迷宫提取
│   ├── final_shape_sym.py     # 对称图辅助线提取
│   └── final_shape_circle.py  # 圆形迷宫提取（新增）
├── output_maze/shape_maze/    # 方形迷宫输出
├── output_sym/shape_sym/      # 对称图输出
├── output_circle/shape_circle/# 圆形迷宫输出
└── shape.py                   # 统一调度脚本（执行并可视化）
```

所有输出掩码与原图保持 **相同的 1201 × 1601 尺寸**，背景像素为 0（黑），前景（被提取的线条）像素为 255（白）。

---


## 三、三个提取模块的职责与输出

### 3.1 方形迷宫提取 —— `final_shape_migong.py`

**输入**：`data/34migong.png`（方形迷宫图，四周带有小猪、房子等卡通装饰，迷宫被一个矩形粗边框包裹）

**主入口**：`extract_maze(image_path, out_dir)`

**处理流程**：

1. **矩形外框定位**：对灰度图做阈值反转 + 膨胀，再用 `keep_largest_rectangle_contour` 找到最大且形状近似矩形（4~8 边）的轮廓，得到迷宫的 ROI 矩形。
2. **线条提取**：CLAHE 增强后，用 **Blackhat 形态学变换** 作为主通道提取暗线，再用灰度阈值做弱线补充，融合得到初步线条掩码。
3. **角落装饰清理**：`clean_corner_decorations` 在右上角（0.88–1.0 宽, 0–0.09 高）与左下角（0–0.1555 宽, 0.91–1.0 高）两个固定子区域内，通过连通域的面积、长宽比、填充率等形状特征，区分"线条型"和"色块型"组件，只删除卡通装饰而保留线条。
4. **外框重建**：`rebuild_outer_border` 在 ROI 内重绘 13 像素粗的矩形外框，修复上一步误删的角落边框。
5. **孤立短线删除**：`remove_isolated_short_segments` 以长度 ≤ 10 像素为阈值清除噪点级短线段。
6. **回填全图**：ROI 内的掩码通过 `put_roi_back` 放回 1201×1601 画布。

**关键输出文件**（位于 `output_maze/shape_maze/`）：

| 文件 | 含义 | 后续算法使用 |
|---|---|---|
| `maze_mask.png` | **迷宫线条二值掩码**（1201×1601） | ✅ 主要输出，即"迷宫标准答案结构" |
| `maze_roi.png` | 迷宫区域彩色裁剪 | 调试查看 |
| `maze_only.png` | 仅保留线条的原图 | 调试查看 |
| `debug/*.png` | 各中间步骤图 | 调试查看 |

---

### 3.2 对称图辅助线提取 —— `final_shape_sym.py`

**⚠ 注意：这个模块提取的不是"对称图形本身"，而是"对称补画任务的辅助参考结构"。**

对称补画题型要求儿童根据一条虚线对称轴，把已给出的一半图形补画成完整的对称图形。原图中 **已给出的一半图形用蓝色折线绘制**，整个画面上还叠加了一套 **浅灰色的网格辅助线**、一条 **水平虚线对称轴** 和一个 **矩形外框**。本模块负责把这几类"参考结构"分别提取出来。

**输入**：`data/35duichen.png`

**主入口**：`extract_symmetry(image_path, out_dir, ignore_side_width=190, expected_width=1201, expected_height=1601, ignore_mode="white")`

**处理流程**：

1. **左右忽略区预处理**：测评图左右各 190 像素为题目说明区，用 `preprocess_ignore_side_regions` 将其涂白或设为透明，防止干扰后续提取。
2. **蓝色折线提取**：`extract_blue_polyline` 通过 HSV 阈值（H∈[90,140]）提取儿童/原图的蓝色绘制部分。
3. **外框提取**：`extract_outer_box` 用灰度反转阈值找到大矩形外框。
4. **原始网格 + 虚线提取**：`extract_grid_and_dashed` 在 ROI 内用 `inRange(190,245)` 提取浅灰色候选像素，再用水平/垂直形态学开运算分离出网格线，并用行积分峰值定位虚线。
5. **网格补全**：`complete_grid_inside_outer_box` 对网格做列和/行和分析，用 `cluster_positions` 聚类确定竖线/横线位置，再：
   - `remove_border_adjacent_lines`：删除与外框过近的首尾"伪辅助线"（阈值 0.75 × 正常间距）；
   - `insert_mid_axis_if_needed`：当发现某个相邻间距明显偏大（> 正常间距的 1.6 倍）时，在中间补一条中轴线。
6. **合并辅助线框**：把外框 + 补全网格 + 虚线合并为 `sym_helper_mask_completed.png`。

**关键输出文件**（位于 `output_sym/shape_sym/`）：

| 文件 | 含义 | 后续算法使用 |
|---|---|---|
| `sym_blue_mask.png` | **蓝色折线二值掩码**（1201×1601），即题目已给出的半边对称图形 | ✅ 用于判断儿童补画的一侧是否关于对称轴与之对应 |
| `sym_helper_mask_completed.png` | **辅助线框二值掩码**：外框 + 补全网格 + 虚线对称轴合并 | ✅ 用于定位对称轴、网格坐标系 |
| `sym_preprocessed_white.png` | 左右忽略区涂白后的彩色原图 | 调试查看 |
| `sym_outer_box_mask.png` | 外框单独掩码 | 调试 |
| `sym_grid_completed_mask.png` | 补全后的网格线单独掩码 | 调试 |
| `sym_dashed_mask.png` | 虚线对称轴单独掩码 | 调试 |
| `sym_grid_keypoints_preview_*.png` | 网格线位置检测结果 | 调试 |

**关键可调参数（模块内顶部常量）**：
- `IGNORE_SIDE_WIDTH = 190`：左右忽略区宽度
- `DARK_THRESH = 100`：深色外框阈值
- `GRID_GRAY_LOW/HIGH = 190/245`：浅灰网格阈值
- `BORDER_NEAR_LINE_RATIO = 0.75`：边缘伪辅助线判定比例
- `MID_AXIS_GAP_RATIO = 1.6`：中轴补线触发比例

---

### 3.3 圆形迷宫提取 —— `final_shape_circle.py`

**输入**：`data/36circle.png`（圆形迷宫图，白色矩形画布位于黑色/透明背景中央，迷宫线为黑色，出入口附近各有一个粉红色箭头）

**主入口**：`extract_maze(image_path, out_dir)`

**与方形迷宫的主要差异**：

| 处理步骤 | 方形迷宫 | 圆形迷宫 |
|---|---|---|
| ROI 定位 | 找最大的深色矩形轮廓 | **找最大白色连通域**（背景是黑/透明，画布是白色矩形） |
| 线条提取 | Blackhat + 灰度阈值 | 双阈值（强黑 + 弱灰），对比度足够时更简单稳定 |
| 干扰物剔除 | 角落坐标法 + 形状分析清除小猪/房子 | **彩色检测**（R/G/B 三通道差异 > 25）剔除彩色箭头 |
| 外框重建 | 重绘矩形外框 | **不需要**（圆形迷宫没有矩形外框）；反而要 `clean_canvas_border` 抹掉画布最外 2 像素的白色画布矩形边残留 |
| 短线清除 | 相同 | 相同 |

**处理流程**：

1. **透明通道处理**：若输入为 RGBA，先把 `alpha == 0` 的像素置为黑色。
2. **白色画布定位**：`locate_white_canvas`（阈值 240，闭运算核 15×15）找到最大白色连通域作为 ROI。
3. **迷宫线提取**：`extract_circle_maze_lines`
   - CLAHE 增强对比度；
   - 双阈值（120 强黑 + 180 弱灰）融合；
   - `build_colorful_mask`（RGB 三通道差异 > 25）识别彩色箭头，膨胀 5×5 后从线条掩码中扣除；
   - 轻微水平/垂直闭运算保证曲线连续；
   - 去除面积 < 15 的小噪点。
4. **画布边缘清理**：`clean_canvas_border` 抹掉 ROI 最外 2 像素（阈值法易把白色画布边识别为线）。
5. **孤立短线删除**：与方形迷宫共用同名逻辑，阈值 10 像素。
6. **回填全图**。

**关键输出文件**（位于 `output_circle/shape_circle/`）：

| 文件 | 含义 | 后续算法使用 |
|---|---|---|
| `circle_mask.png` | **圆形迷宫线条二值掩码**（1201×1601） | ✅ 主要输出 |
| `circle_roi.png` | 画布区域彩色裁剪 | 调试查看 |
| `circle_only.png` | 仅保留线条的原图 | 调试查看 |
| `debug/*.png` | 各中间步骤图 | 调试查看 |

---

## 四、统一调度脚本 —— `shape.py`

提供一个 `main()` 入口，依次：
1. 调用 `run_maze()` → 生成方形迷宫输出；
2. 调用 `run_sym()` → 生成对称图输出；
3. 调用 `run_circle()` → 生成圆形迷宫输出；
4. 用 matplotlib 并排展示：
   - 第一组：原始方形迷宫 vs 方形迷宫线条
   - 第二组：原始对称游戏 vs 对称图形（蓝色折线） vs 辅助线框
   - 第三组：原始圆形迷宫 vs 圆形迷宫线条


---

## 五、后续算法的对接规范


后续开发（笔迹质量评估、路径偏差度量、对称性打分等）可以 **直接读取上述 `*_mask.png` 文件** 作为"标准答案结构"，无需重新运行提取：

- **迷宫类任务**（方形、圆形）：把 `maze_mask.png` / `circle_mask.png` 作为墙壁约束，儿童笔迹越过这些像素即算作 **碰墙**；用距离变换可得到 **离墙距离** 作为稳度指标。
- **对称补画任务**：
  - `sym_blue_mask.png` 为题目已有的一半图形；
  - `sym_helper_mask_completed.png` 中的虚线位置即对称轴；
  - 儿童笔迹沿对称轴翻转后，与 `sym_blue_mask.png` 做 IoU / Chamfer 距离即可量化对称性。

所有掩码均为 **8-bit 单通道 PNG**，前景 255 / 背景 0，尺寸严格 1201×1601，与原始测评图像素级对齐，可直接按 `(x, y)` 坐标使用，无需额外仿射。

---

# 模块2：对称游戏模块



