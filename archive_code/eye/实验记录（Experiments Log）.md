# 实验记录（Experiments Log）

> 记录所有分支实验目的、改动、状态，便于复现与汇报。

| 分支名                       | 模式   | 核心功能                  | 关键改动文件                                  | 状态   | 备注            |
| ------------------------- | ---- | --------------------- | --------------------------------------- | ---- | ------------- |
| `batch-refine-eval`       | 批量评估 | 随机抽样 50 次评估 refine 效果 | `main`, `analyze`                       | ✅ 保留 | 无绘图，用于统计指标    |
| `single-vis-interactive`  | 单例交互 | 单文件输入 + 可视化           | `main`, `trajectory-plotter`            | ✅ 保留 | 基础调试用         |
| `single-refine-highlight` | 单例交互 | refine 结果高亮显示         | `main`, `analyze`, `trajectory-plotter` | ✅ 保留 | 可视化 refine 区域 |

## 命名规范

- 模式：`batch`（批量）、`single`（单例）、`ablation`（消融）、`compare`（对比）
- 功能关键词：`refine`、`highlight`、`vis`、`eval` 等
- 分支格式：`[模式]-[功能]-[可选标识]`

## Commit 规范

- 格式：`类型: 描述` + `- 文件改动`
- 类型：`feat` / `refactor` / `fix` / `exp` / `config`
- 例子：feat：新增功能（如高亮）
  refactor：重构（如把 plotter 逻辑移入 main）
  fix：修复 bug
  exp：实验性改动（不确定是否保留）
  config：配置/参数调整




