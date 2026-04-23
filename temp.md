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
