# shap_m3_visualize.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt

# -------------------------------
# 1. 读取特征 CSV
# -------------------------------
csv_path = 'data/feature/all.csv'
df = pd.read_csv(csv_path)

FEATURE_NAMES = ['F1', 'F2', 'F3', 'F4', 'C1', 'C2', 'C3']

# 样本信息
sample_ids = df['sample_id'].values
games = df['game'].values
y = df['label'].values
X_raw = df[FEATURE_NAMES].values

# -------------------------------
# 2. 简单归一化 (Robust z-score)
# -------------------------------
# 使用全部 label=0 样本估计 median/MAD
X_z = X_raw.copy()
for i, f in enumerate(FEATURE_NAMES):
    ref_vals = X_raw[y == 0, i]
    med = np.median(ref_vals)
    mad = np.median(np.abs(ref_vals - med))
    scale = max(1.4826 * mad, 1e-3)
    # 统一方向：z = (x - med) / scale
    X_z[:, i] = (X_raw[:, i] - med) / scale

# -------------------------------
# 3. 训练 M3 (L2 Logistic Regression)
# -------------------------------
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    class_weight='balanced',
    solver='lbfgs',
    max_iter=1000
)
model.fit(X_z, y)

# -------------------------------
# 4. 创建 SHAP Explainer
# -------------------------------
explainer = shap.LinearExplainer(model, X_z, feature_perturbation="interventional")

# -------------------------------
# 5. 选择样本做可视化
# -------------------------------
sample_idx = 0  # 改成你想查看的样本索引
X_sample = X_z[sample_idx:sample_idx+1]
sample_id = sample_ids[sample_idx]

shap_values = explainer.shap_values(X_sample)

# -------------------------------
# 6. 绘制 SHAP Force Plot
# -------------------------------
print(f"SHAP visualization for sample_id={sample_id}")
shap.initjs()
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_sample[0],
    feature_names=FEATURE_NAMES,
    matplotlib=True
)
plt.show()

# -------------------------------
# 可选: 绘制 Summary Plot (全局)
# -------------------------------
shap_values_all = explainer.shap_values(X_z)
shap.summary_plot(shap_values_all, X_z, feature_names=FEATURE_NAMES)
