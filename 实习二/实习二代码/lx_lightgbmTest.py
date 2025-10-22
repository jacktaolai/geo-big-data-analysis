import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

import json
from datetime import datetime

# 读取已清理的数据
data = pd.read_csv('cleaned_nyc_taxi_data.csv')

# ====== 特征工程：时间与地理特征 ======
# 时间特征（若存在 pickup_datetime 列）
extra_cols_created = []
if 'pickup_datetime' in data.columns:
    dt = pd.to_datetime(data['pickup_datetime'])
    data['pickup_hour'] = dt.dt.hour
    data['pickup_dow'] = dt.dt.dayofweek
    data['is_weekend'] = (dt.dt.weekday >= 5).astype(int)
    data['is_rush'] = data['pickup_hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    extra_cols_created += ['pickup_hour', 'pickup_dow', 'is_weekend', 'is_rush']

# 空间差分、曼哈顿距离与方位角
data['lon_diff'] = data['dropoff_longitude'] - data['pickup_longitude']
data['lat_diff'] = data['dropoff_latitude']  - data['pickup_latitude']
data['abs_lon_diff'] = data['lon_diff'].abs()
data['abs_lat_diff'] = data['lat_diff'].abs()

# 近似曼哈顿距离（km）：经度按纬度缩放
lat_scale = 111.0
lon_scale = np.cos(np.radians(data['pickup_latitude'])).abs() * 111.0
data['manhattan_km'] = data['abs_lon_diff'] * lon_scale + data['abs_lat_diff'] * lat_scale

# 方位角（0-360°）
phi1 = np.radians(data['pickup_latitude'])
phi2 = np.radians(data['dropoff_latitude'])
dlambda = np.radians(data['dropoff_longitude'] - data['pickup_longitude'])
y_b = np.sin(dlambda) * np.cos(phi2)
x_b = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
bearing = np.degrees(np.arctan2(y_b, x_b))
data['bearing'] = (bearing + 360) % 360

extra_cols_created += ['abs_lon_diff', 'abs_lat_diff', 'manhattan_km', 'bearing']

# 交互项：高峰期 × 距离
if 'is_rush' in data.columns:
    data['distance_rush'] = data['distance'] * data['is_rush']
    extra_cols_created.append('distance_rush')

# 选择特征列（动态按存在字段选择）
base_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'distance', 'calculated_speed']
candidate_cols = base_cols + extra_cols_created
feature_cols = [c for c in candidate_cols if c in data.columns]

X = data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y = data['trip_duration']

print(f"[Feature Engineering] 使用特征数: {len(feature_cols)}")
print("特征列:", feature_cols)

# 目标变量的对数变换，避免过拟合
y_log = np.log1p(y)

# 数据分割：训练集、验证集和测试集（60/20/20）
X_temp, X_test, y_temp, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# 设置 LightGBM (sklearn API) 超参数并训练
params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'max_depth': -1,
    'colsample_bytree': 0.9,   # 对应 feature_fraction
    'subsample': 0.8,          # 对应 bagging_fraction
    'subsample_freq': 5,       # 对应 bagging_freq
    'n_estimators': 10000,     # 用较大的上限，配合早停
    'verbose': -1
}
model = LGBMRegressor(**params)

# 设置回调函数（早停与日志）
callbacks = [
    early_stopping(stopping_rounds=50),
    log_evaluation(period=10)
]

# 训练：在验证集上早停
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='rmse',
    callbacks=callbacks
)

# 模型预测（在最佳迭代数上）
y_valid_pred_log = model.predict(X_valid, num_iteration=model.best_iteration_)
y_test_pred_log  = model.predict(X_test,  num_iteration=model.best_iteration_)

# 对数反变换，得到以“秒”为单位的预测时长
y_valid_pred = np.expm1(y_valid_pred_log)
y_test_pred  = np.expm1(y_test_pred_log)

# 评估模型表现：以秒为单位（对数反变换后）
valid_mae  = mean_absolute_error(np.expm1(y_valid), y_valid_pred)
valid_rmse = np.sqrt(mean_squared_error(np.expm1(y_valid), y_valid_pred))

test_mae   = mean_absolute_error(np.expm1(y_test),  y_test_pred)
test_rmse  = np.sqrt(mean_squared_error(np.expm1(y_test),  y_test_pred))

print(f"Validation MAE (s): {valid_mae}")
print(f"Validation RMSE (s): {valid_rmse}")
print(f"Test MAE (s): {test_mae}")
print(f"Test RMSE (s): {test_rmse}")

# 将RMSE结果保存为txt文件
with open('rmse_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Validation RMSE (seconds): {valid_rmse}\n")
    f.write(f"Test RMSE (seconds): {test_rmse}\n")

# 从训练好的模型中提取 feature_importances_ 并可视化
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# 计算 Top-5 特征及占比，并导出供 LLM 使用的上下文
fi_df = feature_importance_df.copy()
fi_df['Percent'] = fi_df['Importance'] / fi_df['Importance'].sum() * 100
top5 = fi_df.head(5)

llm_payload = {
    "generated_at": datetime.now().isoformat(),
    "target": "trip_duration (seconds)",
    "log_transform": "trained on log1p(target); metrics computed after expm1",
    "metrics": {
        "validation_rmse_seconds": float(valid_rmse),
        "test_rmse_seconds": float(test_rmse),
        "validation_mae_seconds": float(valid_mae),
        "test_mae_seconds": float(test_mae)
    },
    "top5_features": [
        {"feature": r.Feature, "importance": int(r.Importance), "percent": float(r.Percent)}
        for r in top5.itertuples(index=False)
    ],
    "notes": {
        "distance": "单位/计算方式（请补充）",
        "calculated_speed": "请确认是否由真实时长计算（若是则存在泄露，应替换为历史均速等）；单位/计算方式（请补充）",
        "lat_lon": "坐标系（如 WGS84）及城市区域范围（请补充）"
    }
}

# 导出 JSON
with open("llm_context.json", "w", encoding="utf-8") as f:
    json.dump(llm_payload, f, ensure_ascii=False, indent=2)

# 导出 Markdown
md_lines = [
    "# LLM 上下文：NYC Taxi 行程时长模型",
    f"- 生成时间: {llm_payload['generated_at']}",
    "- 目标: trip_duration（秒），训练采用 log1p，评估前已 expm1 还原",
    "## 评估指标",
    f"- Validation RMSE (s): {llm_payload['metrics']['validation_rmse_seconds']:.4f}",
    f"- Test RMSE (s): {llm_payload['metrics']['test_rmse_seconds']:.4f}",
    f"- Validation MAE (s): {llm_payload['metrics']['validation_mae_seconds']:.4f}",
    f"- Test MAE (s): {llm_payload['metrics']['test_mae_seconds']:.4f}",
    "## Top-5 特征重要性",
]
for item in llm_payload["top5_features"]:
    md_lines.append(f"- **{item['feature']}**: {item['importance']} （{item['percent']:.2f}%）")
md_lines.extend([
    "## 特征单位/备注（请按实际补充）",
    "- distance: 单位/计算方式",
    "- calculated_speed: 单位/计算方式；若由真实时长计算存在泄露，请替换",
    "- 经纬度: 坐标系与范围"
])
with open("llm_context.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print("已导出：llm_context.json 与 llm_context.md")

# 保存完整特征重要性为 CSV
fi_df.to_csv("feature_importances.csv", index=False, encoding="utf-8")
print("已导出：feature_importances.csv")

# 使用 matplotlib 绘制前 k 个最重要特征的条形图（k 为不超过 10 的可用特征数）
plt.figure(figsize=(10, 6))
k = min(10, len(feature_importance_df))
top_df = feature_importance_df.head(k)
plt.bar(top_df['Feature'], top_df['Importance'])
plt.xticks(rotation=45, ha='right')
plt.ylabel('Importance (split count)')
plt.title(f'LightGBM Feature Importances (Top {k} of {len(feature_importance_df)})')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()