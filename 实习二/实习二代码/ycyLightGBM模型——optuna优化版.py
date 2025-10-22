import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler

# ============================
# 1. 设置路径与加载数据
# ============================
input_path = r"D:\实习\地理大数据实习\实习二\数据\NYC_engineered1.csv"

print("🚀 正在加载特征工程后数据...")
if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ 文件不存在！请检查路径：\n{input_path}")

df = pd.read_csv(input_path, low_memory=False)
print(f"✅ 加载成功！共 {len(df)} 行\n")

# ============================
# 2. 定义特征列与目标变量
# ============================
features = [
    'distance_km',
    'pickup_minute',
    'pickup_time',
    'is_weekend',
    'is_rest',
    'Temp.',
    'Visibility',
    'events_weather',
    'Precip',
    'pickup_cluster',
    'dropoff_cluster',
    'pickup_pca0',
    'pickup_pca1',
    'dropoff_pca0',
    'dropoff_pca1',
    'pickup_hour',
    'pickup_dayofweek',
    'pickup_month',
    'passenger_count',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'ti_day',
    'ti_evening_peak',
    'ti_morning_peak',
    'ti_night'
]
target = 'trip_duration'

X = df[features]
y = np.log1p(df[target])  # 对数变换目标变量

# ============================
# 3. 划分训练集/验证集/测试集（60%:20%:20%）
# ============================
print("🔢 正在划分数据集（训练:验证:测试 = 60%:20%:20%）...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"📊 训练集大小：{len(X_train)}")
print(f"📊 验证集大小：{len(X_val)}")
print(f"📊 测试集大小：{len(X_test)}\n")

# ============================
# 4. 定义 Optuna 目标函数
# ============================
def objective(trial):
    # 定义超参数空间
    params = {
        'objective': 'regression_l1',      # L1损失（MAE）
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9),
        'verbose': -1
        # 'objective': 'regression_l1',        # L2 损失（RMSE）
        # 'metric': 'rmse',
        # 'n_estimators': 1000, #2000 1000
        # 'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        # 'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        # 'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        # 'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        # 'learning_rate': trial.suggest_categorical('learning_rate', [0.006, 0.008, 0.01, 0.014, 0.017, 0.02, 0.05]),
        # 'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20, 50]),
        # 'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
        # 'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        # 'cat_smooth': trial.suggest_int('cat_smooth', 1, 100),
        # 'random_state': 42,
        # 'verbose': -1
    }

    # 初始化模型
    model = lgb.LGBMRegressor(**params,n_jobs=-1)

    # 训练模型（启用早停）
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False)
        ],
        categorical_feature=None
    )

    # 在验证集上预测
    y_pred_log = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_log))

    return rmse  # 最小化 RMSE

# ============================
# 5. 启动 Optuna 优化
# ============================
print("🤖 正在使用 Optuna 优化超参数...")

# 创建研究对象（study）
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))

# 开始优化（最多运行 50 次试验）
study.optimize(objective, n_trials=50)

print("\n🎉 优化完成！最佳参数：")
best_params = study.best_params
for k, v in best_params.items():
    print(f"  • {k}: {v}")

# ============================
# 6. 使用最佳参数训练最终模型
# ============================
print("\n🏋️ 使用最佳参数训练最终模型...")

final_model = lgb.LGBMRegressor(**best_params)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=20, verbose=False)
    ],
    categorical_feature=None
)

# ============================
# 7. 在测试集上预测 & 评估性能
# ============================
print("\n🔮 正在测试集上进行预测...")
y_pred_log = final_model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # 逆变换
y_true = np.expm1(y_test.values)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"🏆 RMSE（均方根误差）：{rmse:.2f} 秒")
print(f"📌 平均行程时长：{np.mean(y_true):.2f} 秒 ({np.mean(y_true)/60:.2f} 分钟)")
print(f"📌 预测平均误差：±{rmse:.2f} 秒 ({rmse/60:.2f} 分钟)\n")

# ============================
# 绘制预测值 vs 真实值折线图（抽样展示）
# ============================
print("📈 正在绘制预测值 vs 真实值折线图...")

# 为避免图形过于密集，随机抽取 200 个样本进行可视化
np.random.seed(42)
sample_size = min(100, len(y_true))
indices = np.random.choice(len(y_true), size=sample_size, replace=False)

# 排序索引以便折线图更清晰（按真实值排序）
sorted_idx = np.argsort(y_true[indices])
x_plot = np.arange(sample_size)
y_true_plot = y_true[indices][sorted_idx]
y_pred_plot = y_pred[indices][sorted_idx]

plt.figure(figsize=(12, 6))
plt.plot(x_plot, y_true_plot, label='真实行程时长', color='blue', linewidth=1.5, alpha=0.8)
plt.plot(x_plot, y_pred_plot, label='预测行程时长', color='red', linestyle='--', linewidth=1.5, alpha=0.8)
plt.title('预测值 vs 真实值对比（测试集抽样）', fontsize=16, weight='bold')
plt.xlabel('样本序号（按真实值排序）')
plt.ylabel('行程时长（秒）')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Prediction_vs_True_.png", dpi=300)
plt.show()

# ============================
# 8. 绘制特征重要性条形图
# ============================
print("📊 正在绘制特征重要性图...")

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance,
    x='Importance',
    y='Feature',
    hue='Feature',
    legend=False,
    palette="viridis"
)
plt.title("LightGBM Feature Importance (Optimized)", fontsize=16, weight="bold")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("LightGBM Feature Importance_7",dpi=300)
plt.show()

# ============================
# 9. 输出最重要的前5个特征
# ============================
print("🔥 最重要的前5个特征：")
print(feature_importance.head())

print("\n🎉 Optuna + LightGBM 模型训练、优化、预测与评估完成！")