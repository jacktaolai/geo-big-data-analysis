import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

#字体配置
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False  #解决负号显示问题
df = pd.read_csv("E:/地理大数据分析/实习二/NYC.csv/NYC.csv")

print("=== 数据基本信息 ===")
print(f"数据形状（行×列）：{df.shape}")
print("\n数据前5行：")
print(df.head())

print("\n=== 数据类型与缺失值 ===")
print(df.info())

print("\n=== 数值型特征统计描述 ===")
print(df.describe())

print("\n=== 各列缺失值数量 ===")
print(df.isnull().sum())

#行程时长计算
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()

print("\n=== 行程时长基本统计 ===")
print(f"最小值：{df['trip_duration'].min()} 秒")
print(f"最大值：{df['trip_duration'].max()} 秒")
print(f"中位数：{df['trip_duration'].median()} 秒")

#数据清洗
print(f"\n清洗前数据量：{len(df)} 条")
df = df[(df['trip_duration'] > 0) & (df['trip_duration'] <= 3600)]
print(f"过滤时长异常后数据量：{len(df)} 条")

#计算距离和速度
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df['distance_km'] = haversine_distance(
    df['pickup_latitude'], df['pickup_longitude'],
    df['dropoff_latitude'], df['dropoff_longitude']
)
df['speed_kmh'] = df['distance_km'] / (df['trip_duration'] / 3600)
df = df[(df['speed_kmh'] > 0) & (df['speed_kmh'] <= 120)]
print(f"过滤速度异常后数据量：{len(df)} 条")

#过滤坐标异常
df = df[
    (df['pickup_longitude'].between(-74.2591, -73.7004)) &
    (df['pickup_latitude'].between(40.5011, 40.9155)) &
    (df['dropoff_longitude'].between(-74.2591, -73.7004)) &
    (df['dropoff_latitude'].between(40.5011, 40.9155))
]
print(f"过滤坐标异常后数据量：{len(df)} 条")
print(f"最终清洗后数据量：{len(df)} 条")

#特征工程
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
df['pickup_month'] = df['pickup_datetime'].dt.month
df['is_rush_hour'] = ((df['pickup_hour'].between(7, 9)) | 
                      (df['pickup_hour'].between(17, 19))).astype(int)
df['is_weekend'] = (df['pickup_dayofweek'] >= 5).astype(int)

df['lon_diff'] = abs(df['dropoff_longitude'] - df['pickup_longitude'])
df['lat_diff'] = abs(df['dropoff_latitude'] - df['pickup_latitude'])
df['is_manhattan_pickup'] = (
    df['pickup_longitude'].between(-74.0266, -73.9712) &
    df['pickup_latitude'].between(40.7033, 40.8756)
).astype(int)

features = [
    'distance_km', 'pickup_hour', 'pickup_dayofweek', 'pickup_month', 
    'is_rush_hour', 'is_weekend', 'lon_diff', 'lat_diff', 
    'is_manhattan_pickup', 'passenger_count'
]
target = 'trip_duration'

#模型训练
X = df[features]
y = np.log1p(df[target])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n训练集形状：{X_train.shape}，验证集形状：{X_val.shape}")

lgb_params = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 6,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

model = lgb.LGBMRegressor(**lgb_params)
print("\n开始训练模型...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
)
print("模型训练完成！")

#模型评估与可视化
y_val_pred_log = model.predict(X_val)
y_val_pred = np.expm1(y_val_pred_log)  #反变换得到预测值
y_val_true = np.expm1(y_val)  #反变换得到真实值

#计算RMSE
rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
print(f"\n验证集RMSE：{rmse:.2f} 秒")
print(f"验证集平均时长：{y_val_true.mean():.2f} 秒")
print(f"RMSE占比：{rmse / y_val_true.mean() * 100:.2f}%")

#预测值与真实值对比
plt.figure(figsize=(12, 6))
#取前100个样本绘制（避免线条过于拥挤）
sample_size = 100
x = range(min(sample_size, len(y_val_true)))  #x轴为样本索引
#绘制真实值折线
plt.plot(x, y_val_true[:sample_size], label='真实行程时长', 
         color='blue', linewidth=2, marker='o', markersize=4)
#绘制预测值折线
plt.plot(x, y_val_pred[:sample_size], label='预测行程时长', 
         color='red', linewidth=2, marker='x', markersize=4)
# 添加标签和标题
plt.xlabel('样本索引')
plt.ylabel('行程时长（秒）')
plt.title('预测值与真实值对比（折线图）')
plt.legend()  # 显示图例
plt.grid(alpha=0.3)
# 保存图片
plt.savefig('预测对比折线图', dpi=300, bbox_inches='tight')
plt.close()

#特征重要性
feature_importance = pd.DataFrame({
    '特征名称': features,
    '重要性得分': model.feature_importances_
}).sort_values('重要性得分', ascending=False)
print("\n特征重要性（降序）：")
print(feature_importance)
plt.figure(figsize=(12, 8))
sns.barplot(x='重要性得分', y='特征名称', data=feature_importance, palette='viridis')
plt.xlabel('重要性得分')
plt.ylabel('特征名称')
plt.title('特征重要性排名')
plt.grid(axis='x', alpha=0.3)
plt.savefig('特征重要性图', dpi=300, bbox_inches='tight')
plt.close()
print("\n代码执行完成！预测对比折线图保存为 '预测对比折线图'，特征重要性图保存为 '特征重要性图'")