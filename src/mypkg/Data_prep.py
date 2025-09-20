import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'shared bikes\data\train.csv')

# 数据预处理
# 提取时间特征
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month

# 选择用于聚类的特征
features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 
'humidity', 'windspeed', 'hour', 'dayofweek', 'casual', 'registered']

# 数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])