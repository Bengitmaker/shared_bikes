from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from Model_Kmeans import df,k
import plt_utils

# 可视化聚类结果
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
fig.suptitle('不同聚类的特征分析')

# 按小时分布
hourly_distribution = df.groupby(['cluster', 'hour']).size().unstack(fill_value=0)
for i in range(k):
    if i < len(hourly_distribution):
        axes[0, 0].plot(hourly_distribution.columns, hourly_distribution.iloc[i], 
                       marker='o', label=f'聚类 {i}')
axes[0, 0].set_xlabel('小时')
axes[0, 0].set_ylabel('样本数量')
axes[0, 0].set_title('各聚类在不同时段的分布')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 按工作日/非工作日分布
workingday_dist = df.groupby(['cluster', 'workingday']).size().unstack(fill_value=0)
workingday_dist.plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_xlabel('聚类')
axes[0, 1].set_ylabel('样本数量')
axes[0, 1].set_title('各聚类在工作日/非工作日的分布')
axes[0, 1].legend(['非工作日', '工作日'])
axes[0, 1].grid(True, axis='y')

# 租赁数量分布
for i in range(k):
    cluster_data = df[df['cluster'] == i]
    axes[1, 0].scatter(cluster_data['casual'], cluster_data['registered'], 
                      label=f'聚类 {i}', alpha=0.6)
axes[1, 0].set_xlabel('临时用户租赁数量')
axes[1, 0].set_ylabel('注册用户租赁数量')
axes[1, 0].set_title('各聚类的用户类型分布')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 天气条件分布
weather_dist = df.groupby(['cluster', 'weather']).size().unstack(fill_value=0)
weather_dist.plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_xlabel('聚类')
axes[1, 1].set_ylabel('样本数量')
axes[1, 1].set_title('各聚类在不同天气条件下的分布')
axes[1, 1].legend(['晴朗', '多云', '小雨/雪', '恶劣天气'])
axes[1, 1].grid(True, axis='y')

plt.tight_layout()
plt.show()