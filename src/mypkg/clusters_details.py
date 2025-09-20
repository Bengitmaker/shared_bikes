from Model_Kmeans import df,k
from Data_prep import features

# 输出每个聚类的简要描述
print("\n聚类描述:")
for i in range(k):
    cluster_data = df[df['cluster'] == i]
    print(f"\n聚类 {i}:")
    print(f"  样本数量: {len(cluster_data)}")
    print(f"  平均临时用户数: {cluster_data['casual'].mean():.2f}")
    print(f"  平均注册用户数: {cluster_data['registered'].mean():.2f}")
    print(f"  最常出现的小时: {cluster_data['hour'].mode().iloc[0]}")
    print(f"  工作日比例: {cluster_data['workingday'].mean():.2%}")

print("聚类0：代表非工作日的深夜和凌晨时段")
print("聚类3：代表工作日下班高峰期")