from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from Data_prep import scaled_data
import plt_utils

# 使用肘部法则确定最佳聚类数
inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertias.append(kmeans.inertia_)

# 绘制肘部法则图
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, marker='o')
plt.xlabel('聚类数 (k)')
plt.ylabel('簇内平方和 (WCSS)')
plt.title('肘部法则确定最佳聚类数')
plt.grid(True)
plt.show()