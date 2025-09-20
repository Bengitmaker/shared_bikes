from sklearn.cluster import KMeans
from Data_prep import scaled_data,features,df

k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# 将聚类结果添加到原数据
df['cluster'] = clusters 