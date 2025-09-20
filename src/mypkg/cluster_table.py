import pandas as pd
from Model_Kmeans import df, k
from Data_prep import features

def generate_cluster_table():
    """
    生成包含所有聚类情况的综合表格
    """
    # 创建一个空的DataFrame来存储聚类统计信息
    cluster_stats = []
    
    # 为每个聚类计算统计信息
    for i in range(k):
        cluster_data = df[df['cluster'] == i]
        stats = {
            'ID': i,
            '样本数量': len(cluster_data),
            'casual': round(cluster_data['casual'].mean(), 2),
            'atemp': round(cluster_data['registered'].mean(), 2),
            'time': cluster_data['hour'].mode().iloc[0] if not cluster_data['hour'].mode().empty else 'N/A',
            'workday': round(cluster_data['workingday'].mean() * 100, 2)
        }
        
        # 添加各特征的平均值
        for feature in features:
            stats[f'{feature}'] = round(cluster_data[feature].mean(), 2)
        
        cluster_stats.append(stats)
    
    # 创建汇总表格
    cluster_table = pd.DataFrame(cluster_stats)
    return cluster_table

def save_cluster_table():
    """
    生成并保存聚类表格到CSV文件
    """
    table = generate_cluster_table()
    
    # 保存到CSV文件
    table.to_csv('聚类分析结果.csv', index=False, encoding='utf-8-sig')
    
    # 打印表格
    print("聚类分析表格:")
    print(table)
    
    return table

# 执行主函数
if __name__ == "__main__":
    cluster_table = save_cluster_table()