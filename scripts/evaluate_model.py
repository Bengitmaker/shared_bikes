"""
模型评估脚本

用于评估训练好的模型性能。
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import argparse

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# 导入配置
from shared_bikes.configs import config
from shared_bikes.utils.visualization import setup_chinese_font

def load_model_and_data(model_dir: str, data_path: str):
    """
    加载模型和数据
    
    Args:
        model_dir (str): 模型目录
        data_path (str): 数据路径
        
    Returns:
        tuple: 模型、标准化器、数据
    """
    # 加载模型和标准化器
    model_path = Path(model_dir) / "kmeans_model.joblib"
    scaler_path = Path(model_dir) / "kmeans_model_scaler.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 加载数据
    if not Path(data_path).exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"加载模型和数据完成")
    return model, scaler, df

def calculate_clustering_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    """
    计算聚类评估指标
    
    Args:
        X_scaled (np.ndarray): 标准化后的特征数据
        labels (np.ndarray): 聚类标签
        
    Returns:
        dict: 评估指标字典
    """
    metrics = {}
    
    # 轮廓系数
    try:
        metrics['silhouette'] = silhouette_score(X_scaled, labels)
    except Exception as e:
        metrics['silhouette'] = np.nan
        
    # Calinski-Harabasz指数
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(X_scaled, labels)
    except Exception as e:
        metrics['calinski_harabasz'] = np.nan
        
    # Davies-Bouldin指数
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(X_scaled, labels)
    except Exception as e:
        metrics['davies_bouldin'] = np.nan
        
    return metrics
def create_evaluation_report(model, scaler, df, output_dir: str):
    """
    创建评估报告
    
    Args:
        model: 训练好的模型
        scaler: 标准化器
        df: 数据
        output_dir (str): 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 准备特征
    feature_columns = ['hour', 'workingday', 'weather', 'temp', 'humidity', 'windspeed']
    df_clean = df.dropna(subset=feature_columns)
    X = df_clean[feature_columns].values
    X_scaled = scaler.transform(X)
    
    # 预测聚类
    labels = model.predict(X_scaled)
    
    # 计算评估指标
    metrics = calculate_clustering_metrics(X_scaled, labels)
    
    # 创建报告内容
    report = f"""# 共享单车聚类模型评估报告

## 基本信息
- 模型类型: K-means
- 聚类数量: {model.n_clusters}
- 样本数量: {len(df_clean)}
- 特征数量: {X.shape[1]}
- 评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 评估指标
- **轮廓系数**: {metrics.get('silhouette', 'N/A'):>.3f}
  - 解释: 值越接近1表示聚类效果越好
- **Calinski-Harabasz指数**: {metrics.get('calinski_harabasz', 'N/A'):>}
  - 解释: 值越大表示聚类效果越好
- **Davies-Bouldin指数**: {metrics.get('davies_bouldin', 'N/A'):>.3f}
  - 解释: 值越小表示聚类效果越好

## 聚类分布
"""
    
    # 添加聚类分布
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(labels)) * 100
        report += f"- 聚类 {cluster_id}: {count} 样本 ({percentage:.1f}%)"

    
    # 保存报告
    report_path = output_path / "evaluation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"评估报告已生成: {report_path}")
    
    # 创建可视化
    create_visualizations(X_scaled, labels, model.n_clusters, output_path)
    
    return report, metrics
def create_visualizations(X_scaled: np.ndarray, labels: np.ndarray, n_clusters: int, output_path: Path):
    """创建可视化图表"""
    # 设置中文字体
    setup_chinese_font()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 选择前两个主成分进行可视化
    if X_scaled.shape[1] >= 2:
        x_axis, y_axis = 0, 1
    else:
        x_axis, y_axis = 0, 0
    
    # 散点图
    scatter = axes[0].scatter(
        X_scaled[:, x_axis], X_scaled[:, y_axis], 
        c=labels, cmap='tab10', alpha=0.6
    )
    axes[0].set_xlabel(f'特征 {x_axis + 1}')
    axes[0].set_ylabel(f'特征 {y_axis + 1}')
    axes[0].set_title('聚类结果可视化')
    plt.colorbar(scatter, ax=axes[0])
    
    # 聚类分布柱状图
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    axes[1].bar(cluster_counts.index, cluster_counts.values)
    axes[1].set_xlabel('聚类ID')
    axes[1].set_ylabel('样本数量')
    axes[1].set_title('各聚类样本数量分布')
    
    # 在每个柱子上显示数值
    for i, v in enumerate(cluster_counts.values):
        axes[1].text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = output_path / "clustering_visualization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存: {plot_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估共享单车聚类模型')
    parser.add_argument('--model-dir', type=str, help='模型目录')
    parser.add_argument('--data-path', type=str, required=True, help='数据路径')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    # 获取配置
    models_config = config.get_path('paths.models')
    logs_config = config.get_path('logging.file').parent
    
    model_dir = args.model_dir or str(models_config)
    output_dir = args.output_dir or str(logs_config / "evaluation")
    
    try:
        # 加载模型和数据
        model, scaler, df = load_model_and_data(model_dir, args.data_path)
        
        # 创建评估报告
        report, metrics = create_evaluation_report(model, scaler, df, output_dir)
        
        # 打印关键指标
        print("\n关键评估指标:")
        for metric_name, value in metrics.items():
            if not pd.isna(value):
                print(f"{metric_name}: {value:.3f}")
            
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()