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

# Project root path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import config manager from configs directory
try:
    from configs.config_manager import ConfigManager
    config = ConfigManager()
except Exception as e:
    print(f"警告: 配置管理器初始化失败: {e}")
    config = None

# Import utility functions
try:
    from utils.visualization import setup_chinese_font
except ImportError:
    # If import fails, add project root to sys.path and try again
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from utils.visualization import setup_chinese_font
    except ImportError:
        setup_chinese_font = lambda: None  # Provide a no-op implementation

def load_model_and_data(model_dir: str, data_path: str):
    """
    Load model and data
    
    Args:
        model_dir (str): Model directory
        data_path (str): Data path
        
    Returns:
        tuple: Model, scaler, data
    """
    # Load model and scaler
    model_path = Path(model_dir) / "kmeans_model.joblib"
    scaler_path = Path(model_dir) / "kmeans_model_scaler.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load data
    if not Path(data_path).exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"加载模型和数据完成")
    return model, scaler, df

def calculate_clustering_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calculate clustering evaluation metrics
    
    Args:
        X_scaled (np.ndarray): Scaled feature data
        labels (np.ndarray): Cluster labels
        
    Returns:
        dict: Evaluation metrics dictionary
    """
    metrics = {}
    
    # Silhouette score
    try:
        metrics['silhouette'] = silhouette_score(X_scaled, labels)
    except Exception as e:
        metrics['silhouette'] = np.nan
        
    # Calinski-Harabasz index
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(X_scaled, labels)
    except Exception as e:
        metrics['calinski_harabasz'] = np.nan
        
    # Davies-Bouldin index
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(X_scaled, labels)
    except Exception as e:
        metrics['davies_bouldin'] = np.nan
        
    return metrics
def create_evaluation_report(model, scaler, df, output_dir: str):
    """
    Create evaluation report
    
    Args:
        model: Trained model
        scaler: Scaler
        df: Data
        output_dir (str): Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare features
    # 如果没有hour列，从datetime列中提取
    if 'hour' not in df.columns:
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    
    feature_columns = ['hour', 'workingday', 'weather', 'temp', 'humidity', 'windspeed']
    
    # 检查特征列是否存在
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"数据中缺少以下特征列: {missing_columns}")
    
    df_clean = df.dropna(subset=feature_columns)
    X = df_clean[feature_columns].values
    X_scaled = scaler.transform(X)
    
    # Predict clusters
    labels = model.predict(X_scaled)
    
    # Calculate evaluation metrics
    metrics = calculate_clustering_metrics(X_scaled, labels)
    
    # Create report content
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
    
    # Add cluster distribution
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(labels)) * 100
        report += f"- 聚类 {cluster_id}: {count} 样本 ({percentage:.1f}%)"

    
    # Save report
    report_path = output_path / "evaluation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"评估报告已生成: {report_path}")
    
    # Create visualizations
    create_visualizations(X_scaled, labels, model.n_clusters, output_path)
    
    return report, metrics
def create_visualizations(X_scaled: np.ndarray, labels: np.ndarray, n_clusters: int, output_path: Path):
    """Create visualization charts"""
    # Set Chinese font
    setup_chinese_font()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Choose first two principal components for visualization
    if X_scaled.shape[1] >= 2:
        x_axis, y_axis = 0, 1
    else:
        x_axis, y_axis = 0, 0
    
    # Scatter plot
    scatter = axes[0].scatter(
        X_scaled[:, x_axis], X_scaled[:, y_axis], 
        c=labels, cmap='tab10', alpha=0.6
    )
    axes[0].set_xlabel(f'特征 {x_axis + 1}')
    axes[0].set_ylabel(f'特征 {y_axis + 1}')
    axes[0].set_title('聚类结果可视化')
    plt.colorbar(scatter, ax=axes[0])
    
    # Cluster distribution bar chart
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    axes[1].bar(cluster_counts.index, cluster_counts.values)
    axes[1].set_xlabel('聚类ID')
    axes[1].set_ylabel('样本数量')
    axes[1].set_title('各聚类样本数量分布')
    
    # Display numbers on top of each bar
    for i, v in enumerate(cluster_counts.values):
        axes[1].text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save chart
    plot_path = output_path / "clustering_visualization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存: {plot_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='评估共享单车聚类模型')
    parser.add_argument('--data-path', type=str, required=True, help='输入数据路径')
    parser.add_argument('--model-dir', type=str, help='模型目录')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    # Get config
    if config is not None:
        models_config = config.get_path('paths.models')
        model_dir = args.model_dir or str(models_config)
        output_dir = args.output_dir or str(config.get_path('paths.output'))
    else:
        model_dir = args.model_dir or "models"
        output_dir = args.output_dir or "output"
    
    try:
        # Load model and data
        model, scaler, df = load_model_and_data(model_dir, args.data_path)
        
        # Create evaluation report
        report, metrics = create_evaluation_report(model, scaler, df, output_dir)
        
        # Print key metrics
        print("\n关键评估指标:")
        for metric_name, value in metrics.items():
            if not pd.isna(value):
                print(f"{metric_name}: {value:.3f}")
            
    except Exception as e:
        print(f"Error 评估失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()