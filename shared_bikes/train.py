"""
Model Training Module

This module handles the training of the K-means clustering model for shared bikes usage pattern analysis.
It loads data, prepares features, trains the model, and saves the trained model and scaler.

The module uses configuration from the project's config.yaml file to set parameters such as:
- Number of clusters
- Maximum iterations
- Random state for reproducibility

Example usage:
    shared-bikes-train --data-path data/raw/train.csv --output-dir models/ --n-clusters 5
"""

import os
import sys
from pathlib import Path
import argparse

# Add project root to path to enable module imports
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import configuration manager
try:
    from configs.config_manager import ConfigManager
    config = ConfigManager()
except Exception as e:
    print(f"警告: 配置管理器初始化失败: {e}")
    config = None

# Import core functions from shared_bikes package
try:
    from shared_bikes.model_trainer import load_data, prepare_features, train_kmeans_model, save_model
except ImportError:
    # If import fails, add project root to sys.path and try again
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from shared_bikes.model_trainer import load_data, prepare_features, train_kmeans_model, save_model


def main(data_path: str = None, output_dir: str = None, n_clusters: int = None):
    """
    Main function to train the K-means clustering model
    
    Args:
        data_path (str, optional): Path to training data. Defaults to config value or data/train.csv
        output_dir (str, optional): Directory to save trained model. Defaults to config value or models/
        n_clusters (int, optional): Number of clusters. Defaults to config value or 5
    """
    print("开始训练共享单车聚类模型...\n")
    
    # Get config values if config manager is available
    if config is not None:
        data_config = config.get_path('paths.data')
        models_config = config.get_path('paths.models')
        
        # Set default parameters from config
        data_path = data_path or str(data_config / "raw" / "train.csv")
        output_dir = output_dir or str(models_config)
        n_clusters = n_clusters or config.get('model.kmeans.n_clusters', 5)
        max_iter = config.get('model.kmeans.max_iter', 300)
        n_init = config.get('model.kmeans.n_init', 10)
        random_state = config.get('model.kmeans.random_state', 42)
    else:
        # Set default values if config is not available
        data_path = data_path or "data/raw/train.csv"
        output_dir = output_dir or "models"
        n_clusters = n_clusters or 5
        max_iter = 300
        n_init = 10
        random_state = 42
    
    try:
        # 1. Load data from specified path
        df = load_data(data_path)
        print(f"加载数据: {data_path} (形状: {df.shape})")
        
        # 2. Prepare features for clustering
        X, scaler, df_clean = prepare_features(df)
        
        # 3. Train K-means model with specified parameters
        model = train_kmeans_model(X, n_clusters, random_state, max_iter, n_init)
        print(f"模型训练完成 - WCSS: {model.inertia_:.2f}")
        
        # 4. Save trained model and scaler
        save_model(model, scaler, output_dir)
        
        print("\nSuccess 模型训练完成！")
        
    except Exception as e:
        print(f"Error 训练失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='训练共享单车聚类模型')
    parser.add_argument('--data-path', type=str, help='训练数据路径')
    parser.add_argument('--output-dir', type=str, help='模型输出目录')
    parser.add_argument('--n-clusters', type=int, help='聚类数量')
    
    args = parser.parse_args()
    main(args.data_path, args.output_dir, args.n_clusters)