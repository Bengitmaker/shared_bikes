"""
Prediction Module

This module handles predictions using a trained K-means clustering model.
It loads a previously trained model and scaler, processes input data,
makes predictions, and saves the results with cluster labels.

The module can be used to predict clusters for new data based on the patterns
learned during model training. It supports customization of model directory
and output path.

Example usage:
    shared-bikes-predict --data-path data/raw/test.csv --model-dir models/ --output-path output/predictions.csv
"""

import os
import sys
from pathlib import Path
import argparse
import pandas as pd

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
    from shared_bikes.predictor import load_model_and_scaler, predict_clusters
except ImportError:
    # If import fails, add project root to sys.path and try again
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from shared_bikes.predictor import load_model_and_scaler, predict_clusters

# Import utility functions for visualization
try:
    from utils.visualization import setup_chinese_font
except ImportError:
    # If import fails, add project root to sys.path and try again
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from utils.visualization import setup_chinese_font
    except ImportError:
        setup_chinese_font = lambda: None  # Provide empty implementation


def main(args=None):
    """
    Main function to make predictions using trained model
    
    Args:
        args (argparse.Namespace, optional): Command line arguments. If None, will parse from sys.argv
    """
    # Parse command line arguments if not provided
    if args is None:
        parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
        parser.add_argument('--data-path', type=str, required=True, help='输入数据路径')
        parser.add_argument('--model-dir', type=str, help='模型目录')
        parser.add_argument('--output-path', type=str, help='输出文件路径')
        
        args = parser.parse_args()
    
    # Get configuration values
    if config is not None:
        models_config = config.get_path('paths.models')
        model_dir = args.model_dir or str(models_config)
    else:
        model_dir = args.model_dir or "models"
    
    try:
        print("开始执行预测...\n")
        
        # 1. Load trained model and scaler from specified directory
        model, scaler = load_model_and_scaler(model_dir)
        print(f"加载模型完成")
        
        # 2. Make predictions on input data
        result_df = predict_clusters(model, scaler, args.data_path)
        print(f"预测完成 - 聚类结果: {result_df['cluster'].unique()}")
        
        # 3. Save prediction results to output file
        if args.output_path is None:
            # Default output path: same directory as input with _with_clusters suffix
            output_path = str(Path(args.data_path).parent / f"{Path(args.data_path).stem}_with_clusters.csv")
        else:
            output_path = args.output_path
            
        result_df.to_csv(output_path, index=False)
        print(f"预测结果已保存: {output_path}")
        
        # 4. Print cluster distribution statistics
        print("\n聚类统计:")
        cluster_counts = result_df['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(result_df)) * 100
            print(f"  聚类 {cluster_id}: {count} 样本 ({percentage:.1f}%)")
        
        print("\nSuccess 预测完成！")
        
    except Exception as e:
        print(f"Error 预测失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()