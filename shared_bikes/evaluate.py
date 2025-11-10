"""
Model Evaluation Module

This module evaluates the performance of a trained K-means clustering model.
It calculates various clustering metrics such as silhouette score, Calinski-Harabasz index,
and Davies-Bouldin index. It also generates a comprehensive evaluation report and visualizations.

The evaluation metrics provide insights into:
- Cluster cohesion and separation (silhouette score)
- Between-cluster to within-cluster variance ratio (Calinski-Harabasz index)
- Average similarity between clusters (Davies-Bouldin index)

Example usage:
    shared-bikes-evaluate --data-path data/raw/test.csv --model-dir models/ --output-dir output/
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
    from shared_bikes.evaluator import load_model_and_data, create_evaluation_report
except ImportError:
    # If import fails, add project root to sys.path and try again
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from shared_bikes.evaluator import load_model_and_data, create_evaluation_report

# Import utility functions for visualization
try:
    from utils.visualization import setup_chinese_font, create_visualizations
except ImportError:
    # If import fails, add project root to sys.path and try again
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from utils.visualization import setup_chinese_font, create_visualizations
    except ImportError:
        setup_chinese_font = lambda: None
        create_visualizations = lambda X_scaled, labels, n_clusters, output_path: None


def main(args=None):
    """
    Main function to evaluate model performance
    
    Args:
        args (argparse.Namespace, optional): Command line arguments. If None, will parse from sys.argv
    """
    # Parse command line arguments if not provided
    if args is None:
        parser = argparse.ArgumentParser(description='评估共享单车聚类模型')
        parser.add_argument('--data-path', type=str, required=True, help='输入数据路径')
        parser.add_argument('--model-dir', type=str, help='模型目录')
        parser.add_argument('--output-dir', type=str, help='输出目录')
        
        args = parser.parse_args()
    
    # Get configuration values
    if config is not None:
        models_config = config.get_path('paths.models')
        model_dir = args.model_dir or str(models_config)
        output_dir = args.output_dir or str(config.get_path('paths.output'))
    else:
        model_dir = args.model_dir or "models"
        output_dir = args.output_dir or "output"
    
    try:
        # Load model, scaler, and data for evaluation
        model, scaler, df = load_model_and_data(model_dir, args.data_path)
        print(f"加载模型和数据完成")
        
        # Create evaluation report with metrics
        report, metrics = create_evaluation_report(model, scaler, df)
        
        # Save report to output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / "evaluation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"评估报告已生成: {report_path}")
        
        # Create visualizations of clustering results
        try:
            # Prepare features for visualization
            if 'hour' not in df.columns:
                df['hour'] = pd.to_datetime(df['datetime']).dt.hour
            
            feature_columns = ['hour', 'workingday', 'weather', 'temp', 'humidity', 'windspeed']
            df_clean = df.dropna(subset=feature_columns)
            X = df_clean[feature_columns].values
            X_scaled = scaler.transform(X)
            labels = model.predict(X_scaled)
            
            create_visualizations(X_scaled, labels, model.n_clusters, output_path)
        except Exception as e:
            print(f"可视化创建失败: {e}")
        
        # Print key metrics to console
        print("\n关键评估指标:")
        for metric_name, value in metrics.items():
            if not pd.isna(value):
                print(f"{metric_name}: {value:.3f}")
            
    except Exception as e:
        print(f"Error 评估失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()