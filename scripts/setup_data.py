import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

# Project root path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configuration manager
try:
    from configs.config_manager import ConfigManager
    config = ConfigManager()
except Exception as e:
    print(f"警告: 配置管理器初始化失败: {e}")
    config = None


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample bike-sharing data
    
    Args:
        n_samples (int): Number of samples
        
    Returns:
        pd.DataFrame: Generated sample data
    """
    np.random.seed(42)
    
    # Generate datetime
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    data = {
        'datetime': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
        'season': np.random.choice([1, 2, 3, 4], n_samples),
        'hour': [d.hour for d in dates],
        'workingday': [1 if d.weekday() < 5 else 0 for d in dates],
        'weather': np.random.choice([1, 2, 3, 4], n_samples, p=[0.6, 0.3, 0.08, 0.02]),
        'temp': np.random.normal(20, 10, n_samples).clip(0, 40),
        'atemp': np.random.normal(22, 10, n_samples).clip(0, 45),
        'humidity': np.random.uniform(20, 100, n_samples),
        'windspeed': np.random.exponential(10, n_samples).clip(0, 67),
        'casual': np.random.poisson(50, n_samples),
        'registered': np.random.poisson(200, n_samples)
    }
    
    df = pd.DataFrame(data)
    # Calculate total rentals
    df['count'] = df['casual'] + df['registered']
    
    return df


def setup_data_directories():
    """Create necessary data directories"""
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # Create directories
    for directory in [data_dir, raw_dir, processed_dir]:
        directory.mkdir(exist_ok=True)
        print(f"创建目录: {directory}")
    
    return data_dir, raw_dir, processed_dir

def main():
    """Main function"""
    print("开始设置项目数据...")
    
    # Create directories
    data_dir, raw_dir, processed_dir = setup_data_directories()
    
    # Generate sample data
    print("生成示例数据...")
    sample_data = create_sample_data(1000)
    
    # Save raw data
    raw_path = raw_dir / "train.csv"
    sample_data.to_csv(raw_path, index=False)
    print(f"保存原始数据到: {raw_path}")
    
    # Save processed data (just copy)
    processed_path = processed_dir / "train_processed.csv"
    sample_data.to_csv(processed_path, index=False)
    print(f"保存处理后数据到: {processed_path}")
    
    # Create test data
    test_data = create_sample_data(200)
    test_path = raw_dir / "test.csv"
    test_data.to_csv(test_path, index=False)
    print(f"保存测试数据到: {test_path}")
    
    print("\nSuccess 数据设置完成！")
    print(f"- 总样本数: {len(sample_data)}")
    print(f"- 测试样本数: {len(test_data)}")
    print(f"- 数据目录结构已创建")

if __name__ == "__main__":
    main()