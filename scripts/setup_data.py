"""
数据准备脚本

用于初始化项目所需的数据目录和示例数据。
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    创建示例共享单车数据
    
    Args:
        n_samples (int): 样本数量
        
    Returns:
        pd.DataFrame: 生成的示例数据
    """
    np.random.seed(42)
    
    # 生成日期时间
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
    # 计算总租赁量
    df['count'] = df['casual'] + df['registered']
    
    return df


def setup_data_directories():
    """创建必要的数据目录"""
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # 创建目录
    for directory in [data_dir, raw_dir, processed_dir]:
        directory.mkdir(exist_ok=True)
        print(f"创建目录: {directory}")
    
    return data_dir, raw_dir, processed_dir

def main():
    """主函数"""
    print("开始设置项目数据...")
    
    # 创建目录
    data_dir, raw_dir, processed_dir = setup_data_directories()
    
    # 生成示例数据
    print("生成示例数据...")
    sample_data = create_sample_data(1000)
    
    # 保存原始数据
    raw_path = raw_dir / "train.csv"
    sample_data.to_csv(raw_path, index=False)
    print(f"保存原始数据到: {raw_path}")
    
    # 保存处理后的数据（这里只是复制）
    processed_path = processed_dir / "train_processed.csv"
    sample_data.to_csv(processed_path, index=False)
    print(f"保存处理后数据到: {processed_path}")
    
    # 创建测试数据
    test_data = create_sample_data(200)
    test_path = raw_dir / "test.csv"
    test_data.to_csv(test_path, index=False)
    print(f"保存测试数据到: {test_path}")
    
    print("\n✅ 数据设置完成！")
    print(f"- 总样本数: {len(sample_data)}")
    print(f"- 测试样本数: {len(test_data)}")
    print(f"- 数据目录结构已创建")

if __name__ == "__main__":
    main()