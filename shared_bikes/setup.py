"""
Data Setup Module

This module prepares the project data by creating necessary directories
and generating sample data for training and testing. It creates the following
directory structure:
- data/
  - raw/
  - processed/

It also generates sample shared bikes usage data with realistic features such as:
- Datetime information
- Weather conditions
- Temperature and humidity
- Usage counts

This module is typically run as the first step in the project pipeline.

Example usage:
    shared-bikes-setup
"""

import os
import sys
from pathlib import Path

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
    from shared_bikes.data_processor import create_sample_data, setup_data_directories
except ImportError:
    # If import fails, add project root to sys.path and try again
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from shared_bikes.data_processor import create_sample_data, setup_data_directories


def main():
    """
    Main function to setup project data
    
    Creates directories and generates sample data for training and testing.
    """
    print("开始设置项目数据...")
    
    # Create necessary data directories
    data_dir, raw_dir, processed_dir = setup_data_directories()
    print(f"创建目录: {data_dir}")
    print(f"创建目录: {raw_dir}")
    print(f"创建目录: {processed_dir}")
    
    # Generate sample training data
    print("生成示例数据...")
    sample_data = create_sample_data(1000)
    
    # Save raw training data
    raw_path = raw_dir / "train.csv"
    sample_data.to_csv(raw_path, index=False)
    print(f"保存原始数据到: {raw_path}")
    
    # Save processed training data (same as raw in this example)
    processed_path = processed_dir / "train_processed.csv"
    sample_data.to_csv(processed_path, index=False)
    print(f"保存处理后数据到: {processed_path}")
    
    # Create and save sample test data
    test_data = create_sample_data(200)
    test_path = raw_dir / "test.csv"
    test_data.to_csv(test_path, index=False)
    print(f"保存测试数据到: {test_path}")
    
    print("\nSuccess 数据设置完成！")
    print(f"- 总样本数: {len(sample_data)}")


if __name__ == "__main__":
    main()