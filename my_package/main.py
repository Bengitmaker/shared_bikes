"""
my_package 主程序示例

展示如何使用预留包中的功能。
"""

import pandas as pd
from shared_bikes.configs import config
from .data_processor import DataProcessor
from .config_extension import ConfigExtension


def main():
    """主函数示例"""
    print("=== my_package 功能演示 ===\n")
    
    # 1. 初始化配置扩展
    config_ext = ConfigExtension(config)
    
    # 2. 验证配置
    if not config_ext.validate_config():
        print("配置验证失败，退出...")
        return
    
    print("配置验证通过！\n")
    
    # 3. 获取配置
    data_config = config_ext.get_data_config()
    print(f"数据配置: {data_config}")
    
    # 4. 创建数据处理器
    processor = DataProcessor()
    
    # 添加示例处理步骤（这里只是演示，不实际处理数据）
    def sample_step(data):
        print(f"数据形状: {data.shape}")
        return data
        
    processor.add_processing_step("初始化检查", sample_step)
    
    print("\nmy_package 准备就绪，可用于未来功能扩展。")

if __name__ == "__main__":
    main()