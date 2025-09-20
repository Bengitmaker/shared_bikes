"""
配置扩展模块

展示如何扩展现有配置系统功能。
"""

from shared_bikes.configs import ConfigManager
from typing import Any, Dict
import os


class ConfigExtension:
    """配置扩展类"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        初始化配置扩展
        
        Args:
            config_manager (ConfigManager): 现有配置管理器实例
        """
        self.config = config_manager
        
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据相关配置"""
        return {
            'data_path': self.config.get_path('paths.data'),
            'output_path': self.config.get_path('paths.output'),
            'chunk_size': self.config.get('data_processing.chunk_size', 10000),
            'date_format': self.config.get('data_processing.date_format', '%Y-%m-%d %H:%M:%S')
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型相关配置"""
        return {
            'n_clusters': self.config.get('model.kmeans.n_clusters', 5),
            'random_state': self.config.get('model.kmeans.random_state', 42),
            'max_iter': self.config.get('model.kmeans.max_iter', 300)
        }
    
    def validate_config(self) -> bool:
        """验证配置完整性"""
        required_paths = [
            'paths.data',
            'paths.output',
            'paths.models'
        ]
        
        for path_key in required_paths:
            path = self.config.get_path(path_key)
            # 检查路径是否可访问（不实际创建）
            try:
                if str(path).strip() == "":
                    print(f"错误: {path_key} 配置为空")
                    return False
            except Exception as e:
                print(f"错误: {path_key} 配置无效: {e}")
                return False
                
        return True