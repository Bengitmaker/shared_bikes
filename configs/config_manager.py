"""
配置管理模块
提供配置文件的加载、解析和访问功能
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path (str): 配置文件路径
        """
        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None
        
        # 确保配置文件存在
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
        self.load_config()
        self.create_directories()
        self.configure_logging()
    
    def load_config(self) -> None:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key (str): 配置键，支持点号分隔的嵌套键，如 'paths.data'
            default (Any): 默认值
            
        Returns:
            Any: 配置值或默认值
        """
        if not self._config:
            return default
            
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_path(self, path_key: str) -> Path:
        """
        获取路径配置
        
        Args:
            path_key (str): 路径配置键
            
        Returns:
            Path: 路径对象
        """
        path_str = self.get(path_key, "")
        return Path(path_str)
    
    def create_directories(self) -> None:
        """根据配置创建必要的目录"""
        if not self._config:
            return
            
        # 创建数据路径
        data_path = self.get_path('paths.data')
        data_path.mkdir(parents=True, exist_ok=True)
        
        # 创建输出路径
        output_path = self.get_path('paths.output')
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建模型路径
        models_path = self.get_path('paths.models')
        models_path.mkdir(parents=True, exist_ok=True)
        
        # 创建日志路径
        log_path = self.get_path('logging.file').parent
        log_path.mkdir(parents=True, exist_ok=True)
    
    def configure_logging(self) -> None:
        """配置日志系统"""
        # 延迟导入logger模块，避免循环导入
        from shared_bikes.logs.logger import configure_root_logger
        configure_root_logger()