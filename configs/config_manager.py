import os
import yaml
import os
import logging
from pathlib import Path
from typing import Any, Dict, Union


class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path (str, optional): 配置文件路径
        """
        # 设置默认配置文件路径
        if config_path is None:
            # 获取项目根目录
            project_root = Path(__file__).parent.parent
            config_path = project_root / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self.configure_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            dict: 配置字典
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _validate_config(self):
        """验证配置文件"""
        required_sections = ['paths', 'data_processing', 'model']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件缺少必需的节: {section}")
        
        # 验证路径配置
        required_paths = ['data', 'models', 'output']
        for path_key in required_paths:
            if path_key not in self.config.get('paths', {}):
                raise ValueError(f"配置文件缺少必需的路径: {path_key}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key_path (str): 配置键路径，使用点号分隔（如"model.kmeans.n_clusters"）
            default (Any, optional): 默认值
            
        Returns:
            Any: 配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_path(self, key_path: str) -> Path:
        """
        获取路径配置值
        
        Args:
            key_path (str): 配置键路径
            
        Returns:
            Path: 路径对象
        """
        path_str = self.get(key_path)
        if path_str is None:
            raise ValueError(f"路径配置不存在: {key_path}")
        
        # 处理相对路径
        path = Path(path_str)
        if not path.is_absolute():
            # 相对于项目根目录
            project_root = self.config_path.parent.parent
            path = project_root / path
        
        # 确保目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def configure_logging(self):
        """配置日志"""
        # 获取日志目录
        try:
            log_dir = self.get_path('paths.logs')
        except ValueError:
            # 如果没有配置日志路径，使用默认路径
            project_root = self.config_path.parent.parent
            log_dir = project_root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志目录
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件路径
        log_file = log_dir / "app.log"
        
        # 配置根日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )


# 创建全局配置实例
config = ConfigManager()
