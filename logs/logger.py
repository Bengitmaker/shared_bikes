"""
日志管理器模块

提供统一的日志配置和管理功能。
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

# 导入配置
from shared_bikes.configs import config

def setup_logger(name: str = __name__, level: Optional[int] = None) -> logging.Logger:
    """
    设置并返回配置好的Logger实例
    
    Args:
        name (str): Logger名称
        level (int, optional): 日志级别，如果为None则使用配置文件中的设置
        
    Returns:
        logging.Logger: 配置好的Logger实例
    """
    # 获取或创建Logger
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 设置日志级别
    log_level = level or getattr(logging, config.get('logging.level', 'INFO'))
    logger.setLevel(log_level)
    
    # 创建格式化器
    formatter = logging.Formatter(config.get('logging.format'))
    
    # 创建文件处理器
    log_file = config.get_path('logging.file')
    max_bytes = config.get('logging.max_bytes', 10485760)  # 10MB
    backup_count = config.get('logging.backup_count', 5)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 防止向上传播到根Logger
    logger.propagate = False
    
    # 记录日志系统初始化
    logger.info(f"Logger '{name}' 已初始化，日志文件: {log_file}")
    
    return logger

def get_logger(name: str = __name__) -> logging.Logger:
    """
    获取已配置的Logger实例
    
    Args:
        name (str): Logger名称
        
    Returns:
        logging.Logger: Logger实例
    """
    # 检查Logger是否已经配置
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 如果没有处理器，则进行配置
        return setup_logger(name)
    return logger

def configure_root_logger():
    """配置根Logger"""
    # 配置根Logger的基本设置
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.get('logging.level', 'INFO')))
    
    # 只有当根Logger没有处理器时才添加
    if not root_logger.handlers:
        formatter = logging.Formatter(config.get('logging.format'))
        
        # 添加控制台处理器到根Logger
        console_handler = logging.StreamHandler()
        console_handler.setLevel(root_logger.level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)