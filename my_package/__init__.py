"""
预留包模块

此目录作为预留的Python包模块，可用于未来功能扩展。
当前项目的核心功能实现在 src/mypkg 目录中。

如果需要添加新功能模块，可以考虑使用此目录。

本包提供了以下示例模块：
- DataProcessor: 通用数据处理器
- ConfigExtension: 配置系统扩展
"""

from .data_processor import DataProcessor
from .config_extension import ConfigExtension
from .main import main

__all__ = ['DataProcessor', 'ConfigExtension', 'main']