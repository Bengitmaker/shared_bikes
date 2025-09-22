"""
Configuration package initialization.
This directory contains configuration files and utilities for the application.
"""

from .config_manager import ConfigManager
from pathlib import Path

# 获取项目根目录
root_dir = Path(__file__).parent.parent.absolute()

# 便捷访问
config = ConfigManager(str(root_dir / "configs" / "config.yaml"))

__all__ = ['ConfigManager', 'config']