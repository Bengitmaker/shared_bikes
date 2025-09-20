"""
Configuration package initialization.
This directory contains configuration files and utilities for the application.
"""

from .config_manager import ConfigManager

# 便捷访问
config = ConfigManager()

__all__ = ['ConfigManager', 'config']