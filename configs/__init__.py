"""
Configuration Package Initialization
Directory with configuration files and utilities.
"""

from .config_manager import ConfigManager
from pathlib import Path

# Project root directory
root_dir = Path(__file__).parent.parent.absolute()

# Create default config instance
config = ConfigManager()

__all__ = ['ConfigManager', 'config']