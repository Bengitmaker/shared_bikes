"""
shared_bikes Package Initialization
"""

import os
import sys
from pathlib import Path

# Package root directory
PACKAGE_ROOT = Path(__file__).parent.parent.absolute()

# Add package root directory to system path
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Import core modules
from configs.config_manager import ConfigManager

# Create global config instance
config = ConfigManager()

__version__ = "2.0.0"
__author__ = "Your Name"

__all__ = ['ConfigManager', 'config']