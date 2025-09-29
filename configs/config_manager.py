import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

# Get project root directory
def get_project_root():
    """Find project root directory by looking for setup.py"""
    current_path = Path(__file__).parent.absolute()
    while current_path != current_path.parent:
        if (current_path / "setup.py").exists():
            return current_path
        current_path = current_path.parent
    # Fallback to default if setup.py not found
    return Path(__file__).parent.parent.absolute()


class ConfigManager:
    """Configuration Manager Class"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config_path: str = "configs/config.yaml"):
        """
        Singleton pattern - ensure only one config manager instance globally
        
        Args:
            config_path (str): Configuration file path
        """
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the configuration manager
        
        Args:
            config_path (str): Configuration file path
        """
        # Prevent re-initialization
        if ConfigManager._initialized:
            return
            
        # Determine project root directory
        self.project_root = self._find_project_root()
        
        # Determine configuration file path
        if Path(config_path).is_absolute():
            self.config_path = Path(config_path)
        else:
            self.config_path = self.project_root / config_path
            
        # Ensure configuration file exists
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        self._config: Optional[Dict[str, Any]] = None
        self.load_config()
        self.create_directories()
        self.configure_logging()
        
        # Mark as initialized
        ConfigManager._initialized = True
    
    def load_config(self) -> None:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key (str): Configuration key, supports dot-separated nested keys, e.g., 'paths.data'
            default (Any): Default value
            
        Returns:
            Any: Configuration value or default value
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
    
    def _find_project_root(self) -> Path:
        """
        Find project root directory
        
        Returns:
            Path: Project root directory path
        """
        # Start from the current file and search upwards for a directory containing 'configs' and 'setup.py'
        current_path = Path(__file__).parent
        while current_path != current_path.parent:  # Stop when reaching the root directory
            if (current_path / "configs").exists() and (current_path / "setup.py").exists():
                return current_path.absolute()
            current_path = current_path.parent
        
        # If not found, use the parent directory of the configuration manager file
        return Path(__file__).parent.parent.absolute()
    
    def get_path(self, path_key: str) -> Path:
        """
        Get path configuration, supports relative and absolute paths
        
        Args:
            path_key (str): Path configuration key
            
        Returns:
            Path: Path object
        """
        path_str = self.get(path_key, "")
        if not path_str:
            return Path()
            
        path = Path(path_str)
        
        # If it's an absolute path, return it directly
        if path.is_absolute():
            return path.absolute()
        
        # If it's a relative path, relative to the project root
        return (self.project_root / path).absolute()
    
    def create_directories(self) -> None:
        """Create necessary directories based on configuration"""
        if not self._config:
            return
            
        # Create all paths defined in the configuration
        paths_config = self.get('paths', {})
        for path_key, path_value in paths_config.items():
            if path_value:
                path = self.get_path(f'paths.{path_key}')
                path.mkdir(parents=True, exist_ok=True)
        
        # Special handling for log directory
        log_file_path = self.get_path('logging.file')
        if log_file_path:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def configure_logging(self) -> None:
        """Configure logging system"""
        try:
            # Try to import project-specific logging configuration
            from shared_bikes.logs.logger import configure_root_logger
            configure_root_logger()
        except ImportError:
            # If import fails, use basic logging configuration
            log_level = self.get('logging.level', 'INFO')
            log_format = self.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                
            # Configure root logger
            logging.basicConfig(
                level=getattr(logging, log_level.upper(), logging.INFO),
                format=log_format,
                handlers=[
                    logging.StreamHandler(),
                ]
            )
                
            # If log file is configured, add file handler
            log_file = self.get_path('logging.file')
            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(logging.Formatter(log_format))
                logging.getLogger().addHandler(file_handler)