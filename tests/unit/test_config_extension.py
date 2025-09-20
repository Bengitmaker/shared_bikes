"""
ConfigExtension 类的单元测试
"""

import pytest
from shared_bikes.configs import ConfigManager
from my_package.config_extension import ConfigExtension
import yaml
import os
from pathlib import Path


def test_config_extension_initialization(test_config):
    """测试ConfigExtension初始化"""
    # 创建临时配置文件
    temp_config = Path("tests/test_config.yaml")
    temp_config.write_text(yaml.dump(test_config))
    
    try:
        # 创建配置管理器
        config_manager = ConfigManager(str(temp_config))
        
        # 创建配置扩展
        config_ext = ConfigExtension(config_manager)
        
        assert isinstance(config_ext, ConfigExtension)
        
    finally:
        # 清理临时文件
        if temp_config.exists():
            temp_config.unlink()


def test_get_data_config(test_config):
    """测试get_data_config方法"""
    # 创建临时配置文件
    temp_config = Path("tests/test_config.yaml")
    temp_config.write_text(yaml.dump(test_config))
    
    try:
        config_manager = ConfigManager(str(temp_config))
        config_ext = ConfigExtension(config_manager)
        
        data_config = config_ext.get_data_config()
        
        # 检查返回值
        assert 'data_path' in data_config
        assert 'output_path' in data_config
        assert 'chunk_size' in data_config
        assert data_config['chunk_size'] == 1000
        
    finally:
        # 清理临时文件
        if temp_config.exists():
            temp_config.unlink()


def test_get_model_config(test_config):
    """测试get_model_config方法"""
    # 创建临时配置文件
    temp_config = Path("tests/test_config.yaml")
    temp_config.write_text(yaml.dump(test_config))
    
    try:
        config_manager = ConfigManager(str(temp_config))
        config_ext = ConfigExtension(config_manager)
        
        model_config = config_ext.get_model_config()
        
        # 检查返回值
        assert 'n_clusters' in model_config
        assert 'random_state' in model_config
        assert 'max_iter' in model_config
        assert model_config['n_clusters'] == 3
        
    finally:
        # 清理临时文件
        if temp_config.exists():
            temp_config.unlink()


def test_validate_config_valid(test_config):
    """测试validate_config方法（有效配置）"""
    # 创建临时配置文件
    temp_config = Path("tests/test_config.yaml")
    temp_config.write_text(yaml.dump(test_config))
    
    try:
        config_manager = ConfigManager(str(temp_config))
        config_ext = ConfigExtension(config_manager)
        
        assert config_ext.validate_config() == True
        
    finally:
        # 清理临时文件
        if temp_config.exists():
            temp_config.unlink()