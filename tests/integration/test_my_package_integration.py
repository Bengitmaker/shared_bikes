"""
my_package 模块集成测试

测试各个模块协同工作的能力。
"""

import pytest
import pandas as pd
import numpy as np
from shared_bikes.configs import ConfigManager
from my_package.data_processor import DataProcessor
from my_package.config_extension import ConfigExtension
from pathlib import Path
import yaml
import os


def test_my_package_integration():
    """测试my_package各模块的集成"""
    # 1. 创建测试数据
    np.random.seed(42)
    n_samples = 50
    test_data = pd.DataFrame({
        'hour': np.random.randint(0, 24, n_samples),
        'workingday': np.random.choice([0, 1], n_samples),
        'casual': np.random.poisson(50, n_samples),
        'registered': np.random.poisson(200, n_samples)
    })
    
    # 2. 创建临时配置文件
    temp_config_path = Path("tests/test_integration_config.yaml")
    test_config = {
        'paths': {
            'data': 'data/',
            'output': 'tests/output/',
            'models': 'models/'
        },
        'data_processing': {
            'chunk_size': 100,
            'date_format': '%Y-%m-%d %H:%M:%S'
        },
        'model': {
            'kmeans': {
                'n_clusters': 3,
                'random_state': 42,
                'max_iter': 100
            }
        }
    }
    
    try:
        # 写入临时配置文件
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f, allow_unicode=True, default_flow_style=False)
        
        # 3. 初始化配置管理器和扩展
        config_manager = ConfigManager(str(temp_config_path))
        config_ext = ConfigExtension(config_manager)
        
        # 4. 验证配置
        assert config_ext.validate_config() == True, "配置验证失败"
        
        # 5. 获取配置
        data_config = config_ext.get_data_config()
        model_config = config_ext.get_model_config()
        
        assert data_config['chunk_size'] == 100
        assert model_config['n_clusters'] == 3
        
        # 6. 创建并使用数据处理器
        processor = DataProcessor()
        
        # 添加处理步骤
        def normalize_data(data):
            data_copy = data.copy()
            if 'casual' in data_copy.columns:
                data_copy['casual_norm'] = data_copy['casual'] / data_copy['casual'].max()
            return data_copy
            
        processor.add_processing_step("normalize", normalize_data)
        
        # 执行处理
        processed_data = processor.process(test_data)
        
        # 验证处理结果
        assert 'casual_norm' in processed_data.columns
        assert processed_data['casual_norm'].max() <= 1.0
        assert processed_data['casual_norm'].min() >= 0.0
        
    finally:
        # 清理临时文件
        if temp_config_path.exists():
            temp_config_path.unlink()