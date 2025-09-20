"""
Pytest 配置文件

提供共享的测试夹具和配置。
"""

import pytest
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 将项目根目录添加到Python路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

@pytest.fixture(scope="session")
def sample_data():
    """创建示例数据用于测试"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'hour': np.random.randint(0, 24, n_samples),
        'workingday': np.random.choice([0, 1], n_samples),
        'weather': np.random.randint(1, 5, n_samples),
        'temp': np.random.uniform(0, 35, n_samples),
        'atemp': np.random.uniform(0, 50, n_samples),
        'humidity': np.random.uniform(20, 100, n_samples),
        'windspeed': np.random.uniform(0, 67, n_samples),
        'casual': np.random.poisson(50, n_samples),
        'registered': np.random.poisson(200, n_samples)
    }
    
    df = pd.DataFrame(data)
    # 确保没有缺失值
    return df

@pytest.fixture(scope="session")
def test_config():
    """测试用的配置字典"""
    return {
        'paths': {
            'data': 'data/',
            'output': 'tests/output/',
            'models': 'tests/models/'
        },
        'data_processing': {
            'chunk_size': 1000,
            'date_format': '%Y-%m-%d %H:%M:%S'
        },
        'model': {
            'kmeans': {
                'n_clusters': 3,
                'random_state': 42,
                'max_iter': 300
            }
        }
    }

@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """创建临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir