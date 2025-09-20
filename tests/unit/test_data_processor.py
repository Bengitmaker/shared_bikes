"""
DataProcessor 类的单元测试
"""

import pytest
import pandas as pd
import numpy as np
from my_package.data_processor import DataProcessor


def test_data_processor_initialization():
    """测试DataProcessor初始化"""
    # 测试默认初始化
    processor = DataProcessor()
    assert isinstance(processor, DataProcessor)
    assert len(processor.processing_steps) == 0
    
    # 测试带配置的初始化
    config = {'test': 'value'}
    processor_with_config = DataProcessor(config)
    assert processor_with_config.config == config


def test_add_processing_step():
    """测试添加处理步骤"""
    processor = DataProcessor()
    
    def sample_function(data):
        return data
    
    processor.add_processing_step("test_step", sample_function)
    
    assert len(processor.processing_steps) == 1
    assert processor.processing_steps[0]['name'] == "test_step"
    assert processor.processing_steps[0]['function'] == sample_function


def test_process_method(sample_data):
    """测试process方法"""
    processor = DataProcessor()
    
    # 添加一个简单的处理函数
    def add_column(data):
        data_copy = data.copy()
        data_copy['new_col'] = 1
        return data_copy
    
    processor.add_processing_step("add_column", add_column)
    
    result = processor.process(sample_data)
    
    # 检查结果
    assert 'new_col' in result.columns
    assert len(result) == len(sample_data)
    assert result['new_col'].sum() == len(result)


def test_get_statistics(sample_data):
    """测试get_statistics方法"""
    processor = DataProcessor()
    stats = processor.get_statistics(sample_data)
    
    # 检查统计信息
    assert stats['shape'] == sample_data.shape
    assert isinstance(stats['missing_values'], dict)
    assert 'hour' in stats['numeric_columns']