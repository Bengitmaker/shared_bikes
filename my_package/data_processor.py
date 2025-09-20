"""
数据处理模块示例

这是一个预留功能模块的示例，展示了如何在my_package中实现新功能。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class DataProcessor:
    """通用数据处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据处理器
        
        Args:
            config (Dict): 配置参数
        """
        self.config = config or {}
        self.processing_steps = []
        
    def add_processing_step(self, step_name: str, function) -> None:
        """
        添加数据处理步骤
        
        Args:
            step_name (str): 步骤名称
            function (callable): 处理函数
        """
        self.processing_steps.append({
            'name': step_name,
            'function': function
        })
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        执行数据处理流程
        
        Args:
            data (pd.DataFrame): 输入数据
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        result = data.copy()
        
        for step in self.processing_steps:
            print(f"执行处理步骤: {step['name']}")
            result = step['function'](result)
            
        return result
    
    def get_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Args:
            data (pd.DataFrame): 输入数据
            
        Returns:
            Dict: 统计信息
        """
        stats = {
            'shape': data.shape,
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['object']).columns)
        }
        
        return stats