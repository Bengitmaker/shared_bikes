"""
模型工具模块

提供常用的机器学习模型工具和评估功能。
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path


def calculate_clustering_metrics(X: np.ndarray, 
                               labels: np.ndarray,
                               metric_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    计算聚类评估指标
    
    Args:
        X (np.ndarray): 特征数据
        labels (np.ndarray): 聚类标签
        metric_names (List[str], optional): 要计算的指标名称列表
        
    Returns:
        Dict[str, float]: 指标名称和值的字典
    """
    if metric_names is None:
        metric_names = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    
    metrics = {}
    
    # 轮廓系数
    if 'silhouette' in metric_names:
        try:
            metrics['silhouette'] = silhouette_score(X, labels)
        except Exception:
            metrics['silhouette'] = np.nan
    
    # Calinski-Harabasz指数
    if 'calinski_harabasz' in metric_names:
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        except Exception:
            metrics['calinski_harabasz'] = np.nan
    
    # Davies-Bouldin指数
    if 'davies_bouldin' in metric_names:
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        except Exception:
            metrics['davies_bouldin'] = np.nan
            
    return metrics


def calculate_regression_metrics(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               metric_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    计算回归评估指标
    
    Args:
        y_true (np.ndarray): 真实值
        y_pred (np.ndarray): 预测值
        metric_names (List[str], optional): 要计算的指标名称列表
        
    Returns:
        Dict[str, float]: 指标名称和值的字典
    """
    if metric_names is None:
        metric_names = ['mse', 'rmse', 'mae', 'r2']
    
    metrics = {}
    
    # 均方误差
    if 'mse' in metric_names:
        metrics['mse'] = mean_squared_error(y_true, y_pred)
    
    # 均方根误差
    if 'rmse' in metric_names:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 平均绝对误差
    if 'mae' in metric_names:
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # R²分数
    if 'r2' in metric_names:
        metrics['r2'] = r2_score(y_true, y_pred)
        
    return metrics


def perform_cross_validation(model, 
                           X: np.ndarray, 
                           y: np.ndarray, 
                           cv: int = 5,
                           scoring: str = 'r2') -> Dict[str, float]:
    """
    执行交叉验证
    
    Args:
        model: 机器学习模型
        X (np.ndarray): 特征数据
        y (np.ndarray): 目标变量
        cv (int): 交叉验证折数
        scoring (str): 评分方法
        
    Returns:
        Dict[str, float]: 包含交叉验证结果的字典
    """
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
    except Exception as e:
        return {
            'mean': np.nan,
            'std': np.nan,
            'scores': [],
            'error': str(e)
        }


def save_model(model, 
               filepath: str, 
               metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    保存模型到文件
    
    Args:
        model: 要保存的模型
        filepath (str): 文件路径
        metadata (Dict, optional): 元数据
        
    Returns:
        bool: 是否保存成功
    """
    try:
        # 创建目录
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        joblib.dump(model, filepath)
        
        # 如果有元数据，也保存
        if metadata:
            metadata_path = Path(filepath).with_suffix('.meta.pkl')
            joblib.dump(metadata, metadata_path)
            
        return True
    except Exception:
        return False


def load_model(filepath: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    从文件加载模型
    
    Args:
        filepath (str): 文件路径
        
    Returns:
        Tuple: (模型, 元数据)
    """
    try:
        # 加载模型
        model = joblib.load(filepath)
        
        # 尝试加载元数据
        metadata_path = Path(filepath).with_suffix('.meta.pkl')
        metadata = None
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            
        return model, metadata
    except Exception:
        return None, None