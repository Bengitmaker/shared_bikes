"""
评估模块的单元测试
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shared_bikes.evaluator import (
    load_model_and_data,
    calculate_clustering_metrics,
    create_evaluation_report
)
from pathlib import Path
import tempfile
import os
import joblib


def test_calculate_clustering_metrics():
    """测试聚类评估指标计算功能"""
    # 创建测试数据
    np.random.seed(42)
    X_scaled = np.random.rand(100, 6)
    labels = np.random.choice([0, 1, 2], 100)
    
    # 计算指标
    metrics = calculate_clustering_metrics(X_scaled, labels)
    
    # 检查返回值
    assert isinstance(metrics, dict)
    assert 'silhouette' in metrics
    assert 'calinski_harabasz' in metrics
    assert 'davies_bouldin' in metrics
    
    # 检查指标值类型
    for metric_value in metrics.values():
        assert isinstance(metric_value, (float, np.floating)) or np.isnan(metric_value)


def test_calculate_clustering_metrics_edge_cases():
    """测试边界情况下的指标计算"""
    # 创建只有一个簇的情况
    X_scaled = np.random.rand(100, 6)
    labels = np.zeros(100)  # 所有样本都属于同一个簇
    
    metrics = calculate_clustering_metrics(X_scaled, labels)
    
    # 检查指标值（某些指标在这种情况下可能无法计算）
    assert isinstance(metrics, dict)


def test_create_evaluation_report():
    """测试评估报告创建功能"""
    # 创建测试数据
    test_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=10, freq='H'),
        'workingday': [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        'weather': [1, 1, 2, 2, 1, 3, 3, 1, 1, 2],
        'temp': [20.0, 21.0, 22.0, 23.0, 24.0, 18.0, 19.0, 20.0, 21.0, 22.0],
        'humidity': [60.0, 65.0, 70.0, 65.0, 60.0, 75.0, 80.0, 60.0, 65.0, 70.0],
        'windspeed': [5.0, 6.0, 7.0, 5.0, 4.0, 8.0, 10.0, 5.0, 6.0, 7.0]
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建并保存测试模型
        np.random.seed(42)
        X = np.random.rand(100, 6)
        scaler = StandardScaler()
        scaler.fit(X)
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)
        
        # 创建评估报告
        report, metrics = create_evaluation_report(model, scaler, test_data)
        
        # 检查返回值
        assert isinstance(report, str)
        assert isinstance(metrics, dict)
        
        # 检查报告内容
        assert "# 共享单车聚类模型评估报告" in report
        assert "模型类型: K-means" in report
        assert "聚类数量: 3" in report
        
        # 检查指标
        assert 'silhouette' in metrics
        assert 'calinski_harabasz' in metrics
        assert 'davies_bouldin' in metrics


def test_create_evaluation_report_missing_columns():
    """测试缺少必要特征列的情况"""
    # 创建缺少特征列的测试数据
    test_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=5, freq='H'),
        'workingday': [1, 1, 0, 0, 1]
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试模型
        np.random.seed(42)
        X = np.random.rand(100, 6)
        scaler = StandardScaler()
        scaler.fit(X)
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)
        
        # 应该抛出ValueError
        with pytest.raises(ValueError, match="数据中缺少以下特征列"):
            create_evaluation_report(model, scaler, test_data)