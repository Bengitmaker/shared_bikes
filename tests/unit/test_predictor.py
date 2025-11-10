"""
预测模块的单元测试
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shared_bikes.predictor import (
    load_model_and_scaler,
    prepare_prediction_data,
    predict_clusters
)
from pathlib import Path
import tempfile
import os
import joblib


def test_prepare_prediction_data():
    """测试预测数据准备功能"""
    # 创建测试数据
    test_data = pd.DataFrame({
        'datetime': ['2023-01-01 10:00:00', '2023-01-01 11:00:00'],
        'workingday': [1, 0],
        'weather': [1, 2],
        'temp': [20.0, 25.0],
        'humidity': [60.0, 70.0],
        'windspeed': [5.0, 10.0]
    })
    
    X, df_clean = prepare_prediction_data(test_data)
    
    # 检查返回值类型
    assert isinstance(X, np.ndarray)
    assert isinstance(df_clean, pd.DataFrame)
    
    # 检查特征数量
    assert X.shape[1] == 6  # 6个特征列
    
    # 检查hour列是否正确提取
    assert 'hour' in df_clean.columns
    assert df_clean['hour'].tolist() == [10, 11]


def test_prepare_prediction_data_missing_columns():
    """测试缺少必要特征列的情况"""
    # 创建缺少特征列的测试数据
    test_data = pd.DataFrame({
        'datetime': ['2023-01-01 10:00:00'],
        'workingday': [1]
    })
    
    # 应该抛出ValueError
    with pytest.raises(ValueError, match="数据中缺少以下特征列"):
        prepare_prediction_data(test_data)


def test_predict_clusters():
    """测试聚类预测功能"""
    # 创建测试数据
    test_data = pd.DataFrame({
        'datetime': ['2023-01-01 10:00:00', '2023-01-01 11:00:00'],
        'workingday': [1, 0],
        'weather': [1, 2],
        'temp': [20.0, 25.0],
        'humidity': [60.0, 70.0],
        'windspeed': [5.0, 10.0]
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建并保存测试模型
        np.random.seed(42)
        X = np.random.rand(100, 6)
        scaler = StandardScaler()
        scaler.fit(X)
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(X)
        
        model_path = Path(temp_dir) / "kmeans_model.joblib"
        scaler_path = Path(temp_dir) / "kmeans_model_scaler.joblib"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # 创建测试CSV文件
        csv_path = Path(temp_dir) / "test_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        # 执行预测
        result_df = predict_clusters(model, scaler, str(csv_path))
        
        # 检查结果
        assert isinstance(result_df, pd.DataFrame)
        assert 'cluster' in result_df.columns
        assert len(result_df) == 2
        assert set(result_df['cluster'].unique()).issubset({0, 1})


def test_predict_clusters_file_not_found():
    """测试数据文件不存在的情况"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试模型
        np.random.seed(42)
        X = np.random.rand(10, 6)
        scaler = StandardScaler()
        scaler.fit(X)
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(X)
        
        # 测试不存在的文件
        with pytest.raises(FileNotFoundError):
            predict_clusters(model, scaler, "non_existent_file.csv")