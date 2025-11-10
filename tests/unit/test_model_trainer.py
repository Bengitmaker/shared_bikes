"""
模型训练模块的单元测试
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shared_bikes.model_trainer import (
    load_data, 
    prepare_features, 
    train_kmeans_model, 
    save_model
)
from pathlib import Path
import tempfile
import os


def test_load_data():
    """测试数据加载功能"""
    # 创建临时CSV文件用于测试
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("col1,col2\n1,2\n3,4\n")
        temp_file = f.name
    
    try:
        # 测试正常加载
        df = load_data(temp_file)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        
        # 测试文件不存在的情况
        with pytest.raises(FileNotFoundError):
            load_data("non_existent_file.csv")
            
    finally:
        # 清理临时文件
        os.unlink(temp_file)


def test_prepare_features():
    """测试特征准备功能"""
    # 创建测试数据
    test_data = pd.DataFrame({
        'datetime': ['2023-01-01 10:00:00', '2023-01-01 11:00:00'],
        'workingday': [1, 0],
        'weather': [1, 2],
        'temp': [20.0, 25.0],
        'humidity': [60.0, 70.0],
        'windspeed': [5.0, 10.0]
    })
    
    X, scaler, df_clean = prepare_features(test_data)
    
    # 检查返回值类型
    assert isinstance(X, np.ndarray)
    assert isinstance(scaler, StandardScaler)
    assert isinstance(df_clean, pd.DataFrame)
    
    # 检查特征数量
    assert X.shape[1] == 6  # 6个特征列


def test_train_kmeans_model():
    """测试K-means模型训练功能"""
    # 创建测试数据
    np.random.seed(42)
    X = np.random.rand(100, 6)
    
    # 训练模型
    model = train_kmeans_model(X, n_clusters=3, random_state=42)
    
    # 检查模型类型
    assert isinstance(model, KMeans)
    
    # 检查聚类数量
    assert model.n_clusters == 3
    
    # 检查预测功能
    labels = model.predict(X)
    assert len(labels) == 100
    assert set(labels).issubset({0, 1, 2})


def test_save_model():
    """测试模型保存功能"""
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试模型和标准化器
        np.random.seed(42)
        X = np.random.rand(100, 6)
        scaler = StandardScaler()
        scaler.fit(X)
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)
        
        # 保存模型
        save_model(model, scaler, temp_dir, "test_model")
        
        # 检查文件是否创建
        model_path = Path(temp_dir) / "test_model.joblib"
        scaler_path = Path(temp_dir) / "test_model_scaler.joblib"
        info_path = Path(temp_dir) / "test_model_info.yaml"
        
        assert model_path.exists()
        assert scaler_path.exists()
        assert info_path.exists()