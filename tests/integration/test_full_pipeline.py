"""
完整管道集成测试

测试从数据准备到模型训练、预测和评估的整个流程。
"""

import pytest
import pandas as pd
import numpy as np
from shared_bikes import (
    create_sample_data,
    load_data,
    prepare_features,
    train_kmeans_model,
    save_model,
    load_model_and_scaler,
    predict_clusters,
    create_evaluation_report
)
from pathlib import Path
import tempfile
import os


def test_full_pipeline():
    """测试完整管道流程"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. 创建示例数据
        print("步骤1: 创建示例数据")
        sample_data = create_sample_data(200)
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) == 200
        
        # 保存训练数据
        train_path = temp_path / "train.csv"
        sample_data.to_csv(train_path, index=False)
        
        # 2. 加载数据
        print("步骤2: 加载数据")
        df = load_data(str(train_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200
        
        # 3. 准备特征
        print("步骤3: 准备特征")
        X, scaler, df_clean = prepare_features(df)
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == len(df_clean)
        assert X.shape[1] == 6  # 6个特征
        
        # 4. 训练模型
        print("步骤4: 训练模型")
        model = train_kmeans_model(X, n_clusters=3, random_state=42)
        assert hasattr(model, 'predict')
        assert model.n_clusters == 3
        
        # 5. 保存模型
        print("步骤5: 保存模型")
        model_dir = temp_path / "models"
        save_model(model, scaler, str(model_dir), "test_kmeans")
        
        # 检查模型文件是否存在
        model_file = model_dir / "test_kmeans.joblib"
        scaler_file = model_dir / "test_kmeans_scaler.joblib"
        info_file = model_dir / "test_kmeans_info.yaml"
        assert model_file.exists()
        assert scaler_file.exists()
        assert info_file.exists()
        
        # 6. 加载模型和标准化器
        print("步骤6: 加载模型和标准化器")
        loaded_model, loaded_scaler = load_model_and_scaler(str(model_dir), "test_kmeans")
        assert loaded_model is not None
        assert loaded_scaler is not None
        
        # 7. 创建测试数据并进行预测
        print("步骤7: 创建测试数据并进行预测")
        test_data = create_sample_data(50)
        test_path = temp_path / "test.csv"
        test_data.to_csv(test_path, index=False)
        
        result_df = predict_clusters(loaded_model, loaded_scaler, str(test_path))
        assert isinstance(result_df, pd.DataFrame)
        assert 'cluster' in result_df.columns
        assert len(result_df) == 50
        assert set(result_df['cluster'].unique()).issubset({0, 1, 2})
        
        # 8. 创建评估报告
        print("步骤8: 创建评估报告")
        report, metrics = create_evaluation_report(loaded_model, loaded_scaler, df)
        assert isinstance(report, str)
        assert isinstance(metrics, dict)
        assert "# 共享单车聚类模型评估报告" in report
        assert len(metrics) >= 3  # 至少包含3个指标
        
        print("完整管道测试通过！")


def test_pipeline_with_missing_data():
    """测试包含缺失数据的管道流程"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建包含缺失值的数据
        sample_data = create_sample_data(100)
        # 随机设置一些值为NaN
        sample_data.loc[5:10, 'temp'] = np.nan
        sample_data.loc[15:20, 'humidity'] = np.nan
        
        # 保存数据
        train_path = temp_path / "train_with_missing.csv"
        sample_data.to_csv(train_path, index=False)
        
        # 执行标准流程
        df = load_data(str(train_path))
        X, scaler, df_clean = prepare_features(df)
        
        # 检查缺失值是否被正确处理（行数应该减少）
        assert len(df_clean) < len(df)
        
        # 训练模型
        model = train_kmeans_model(X, n_clusters=2, random_state=42)
        assert model.n_clusters == 2
        
        print("包含缺失数据的管道测试通过！")