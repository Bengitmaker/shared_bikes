"""
预测脚本

使用训练好的模型进行预测。
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import argparse

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# 导入配置
from shared_bikes.configs import config

def load_model_and_scaler(model_dir: str, model_name: str = "kmeans_model"):
    """
    加载模型和标准化器
    
    Args:
        model_dir (str): 模型目录
        model_name (str): 模型名称
        
    Returns:
        tuple: 模型和标准化器
    """
    model_path = Path(model_dir) / f"{model_name}.joblib"
    scaler_path = Path(model_dir) / f"{model_name}_scaler.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"加载模型: {model_path}")
    print(f"加载标准化器: {scaler_path}")
    
    return model, scaler

def prepare_prediction_data(df: pd.DataFrame) -> np.ndarray:
    """
    准备预测数据
    
    Args:
        df (pd.DataFrame): 原始数据
        
    Returns:
        np.ndarray: 准备好的特征数据
    """
    # 选择与训练时相同的特征
    feature_columns = ['hour', 'workingday', 'weather', 'temp', 'humidity', 'windspeed']
    
    # 处理缺失值
    df_clean = df.dropna(subset=feature_columns)
    
    # 提取特征
    X = df_clean[feature_columns].values
    
    print(f"预测数据准备完成: {X.shape}")
    return X

def predict_clusters(data_path: str, model_dir: str, output_path: str):
    """
    执行预测
    
    Args:
        data_path (str): 数据路径
        model_dir (str): 模型目录
        output_path (str): 输出路径
    """
    print("开始执行预测...\n")
    
    # 1. 加载数据
    if not Path(data_path).exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"加载数据: {data_path} (形状: {df.shape})")
    
    # 2. 加载模型和标准化器
    model, scaler = load_model_and_scaler(model_dir)
    
    # 3. 准备预测数据
    X, df_clean = prepare_prediction_data(df)
    X_scaled = scaler.transform(X)
    
    # 4. 执行预测
    predictions = model.predict(X_scaled)
    print(f"预测完成 - 聚类结果: {np.unique(predictions)}")
    
    # 5. 合并结果
    result_df = df_clean.copy()
    result_df['cluster'] = predictions
    
    # 6. 保存结果
    if output_path is None:
        output_path = str(Path(data_path).parent / f"{Path(data_path).stem}_with_clusters.csv")
    
    result_df.to_csv(output_path, index=False)
    print(f"预测结果已保存: {output_path}")
    
    # 7. 打印统计信息
    print("\n聚类统计:")
    cluster_counts = result_df['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(result_df)) * 100
        print(f"  聚类 {cluster_id}: {count} 样本 ({percentage:.1f}%)")
    
    print("\n✅ 预测完成！")
    return result_df

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    parser.add_argument('--data-path', type=str, required=True, help='输入数据路径')
    parser.add_argument('--model-dir', type=str, help='模型目录')
    parser.add_argument('--output-path', type=str, help='输出文件路径')
    
    args = parser.parse_args()
    
    # 获取配置
    models_config = config.get_path('paths.models')
    model_dir = args.model_dir or str(models_config)
    
    try:
        predict_clusters(args.data_path, model_dir, args.output_path)
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()