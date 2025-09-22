"""
模型训练脚本

用于训练共享单车使用模式的聚类模型。
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_dir))

# 导入配置
from configs.config_manager import ConfigManager

# 创建配置实例
config = ConfigManager(str(root_dir / "configs" / "config.yaml"))

def load_data(data_path: str) -> pd.DataFrame:
    """
    加载数据
    
    Args:
        data_path (str): 数据文件路径
        
    Returns:
        pd.DataFrame: 加载的数据
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"加载数据: {data_path} (形状: {df.shape})")
    return df

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    准备特征数据
    
    Args:
        df (pd.DataFrame): 原始数据
        
    Returns:
        tuple: 特征数据和标准化器
    """
    # 从datetime列提取hour信息
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    
    # 选择用于聚类的特征
    feature_columns = ['hour', 'workingday', 'weather', 'temp', 'humidity', 'windspeed']
    
    # 处理缺失值
    df_clean = df.dropna(subset=feature_columns)
    
    # 提取特征
    X = df_clean[feature_columns].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"特征准备完成: {X_scaled.shape}")
    return X_scaled, scaler, df_clean

def train_kmeans_model(X: np.ndarray, n_clusters: int, random_state: int = 42) -> KMeans:
    """
    训练K-means模型
    
    Args:
        X (np.ndarray): 特征数据
        n_clusters (int): 聚类数量
        random_state (int): 随机种子
        
    Returns:
        KMeans: 训练好的模型
    """
    print(f"训练K-means模型 (n_clusters={n_clusters})...")
    
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=config.get('model.kmeans.max_iter', 300),
        n_init=config.get('model.kmeans.n_init', 10)
    )
    
    model.fit(X)
    print(f"模型训练完成 - WCSS: {model.inertia_:.2f}")
    
    return model

def save_model(model: KMeans, scaler: StandardScaler, output_dir: str, model_name: str = "kmeans_model"):
    """
    保存模型和标准化器
    
    Args:
        model (KMeans): 训练好的模型
        scaler (StandardScaler): 标准化器
        output_dir (str): 输出目录
        model_name (str): 模型名称
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = output_path / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"模型已保存: {model_path}")
    
    # 保存标准化器
    scaler_path = output_path / f"{model_name}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"标准化器已保存: {scaler_path}")
    
    # 保存模型信息
    info = {
        'model_name': model_name,
        'n_clusters': model.n_clusters, # type: ignore
        'inertia': model.inertia_,
        'timestamp': datetime.now().isoformat(),
        'feature_columns': [
            'hour', 'workingday', 'weather', 'temp', 'humidity', 'windspeed'
        ]
    }
    
    info_path = output_path / f"{model_name}_info.yaml"
    with open(info_path, 'w', encoding='utf-8') as f:
        yaml.dump(info, f, allow_unicode=True, default_flow_style=False)
    print(f"模型信息已保存: {info_path}")

def main(data_path: str = None, output_dir: str = None, n_clusters: int = None):  # type: ignore
    """主函数"""
    print("开始训练共享单车聚类模型...\n")
    
    # 获取配置
    data_config = config.get_path('paths.data')
    models_config = config.get_path('paths.models')
    
    # 设置默认参数
    data_path = data_path or str(data_config / "train.csv")  # 修改为正确的路径
    output_dir = output_dir or str(models_config)
    n_clusters = n_clusters or config.get('model.kmeans.n_clusters', 5)
    
    try:
        # 1. 加载数据
        df = load_data(data_path)
        
        # 2. 准备特征
        X, scaler, df_clean = prepare_features(df)
        
        # 3. 训练模型
        model = train_kmeans_model(X, n_clusters)
        
        # 4. 保存模型
        save_model(model, scaler, output_dir)
        
        print("\n✅ 模型训练完成！")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练共享单车聚类模型')
    parser.add_argument('--data-path', type=str, help='训练数据路径')
    parser.add_argument('--output-dir', type=str, help='模型输出目录')
    parser.add_argument('--n-clusters', type=int, help='聚类数量')
    
    args = parser.parse_args()
    main(args.data_path, args.output_dir, args.n_clusters)