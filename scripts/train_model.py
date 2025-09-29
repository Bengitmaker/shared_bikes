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

# Project root path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import config manager from configs directory
try:
    from configs.config_manager import ConfigManager
    config = ConfigManager()
except Exception as e:
    print(f"警告: 配置管理器初始化失败: {e}")
    config = None

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data
    
    Args:
        data_path (str): Data file path
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"加载数据: {data_path} (形状: {df.shape})")
    return df

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature data
    
    Args:
        df (pd.DataFrame): Raw data
        
    Returns:
        tuple: Feature data and scaler
    """
    # Extract hour information from datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    
    # Select features for clustering
    feature_columns = ['hour', 'workingday', 'weather', 'temp', 'humidity', 'windspeed']
    
    # Handle missing values
    df_clean = df.dropna(subset=feature_columns)
    
    # Extract features
    X = df_clean[feature_columns].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"特征准备完成: {X_scaled.shape}")
    return X_scaled, scaler, df_clean

def train_kmeans_model(X: np.ndarray, n_clusters: int, random_state: int = 42) -> KMeans:
    """
    Train K-means model
    
    Args:
        X (np.ndarray): Feature data
        n_clusters (int): Number of clusters
        random_state (int): Random seed
        
    Returns:
        KMeans: Trained model
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
    Save model and scaler
    
    Args:
        model (KMeans): Trained model
        scaler (StandardScaler): Scaler
        output_dir (str): Output directory
        model_name (str): Model name
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"模型已保存: {model_path}")
    
    # Save scaler
    scaler_path = output_path / f"{model_name}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"标准化器已保存: {scaler_path}")
    
    # Save model information
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
    """Main function"""
    print("开始训练共享单车聚类模型...\n")
    
    # Get config
    data_config = config.get_path('paths.data')
    models_config = config.get_path('paths.models')
    
    # Set default parameters
    data_path = data_path or str(data_config / "train.csv")  # 修改为正确的路径
    output_dir = output_dir or str(models_config)
    n_clusters = n_clusters or config.get('model.kmeans.n_clusters', 5)
    
    try:
        # 1. Load data
        df = load_data(data_path)
        
        # 2. Prepare features
        X, scaler, df_clean = prepare_features(df)
        
        # 3. Train model
        model = train_kmeans_model(X, n_clusters)
        
        # 4. Save model
        save_model(model, scaler, output_dir)
        
        print("\nSuccess 模型训练完成！")
        
    except Exception as e:
        print(f"Error 训练失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练共享单车聚类模型')
    parser.add_argument('--data-path', type=str, help='训练数据路径')
    parser.add_argument('--output-dir', type=str, help='模型输出目录')
    parser.add_argument('--n-clusters', type=int, help='聚类数量')
    
    args = parser.parse_args()
    main(args.data_path, args.output_dir, args.n_clusters)