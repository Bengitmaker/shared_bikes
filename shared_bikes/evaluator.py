"""
Model Evaluation Core Module

This module contains the core logic for evaluating the performance of a
trained K-means clustering model. It provides functions for loading models
and data, calculating clustering metrics, and generating evaluation reports.

Key evaluation metrics:
- Silhouette Score: Measures how similar an object is to its own cluster compared to other clusters
- Calinski-Harabasz Index: Measures cluster cohesion and separation
- Davies-Bouldin Index: Measures the average similarity between each cluster and its most similar cluster

The module also generates comprehensive evaluation reports and can be extended
to include visualization capabilities.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pathlib import Path
from typing import Dict, Tuple, Any


def load_model_and_data(model_dir: str, data_path: str) -> Tuple[Any, Any, pd.DataFrame]:
    """
    Load a trained model, scaler, and data for evaluation
    
    Args:
        model_dir (str): Directory containing the model files
        data_path (str): Path to the data file for evaluation
        
    Returns:
        tuple: (model, scaler, data) Loaded model, scaler, and data
        
    Raises:
        FileNotFoundError: If model, scaler, or data files do not exist
    """
    # Construct file paths for model and scaler
    model_path = Path(model_dir) / "kmeans_model.joblib"
    scaler_path = Path(model_dir) / "kmeans_model_scaler.joblib"
    
    # Check if model and scaler files exist
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load data for evaluation
    if not Path(data_path).exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    return model, scaler, df


def calculate_clustering_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate clustering evaluation metrics
    
    This function calculates three key metrics:
    1. Silhouette Score: Ranges from -1 to 1, higher values indicate better clustering
    2. Calinski-Harabasz Index: Higher values indicate better clustering
    3. Davies-Bouldin Index: Lower values indicate better clustering
    
    Args:
        X_scaled (np.ndarray): Standardized feature matrix
        labels (np.ndarray): Cluster labels assigned to each data point
        
    Returns:
        dict: Dictionary containing the calculated metrics
    """
    metrics = {}
    
    # Calculate Silhouette Score (range: [-1, 1], higher is better)
    try:
        metrics['silhouette'] = float(silhouette_score(X_scaled, labels))
    except Exception:
        # Silhouette score may fail with certain cluster configurations
        metrics['silhouette'] = np.nan
        
    # Calculate Calinski-Harabasz Index (higher is better)
    try:
        metrics['calinski_harabasz'] = float(calinski_harabasz_score(X_scaled, labels))
    except Exception:
        metrics['calinski_harabasz'] = np.nan
        
    # Calculate Davies-Bouldin Index (lower is better)
    try:
        metrics['davies_bouldin'] = float(davies_bouldin_score(X_scaled, labels))
    except Exception:
        metrics['davies_bouldin'] = np.nan
        
    return metrics


def create_evaluation_report(model, scaler, df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    """
    Create a comprehensive evaluation report for the clustering model
    
    This function:
    1. Prepares the data for evaluation
    2. Makes predictions using the model
    3. Calculates clustering metrics
    4. Generates a formatted report with results
    
    Args:
        model: Trained clustering model
        scaler: Fitted scaler used during training
        df (pd.DataFrame): Data to evaluate the model on
        
    Returns:
        tuple: (report, metrics)
            - report (str): Formatted evaluation report in Markdown format
            - metrics (dict): Dictionary of calculated metrics
    """
    # Prepare features for evaluation
    # Extract hour information from datetime column if hour column is not present
    if 'hour' not in df.columns:
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    
    # Select features used during training
    feature_columns = ['hour', 'workingday', 'weather', 'temp', 'humidity', 'windspeed']
    
    # Check if all required feature columns are present
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"数据中缺少以下特征列: {missing_columns}")
    
    # Handle missing values by removing rows with missing features
    df_clean = df.dropna(subset=feature_columns)
    X = df_clean[feature_columns].values
    X_scaled = scaler.transform(X)
    
    # Make predictions
    labels = model.predict(X_scaled)
    
    # Calculate evaluation metrics
    metrics = calculate_clustering_metrics(X_scaled, labels)
    
    # Create formatted evaluation report
    report = f"""# 共享单车聚类模型评估报告

## 基本信息
- 模型类型: K-means
- 聚类数量: {model.n_clusters}
- 样本数量: {len(df_clean)}
- 特征数量: {X.shape[1]}
- 评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 评估指标
- **轮廓系数**: {metrics.get('silhouette', 'N/A'):>.3f}
  - 解释: 值越接近1表示聚类效果越好
- **Calinski-Harabasz指数**: {metrics.get('calinski_harabasz', 'N/A'):>.3f}
  - 解释: 值越大表示聚类效果越好
- **Davies-Bouldin指数**: {metrics.get('davies_bouldin', 'N/A'):>.3f}
  - 解释: 值越小表示聚类效果越好

## 聚类分布
"""
    
    # Add cluster distribution information
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(labels)) * 100
        report += f"- 聚类 {cluster_id}: {count} 样本 ({percentage:.1f}%)\n"

    return report, metrics