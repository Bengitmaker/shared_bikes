"""
Model Training Core Module

This module contains the core logic for training the K-means clustering model
for shared bikes usage pattern analysis. It provides functions for data loading,
feature preparation, model training, and model saving.

Key features:
- Data loading from CSV files
- Feature engineering and preprocessing
- K-means model training with configurable parameters
- Model persistence using joblib
- Model metadata saving in YAML format

The module is designed to be used as part of a larger pipeline but can also
be used independently for model training tasks.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file
    
    Args:
        data_path (str): Path to the CSV file containing the data
        
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame
        
    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
    df = pd.read_csv(data_path)
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """
    Prepare features for clustering analysis
    
    This function:
    1. Extracts hour information from the datetime column
    2. Selects relevant features for clustering
    3. Handles missing values by removing affected rows
    4. Standardizes features using StandardScaler
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        tuple: (scaled_features, scaler, cleaned_data)
            - scaled_features (np.ndarray): Standardized feature matrix
            - scaler (StandardScaler): Fitted scaler for future use
            - cleaned_data (pd.DataFrame): Data with features and no missing values
    """
    # Extract hour information from datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    
    # Select features for clustering
    feature_columns = ['hour', 'workingday', 'weather', 'temp', 'humidity', 'windspeed']
    
    # Handle missing values by removing rows with missing features
    df_clean = df.dropna(subset=feature_columns)
    
    # Extract features as numpy array
    X = df_clean[feature_columns].values
    
    # Standardize features to have zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, df_clean


def train_kmeans_model(X: np.ndarray, n_clusters: int, random_state: int = 42, 
                       max_iter: int = 300, n_init: int = 10) -> KMeans:
    """
    Train a K-means clustering model
    
    Args:
        X (np.ndarray): Feature matrix to train on
        n_clusters (int): Number of clusters to form
        random_state (int): Random seed for reproducibility
        max_iter (int): Maximum number of iterations
        n_init (int): Number of times the algorithm will be run with different centroid seeds
        
    Returns:
        KMeans: Trained K-means model
    """
    # Initialize and configure the K-means model
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init
    )
    
    # Fit the model to the data
    model.fit(X)
    return model


def save_model(model: KMeans, scaler: StandardScaler, output_dir: str, model_name: str = "kmeans_model"):
    """
    Save the trained model, scaler, and model metadata
    
    This function saves three files:
    1. The trained model (.joblib)
    2. The fitted scaler (.joblib)
    3. Model metadata (.yaml)
    
    Args:
        model (KMeans): Trained K-means model
        scaler (StandardScaler): Fitted scaler
        output_dir (str): Directory to save the files
        model_name (str): Base name for the saved files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the trained model
    model_path = output_path / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    
    # Save the fitted scaler
    scaler_path = output_path / f"{model_name}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    
    # Save model metadata
    info = {
        'model_name': model_name,
        'n_clusters': int(model.n_clusters),
        'inertia': float(model.inertia_),
        'timestamp': datetime.now().isoformat(),
        'feature_columns': [
            'hour', 'workingday', 'weather', 'temp', 'humidity', 'windspeed'
        ]
    }
    
    info_path = output_path / f"{model_name}_info.yaml"
    with open(info_path, 'w', encoding='utf-8') as f:
        yaml.dump(info, f, allow_unicode=True, default_flow_style=False)