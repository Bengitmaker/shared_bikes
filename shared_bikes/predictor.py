"""
Prediction Core Module

This module contains the core logic for making predictions using a trained
K-means clustering model. It provides functions for loading models, preparing
data for prediction, and generating cluster predictions.

Key features:
- Loading trained models and scalers from disk
- Preparing new data for prediction with the same preprocessing pipeline
- Making cluster predictions on new data
- Returning results with cluster labels attached

The module is designed to be used as part of a larger pipeline but can also
be used independently for prediction tasks.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Any


def load_model_and_scaler(model_dir: str, model_name: str = "kmeans_model") -> Tuple[Any, Any]:
    """
    Load a trained model and scaler from disk
    
    Args:
        model_dir (str): Directory containing the model files
        model_name (str): Base name of the model files
        
    Returns:
        tuple: (model, scaler) Loaded model and scaler
        
    Raises:
        FileNotFoundError: If model or scaler files do not exist
    """
    # Construct file paths for model and scaler
    model_path = Path(model_dir) / f"{model_name}.joblib"
    scaler_path = Path(model_dir) / f"{model_name}_scaler.joblib"
    
    # Check if files exist
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler


def prepare_prediction_data(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Prepare data for prediction using the same preprocessing pipeline as training
    
    This function:
    1. Extracts hour information from datetime column if not present
    2. Selects the same features used during training
    3. Handles missing values by removing affected rows
    
    Args:
        df (pd.DataFrame): Input data for prediction
        
    Returns:
        tuple: (features, cleaned_data)
            - features (np.ndarray): Feature matrix for prediction
            - cleaned_data (pd.DataFrame): Data with features and no missing values
            
    Raises:
        ValueError: If required feature columns are missing
    """
    # Extract hour information from datetime column if hour column is not present
    if 'hour' not in df.columns:
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    
    # Select the same features used during training
    feature_columns = ['hour', 'workingday', 'weather', 'temp', 'humidity', 'windspeed']
    
    # Check if all required feature columns are present
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"数据中缺少以下特征列: {missing_columns}")
    
    # Handle missing values by removing rows with missing features
    df_clean = df.dropna(subset=feature_columns)
    
    # Extract features as numpy array
    X = df_clean[feature_columns].values
    
    return X, df_clean


def predict_clusters(model, scaler, data_path: str) -> pd.DataFrame:
    """
    Make cluster predictions on data from a CSV file
    
    This function:
    1. Loads data from a CSV file
    2. Prepares the data for prediction
    3. Scales the features using the loaded scaler
    4. Makes predictions using the loaded model
    5. Returns the original data with cluster labels attached
    
    Args:
        model: Trained clustering model
        scaler: Fitted scaler used during training
        data_path (str): Path to CSV file containing data for prediction
        
    Returns:
        pd.DataFrame: Original data with 'cluster' column added
        
    Raises:
        FileNotFoundError: If data file does not exist
    """
    # Load data from CSV file
    if not Path(data_path).exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Prepare data for prediction
    X, df_clean = prepare_prediction_data(df)
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Attach predictions to original data
    result_df = df_clean.copy()
    result_df['cluster'] = predictions
    
    return result_df