"""
Data Processing Core Module

This module contains the core logic for data generation and preprocessing
in the shared bikes usage pattern analysis project. It provides functions
for creating sample data and setting up the required directory structure.

Key features:
- Generation of realistic sample shared bikes usage data
- Creation of necessary project directories
- Reproducible data generation with configurable parameters

The sample data includes realistic features such as:
- Datetime information
- Seasonal information
- Working day indicators
- Weather conditions
- Temperature and humidity
- Wind speed
- Casual and registered user counts
- Total usage counts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample shared bikes usage data with realistic patterns
    
    This function generates synthetic data that mimics real shared bikes usage,
    including temporal patterns, weather effects, and user behavior patterns.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Generated sample data with shared bikes usage features
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate datetime information
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate realistic features with appropriate distributions
    data = {
        'datetime': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
        'season': np.random.choice([1, 2, 3, 4], n_samples),
        'hour': [d.hour for d in dates],
        'workingday': [1 if d.weekday() < 5 else 0 for d in dates],
        'weather': np.random.choice([1, 2, 3, 4], n_samples, p=[0.6, 0.3, 0.08, 0.02]),
        'temp': np.random.normal(20, 10, n_samples).clip(0, 40),
        'atemp': np.random.normal(22, 10, n_samples).clip(0, 45),
        'humidity': np.random.uniform(20, 100, n_samples),
        'windspeed': np.random.exponential(10, n_samples).clip(0, 67),
        'casual': np.random.poisson(50, n_samples),
        'registered': np.random.poisson(200, n_samples)
    }
    
    # Create DataFrame and calculate total count
    df = pd.DataFrame(data)
    # Calculate total rental count as sum of casual and registered users
    df['count'] = df['casual'] + df['registered']
    
    return df


def setup_data_directories() -> Tuple[Path, Path, Path]:
    """
    Create the necessary data directories for the project
    
    This function creates the following directory structure:
    - data/
      - raw/
      - processed/
      
    Args:
        None
        
    Returns:
        tuple: (data_dir, raw_dir, processed_dir) Paths to the created directories
    """
    # Define directory structure
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # Create directories if they don't exist
    for directory in [data_dir, raw_dir, processed_dir]:
        directory.mkdir(exist_ok=True)
    
    return data_dir, raw_dir, processed_dir