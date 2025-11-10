"""
Shared Bikes Analysis Package

This is the main package for the shared bikes usage pattern analysis project.
It provides a complete solution for analyzing bike sharing usage patterns through
clustering algorithms, including data processing, model training, prediction,
and evaluation capabilities.

The package is organized into the following modules:
- data_processor: Data generation and preprocessing utilities
- model_trainer: Core model training functionality
- predictor: Prediction using trained models
- evaluator: Model evaluation and reporting
"""

import os
import sys
from pathlib import Path

# Package root directory
PACKAGE_ROOT = Path(__file__).parent.parent.absolute()

# Add package root directory to system path
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Import core modules
from configs.config_manager import ConfigManager

# Create global config instance
config = ConfigManager()

__version__ = "2.0.0"
__author__ = "Your Name"

# 导出核心功能模块
from .model_trainer import load_data, prepare_features, train_kmeans_model, save_model
from .predictor import load_model_and_scaler, prepare_prediction_data, predict_clusters
from .evaluator import load_model_and_data, calculate_clustering_metrics, create_evaluation_report
from .data_processor import create_sample_data, setup_data_directories

__all__ = [
    'ConfigManager', 
    'config',
    'load_data',
    'prepare_features', 
    'train_kmeans_model', 
    'save_model',
    'load_model_and_scaler',
    'prepare_prediction_data',
    'predict_clusters',
    'load_model_and_data',
    'calculate_clustering_metrics',
    'create_evaluation_report',
    'create_sample_data',
    'setup_data_directories'
]