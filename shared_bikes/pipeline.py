"""
Pipeline Execution Module

This module orchestrates the complete shared bikes usage pattern analysis workflow.
It can run the full pipeline or individual steps such as data setup, model training,
prediction, and evaluation.

The pipeline consists of the following steps:
1. Data setup - Creates directories and generates sample data
2. Model training - Trains the K-means clustering model
3. Prediction - Makes predictions on test data
4. Evaluation - Evaluates model performance and generates reports

This module provides a high-level interface to execute the entire analysis process
with a single command, or to run individual steps separately.

Example usage:
    shared-bikes-run --step all
    shared-bikes-run --step train
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess

# Add project root to path to enable module imports
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def run_step(script_name, args=None):
    """
    Run a single step of the pipeline using subprocess
    
    Args:
        script_name (str): Name of the script/module to run
        args (list, optional): List of arguments to pass to the script
        
    Returns:
        bool: True if step completed successfully, False otherwise
    """
    if args is None:
        args = []
    
    # Construct command to run the script
    cmd = [sys.executable, "-m", script_name] + args
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # Execute the command and capture output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False


def run_pipeline(step="all"):
    """
    Run the complete shared bikes analysis pipeline
    
    Args:
        step (str): Which step to run ('all', 'setup', 'train', 'predict', 'evaluate')
        
    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    print("开始执行共享单车使用模式分析管道...")
    
    # Step 1: Data setup
    if step == "all" or step == "setup":
        print("\n=== 步骤 1: 数据准备 ===")
        if not run_step("shared_bikes.setup"):
            print("数据准备步骤失败")
            return False
    
    # Step 2: Model training
    if step == "all" or step == "train":
        print("\n=== 步骤 2: 模型训练 ===")
        if not run_step("shared_bikes.train"):
            print("模型训练步骤失败")
            return False
    
    # Step 3: Prediction
    if step == "all" or step == "predict":
        print("\n=== 步骤 3: 模型预测 ===")
        predict_args = ["--data-path", "data/raw/test.csv"]
        if not run_step("shared_bikes.predict", predict_args):
            print("模型预测步骤失败")
            return False
    
    # Step 4: Evaluation
    if step == "all" or step == "evaluate":
        print("\n=== 步骤 4: 模型评估 ===")
        evaluate_args = ["--data-path", "data/raw/test.csv"]
        if not run_step("shared_bikes.evaluate", evaluate_args):
            print("模型评估步骤失败")
            return False
    
    print("\nSuccess 管道执行完成！")
    return True


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='运行完整管道')
    parser.add_argument('--step', type=str, default='all', 
                       choices=['all', 'setup', 'train', 'predict', 'evaluate'],
                       help='执行步骤')
    
    args = parser.parse_args()
    run_pipeline(args.step)