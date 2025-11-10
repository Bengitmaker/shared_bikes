"""
Command Line Interface for Shared Bikes Analysis Project

This module provides the command-line interface for the shared bikes usage pattern analysis project.
It allows users to run the full pipeline or individual steps such as data setup, model training,
prediction, and evaluation.

The CLI supports the following commands:
- setup: Prepare project data
- train: Train the clustering model
- predict: Make predictions using the trained model
- evaluate: Evaluate model performance
- run: Run the complete pipeline

Example usage:
    shared-bikes-setup
    shared-bikes-train
    shared-bikes-predict --data-path data/raw/test.csv
    shared-bikes-evaluate --data-path data/raw/test.csv
    shared-bikes-run --step all
"""

import argparse
import sys
from pathlib import Path

# Add project root to path to enable module imports
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import main functions from submodules
from shared_bikes.train import main as train_main
from shared_bikes.predict import main as predict_main
from shared_bikes.evaluate import main as evaluate_main
from shared_bikes.setup import main as setup_main


def main():
    """
    Main CLI entry point
    
    Parses command line arguments and executes the corresponding function.
    """
    parser = argparse.ArgumentParser(
        description="共享单车使用模式聚类分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
命令:
    setup      设置项目数据
    train      训练聚类模型
    predict    使用模型进行预测
    evaluate   评估模型性能
    run        运行完整管道
    
示例:
    shared-bikes-setup
    shared-bikes-train
    shared-bikes-predict --data-path data/raw/test.csv
    shared-bikes-evaluate --data-path data/raw/test.csv
    shared-bikes-run --step all
        """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # Setup subcommand - prepares project data
    setup_parser = subparsers.add_parser('setup', help='设置项目数据')
    
    # Train subcommand - trains the clustering model
    train_parser = subparsers.add_parser('train', help='训练聚类模型')
    train_parser.add_argument('--data-path', type=str, help='训练数据路径')
    train_parser.add_argument('--output-dir', type=str, help='模型输出目录')
    train_parser.add_argument('--n-clusters', type=int, help='聚类数量')
    
    # Predict subcommand - makes predictions using trained model
    predict_parser = subparsers.add_parser('predict', help='使用模型进行预测')
    predict_parser.add_argument('--data-path', type=str, required=True, help='输入数据路径')
    predict_parser.add_argument('--model-dir', type=str, help='模型目录')
    predict_parser.add_argument('--output-path', type=str, help='输出文件路径')
    
    # Evaluate subcommand - evaluates model performance
    evaluate_parser = subparsers.add_parser('evaluate', help='评估模型性能')
    evaluate_parser.add_argument('--data-path', type=str, required=True, help='输入数据路径')
    evaluate_parser.add_argument('--model-dir', type=str, help='模型目录')
    evaluate_parser.add_argument('--output-dir', type=str, help='输出目录')
    
    # Run pipeline subcommand - runs the complete pipeline
    run_parser = subparsers.add_parser('run', help='运行完整管道')
    run_parser.add_argument('--step', type=str, default='all', 
                           choices=['all', 'setup', 'train', 'predict', 'evaluate'],
                           help='执行步骤')
    
    # Parse arguments and execute corresponding function
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_main()
    elif args.command == 'train':
        train_main(args.data_path, args.output_dir, args.n_clusters)
    elif args.command == 'predict':
        predict_main(args)
    elif args.command == 'evaluate':
        evaluate_main(args)
    elif args.command == 'run':
        # Import and run pipeline
        try:
            from shared_bikes.pipeline import run_pipeline
            run_pipeline(args.step)
        except ImportError:
            print("Error: pipeline module not found")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()