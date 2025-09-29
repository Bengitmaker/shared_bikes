import os
import sys
from pathlib import Path
import argparse
import subprocess
import logging

# Project root path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configuration manager
from configs.config_manager import ConfigManager

# Create global config instance
try:
    config = ConfigManager()
except Exception as e:
    print(f"警告: 配置管理器初始化失败: {e}")
    config = None

def setup_logging():
    """设置日志"""
    try:
        if config:
            log_file = config.get_path('logging.file')
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, encoding='utf-8'),
                    logging.StreamHandler(sys.stdout)
                ]
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(sys.stdout)
                ]
            )
    except Exception as e:
        print(f"警告: 日志配置失败: {e}")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    return logging.getLogger(__name__)

def run_script(script_name: str, args: list) -> bool:
    """
    运行指定的脚本
    
    Args:
        script_name (str): 脚本名称
        args (list): 参数列表
        
    Returns:
        bool: 是否成功
    """
    logger = logging.getLogger(__name__)
    
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)]
    
    if args:
        cmd.extend(args)
    
    try:
        logger.info(f"运行脚本: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        logger.info(f"{script_name} 执行成功")
        if result.stdout:
            logger.debug(f"标准输出: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"{script_name} 执行失败: {e}")
        if e.stdout:
            logger.error(f"标准输出: {e.stdout}")
        if e.stderr:
            logger.error(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"运行 {script_name} 时发生异常: {e}")
        return False

def main():
    """主函数"""
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description='运行共享单车分析管道')
    parser.add_argument('--step', type=str, choices=['all', 'setup', 'train', 'predict', 'evaluate'], 
                       default='all', help='要执行的步骤')
    parser.add_argument('--data-path', type=str, help='数据路径')
    parser.add_argument('--model-dir', type=str, help='模型目录')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    # 构建参数
    common_args = []
    if args.data_path:
        common_args.extend(['--data-path', args.data_path])
    if args.model_dir:
        common_args.extend(['--model-dir', args.model_dir])
    if args.output_dir:
        common_args.extend(['--output-dir', args.output_dir])
    
    success = True
    
    try:
        if args.step in ['all', 'setup']:
            logger.info("=== 步骤1: 数据准备 ===")
            if not run_script('setup_data.py', []):  # 修复：添加缺失的args参数
                success = False
                if args.step != 'all':
                    return
            
        if args.step in ['all', 'train']:
            logger.info("=== 步骤2: 模型训练 ===")
            train_args = common_args.copy()
            if not run_script('train_model.py', train_args):
                success = False
                if args.step != 'all':
                    return
            
        if args.step in ['all', 'predict']:
            logger.info("=== 步骤3: 预测 ===")
            predict_args = common_args.copy()
            # 使用正确的测试数据路径
            predict_args.extend(['--data-path', 'data/test.csv'])
            if not run_script('predict.py', predict_args):
                success = False
                if args.step != 'all':
                    return
            
        if args.step in ['all', 'evaluate']:
            logger.info("=== 步骤4: 模型评估 ===")
            eval_args = common_args.copy()
            eval_args.extend(['--data-path', 'data/train.csv'])
            if not run_script('evaluate_model.py', eval_args):
                success = False
                
    except KeyboardInterrupt:
        logger.error("管道执行被用户中断")
        success = False
    except Exception as e:
        logger.error(f"管道执行发生未预期的错误: {e}")
        success = False
    
    # 只有当所有指定步骤都成功执行时才报告成功
    # 如果某个步骤失败，则success为False，管道执行失败
    if success:
        logger.info("Success 管道执行完成！")
    else:
        logger.error("Error 管道执行失败")
        sys.exit(1)

if __name__ == "__main__":
    main()