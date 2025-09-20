"""
管道运行脚本

协调数据准备、模型训练、预测和评估的完整流程。
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess
import logging

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# 导入配置
from shared_bikes.configs import config

def setup_logging():
    """设置日志"""
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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"{script_name} 执行成功")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"{script_name} 执行失败: {e}")
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
            if not run_script('setup_data.py'): # type: ignore
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
            if args.data_path:
                predict_args.remove('--data-path')
                predict_args.remove(args.data_path)
            predict_args.extend(['--data-path', 'data/raw/test.csv'])
            if not run_script('predict.py', predict_args):
                success = False
                if args.step != 'all':
                    return
            
        if args.step in ['all', 'evaluate']:
            logger.info("=== 步骤4: 模型评估 ===")
            eval_args = common_args.copy()
            eval_args.extend(['--data-path', 'data/raw/train.csv'])
            if not run_script('evaluate_model.py', eval_args):
                success = False
                
    except KeyboardInterrupt:
        logger.error("管道执行被用户中断")
        success = False
    except Exception as e:
        logger.error(f"管道执行发生未预期的错误: {e}")
        success = False
    
    if success:
        logger.info("✅ 管道执行完成！")
    else:
        logger.error("❌ 管道执行失败")
        sys.exit(1)

if __name__ == "__main__":
    main()