"""
文件工具模块

提供常用的文件操作功能。
"""

import os
import json
import yaml
import pickle
from pathlib import Path
from typing import Union, Any, Optional, Dict, List
import shutil


def ensure_directory_exists(path: Union[str, Path]) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path (Union[str, Path]): 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    读取JSON文件
    
    Args:
        filepath (Union[str, Path]): 文件路径
        
    Returns:
        Dict: JSON数据
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式错误
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(data: Dict[str, Any], 
               filepath: Union[str, Path], 
               indent: int = 2) -> None:
    """
    写入JSON文件
    
    Args:
        data (Dict): 要写入的数据
        filepath (Union[str, Path]): 文件路径
        indent (int): 缩进空格数
    """
    filepath = Path(filepath)
    ensure_directory_exists(filepath.parent)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    读取YAML文件
    
    Args:
        filepath (Union[str, Path]): 文件路径
        
    Returns:
        Dict: YAML数据
        
    Raises:
        FileNotFoundError: 文件不存在
        yaml.YAMLError: YAML格式错误
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"YAML文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_yaml(data: Dict[str, Any], 
               filepath: Union[str, Path]) -> None:
    """
    写入YAML文件
    
    Args:
        data (Dict): 要写入的数据
        filepath (Union[str, Path]): 文件路径
    """
    filepath = Path(filepath)
    ensure_directory_exists(filepath.parent)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def read_pickle(filepath: Union[str, Path]) -> Any:
    """
    读取Pickle文件
    
    Args:
        filepath (Union[str, Path]): 文件路径
        
    Returns:
        Any: 反序列化对象
        
    Raises:
        FileNotFoundError: 文件不存在
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle文件不存在: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def write_pickle(data: Any, 
                 filepath: Union[str, Path]) -> None:
    """
    写入Pickle文件
    
    Args:
        data (Any): 要序列化的对象
        filepath (Union[str, Path]): 文件路径
    """
    filepath = Path(filepath)
    ensure_directory_exists(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def get_file_size(filepath: Union[str, Path]) -> int:
    """
    获取文件大小（字节）
    
    Args:
        filepath (Union[str, Path]): 文件路径
        
    Returns:
        int: 文件大小（字节）
    """
    filepath = Path(filepath)
    return filepath.stat().st_size if filepath.exists() else 0


def list_files(directory: Union[str, Path], 
               pattern: Optional[str] = None) -> List[Path]:
    """
    列出目录中的文件
    
    Args:
        directory (Union[str, Path]): 目录路径
        pattern (str, optional): 文件模式（如 '*.txt'）
        
    Returns:
        List[Path]: 文件路径列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if pattern:
        return list(directory.glob(pattern))
    else:
        return list(directory.iterdir())


def safe_remove(filepath: Union[str, Path]) -> bool:
    """
    安全删除文件或目录
    
    Args:
        filepath (Union[str, Path]): 文件或目录路径
        
    Returns:
        bool: 是否删除成功
    """
    try:
        filepath = Path(filepath)
        if filepath.is_file():
            filepath.unlink()
        elif filepath.is_dir():
            shutil.rmtree(filepath)
        return True
    except Exception:
        return False