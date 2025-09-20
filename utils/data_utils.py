"""
数据处理工具模块

提供常用的数据处理和转换功能。
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
from pathlib import Path


def load_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    加载数据文件
    
    Args:
        file_path (Union[str, Path]): 数据文件路径
        **kwargs: 传递给pd.read_csv的其他参数
        
    Returns:
        pd.DataFrame: 加载的数据
        
    Raises:
        FileNotFoundError: 文件不存在
        pd.errors.EmptyDataError: 文件为空
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    # 尝试不同的分隔符
    separators = [',', ';', '\t', ' ']
    exceptions = []
    
    for sep in separators:
        try:
            df = pd.read_csv(file_path, sep=sep, **kwargs)
            return df
        except Exception as e:
            exceptions.append(f"分隔符 '{sep}': {str(e)}")
            continue
    
    # 如果所有分隔符都失败，则抛出最后一个异常
    raise ValueError(f"无法加载数据文件 {file_path}:\n" + "\n".join(exceptions))


def clean_data(df: pd.DataFrame, 
               drop_na: bool = True, 
               drop_duplicates: bool = True,
               columns_to_drop: Optional[List[str]] = None) -> pd.DataFrame:
    """
    清理数据
    
    Args:
        df (pd.DataFrame): 输入数据
        drop_na (bool): 是否删除包含缺失值的行
        drop_duplicates (bool): 是否删除重复行
        columns_to_drop (List[str], optional): 要删除的列名列表
        
    Returns:
        pd.DataFrame: 清理后的数据
    """
    df_clean = df.copy()
    
    # 删除指定列
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')
    
    # 删除缺失值
    if drop_na:
        df_clean = df_clean.dropna()
    
    # 删除重复行
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    return df_clean


def normalize_features(df: pd.DataFrame, 
                      feature_columns: List[str],
                      method: str = 'minmax') -> pd.DataFrame:
    """
    标准化特征
    
    Args:
        df (pd.DataFrame): 输入数据
        feature_columns (List[str]): 要标准化的特征列名
        method (str): 标准化方法 ('minmax' 或 'zscore')
        
    Returns:
        pd.DataFrame: 标准化后的数据
    """
    df_norm = df.copy()
    
    if method == 'minmax':
        for col in feature_columns:
            if col in df_norm.columns:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val != min_val:
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
                else:
                    df_norm[col] = 0
                    
    elif method == 'zscore':
        for col in feature_columns:
            if col in df_norm.columns:
                mean_val = df_norm[col].mean()
                std_val = df_norm[col].std()
                if std_val != 0:
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
                else:
                    df_norm[col] = 0
                    
    return df_norm


def encode_categorical_features(df: pd.DataFrame, 
                              categorical_columns: List[str]) -> pd.DataFrame:
    """
    对分类特征进行编码
    
    Args:
        df (pd.DataFrame): 输入数据
        categorical_columns (List[str]): 分类特征列名
        
    Returns:
        pd.DataFrame: 编码后的数据
    """
    df_encoded = df.copy()
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            # 使用pandas的get_dummies进行独热编码
            dummies = pd.get_dummies(df_encoded[col], prefix=col)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            
    return df_encoded


def split_data(df: pd.DataFrame, 
               test_size: float = 0.2, 
               random_state: int = 42) -> tuple:
    """
    分割数据为训练集和测试集
    
    Args:
        df (pd.DataFrame): 输入数据
        test_size (float): 测试集比例
        random_state (int): 随机种子
        
    Returns:
        tuple: (train_df, test_df)
    """
    np.random.seed(random_state)
    
    # 随机打乱数据
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # 计算分割点
    split_idx = int(len(df_shuffled) * (1 - test_size))
    
    # 分割数据
    train_df = df_shuffled[:split_idx]
    test_df = df_shuffled[split_idx:]
    
    return train_df, test_df