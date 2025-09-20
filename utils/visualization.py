"""
可视化工具模块

提供常用的可视化功能和中文字体支持。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Union
import platform
import matplotlib.font_manager as fm


def setup_chinese_font() -> bool:
    """
    设置中文字体支持
    
    Returns:
        bool: 是否成功设置中文字体
    """
    # 常见中文字体列表
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'STHeiti', 'Songti SC', 
        'Arial Unicode MS', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'
    ]
    
    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找可用的中文字体
    found_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            found_font = font
            break
    
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font]
        plt.rcParams['axes.unicode_minus'] = False
        return True
    else:
        # 如果找不到中文字体，使用默认字体并警告
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False


def plot_distribution(df: pd.DataFrame, 
                     column: str, 
                     title: Optional[str] = None,
                     figsize: tuple = (10, 6)) -> plt.Figure:
    """
    绘制单个列的分布图
    
    Args:
        df (pd.DataFrame): 数据
        column (str): 列名
        title (str, optional): 图表标题
        figsize (tuple): 图表大小
        
    Returns:
        plt.Figure: matplotlib图表对象
    """
    setup_chinese_font()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 检查数据类型
    if df[column].dtype in ['object', 'category']:
        # 分类数据使用条形图
        value_counts = df[column].value_counts()
        ax.bar(range(len(value_counts)), value_counts.values)
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45)
        ax.set_ylabel('频次')
    else:
        # 数值数据使用直方图
        ax.hist(df[column].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax.set_ylabel('频次')
        ax.set_xlabel(column)
    
    ax.set_title(title or f'{column} 分布图')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, 
                          figsize: tuple = (12, 10),
                          method: str = 'pearson') -> plt.Figure:
    """
    绘制相关性矩阵热力图
    
    Args:
        df (pd.DataFrame): 数据
        figsize (tuple): 图表大小
        method (str): 相关性计算方法
        
    Returns:
        plt.Figure: matplotlib图表对象
    """
    setup_chinese_font()
    
    # 选择数值列
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 计算相关性矩阵
    corr_matrix = numeric_df.corr(method=method)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                ax=ax)
    
    ax.set_title('特征相关性矩阵')
    plt.tight_layout()
    
    return fig


def plot_time_series(df: pd.DataFrame,
                    date_column: str,
                    value_column: str,
                    title: Optional[str] = None,
                    figsize: tuple = (12, 6)) -> plt.Figure:
    """
    绘制时间序列图
    
    Args:
        df (pd.DataFrame): 数据
        date_column (str): 日期列名
        value_column (str): 值列名
        title (str, optional): 图表标题
        figsize (tuple): 图表大小
        
    Returns:
        plt.Figure: matplotlib图表对象
    """
    setup_chinese_font()
    
    # 确保日期列是datetime类型
    df_plot = df.copy()
    df_plot[date_column] = pd.to_datetime(df_plot[date_column])
    
    # 按日期排序
    df_plot = df_plot.sort_values(date_column)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制时间序列
    ax.plot(df_plot[date_column], df_plot[value_column], marker='o', markersize=3)
    
    ax.set_xlabel('时间')
    ax.set_ylabel(value_column)
    ax.set_title(title or f'{value_column} 时间序列')
    ax.grid(True, alpha=0.3)
    
    # 自动旋转日期标签
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_cluster_scatter(df: pd.DataFrame,
                        x_column: str,
                        y_column: str,
                        cluster_column: str,
                        title: Optional[str] = None,
                        figsize: tuple = (10, 8)) -> plt.Figure:
    """
    绘制聚类散点图
    
    Args:
        df (pd.DataFrame): 数据
        x_column (str): X轴列名
        y_column (str): Y轴列名
        cluster_column (str): 聚类标签列名
        title (str, optional): 图表标题
        figsize (tuple): 图表大小
        
    Returns:
        plt.Figure: matplotlib图表对象
    """
    setup_chinese_font()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 获取唯一的聚类标签
    clusters = df[cluster_column].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    
    # 为每个聚类绘制散点图
    for i, cluster in enumerate(clusters):
        cluster_data = df[df[cluster_column] == cluster]
        ax.scatter(cluster_data[x_column], 
                  cluster_data[y_column], 
                  c=[colors[i]], 
                  label=f'聚类 {cluster}',
                  alpha=0.7,
                  s=50)
    
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(title or '聚类结果散点图')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig