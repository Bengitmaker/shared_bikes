# 共享单车使用模式聚类分析

## 项目简介

本项目通过聚类分析方法，对共享单车的使用模式进行分析，识别不同的使用场景和用户行为模式。项目使用K-means算法对共享单车数据进行聚类，帮助理解在不同时间、天气和环境条件下单车的使用规律。

## 目录结构

```
shared_bikes/
├── configs/                 # 配置文件目录
│   ├── config.yaml          # 主配置文件
│   └── config_manager.py    # 配置管理器
├── data/                    # 数据目录
│   ├── raw/                 # 原始数据
│   └── processed/           # 处理后数据
├── docs/                    # 文档目录
├── models/                  # 模型保存目录
├── notebooks/               # Jupyter笔记本目录
├── output/                  # 输出目录
├── shared_bikes/            # 核心代码包
│   ├── __init__.py          # 包初始化文件
│   ├── cli.py               # 命令行接口
│   ├── train.py             # 模型训练模块
│   ├── predict.py           # 预测模块
│   ├── evaluate.py          # 评估模块
│   ├── setup.py             # 数据设置模块
│   ├── pipeline.py          # 管道执行模块
│   ├── model_trainer.py     # 模型训练核心功能
│   ├── predictor.py         # 预测核心功能
│   ├── evaluator.py         # 评估核心功能
│   └── data_processor.py    # 数据处理核心功能
├── tests/                   # 测试目录
├── utils/                   # 工具函数目录
├── scripts/                 # 脚本目录（遗留）
├── requirements.txt         # 依赖包列表
└── setup.py                # 项目安装配置
```

## 核心算法

### K-means聚类算法

本项目使用K-means聚类算法对共享单车使用数据进行分析。K-means是一种无监督学习算法，通过将数据点分配到K个簇中，使得每个簇内的数据点尽可能相似，而不同簇之间的数据点尽可能不同。

### 特征选择

项目使用以下6个特征进行聚类分析：
- `hour`: 小时（从datetime字段中提取）
- `workingday`: 是否工作日
- `weather`: 天气状况
- `temp`: 温度
- `humidity`: 湿度
- `windspeed`: 风速

### 模型参数

- 聚类数量: 5
- 最大迭代次数: 300
- 初始化方法: k-means++（默认）
- 随机种子: 42

## 技术难点

### 1. 数据预处理

- 时间特征提取：从datetime字段中提取hour特征
- 缺失值处理：删除包含缺失值的样本
- 特征标准化：使用StandardScaler对特征进行标准化处理

### 2. 模型评估

项目使用多种评估指标来评估聚类效果：

- **轮廓系数(Silhouette Score)**: 衡量聚类的紧密度和分离度，值越接近1表示聚类效果越好
- **Calinski-Harabasz指数**: 衡量簇间的分离度与簇内的紧密度之比，值越大表示聚类效果越好
- **Davies-Bouldin指数**: 衡量簇内的紧密度与簇间的分离度之比，值越小表示聚类效果越好

### 3. 可视化展示

项目使用matplotlib和seaborn库进行数据可视化，包括：
- 聚类结果散点图
- 特征分布直方图
- 聚类中心雷达图

## 运行结果

执行完整管道后，将在output目录下生成以下文件：
- `evaluation_report.md`: 模型评估报告
- `cluster_visualization.png`: 聚类结果可视化图
- `feature_distribution.png`: 特征分布图

### 性能指标

在测试数据集上（200个样本），模型表现如下：

| 指标 | 值 |
|------|----|
| 轮廓系数(Silhouette Score) | 0.170 |
| Calinski-Harabasz指数 | 34.187 |
| Davies-Bouldin指数 | 1.584 |
| 训练时间 | < 1秒 |
| 预测时间 | < 1秒 |

### 聚类分布

模型将数据分为5个聚类，分布如下：

| 聚类ID | 样本数 | 占比 |
|--------|--------|------|
| 0 | 46 | 23.0% |
| 1 | 55 | 27.5% |
| 2 | 20 | 10.0% |
| 3 | 26 | 13.0% |
| 4 | 53 | 26.5% |

### 结果分析

1. **聚类效果**: 轮廓系数为0.170，表明聚类结果存在一定的区分度，但仍有改进空间。这可能是因为特征维度较低或数据本身的聚类结构不够明显。

2. **聚类分布**: 聚类分布相对均匀，没有出现极端不平衡的情况，说明模型能够较好地识别不同的使用模式。

3. **效率表现**: 模型训练和预测速度非常快，适合实时或批量处理任务。

评估报告包含以下关键信息：
- 模型基本信息（聚类数量、样本数量等）
- 评估指标值（轮廓系数、Calinski-Harabasz指数、Davies-Bouldin指数）
- 聚类分布情况

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 安装项目
```bash
pip install -e .
```

### 运行完整管道
```bash
shared-bikes-run run
```

或者分步执行：

```bash
# 数据准备
shared-bikes-setup

# 模型训练
shared-bikes-train

# 模型预测
shared-bikes-predict --data-path data/raw/test.csv

# 模型评估
shared-bikes-evaluate --data-path data/raw/test.csv
```

### 预测模块使用方法

预测模块用于使用训练好的模型对新数据进行聚类预测：

```bash
shared-bikes-predict --data-path <数据文件路径> [--model-dir <模型目录>] [--output-path <输出文件路径>]
```

参数说明：
- `--data-path`: 输入数据文件路径（必须）
- `--model-dir`: 模型目录，默认为models/
- `--output-path`: 输出文件路径，默认为输入文件同目录下添加"_with_clusters"后缀的CSV文件

输入数据文件应包含以下列：
- datetime: 时间戳
- workingday: 是否工作日
- weather: 天气状况
- temp: 温度
- humidity: 湿度
- windspeed: 风速

输出文件将包含输入数据的所有列，以及额外的cluster列，表示预测的聚类标签。
```
