# shared_bikes 项目文档

## 目录
- [项目概述](#项目概述)
- [安装指南](#安装指南)
- [使用方法](#使用方法)
- [配置说明](#配置说明)
- [API文档](#api文档)
- [开发规范](#开发规范)
- [贡献指南](#贡献指南)

## 项目概述

shared_bikes是一个共享单车数据分析与预测系统，旨在通过机器学习算法对共享单车使用模式进行分析和预测。

### 主要功能
- 数据预处理与清洗
- 用户聚类分析
- 使用量预测
- 可视化展示

### 技术栈
- Python 3.8+
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- YAML (配置管理)

## 安装指南

### 环境要求
- Python 3.8 或更高版本
- pip 包管理器

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/yourusername/shared_bikes.git

cd shared_bikes

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/MacOS
source .venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 运行主程序

```bash
python src/mypkg/main.py
```

### 数据处理

```bash
python src/mypkg/Data_prep.py
```

### 聚类分析

```bash
python src/mypkg/Model_Kmeans.py
```

### 使用脚本工具

```bash
# 运行完整分析管道
python scripts/run_pipeline.py

# 训练模型
python scripts/train_model.py

# 执行预测
python scripts/predict.py --data-path data/raw/test.csv

# 评估模型
python scripts/evaluate_model.py --data-path data/raw/train.csv
```

## 配置说明

项目配置文件位于 `configs/config.yaml`，包含以下主要配置项：

### 路径配置
```yaml
paths:
  data: "data/"
  output: "output/"
  models: "models/"
```

### 数据处理配置
```yaml
data_processing:
  chunk_size: 10000
  date_format: "%Y-%m-%d %H:%M:%S"
  timezone: "Asia/Shanghai"
```

### 模型配置
```yaml
model:
  kmeans:
    n_clusters: 5
    random_state: 42
    max_iter: 300
    n_init: 10
```

### 日志配置
```yaml
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/app.log"
  max_bytes: 10485760
  backup_count: 5
```

## API文档

### 核心模块

#### Data_prep.py
数据预处理模块，负责数据加载、清洗和特征工程。

#### Model_Kmeans.py
K-means聚类模型，用于用户分群分析。

#### main.py
主程序入口，协调各个模块的执行流程。

## 开发规范

### 代码风格
遵循 PEP 8 编码规范。

### 注释要求
- 函数和类必须有文档字符串
- 复杂逻辑需要添加行内注释

## 贡献指南

欢迎提交Issue和Pull Request。