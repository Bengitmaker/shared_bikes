# 共享单车使用模式聚类分析项目

本项目通过对共享单车数据进行聚类分析，识别不同用户群体的使用模式和行为特征。

## 项目概述

本项目使用K-means聚类算法对共享单车的使用数据进行分析，根据多种特征（如时间、天气、用户类型等）将使用模式分为不同类别，帮助理解用户行为并为运营决策提供支持。

## 数据说明

项目使用以下数据文件：
- `data/train.csv`: 训练数据集，包含共享单车的历史使用记录
- `data/test.csv`: 测试数据集

## 项目结构

```
shared_bikes/
├── configs/              # 配置文件目录
│   ├── config.yaml       # YAML格式配置文件
│   └── config_manager.py # 配置管理模块
├── data/                 # 数据文件目录
│   ├── train.csv         # 训练数据
│   └── test.csv          # 测试数据
├── docs/                 # 文档目录
│   ├── README.md         # 详细项目文档
│   └── doc_generator.py  # 文档生成工具
├── logs/                 # 日志文件目录
├── models/               # 模型文件目录
├── notebooks/            # Jupyter笔记本目录
├── output/               # 输出文件目录
├── scripts/              # 脚本文件目录
│   ├── run_pipeline.py       # 运行完整管道
│   ├── setup_data.py         # 数据设置脚本
│   ├── train_model.py        # 模型训练脚本
│   ├── predict.py            # 预测脚本
│   └── evaluate_model.py     # 模型评估脚本
├── shared_bikes/         # 主包目录
├── tests/                # 测试目录
├── utils/                # 工具函数目录
├── .gitignore
├── LICENSE
├── README.md             # 项目说明文档
├── requirements.txt      # 依赖包列表
├── setup.py              # 包安装配置
└── VERSION               # 版本文件
```

## 目录说明

### configs - 配置管理
包含项目的配置文件和管理模块：
```
