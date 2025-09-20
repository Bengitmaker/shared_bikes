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
├── notebooks/            # Jupyter笔记本目录
├── scripts/              # 脚本文件目录
├── src/                  # 源代码目录
│   └── mypkg/            # Python模块目录
│       ├── Data_prep.py      # 数据预处理
│       ├── Elbow_anal.py     # 肘部法则分析最佳聚类数
│       ├── Model_Kmeans.py   # K-means聚类实现
│       ├── cluster_table.py  # 聚类结果统计表生成
│       ├── clusters_details.py # 聚类详情分析
│       ├── main.py           # 主程序和可视化
│       └── plt_utils.py      # 绘图工具
├── tests/                # 测试目录
├── .gitignore
├── LICENSE
├── README.md             # 项目说明文档
├── requirements.txt      # 依赖包列表
└── setup.py              # 包安装配置
```

## 目录说明

### configs - 配置管理
包含项目的配置文件和管理模块：
- `config.yaml`: 项目配置文件，定义路径、数据处理参数、模型参数和日志配置
- `config_manager.py`: 配置管理器，提供配置加载、解析和访问功能

### data - 数据文件
存放项目所需的数据文件，包括训练数据和测试数据。

### docs - 文档
存放项目的详细技术文档和文档生成工具：
- `README.md`: 详细技术文档
- `doc_generator.py`: 自动化文档生成工具

### logs - 日志
项目日志文件目录，包含运行时日志记录：
- 支持日志滚动和备份
- 可配置日志级别和格式
- 同时输出到文件和控制台

### notebooks - Jupyter笔记本
用于数据分析和探索的Jupyter笔记本文件目录。

### scripts - 脚本工具
包含各种实用脚本，用于数据准备、模型训练、预测和评估：
- `setup_data.py`: 数据准备脚本
- `train_model.py`: 模型训练脚本
- `predict.py`: 预测脚本
- `evaluate_model.py`: 模型评估脚本
- `run_pipeline.py`: 管道运行脚本

### src/mypkg - 核心源代码
项目的核心功能实现目录，包含了所有实际运行的代码模块：
- 数据预处理
- 聚类分析
- 结果可视化

### src/mypkg vs my_package

项目中有两个包目录，它们有不同的用途：

- **src/mypkg**: 当前项目的核心功能实现目录，包含了所有实际运行的代码模块。这是项目的主要源代码位置。

- **my_package**: 预留的扩展模块目录，可用于未来功能开发。当前包含示例模块作为扩展参考。

这种设计允许项目在保持当前功能稳定的同时，为未来的模块化扩展提供便利。

### tests - 测试
包含项目的单元测试和集成测试：
- `unit/`: 单元测试
- `integration/`: 集成测试
- `run_tests.py`: 测试运行脚本

## 功能模块

1. **数据预处理** ([Data_prep.py](file:///e:/代码/2025/项目/shared%20bikes/src/mypkg/Data_prep.py))
   - 从原始数据中提取时间特征（小时、星期、月份等）
   - 选择用于聚类的特征
   - 对数据进行标准化处理

2. **最佳聚类数分析** ([Elbow_anal.py](file:///e:/代码/2025/项目/shared%20bikes/src/mypkg/Elbow_anal.py))
   - 使用肘部法则确定最佳聚类数量
   - 绘制WCSS（簇内平方和）随聚类数变化的曲线

3. **K-means聚类** ([Model_Kmeans.py](file:///e:/代码/2025/项目/shared%20bikes/src/mypkg/Model_Kmeans.py))
   - 实现K-means聚类算法
   - 将聚类结果添加到原始数据中

4. **聚类结果分析**
   - [cluster_table.py](file:///e:/代码/2025/项目/shared%20bikes/src/mypkg/cluster_table.py): 生成聚类统计表并保存为CSV文件
   - [clusters_details.py](file:///e:/代码/2025/项目/shared%20bikes/src/mypkg/clusters_details.py): 输出每个聚类的详细描述

5. **可视化分析** ([main.py](file:///e:/代码/2025/项目/shared%20bikes/src/mypkg/main.py))
   - 各聚类在不同时段的分布
   - 各聚类在工作日/非工作日的分布
   - 各聚类的用户类型分布
   - 各聚类在不同天气条件下的分布

6. **工具模块** ([plt_utils.py](file:///e:/代码/2025/项目/shared%20bikes/src/mypkg/plt_utils.py))
   - 设置matplotlib中文字体显示

## 运行方法

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
.venv\\Scripts\\activate
# Linux/MacOS
source .venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt
```

### 运行程序

```bash
# 运行主程序
cd src/mypkg
python main.py

# 重新生成聚类分析结果
python cluster_table.py

# 运行完整分析管道
python scripts/run_pipeline.py

# 训练模型
python scripts/train_model.py

# 执行预测
python scripts/predict.py --data-path data/raw/test.csv

# 评估模型
python scripts/evaluate_model.py --data-path data/raw/train.csv
```

## 分析结果

项目将数据分为4个聚类：

- **聚类0**: 代表非工作日的深夜和凌晨时段，主要由临时用户在晴朗天气下使用
- **聚类1**: 代表工作日的早高峰时段，主要是注册用户在上班通勤时使用
- **聚类2**: 代表工作日的午间和下午时段，用户类型混合，受天气影响较小
- **聚类3**: 代表工作日下班高峰期，注册用户占比较高，是全天使用量最大的时段

每个聚类在以下维度上具有不同特征：
- 时间分布（小时）
- 工作日/非工作日分布
- 临时用户与注册用户比例
- 天气条件影响

## 输出文件

- `聚类分析结果.csv`: 包含各聚类详细统计信息的表格
- 多个可视化图表，展示各聚类在不同维度上的特征分布
- 日志文件，记录程序运行状态和错误信息

## 依赖库

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 文档管理

项目采用自动化文档管理：

```bash
# 更新API文档
cd src/mypkg
echo "from shared_bikes.docs import generator; generator.update_documentation()" > temp_doc_update.py
python temp_doc_update.py
rm temp_doc_update.py
```

详细文档请参阅 [docs/README.md](docs/README.md)。