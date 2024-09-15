
## 项目结构

### 代码库结构
```
emotion-recogiition
├── recognize/
│   ├── dataset/  # 针对不同数据集的封装
│   │   ├── ...
│   │   └── base.py
│   ├── model/  # 实现不同类型的网络
│   │   ├── base.py
│   │   ├── unimodal.py
│   │   └── multimodal.py
│   ├── module/  # 可复用且相对独立的模块
│   │   ├── ...
│   │   └── utils.py
│   ├── typing.py  # 一些类型定义
│   ├── config.py  # 配置格式定义
│   ├── estimate.py  # 面向应用的封装好的情绪评估类器
│   ├── cache.py  # 主干网络冻结时的训练缓存机制
│   ├── evaluate.py  # 评估模型
│   ├── preprocess.py  # 数据预处理
│   └── utils.py  # 对外暴露的主要接口
├── cli/
│   ├── cli_recognize.py  # 模型相关的命令行接口
│   └── cli_tool.py  # 工具类命令行接口
├── configs/
│   ├── ...
│   └── T--F.toml
├── docs/
└── README.md
```

### 数据集结构
```
MELD
├── audios/
│   ├── dev/
│   ├── test/
│   └── train/
├── videos/
│   ├── dev/
│   ├── test/
│   └── train/
├── dev_sent_emo.csv
├── test_sent_emo.csv
└── train_sent_emo.csv
```
