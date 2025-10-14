# 项目结构文档

## 代码库结构

```
emotion-recognition/
├── src/
│   ├── recognize/                   # 核心识别模块
│   │   ├── dataset/                 # 数据集封装
│   │   │   ├── base.py              # 数据集基类
│   │   │   ├── iemocap.py           # IEMOCAP数据集
│   │   │   ├── meld.py              # MELD数据集
│   │   │   ├── sims.py              # SIMS数据集
│   │   │   └── utils.py             # 数据集工具
│   │   ├── model/                   # 模型实现
│   │   │   ├── base.py              # 模型基类
│   │   │   ├── backbone.py          # 骨干网络
│   │   │   ├── inputs.py            # 输入处理
│   │   │   └── outputs.py           # 输出处理
│   │   ├── module/                  # 可复用模块
│   │   │   ├── basic.py             # 基础模块
│   │   │   ├── fusion.py            # 融合模块
│   │   │   ├── loss.py              # 损失函数
│   │   │   ├── moe.py               # 混合专家模型
│   │   │   ├── router.py            # 路由器
│   │   │   └── utils.py             # 工具函数
│   │   ├── cache.py                 # 特征缓存机制
│   │   ├── config.py                # 配置定义
│   │   ├── estimator.py             # 情绪评估器
│   │   ├── evaluate.py              # 模型评估
│   │   ├── preprocessor.py          # 数据预处理
│   │   ├── trainer.py               # 模型训练器
│   │   ├── typing.py                # 类型定义
│   │   └── utils.py                 # 通用工具
│   └── recognize_cli/               # 命令行接口
│       ├── cli_recognize.py         # 识别命令
│       └── cli_tool.py              # 工具命令
├── configs/                         # 配置文件
│   ├── classifier/                  # 分类器配置
│   ├── dataset/                     # 数据集配置
│   ├── encoders/                    # 编码器配置
│   ├── fusion/                      # 融合策略配置
│   └── losses/                      # 损失函数配置
├── experiments/                     # 实验配置
├── packages/                        # 子包
│   └── emotion-recognition-utils/   # 工具包
├── docs/                            # 文档
└── README.md                        # 项目说明
```

## 配置系统结构

### 编码器配置 (`configs/encoders/`)
- `T.toml` - 文本编码器
- `A.toml` - 音频编码器
- `V.toml` - 视频编码器
- `T+A.toml` - 文本+音频组合
- `T+V.toml` - 文本+视频组合
- `A+V.toml` - 音频+视频组合
- `T+A+V.toml` - 三模态组合

### 融合策略配置 (`configs/fusion/`)
- `vallina.toml` - 基础融合
- `deep-vallina.toml` - 深度融合
- `private.toml` - 私有特征融合
- `shared-*.toml` - 共享特征融合
- `DF-*.toml` - 深度融合变体

### 损失函数配置 (`configs/losses/`)
- `classification/` - 分类损失
  - `focal.toml` - Focal Loss
  - `weight.toml` - 加权交叉熵
- `apcl.toml` - Adaptive Prototypical Contrastive Learning
- `spcl.toml` - Supervised Prototypical Contrastive Learning
- `scl.toml` - Supervised Contrastive Learning
- `cmcl.toml` - Cross-Modal Contrastive Learning
- `frl-*.toml` - Feature Representation Learning

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
