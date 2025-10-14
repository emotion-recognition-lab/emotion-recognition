# 多模态情绪识别 (Multimodal Emotion Recognition)

> 基于深度学习的多模态情绪识别框架，支持文本、音频、视频三种模态的情绪分类与情感分析任务。

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 项目简介

本项目是一个用于多模态情绪识别的深度学习框架，支持以下功能：

- 🎯 **多模态融合**：支持文本(T)、音频(A)、视频(V)三种模态的单独和组合使用
- 📊 **多数据集支持**：支持MELD、IEMOCAP、SIMS等主流情绪数据集
- 🚀 **高效训练**：集成缓存机制、混合精度训练、知识蒸馏等优化技术
- ⚙️ **灵活配置**：基于TOML配置文件的模块化设计，支持各种实验配置
- 📈 **完整流程**：从数据预处理到模型训练、评估的完整工具链

如果需要了解项目的详细结构，请查看[项目结构文档](docs/structure.md)。


## 快速开始

### 环境要求

- Python 3.12+
- PyTorch 2.4.1
- CUDA 12.1 (可选，用于GPU加速)

### 安装

1. 克隆项目仓库：
```bash
git clone https://github.com/zrr1999/emotion-recognition.git
cd emotion-recognition
```

2. 安装依赖：
```bash
# 使用uv安装（推荐）
uv sync --all-extras --dev

# 国内用户可使用清华源加速
uv sync --all-extras --dev --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121
```

3. 准备数据集：
```bash
# 创建数据集目录
mkdir datasets

# 下载并链接MELD数据集
ln -s /path/to/MELD datasets/MELD
```

数据集需要按照[项目结构文档](docs/structure.md#数据集结构)中的格式组织。

### 基本使用

#### 命令行工具

项目提供了两个命令行工具：

```bash
# 情绪识别工具
emotion-recognize --help

# 数据处理工具
emotion-tool --help
```

#### 训练模型

使用[nanoflow](https://github.com/zrr-lab/nanoflow)进行实验：

```bash
# 运行基础实验
uvx nanoflow run experiments/mdn.toml
uvx nanoflow run experiments/makd.toml

# 运行消融实验
uvx nanoflow run experiments/mdn-ablation.toml
uvx nanoflow run experiments/makd-ablation.toml
```

## 核心特性

### 🚀 高效训练机制

#### 特征缓存机制
由于当特征提取模块冻结时，相同输入的特征提取模块的输出不会发生变化，因此可以将特征提取模块的输出缓存下来，以减少重复计算。本项目使用[SafeTensors](https://github.com/huggingface/safetensors)格式进行特征缓存，相比于其他格式，SafeTensors拥有更好的性能和安全性。

#### 透明模型存储
在实验过程中，我们通常会保存许多模型的检查点，也会通过修改参数训练多个不同的模型。一般情况下，每一份的模型都需要保存一份参数，这样会导致存储空间的浪费。为了解决这个问题，我们使用软连接将相同的模型参数链接到不同的模型文件夹中，以减少存储空间的浪费。同时，这种方法并不会破坏原本的文件夹结构，使得整体结构更加清晰。

#### 混合精度训练
集成自动混合精度(AMP)支持，在保持模型精度的同时显著加速训练过程并减少显存占用。

### ⚙️ 灵活配置系统

#### 动态配置理念
在实验过程中，为了比较不同的模型，往往需要频繁的修改参数以便训练多个不同的模型。本项目采用配置文件方案，既可以保持代码的整洁，又可以方便的修改参数。配置文件支持列表、字典等复杂数据结构，表达能力远超命令行参数。

采用TOML配置文件实现模块化设计：

- `configs/encoders/`：编码器配置（T、A、V及其组合）
- `configs/fusion/`：融合策略配置
- `configs/losses/`：损失函数配置
- `configs/dataset/`：数据集配置

技术依赖：[Pydantic](https://pydantic-docs.helpmanual.io/) 用于配置验证和类型检查。

### 🔄 知识蒸馏优化

在[TelME](https://github.com/yuntaeyang/TelME)的实现中，使用知识蒸馏训练不同的模态需要进行多次，这样会导致训练时间过长（即使使用了本项目的缓存技术）。为了解决这个问题，我们将多个模态的知识蒸馏训练合并到一次训练中，这样可以大幅度减少训练时间。

### 📊 多数据集支持

支持主流情绪识别数据集：
- **MELD**：情绪分类和情感分析
- **IEMOCAP**：情绪分类
- **SIMS**：情感分析

### 🔧 持续集成

本项目实现了基础的持续集成，具体可以参考[train-and-eval.yml](.github/workflows/train-and-eval.yml)。

## 项目约定

### 检查点命名规范

检查点命名格式为`{训练类别}/{数据集}/{训练方式}--{批大小}--{分类损失函数}/{网络摘要}/{网络哈希}--{随机种子}`，例如`training/MELD--E/trainable--2--{loss}/1xE--T/51fe7ba3--114`。

#### 模态简称
- **T** (Text): 文本模态
- **A** (Audio): 音频模态
- **V** (Video): 视频模态

#### 训练方式简称
- **T** (Full Tuning): 全参数微调
- **L** (LoRA): 低秩适应
- **F** (Froze Backbones): 冻结骨干网络

#### 数据集类型简称
- **E** (Emotion): 情绪分类任务
- **S** (Sentiment): 情感分析任务

## 技术架构

### 核心依赖

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.12+ | 运行环境 |
| PyTorch | 2.4.1 | 深度学习框架 |
| Transformers | 4.48+ | 预训练模型支持 |
| Pydantic | 2.0+ | 配置验证 |
| SafeTensors | 0.4+ | 高效特征缓存 |

### 模型架构

```
多模态输入 → 特征编码器 → 特征融合 → 分类器 → 情绪预测
    ↓           ↓          ↓        ↓         ↓
  T/A/V    BERT/Whisper  Deep/MoE  Linear   Classes
```

#### 编码器支持
- **文本(T)**：BERT、RoBERTa、ModernBERT等
- **音频(A)**：Whisper、Distil-Whisper等
- **视频(V)**：OpenCV特征提取

#### 融合策略
- **Vanilla融合**：简单拼接
- **Deep融合**：深度神经网络
- **MoE融合**：混合专家模型
- **注意力融合**：自注意力机制

### 性能优化

- **视频读取**：OpenCV (0.08s/帧) vs PyAV (0.17s/帧)
- **特征缓存**：使用SafeTensors格式，冻结时避免重复计算
- **存储优化**：软连接机制减少模型存储空间

## 实验结果

> 以下结果基于MELD数据集的情绪分类任务

### 单模态性能

#### 文本模态 (Text-only)

| 方法 | 随机种子 | 准确率 | Weighted-F1 |
|------|----------|--------|-------------|
| APCL (temp=0.08, β=0.1, γ=0.1) | 43 | 67.74% | 67.04% |
| APCL (temp=0.08, β=0.1, γ=0.1) | 42 | 68.05% | 66.91% |
| APCL (temp=0.08, β=0.1, γ=0.1) | 114 | 67.59% | 66.55% |
| APCL (temp=0.08, β=0.1, γ=0.1) | 0 | 67.93% | 66.92% |
| SPCL (temp=0.08, pool=512, support=64) | 42 | 68.31% | 67.31% |
| SPCL (temp=0.08, pool=512, support=64) | 114 | 67.32% | 66.55% |
| SPCL (temp=0.08, pool=512, support=64) | 0 | 66.63% | 66.50% |

### 多模态性能

| 模态组合 | 方法 | 准确率 | Weighted-F1 | 备注 |
|----------|------|--------|-------------|------|
| T+A+V | 待补充 | - | - | 待补充 |
| T+A | 待补充 | - | - | 待补充 |
| T+V | 待补充 | - | - | 待补充 |


## 贡献指南

欢迎贡献代码和想法！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 开发环境

```bash
# 安装开发依赖
uv sync --dev

# 运行代码检查
uv run ruff check src/
uv run pyright src/

# 运行格式化
uv run ruff format src/
```

## 参考文献

### 预训练模型
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/pdf/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692)
- [ModernBERT: Modernizing the BERT Architecture](https://github.com/AnswerDotAI/ModernBERT)

### 模型压缩技术
- [LoRA: Low-Rank Adaptation of Large Language Models](https://huggingface.co/docs/peft/task_guides/lora_based_methods)

### 知识蒸馏
- [Knowledge Distillation for Deep Learning](https://arxiv.org/pdf/2104.09044)
- [DIST: Distillation with Student-Teacher Networks](https://arxiv.org/pdf/2205.10536)
- [Cross-Modal Knowledge Distillation](https://arxiv.org/pdf/2401.12987v2)

### 混合专家模型
- [TGMoE: A Text Guided Mixture of Experts Model](https://ftp.saiconference.com/Downloads/Volume15No8/Paper_119-TGMoE_A_Text_Guided_Mixture_of_Experts_Model.pdf)

### 语音识别
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [Distil-Whisper: Robust Knowledge Distillation](https://arxiv.org/abs/2311.00430)
- [Distil-Large-V3](https://huggingface.co/distil-whisper/distil-large-v3)
- [BELLE: Be Everyone's Large Language model Engine](https://github.com/LianjiaTech/BELLE)

## 其他参考资料

### 技术博客与教程
- [多模态情绪识别综述](https://zhuanlan.zhihu.com/p/694747931) - 多模态情绪识别技术的全面介绍
- [自动混合精度AMP训练详解](https://zhuanlan.zhihu.com/p/408610877/) - PyTorch AMP技术说明和最佳实践

### 数据集与基准
- [情绪识别数据集对比](https://paperswithcode.com/task/emotion-recognition-in-conversation) - Papers with Code上的相关资源
- [MELD数据集官方](https://affective-meld.github.io/) - MELD数据集官方网站
- [IEMOCAP数据集](https://sail.usc.edu/iemocap/) - IEMOCAP情绪数据库

### 学术资源
- [多模态学习方法综述](https://arxiv.org/abs/2209.05025) - 多模态深度学习最新进展
- [情绪计算会议(ACII)](http://acii-conf.org/) - 情感计算领域顶级会议
- [多模态机器学习教程](https://cmu-multicomp-lab.github.io/mmml-tutorial/) - CMU多模态课程

### 工具与框架
- [Transformers库文档](https://huggingface.co/docs/transformers/) - HuggingFace Transformers使用指南
- [PyTorch官方教程](https://pytorch.org/tutorials/) - PyTorch深度学习教程
- [Nanoflow工作流引擎](https://github.com/zrr-lab/nanoflow) - 实验管理工具

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 致谢

感谢所有为本项目提供指导和想法的同学和老师们！

---

如有问题或建议，欢迎提交 [Issue](https://github.com/zrr1999/emotion-recognition/issues) 或 [Pull Request](https://github.com/zrr1999/emotion-recognition/pulls)。
