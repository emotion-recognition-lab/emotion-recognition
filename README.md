# 多模态情绪评估实验 (multimodal emotion recoginition)

如果你需要了解源码的更多信息，请查看[项目结构](docs/structure.md)。

## 项目约定

### 检查点命名(待定)

每个训练任务的检查点对应一个文件夹，
对于单一模态任务，文件夹命名为`{模态}--{分类任务}{训练方式}({骨干网络})`，
对于多模态任务，文件夹命名为`{模态}--{分类任务}{训练方式}--{融合网络}({骨干网络})`。

其中模态分为`text`、`audio`和`video`，简称如下：

- text: T
- audio: A
- video: V

其中分类任务分为`emotion`和`sentiment`，简称如下：

- emotion: E
- sentiment: S

训练方式分为`Full Tuning`、`LoRA` 和 `Froze Backbones`，简称如下：

- Full Tuning: T
- LoRA: L
- Froze Backbones: F


## 技术选型

### 视频读取
- opencv
   首次调用耗时: 0.11s
   平均耗时: 0.08s
- pyav
   首次调用耗时: 0.22s
   平均耗时: 0.17s

经测试，OpenCV 比 PyAV 速度更快。

## 机制介绍

### 特征缓存机制

由于当特征提取模块冻结时，相同输入的特征提取模块的输出不会发生变化，因此可以将特征提取模块的输出缓存下来，以减少重复计算。
`safetensors` 是 `huggingface` 推出的一个储存和分发张量的格式，相比于其他格式，`safetensors` 拥有更好的性能。

依赖技术：

- [safetensors](https://github.com/huggingface/safetensors)

### 透明模型存储

在实验过程中，我们通常会保存许多模型的检查点，也会通过修改参数训练多个不同的模型。
一般情况下，每一份的模型都需要保存一份参数，这样会导致存储空间的浪费。
为了解决这个问题，我们使用软连接将相同的模型参数链接到不同的模型文件夹中，以减少存储空间的浪费。
同时，这种方法并不会破坏原本的文件夹结构，使得整体结构更加清晰。

### 动态配置

在实验过程中，为了比较不同的模型，往往需要频繁的修改参数以便训练多个不同的模型。
一般情况下，有三种可选的方案：

1. 直接修改源码；
2. 设置合适的命令行参数；
3. 使用配置文件。

直接修改源码并不适合一个工程化的项目，因为这样会导致代码的混乱，不利于维护。
设置命令行参数虽然可以解决这个问题，但是当参数较多时，命令行参数会变得很长，不利于阅读，且命令行参数的表达能力有限，
例如很难表示列表、字典等数据结构。
因此，我们最终选择使用配置文件来确定模型的参数，这样既可以保持代码的整洁，又可以方便的修改参数。

依赖技术：
- [Pydantic](https://pydantic-docs.helpmanual.io/)

### 知识蒸馏

在 [TelME](https://github.com/yuntaeyang/TelME) 的实现中，使用知识蒸馏训练不同的模态需要进行多次，
这样会导致训练时间过长（即使使用了本项目的缓存技术）。
为了解决这个问题，我们使用将多个模态的知识蒸馏训练合并到一次训练中，这样可以大幅度减少训练时间。

### 持续集成

本项目实现了基础的持续集成，具体可以参考 [train-and-eval.yml](.github/workflows/train-and-eval.yml)。

## 实验复现
1. 下载 MELD 数据集。
2. 将原始 MELD 数据集中视频文件（mp4）中的音频以 flac 格式提取出来，并按照规定的[数据集结构](docs/structure.md#数据集结构)组织文件。
3. 链接数据集到项目 `datasets` 文件夹。
```sh
mkdir datasets
ln -s /path/to/MELD datasets/MELD
```
4. 安装依赖。
```sh
uv sync --all-extras --dev
# 如果你在中国大陆，可以使用清华源加速下载
# uv sync --all-extras --dev --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121
```
5. 接下来按照下列文档的内容进行实验，注意部分内容需要根据文档中给出的 `git commit SHA` 切换到指定 `commit` 进行实验。

- [basic.md](experiments/basic.md)
- [dependency-finetuning.md](experiments/dependency-finetuning.md)


## 实验结果（早期）

### 单一模态

|  模态   | 骨干网络 |            训练方式            |       准确率        | 精确率 | 召回率 |     weighted-F1     |
| :-----: | :------: | :----------------------------: | :-----------------: | :----: | :----: | :-----------------: |
| 文本(T) |   BERT   |          Full Tuning           |       55.40%        |   -    |   -    |       53.05%        |
| 文本(T) | RoBERTa  |          Full Tuning           |       60.07%        |   -    |   -    |       57.99%        |
| 文本(T) |  MPNet   |          Full Tuning           | 60.84%(1)/59.66%(2) |   -    |   -    | 58.56%(1)/59.75%(2) |
| 文本(T) |  MPNet   |        Froze Backbones         |      58.01%(1)      |   -    |   -    |      56.33%(1)      |
| 文本(T) |  MPNet   | Full Tuning -> Froze Backbones |      64.18%(2)      |   -    |   -    |      62.33%(2)      |

### 多模态

|      模态      |    骨干网络    | 融合网络 |                        训练方式                         |      准确率      | 精确率 | 召回率 |   weighted-F1    |
| :------------: | :------------: | :------: | :-----------------------------------------------------: | :--------------: | :----: | :----: | :--------------: |
| 文本+语音(T+A) | MPNet+Wav2Vec2 |   LMF    |        Text-only Full Tuning -> Froze Backbones         |      58.62%      |   -    |   -    |      58.64%      |
| 文本+语音(T+A) | MPNet+Wav2Vec2 |   LMF    |          Text-only Full Tuning -> Full Tuning           | 58.74%/63.29%(2) |   -    |   -    | 58.84%/62.90%(2) |
| 文本+语音(T+A) | MPNet+Wav2Vec2 |   LMF    | Text-only Full Tuning -> Full Tuning -> Froze Backbones | 59.89%/62.64%(2) |   -    |   -    | 59.55%/62.66%(2) |

## 相关技术与参考文献

### 模型压缩
- [LoRA](https://huggingface.co/docs/peft/task_guides/lora_based_methods)

### 知识蒸馏
- [KD](https://arxiv.org/pdf/2104.09044)
- [DIST](https://arxiv.org/pdf/2205.10536)
- [跨模态知识蒸馏](https://arxiv.org/pdf/2401.12987v2)

### 混合专家模型
- [TGMoE](https://ftp.saiconference.com/Downloads/Volume15No8/Paper_119-TGMoE_A_Text_Guided_Mixture_of_Experts_Model.pdf)

### 语音识别
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3)
- [Distil-Whisper](https://arxiv.org/abs/2311.00430)
- [BELLE](https://github.com/LianjiaTech/BELLE)
