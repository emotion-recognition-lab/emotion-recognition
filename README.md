# 多模态情绪评估实验 (multimodal emotion recoginition)

## 项目结构

```
emotion-recogiition
├── recognize/
│   ├── dataset/
│   │   ├── base.py
│   │   ├── ...
│   │   └── meld.py
│   ├── model/
│   │   ├── base.py
│   │   ├── unimodal.py
│   │   └── multimodal.py
│   ├── module/
│   │   ├── ...
│   │   └── utils.py
│   ├── cache.py
│   ├── cli.py
│   ├── typing.py
│   ├── estimate.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── utils.py
├── cli/
│   ├── cli_recognize.py
│   └── cli_tool.py
├── configs/
│   ├── ...
│   └── MELD--T--EF.toml
└── README.md
```

## 复现方法

1. 下载 MELD 数据集。
2. 将原始 MELD 数据集中视频文件（mp4）中的音频以 flac 格式提取出来，并按照如下结构组织文件：
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
3. 链接数据集到项目 `datasets` 文件夹。
   ```sh
   mkdir datasets
   ln -s /path/to/MELD datasets/MELD
   ```
4. 开始训练。
   ```sh
   pdm install
   pdm run emotion-recognize train datasets/MELD --modal T --label-type sentiment
   ```
   ```sh
   pip install -e .
   python recognize/cli.py train datasets/MELD --modal T --label-type sentiment
   ```


## 技术选型

### 视频读取
经测试，OpenCV 比 PyAV 速度更快。

## 机制介绍

### 特征缓存机制

由于当特征提取模块冻结时，相同输入的特征提取模块的输出不会发生变化，因此可以将特征提取模块的输出缓存下来，以减少重复计算。
`safetensors` 是 `huggingface` 推出的一个储存和分发张量的格式，相比于其他格式，`safetensors` 拥有更好的性能。

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

## 项目约定

### 检查点命名(待定)

每个训练任务的检查点对应一个文件夹，
对于单一模态任务，文件夹命名为`{模态}--{分类任务}{训练方式}({骨干网络})`，
对于多模态任务，文件夹命名为`{模态}--{分类任务}{训练方式}--{融合网络}({骨干网络})`。

其中分类任务分为`emotion`和`sentiment`，简称如下：

- emotion: E
- sentiment: S

训练方式分为`Full Tuning`、`LoRA` 和 `Froze Backbones`，简称如下：

- Full Tuning: T
- LoRA: L
- Froze Backbones: F

## 实验结果

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

## 进行中实验

|  模态   | 骨干网络 |    训练方式     |          准确率           | 精确率 | 召回率 |        weighted-F1         |
| :-----: | :------: | :-------------: | :-----------------------: | :----: | :----: | :------------------------: |
| 视频(V) |  ViViT   | Froze Backbones | 49.5%(验证集十折交叉验证) |   -    |   -    | 41.36%(验证集十折交叉验证) |

## 相关技术

- [LoRA](https://huggingface.co/docs/peft/task_guides/lora_based_methods)

## 参考文献

### 知识蒸馏

- [KD](https://arxiv.org/pdf/2104.09044)
- [DIST](https://arxiv.org/pdf/2205.10536)
- [跨模态知识蒸馏](https://arxiv.org/pdf/2401.12987v2)
