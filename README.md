# 多模态情绪评估实验 (multimodal emotion recoginition)

## 项目结构

```
emotion-recogiition
├── recognize/
│   ├── dataset/
│   ├── model/
│   ├── ...
│   └── cache.py
├── training/
│   ├── ...
│   └── text_bert.py
├── evaluation/
│   └── TODO
└── README.md
```

## 复现方法

1. 下载 MELD 数据集。
2. 将原始 MELD 数据集中视频文件（mp4）中的音频以 flac 格式提取出来，并按照如下结构组织文件：
   ```
   MELD
   ├── audios/
   │   ├── dev
   │   ├── test
   │   └── train
   ├── videos/
   │   ├── dev
   │   ├── test
   │   └── train
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

## 项目约定

### 检查点命名

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

|  模态   | 骨干网络 |               训练方式               |       准确率        | 精确率 | 召回率 |     weighted-F1     |
| :-----: | :------: | :----------------------------------: | :-----------------: | :----: | :----: | :-----------------: |
| 文本(T) |   BERT   |             Full Tuning              |       55.40%        |   -    |   -    |       53.05%        |
| 文本(T) | RoBERTa  |             Full Tuning              |       60.07%        |   -    |   -    |       57.99%        |
| 文本(T) |  MPNet   |             Full Tuning              | 60.84%(1)/59.66%(2) |   -    |   -    | 58.56%(1)/59.75%(2) |
| 文本(T) |  MPNet   |        Traing Classification         |      58.01%(1)      |   -    |   -    |      56.33%(1)      |
| 文本(T) |  MPNet   | Full Tuning -> Traing Classification |      64.18%(2)      |   -    |   -    |      62.33%(2)      |

### 多模态

|      模态      |    骨干网络    | 融合网络 |                           训练方式                            |      准确率      | 精确率 | 召回率 |   weighted-F1    |
| :------------: | :------------: | :------: | :-----------------------------------------------------------: | :--------------: | :----: | :----: | :--------------: |
| 文本+语音(T+A) | MPNet+Wav2Vec2 |   LMF    |        Text-only Full Tuning -> Traing Classification         |      58.62%      |   -    |   -    |      58.64%      |
| 文本+语音(T+A) | MPNet+Wav2Vec2 |   LMF    |             Text-only Full Tuning -> Full Tuning              | 58.74%/63.29%(2) |   -    |   -    | 58.84%/62.90%(2) |
| 文本+语音(T+A) | MPNet+Wav2Vec2 |   LMF    | Text-only Full Tuning -> Full Tuning -> Traing Classification | 59.89%/62.64%(2) |   -    |   -    | 59.55%/62.66%(2) |

## 进行中实验

|  模态   | 骨干网络 |       训练方式        |          准确率           | 精确率 | 召回率 |        weighted-F1         |
| :-----: | :------: | :-------------------: | :-----------------------: | :----: | :----: | :------------------------: |
| 视频(V) |  ViViT   | Traing Classification | 49.5%(验证集十折交叉验证) |   -    |   -    | 41.36%(验证集十折交叉验证) |

## 相关技术

- [LoRA](https://huggingface.co/docs/peft/task_guides/lora_based_methods)

## 参考文献

### 知识蒸馏

- [KD](https://arxiv.org/pdf/2104.09044)
- [DIST](https://arxiv.org/pdf/2205.10536)
- [跨模态知识蒸馏](https://arxiv.org/pdf/2401.12987v2)
