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

## 相关技术

LoRA

## 实验结果

### 单一模态

|  模态   | 骨干网络 |       训练方式        |          准确率           | 精确率 | 召回率 |           F1 值            |
| :-----: | :------: | :-------------------: | :-----------------------: | :----: | :----: | :------------------------: |
| 文本(T) |   BERT   |      Full Tuning      |          55.40%           |   -    |   -    |           53.05%           |
| 文本(T) | RoBERTa  |      Full Tuning      |          60.07%           |   -    |   -    |           57.99%           |
| 文本(T) |  MPNet   |      Full Tuning      |          60.84%           |   -    |   -    |           58.56%           |
| 文本(T) |  MPNet   | Traing Classification |          58.01%           |   -    |   -    |           56.33%           |
| 视频(V) |  ViViT   | Traing Classification | 49.5%(验证集十折交叉验证) |   -    |   -    | 41.36%(验证集十折交叉验证) |

### 多模态

|      模态      |    骨干网络    | 融合网络 |                           训练方式                            | 准确率 | 精确率 | 召回率 | F1 值  |
| :------------: | :------------: | :------: | :-----------------------------------------------------------: | :----: | :----: | :----: | :----: |
| 文本+语音(T+A) | MPNet+Wav2Vec2 |   LMF    |        Text-only Full Tuning -> Traing Classification         | 58.62% |   -    |   -    | 58.64% |
| 文本+语音(T+A) | MPNet+Wav2Vec2 |   LMF    |             Text-only Full Tuning -> Full Tuning              | 58.74% |   -    |   -    | 58.84% |
| 文本+语音(T+A) | MPNet+Wav2Vec2 |   LMF    | Text-only Full Tuning -> Full Tuning -> Traing Classification | 59.89% |   -    |   -    | 59.55% |
