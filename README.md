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

### 文本模态(T)

| 骨干网络 |  训练方式   | 准确率 | 精确率 | 召回率 | F1 值  |
| :------: | :---------: | :----: | :----: | :----: | :----: |
|   BERT   | Full Tuning | 55.40% |   -    |   -    | 53.05% |
| RoBERTa  | Full Tuning | 60.07% |   -    |   -    | 57.99% |
|  MPNet   | Full Tuning | 60.84% |   -    |   -    | 58.56% |

### 视频模态(V)

| 骨干网络 | 训练方式 |          准确率           | 精确率 | 召回率 |           F1 值            |
| :------: | :------: | :-----------------------: | :----: | :----: | :------------------------: |
|  ViViT   |  Frozen  | 49.5%(验证集十折交叉验证) |   -    |   -    | 41.36%(验证集十折交叉验证) |

### 文本+语音模态(T+A)

|    骨干网络    |                           训练方式                            | 准确率 | 精确率 | 召回率 | F1 值  |
| :------------: | :-----------------------------------------------------------: | :----: | :----: | :----: | :----: |
| MPNet+Wav2Vec2 |        Text-only Full Tuning -> Traing Classification         | 58.62% |   -    |   -    | 58.64% |
| MPNet+Wav2Vec2 |             Text-only Full Tuning -> Full Tuning              | 58.74% |   -    |   -    | 58.84% |
| MPNet+Wav2Vec2 | Text-only Full Tuning -> Full Tuning -> Traing Classification | 59.89% |   -    |   -    | 59.55% |
