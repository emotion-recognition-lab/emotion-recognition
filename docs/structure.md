
## 项目结构


### 代码库结构
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
