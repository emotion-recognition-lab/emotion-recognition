### 单独微调主干网络的多模态实验复现
git commit SHA:

```sh
uv run emotion-recognize train configs/T+A--T.toml configs/dataset/MELD--E.toml --seed 114
uv run emotion-recognize train configs/T+A+V--T.toml configs/dataset/MELD--E.toml --seed 114
```

| 模态  | 准确率 | weighted-F1 |
| :---: | :----: | :---------: |
|  T+A  |        |             |
|  T+V  |        |             |
| T+A+V |        |             |
