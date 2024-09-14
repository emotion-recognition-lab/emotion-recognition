### 所有模态同时微调的各模态组合实验复现

git commit SHA: 0984ab42bb82ee63bdf567112bc844043edfcb4f

```sh
uv run emotion-recognize train configs/T--T.toml configs/dataset/MELD--E.toml --seed 114
uv run emotion-recognize train configs/A--T.toml configs/dataset/MELD--E.toml --seed 114
uv run emotion-recognize train configs/V--T.toml configs/dataset/MELD--E.toml --seed 114
uv run emotion-recognize train configs/T+A--T.toml configs/dataset/MELD--E.toml --seed 114
uv run emotion-recognize train configs/T+A+V--T.toml configs/dataset/MELD--E.toml --seed 114
```

|  模态  | 准确率 | weighted-F1 |
| :----: | :----: | :---------: |
| T(768) | 62.64% |   62.89%    |
| A(256) | 39.31% |   38.12%    |
| V(32)  | 48.12% |   31.27%    |
| V(256) | 39.00% |   34.86%    |
|  T+A   | 63.18% |   63.77%    |
|  A+V   |        |             |
| T+A+V  |        |             |
