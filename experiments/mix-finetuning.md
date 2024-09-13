### 所有模态同时微调的各模态组合实验复现

git commit SHA: fd08009357388c847eb417e7ad8bc12436c6df36

```sh
uv run emotion-recognize train configs/MELD--T--ET.toml --seed 114
uv run emotion-recognize train configs/MELD--A--ET.toml --seed 114
uv run emotion-recognize train configs/MELD--V--ET.toml --seed 114
uv run emotion-recognize train configs/MELD--T+A--ET.toml --seed 114
uv run emotion-recognize train configs/MELD--T+A+V--ET.toml --seed 114
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
