### 各模态组合的基础实验复现

git commit SHA: 02e87a5a3a15b69c0a26d6471fd4dfd06b76415d

```sh
uvx nanoflow experiments/basic.toml --use-tui
```

| 模态  | 准确率 | weighted-F1 |
| :---: | :----: | :---------: |
|   T   | 66.44% |   66.50%    |
|   A   | 45.25% |   42.76%    |
|   V   | 38.35% |   34.26%    |
|  T+A  | 66.05% |   66.35%    |
|  T+V  | 66.86% |   65.47%    |
|  A+V  |   -    |      -      |
| T+A+V | 67.32% |   66.61%    |
