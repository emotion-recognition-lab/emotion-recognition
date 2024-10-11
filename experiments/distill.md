### 各模态组合针对文本模态蒸馏的实验复现

git commit SHA: 02e87a5a3a15b69c0a26d6471fd4dfd06b76415d

```sh
uvx nanoflow experiments/distill.toml --use-tui
```

| 模态  | 准确率 | weighted-F1 |
| :---: | :----: | :---------: |
|  T+A  | 66.70% |   66.06%    |
|  T+V  |        |             |
| T+A+V |        |             |

#### TODO:

- [ ] 对音频蒸馏
- [ ] 对视觉蒸馏
- [ ] 对组合模态蒸馏
- [ ] 多步骤蒸馏
