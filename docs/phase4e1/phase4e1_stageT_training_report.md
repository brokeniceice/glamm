# Phase 4E-1 Stage T 训练报告

Teacher 使用全部 8,836 train Fake，5 个完整 epoch；selector 仅使用 validation Fake canonical TF mean FG IoU。

| Epoch | Optimizer updates | Train loss | Validation mean FG IoU | Validation global FG IoU |
|---:|---:|---:|---:|---:|
| 0 | 0 | — | 0.043025 | 0.038949 |
| 1 | 1105 | 1.1028755478537315 | 0.215157 | 0.276409 |
| 2 | 2210 | 0.942799312920525 | 0.251525 | 0.331014 |
| 3 | 3315 | 0.6178186444633954 | 0.256519 | 0.341699 |
| 4 | 4420 | 0.41688364645699777 | 0.261618 | 0.347776 |
| 5 | 5525 | 0.3970457253654708 | 0.266076 | 0.350611 |

Selected epoch：**5**。internal test 与 official1000 未使用。
