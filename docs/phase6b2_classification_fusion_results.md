# Phase 6B.2 — CLIP CLS + LLM [CLS] Fusion

结论：冻结最佳候选 `F2-fixed`。本阶段只使用 internal train/validation，未访问 OOD。

## Complementarity

| Class | N | Both correct | LLM-only | CLIP-only | Both wrong |
|---|---:|---:|---:|---:|---:|
| all | 2212 | 2161 | 19 | 23 | 9 |
| Real | 1106 | 1084 | 14 | 3 | 5 |
| Fake | 1106 | 1077 | 5 | 20 | 4 |

Learned alpha: `0.77447295`; F3 gate: `PASS`.

## Validation metrics

| Arm | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|
| F0 | 0.985533 | 0.998649 | 0.978300 | 0.992767 | 0.007233 | 0.985428 |
| F1 | 0.987342 | 0.998822 | 0.991863 | 0.982821 | 0.017179 | 0.987399 |
| F2-fixed | 0.990958 | 0.999346 | 0.991863 | 0.990054 | 0.009946 | 0.990967 |
| F2-alpha | 0.991410 | 0.999230 | 0.993671 | 0.989150 | 0.010850 | 0.991430 |
| F3 | 0.985986 | 0.999299 | 0.982821 | 0.989150 | 0.010850 | 0.985941 |

阶段状态：`COMPLETE_AND_STOPPED_BEFORE_OOD`。
