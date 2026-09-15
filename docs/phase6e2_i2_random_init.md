# Phase 6E.2 — I2 random utility + random rectifier

| Arm | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 | SEG trigger |
|---|---:|---:|---:|---:|---:|
| P1 | 0.229544 | 0.319323 | 0.236949 | 0.383118 | 0.968000 |
| P1+old_R1 | 0.286588 | 0.393974 | 0.267042 | 0.421521 | 0.968000 |
| C1 | 0.206240 | 0.291229 | 0.190361 | 0.319838 | 0.992000 |
| C1+old_R1 | 0.300479 | 0.415358 | 0.283588 | 0.441867 | 0.992000 |
| C1+I2_random_R1 | 0.310387 | 0.427922 | 0.301102 | 0.462842 | 0.992000 |

## Decisions

- `I2_RANDOM_R1_BEATS_OLD_R1 = YES`
- `I2_RANDOM_ADAPTATION_GAIN = 0.009908309242559643`

Official1000仅在internal-validation selector冻结后访问；完成后STOP。
