# Phase 6E.2 — I1 random utility + Phase4F epoch9 rectifier

| Arm | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 | SEG trigger |
|---|---:|---:|---:|---:|---:|
| P1 | 0.229544 | 0.319323 | 0.236949 | 0.383118 | 0.968000 |
| P1+old_R1 | 0.286588 | 0.393974 | 0.267042 | 0.421521 | 0.968000 |
| C1 | 0.206240 | 0.291229 | 0.190361 | 0.319838 | 0.992000 |
| C1+old_R1 | 0.300479 | 0.415358 | 0.283588 | 0.441867 | 0.992000 |
| C1+I1_random_utility_R1 | 0.317788 | 0.437676 | 0.332440 | 0.498995 | 0.992000 |

## Decisions

- `I1_RANDOM_UTILITY_R1_BEATS_OLD_R1 = YES`
- `I1_RANDOM_UTILITY_ADAPTATION_GAIN = 0.01730914046829063`

Official1000仅在internal-validation selector冻结后访问；完成后STOP。
