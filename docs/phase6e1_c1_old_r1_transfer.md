# Phase 6E.1 — C1 + old R1 frozen transfer

Official1000 canonical G0 only; all parameters frozen.

| Arm | Mean FG IoU | Global FG IoU | Mean FG F1 | Global FG F1 | SEG trigger rate |
|---|---:|---:|---:|---:|---:|
| P1 | 0.229544 | 0.236949 | 0.319323 | 0.383118 | 0.968000 |
| P1+old_R1 | 0.286588 | 0.267042 | 0.393974 | 0.421521 | 0.968000 |
| C1 | 0.206240 | 0.190361 | 0.291229 | 0.319838 | 0.992000 |
| C1+old_R1 | 0.300479 | 0.283588 | 0.415358 | 0.441867 | 0.992000 |

## Decisions

- `OLD_R1_TRANSFERS_TO_C1 = YES`
- `C1_CHANGES_BASE_LOCALIZATION = YES`
- `SEG_QUERY_SHIFT_OBSERVED = YES`
- `RETRAIN_R1_ON_C1_NEEDED = NO`
