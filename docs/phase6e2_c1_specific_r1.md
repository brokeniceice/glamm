# Phase 6E.2 — C1-specific new R1

| Arm | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 | SEG trigger |
|---|---:|---:|---:|---:|---:|
| P1 | 0.229544 | 0.319323 | 0.236949 | 0.383118 | 0.968000 |
| P1+old_R1 | 0.286588 | 0.393974 | 0.267042 | 0.421521 | 0.968000 |
| C1 | 0.206240 | 0.291229 | 0.190361 | 0.319838 | 0.992000 |
| C1+old_R1 | 0.300479 | 0.415358 | 0.283588 | 0.441867 | 0.992000 |
| C1+new_R1 | 0.319867 | 0.440385 | 0.343538 | 0.511393 | 0.992000 |

## Decisions

- `C1_SPECIFIC_R1_BEATS_OLD_R1 = YES`
- `C1_SPECIFIC_ADAPTATION_GAIN = 0.019388718274094885`
- `FINAL_R1_CANDIDATE = NEW_R1`

Official1000仅在internal-validation selector冻结后访问；完成后STOP。

## Supplemental I2 — random utility + random rectifier

I1 was not run. I2 reused the identical frozen C1 G0 query cache and used the same 10-epoch recipe/validation selector.

| Arm | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 | SEG trigger |
|---|---:|---:|---:|---:|---:|
| C1+I2 random R1 | 0.310387 | 0.427922 | 0.301102 | 0.462842 | 0.992000 |

## Supplemental I1 — random utility + Phase4F epoch9 rectifier

I1仅将utility恢复为seed-3407随机初始化；rectifier严格复用Phase4F selected epoch9。其余训练、validation selector与Official1000协议均与Phase6E.2一致。

| Arm | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 | SEG trigger |
|---|---:|---:|---:|---:|---:|
| C1+I1 random-utility R1 | 0.317788 | 0.437676 | 0.332440 | 0.498995 | 0.992000 |
