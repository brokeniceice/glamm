# Invalid G0 与 no-oracle-leakage

## Invalid canonical G0

canonical G0没有合法 `[SEG]` 时：localization invalid，formal score固定0。强 forensic evidence也不得 rescue；不得使用zero hidden、last token、TF hidden或authoritative phrase replacement。

synthetic test使用 invalid language stream与强、present、non-vacuous forensic stream。输出 `formal_valid=false`、formal score 0，测试通过：

`INVALID_G0_POLICY = PASS`。

## Leakage audit

PCERF forward API逐项只有 `S64,q_seg,z_L,F24,z_F24,valid_g0,forensic_present,forensic_vacuous,forensic_off,clip_geometry,reliability_permutation,ablation`。检查结果：

| 项 | inference path |
|---|---|
| input tensor | 仅上述 deployment-time tensor |
| training target | 仅在 loss harness，未进入 forward |
| calibration target | 仅 TRAIN-CAL evaluator侧 source error |
| fusion signal | belief、uncertainty、conflict、support |
| routing signal | source availability、vacuous/off、discount |
| metadata | geometry与valid-SEG；无condition identity |

GT、TF identity、Phrase identity、authoritative phrase、GT polygon与official evaluation label均无法通过 forward signature传入。`NO_ORACLE_LEAKAGE = PASS`。

机器可读结果见 `outputs/phase4g05/invalid_g0_policy.json`。
