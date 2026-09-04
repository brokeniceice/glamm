# Phase 4G-1Q CSCU-LF Conditional-Utility Validity Preflight

## 1. Protocol freeze

Phase 4G-1P 的 architecture、target、tau=0.0417200699、split、corruption、permutation、bootstrap 与 gate threshold 均保持冻结。本阶段没有 architecture、loss、tau、learning-rate、checkpoint 或 threshold sweep。

## 2. Population and firewall

UTILITY-FIT 5,258（valid 5,171 / invalid 87）；UTILITY-CAL 1,127（1,108 / 19）；UTILITY-AUDIT 1,126（1,108 / 18）。AUDIT 总物理读取数=2：原中止读取1次、用户明确授权的 recovery read 1次；没有第三次读取。旧 G1-C TRAIN-AUDIT、development validation、internal test、official1000 均未访问。

## 3. Training configuration

Seed 3407；AdamW lr=1e-4、weight decay=1e-4；batch=8；10 epochs；grad clip=1；无 scheduler、early stopping 或 model selection。只优化冻结的 371,803-parameter context/interaction/U branch；source experts始终 frozen。

## 4. FIT results

`UTILITY_FIT_COMPLETE=YES`。Epoch 1 loss=0.6580987782340935（完整逐 epoch 轨迹见 fit_history.csv）；epoch 10 final loss=0.6411805791588361。正式 checkpoint 仅为 epoch 10 final。

## 5. CAL results

`CALIBRATION_STATUS=PASS`；T_U=1.16673851；image-balanced soft BCE 0.64849919 → 0.64755821。

## 6. Utility relative-loss validity

Primary pooled-pixel Spearman=0.745942，image-cluster bootstrap 95% CI=[0.7278191581201015, 0.7632723683930914]；`UTILITY_RELATIVE_LOSS_RELATION=PASS`。Foreground/background/boundary 与 image-level结果均保存在 results.json，未替换 primary endpoint。

## 7. Four-state diagnostic

L-correct/F-wrong mean U=0.530146598815918；L-wrong/F-correct mean U=0.609617292881012。四态完整 count、quartile 与 histogram见 results.json。

## 8. Cross-image corruption

Cross−matched mean=0.002729，95% CI=[-0.0010504298039883482, 0.0063942777974672245]；`CROSS_IMAGE_UTILITY_RESPONSE=FAIL`。

## 9. Spatial-shuffle corruption

Shuffle−matched mean=0.035431，95% CI=[0.034037979915469134, 0.03680287725712418]；`SPATIAL_SHUFFLE_UTILITY_RESPONSE=FAIL`。

## 10. QMF conditional relation

Effective forensic weight vs calibrated forensic NLL：Pearson={'point': -0.11003408391908902, 'ci95': [-0.18093105571201915, -0.03359233738473956]}；Spearman={'point': -0.10261101269836306, 'ci95': [-0.1604604344306503, -0.042031412523076316]}；`QMF_CONDITIONAL_RELATION=PASS`。该项仅为 conditional weight-quality criterion，不作 intrinsic-uncertainty claim。

## 11. Utility intervention identity

`UTILITY_INTERVENTION_IDENTITY=PASS`。Source posteriors、masses、aligned L/F features 的 before/after hash bit-exact，support外 U=0。

## 12. Utility permutation causal test

`UTILITY_PERMUTATION_SENSITIVITY=PASS`；完整 image/spatial fused-FG MAD 与 dominance-change 数值见 results.json。

## 13. Vacuous / invalid-G0 checks

`VACUOUS_EXACT_RECOVERY=PASS`；`INVALID_G0_POLICY=PASS`；`NO_ORACLE_LEAKAGE=PASS`。Absent/off/vacuous/unsupported 均采用 direct identity dispatch，invalid G0不得被 forensic rescue。

## 14. Gate summary

```json
{
  "schema": "phase4g1q_gate_summary_v1",
  "UTILITY_FIT_COMPLETE": "YES",
  "CALIBRATION_STATUS": "PASS",
  "UTILITY_RELATIVE_LOSS_RELATION": "PASS",
  "CROSS_IMAGE_UTILITY_RESPONSE": "FAIL",
  "SPATIAL_SHUFFLE_UTILITY_RESPONSE": "FAIL",
  "QMF_CONDITIONAL_RELATION": "PASS",
  "UTILITY_INTERVENTION_IDENTITY": "PASS",
  "UTILITY_PERMUTATION_SENSITIVITY": "PASS",
  "VACUOUS_EXACT_RECOVERY": "PASS",
  "NO_ORACLE_LEAKAGE": "PASS",
  "INVALID_G0_POLICY": "PASS",
  "SOURCE_HASH_INTEGRITY": "PASS",
  "UTILITY_AUDIT_ACCESS_COUNT": 2,
  "AUTHORIZED_RECOVERY_AUDIT_READ_COUNT": 1,
  "DEVELOPMENT_VALIDATION_ACCESSED": "NO",
  "INTERNAL_TEST_ACCESSED": "NO",
  "OFFICIAL1000_ACCESSED": "NO",
  "CONDITIONAL_UTILITY_PREFLIGHT": "FAIL",
  "FORMAL_TRAINING_JUSTIFIED": "NO",
  "FORMAL_TRAINING_EXECUTED": "NO"
}
```

## 15. Scientific interpretation

旧 Intrinsic-PCERF 的 uncertainty-error 与 spatial-shuffle failure保持历史原结论且未重跑。CSCU-LF 学到了与 frozen relative loss 强正相关的 utility，并通过 QMF、identity、permutation 与 fallback gates；但 matched utility 在 cross-image 与 spatial-shuffle 后没有按预注册方向下降，反而点估计上升。因此 cross-source mismatch sensitivity 未获支持，不能声称 G1-C failure mode 已被 conditional-utility redesign 机制性缓解，也不能授权正式 localization training。本阶段没有评估或声称最终 G0、Phrase、TF 或 localization 改善。

## 16. Final authorization decision

`CONDITIONAL_UTILITY_PREFLIGHT=FAIL`  
`FORMAL_TRAINING_JUSTIFIED=NO`

即使 justified=YES，也仅表示可以申请下一阶段授权。本阶段到此 STOP；没有自动启动正式 CSCU-LF localization training 或任何封存评估。
