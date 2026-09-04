# Phase 4G-1S mismatch-aware CSCU-LF utility retrain

## Protocol and firewall

CSCU-LF graph保持完全不变：d64/window7/heads4/blocks1，371,803 trainable parameters，REFINEMENT_INCLUDED=NO。Warm-start为 Phase4G-1Q epoch10。一次冻结 margin=0.1、relative:ranking=1:1、cross:shuffle=1:1、seed3407、AdamW lr1e-4/wd1e-4、batch8、10 epochs、grad clip1；无 sweep、scheduler、early stopping、fused segmentation loss或QMF regularizer。只用 UTIL-FIT训练、UTIL-CAL校准；dev/internal test/official1000均未访问。

本次同一 UTIL-AUDIT 是由先前 mismatch failure 驱动的 **confirmatory re-audit，不是 blind audit**；它没有参与参数更新或 calibration。

## FIT and CAL

10 epochs共 6470 updates。Epoch1 checkpoint 在日志目录错误前已保存且未重放，其逐项指标未持久化；可用日志从 epoch2 开始。Total loss 0.6503903147930381 → 0.6389412888012548；relative loss 0.6471324373181427 → 0.6382074947399596；ranking loss 0.003257877024565468 → 0.0007337935369484071。最终 FIT cross/shuffle mean ΔU分别为 -0.40070951048928105 / -0.44786883979384523。CAL `T_U=1.16089785`，objective 0.64851171 → 0.64763212。

## Preregistered gates

- Relative-loss pooled-pixel Spearman=0.729846，95% CI=[0.7109572591434604, 0.7480173467192976]：**PASS**。
- Cross−matched mean ΔU=-0.388616，95% CI=[-0.39293429177516687, -0.3841989778688292]：**PASS**。
- Shuffle−matched mean ΔU=-0.442651，95% CI=[-0.44593373124928143, -0.43928062261253703]：**PASS**。
- QMF conditional relation：**PASS**。
- Utility intervention identity：**PASS**。
- Utility permutation sensitivity：**PASS**。
- Vacuous exact recovery：**PASS**。
- Invalid G0 policy：**PASS**。

完整 four-state、scoped relation、QMF correlations、permutation metrics与hash identity集中在 `outputs/phase4g1s/results.json`。

## Gate summary

```json
{
  "schema": "phase4g1s_gate_summary_v1",
  "MISMATCH_AWARE_UTILITY_FIT_COMPLETE": "YES",
  "CALIBRATION_STATUS": "PASS",
  "UTILITY_RELATIVE_LOSS_RELATION": "PASS",
  "CROSS_IMAGE_UTILITY_RESPONSE": "PASS",
  "SPATIAL_SHUFFLE_UTILITY_RESPONSE": "PASS",
  "QMF_CONDITIONAL_RELATION": "PASS",
  "UTILITY_INTERVENTION_IDENTITY": "PASS",
  "UTILITY_PERMUTATION_SENSITIVITY": "PASS",
  "VACUOUS_EXACT_RECOVERY": "PASS",
  "NO_ORACLE_LEAKAGE": "PASS",
  "INVALID_G0_POLICY": "PASS",
  "SOURCE_HASH_INTEGRITY": "PASS",
  "UTILITY_AUDIT_ACCESS_COUNT": 1,
  "AUDIT_INTERPRETATION": "CONFIRMATORY_REAUDIT_NOT_BLIND",
  "DEVELOPMENT_VALIDATION_ACCESSED": "NO",
  "INTERNAL_TEST_ACCESSED": "NO",
  "OFFICIAL1000_ACCESSED": "NO",
  "MISMATCH_AWARE_UTILITY_PREFLIGHT": "PASS",
  "FORMAL_TRAINING_JUSTIFIED": "YES",
  "FORMAL_TRAINING_EXECUTED": "NO"
}
```

## Interpretation and decision

Phase4G-1Q历史结果保持不变。本阶段只判断 mismatch-aware utility retrain 是否同时保留 relative utility并修复 cross/spatial response；不计算或声称 G0/Phrase/TF/localization performance。任一核心 gate FAIL即停止。

`MISMATCH_AWARE_UTILITY_PREFLIGHT=PASS`  
`FORMAL_TRAINING_JUSTIFIED=YES`

即使全部 PASS，也只输出授权建议，不自动运行正式 localization training。Phase 4G-1S 到此 STOP。
