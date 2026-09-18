# Phase 6G.6 — Adapter + Rectifier + Utility Joint Training

Status: **COMPLETE STOP**. C1, CLIP, selected block11+17 fusion and SAM remained frozen. The original Phase4C-A adapter, random I2 Rectifier and random I2 Utility were optimized jointly from their matched initializations. No standalone adapter dense-head loss was added.

| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|
| A0 current block22 new R1 | 0.202488 | 0.104586 | 0.283836 | 0.222581 | 0.364117 |
| A1 joint-from-init block11+17 | 0.207724 | 0.109879 | 0.288003 | 0.215205 | 0.354187 |

- paired: `{'n': 1106, 'iou': {'mean_delta': 0.005235684696563309, 'median_delta': 0.0, 'bootstrap_95_ci': [-0.0031345624702702572, 0.013524691251068123], 'wins': 460, 'ties': 187, 'losses': 459, 'wilcoxon_statistic': 202384.0, 'wilcoxon_pvalue': 0.2642403143368106}, 'f1': {'mean_delta': 0.0041668753982132475, 'median_delta': 0.0, 'bootstrap_95_ci': [-0.006210804148817776, 0.014376345024901017], 'wins': 460, 'ties': 187, 'losses': 459, 'wilcoxon_statistic': 204153.0, 'wilcoxon_pvalue': 0.3699091919098352}}`
- SEG trigger invariance: `{'validation_samples': 1106, 'valid_seg_samples': 1090, 'trigger_rate': 0.9855334758758545, 'A0_A1_exact': True, 'basis': 'same frozen C1 canonical G0 cache and sample-valid mask'}`
- selected parameter updates: `{'adapter': {'absolute_l2': 9.273143301067599, 'relative_l2': 0.24835260290237215, 'max_abs': 0.07184070348739624}, 'rectifier': {'absolute_l2': 7.587921545801531, 'relative_l2': 0.24743813729475078, 'max_abs': 0.10948293656110764}, 'utility': {'absolute_l2': 7.6414898435702945, 'relative_l2': 0.19407832983403137, 'max_abs': 0.2406303882598877}}`
- reference rows are preserved in `summary.json` and were not used for selection.

```text
END_TO_END_FORENSIC_JOINT_TRAINING_NOT_STABLY_BETTER
```
