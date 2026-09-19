# Phase 6G.12 — Full Rectifier Training with Complementary Side Correction

Status: **COMPLETE STOP**.

| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 |
|---|---:|---:|---:|
| A0 main-only | 0.181244 | 0.086192 | 0.256231 |
| A1 frozen main + side (ref) | 0.185546 | 0.094848 | 0.263737 |
| A2 joint main + side | 0.185791 | 0.081580 | 0.261359 |

- A2 - A0: `{'mean': 0.004547376546922753, 'median': 0.0, 'ci': [0.001371278005670407, 0.007836888608119244], 'wins': 438, 'ties': 268, 'losses': 400, 'wilcoxon_p': 0.010003357044750543}`
- A2 - A1: `{'mean': 0.0002457495028824632, 'median': 0.0, 'ci': [-0.0032727240522887357, 0.0037137195086709885], 'wins': 425, 'ties': 257, 'losses': 424, 'wilcoxon_p': 0.41313016843199446}`
- A1 - A0 reference: `{'n': 1106, 'mean_difference': 0.004301626757551539, 'median_difference': 0.0, 'bootstrap_95_ci': [0.0009165758923755294, 0.007822999296083115], 'wins': 420, 'ties': 263, 'losses': 423, 'wilcoxon_statistic': 170607.0, 'wilcoxon_pvalue': 0.30421110480545643}`

```text
FROZEN_MAIN_SIDE_CORRECTION_PREFERRED
```

No Utility, joint R1, internal test, Official1000, or OOD was accessed.
