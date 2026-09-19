# Phase 6G.13 — Utility Training on Frozen Main+Side Rectifier

Status: **COMPLETE STOP**.

| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 |
|---|---:|---:|---:|
| A0 frozen main+side (no Utility) | 0.185546 | - | - |
| A1 same Rectifier + trained Utility | 0.192480 | 0.100842 | 0.272014 |

- paired A1 - A0: `{'n': 1106, 'mean_difference': 0.0069349351029504794, 'median_difference': 0.00016103081725304946, 'bootstrap_95_ci': [0.0045091459305024, 0.009335789128906662], 'wins': 561, 'ties': 245, 'losses': 300, 'wilcoxon_statistic': 121922.0, 'wilcoxon_pvalue': 2.877895835954255e-18}`
- utility diagnostics: `{'n': 1106, 'gate': {'mean': 0.556641579725212, 'median': 0.5575058162212372, 'q10': 0.4969274401664734, 'q25': 0.5260119140148163, 'q75': 0.5992407947778702, 'q90': 0.6417480409145355}, 'side_benefit': {'mean': 0.00430162704404029, 'median': 0.0, 'positive_fraction': 0.379746835443038}, 'correlation': {'pearson': 0.00675214913459158, 'spearman': 0.02026932883156285}, 'side_help': {'n': 420, 'gate_mean': 0.5714593712063063, 'gate_median': 0.5647298097610474}, 'side_harm': {'n': 423, 'gate_mean': 0.5657847925280848, 'gate_median': 0.5598745346069336}}`
- SEG trigger invariance: `{'valid': 1090, 'n': 1106, 'rate': 0.9855334758758545}`

```text
UTILITY_AMPLIFIES_COMPLEMENTARY_CORRECTION
```

No Rectifier/side/Adapter/fusion training, internal test, Official1000, or OOD was accessed.
