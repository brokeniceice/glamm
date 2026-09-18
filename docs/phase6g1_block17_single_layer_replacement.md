# Phase 6G.1 — CLIP Block-17 Single-Layer Replacement

Status: **COMPLETE STOP**. The only representation change is legacy center-crop CLIP block22 to block17. The Phase4C-A adapter was retrained with its matched original protocol; R1 restarted from Phase4H-C A2 epoch3 Utility + Phase4F epoch9 Rectifier, never from current selected new R1.

| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|
| A0 block22 current new R1 | 0.202488 | 0.104586 | 0.283836 | 0.222581 | 0.364117 |
| A1 block17 matched new R1 | 0.205624 | 0.110012 | 0.286640 | 0.223476 | 0.365314 |

- IoU paired: `{'mean_delta': 0.003135988806098258, 'median_delta': 0.0, 'bootstrap_95_ci': [-0.004077809004240429, 0.010367878570635336], 'wins': 471, 'ties': 190, 'losses': 445, 'wilcoxon_statistic': 195784.0, 'wilcoxon_pvalue': 0.07606187840260133}`
- F1 paired: `{'mean_delta': 0.0028040483873446203, 'median_delta': 0.0, 'bootstrap_95_ci': [-0.005783223299201451, 0.011413519450064075], 'wins': 471, 'ties': 190, 'losses': 445, 'wilcoxon_statistic': 197176.0, 'wilcoxon_pvalue': 0.10955087034079461}`

```text
BLOCK17_R1_NOT_STABLY_BETTER
```

No test, Official1000, OOD, block11 or multi-level fusion was run.
