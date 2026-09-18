# Phase 6G.4 — Multi-Scale Local Forensic Adapter

Status: **COMPLETE STOP**. The selected block11+17 fusion was frozen. Rectifier, Utility, test, Official1000 and OOD were not accessed.

| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|
| A0 | 0.184992 | 0.095747 | 0.264673 | 0.214721 | 0.353531 |
| A1 | 0.184364 | 0.100383 | 0.265308 | 0.209539 | 0.346477 |

Upstream attention linear probe mean IoU: **0.186741**.

- IoU paired: `{'n': 1106, 'mean_difference': -0.0006280623748762641, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.0038786990521313316, 0.002689715756305467], 'wins': 447, 'ties': 246, 'losses': 413, 'wilcoxon_statistic': 183532.0, 'wilcoxon_pvalue': 0.8280191675210573}`
- F1 paired: `{'n': 1106, 'mean_difference': 0.000635023241814632, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.0034089150406980393, 0.0047132202355805244], 'wins': 447, 'ties': 246, 'losses': 413, 'wilcoxon_statistic': 182537.0, 'wilcoxon_pvalue': 0.72349520342053}`
- A1 diagnostics: `{'base_patch_l2_mean': 11.071824950436069, 'residual_patch_l2_mean': 0.18470037666123051, 'residual_to_base_norm_ratio': 0.016682017417007304, 'selected_gamma': 0.020844371989369392}`
- complexity: `{'A0_parameters': 469249, 'A0_MACs_per_image_approx': 268369920, 'A0_FLOPs_per_image_approx': 536739840, 'A1_parameters': 330882, 'A1_MACs_per_image_approx': 189886464, 'A1_FLOPs_per_image_approx': 379772928, 'A1_minus_A0_parameters': -138367, 'A1_minus_A0_MACs_per_image_approx': -78483456, 'convention': 'one MAC equals two FLOPs; GN, GELU and residual addition excluded'}`

```text
MULTISCALE_FORENSIC_ADAPTER_NOT_SUPPORTED
MULTISCALE_FORENSIC_ADAPTER_NOT_SUPPORTED
```
