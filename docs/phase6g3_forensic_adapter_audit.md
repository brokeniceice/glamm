# Phase 6G.3 — Forensic Adapter Architecture Audit

Status: **COMPLETE STOP**. Selected Phase6G.2A fusion and CLIP were frozen. No Rectifier, Utility, test, Official1000 or OOD access occurred.

| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|
| A0 original Phase4C-A | 0.184992 | 0.095747 | 0.264673 | 0.214721 | 0.353531 |
| A1 spatial-prior interaction | 0.185957 | 0.101209 | 0.266211 | 0.214902 | 0.353776 |

- IoU paired: `{'n': 1106, 'mean_difference': 0.000965028717086933, 'median_difference': 0.0, 'bootstrap_95_ci': [0.00036438593600982607, 0.0015896034713304321], 'wins': 458, 'ties': 280, 'losses': 368, 'wilcoxon_statistic': 150856.0, 'wilcoxon_pvalue': 0.0036836335333288546}`
- F1 paired: `{'n': 1106, 'mean_difference': 0.0015379364415898766, 'median_difference': 0.0, 'bootstrap_95_ci': [0.0007096469938009989, 0.002408126181344172], 'wins': 458, 'ties': 280, 'losses': 368, 'wilcoxon_statistic': 148474.0, 'wilcoxon_pvalue': 0.0011486814354446633}`
- A1 diagnostics: `{'clip_token_l2_mean': 9.37022885841276, 'spatial_residual_token_l2_mean': 0.10982119378397832, 'residual_to_clip_norm_ratio': 0.011720225348111842, 'attention_entropy_mean': 5.926189978294687, 'attention_max_weight_mean': 0.006943861188384599, 'attention_diagonal_mass_mean': 0.0017788932041779852, 'gamma_spatial': 0.01637990213930607}`
- complexity: `{'A0_parameters': 469249, 'A1_parameters': 778370, 'incremental_parameters': 309121, 'incremental_MACs_per_image_approx': 491337216, 'incremental_FLOPs_per_image_approx': 982674432, 'convention': 'one MAC equals two FLOPs; normalization, GELU, interpolation and softmax excluded'}`
- rescue samples: `0`

```text
SPATIAL_PRIOR_ADAPTER_SUPPORTED
```
