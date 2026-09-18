# Phase 6G.3A — Spatial Prior Stem Diagnostic

Status: **COMPLETE STOP**. Only internal TRAIN/validation were used. Stem and CLIP-side sources were frozen; Rectifier/Utility and all external sets were untouched.

| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|
| block11 | 0.151869 | 0.053200 | 0.218890 | 0.173986 | 0.296403 |
| block17 | 0.154149 | 0.060752 | 0.222925 | 0.185968 | 0.313614 |
| block22 | 0.143842 | 0.048468 | 0.209971 | 0.178934 | 0.303553 |
| block11_17_attention | 0.186741 | 0.109707 | 0.269635 | 0.210842 | 0.348257 |
| spatial_prior_stem_only | 0.004547 | 0.000000 | 0.008323 | 0.005831 | 0.011594 |
| global_spatial_attention_adapter | 0.185957 | 0.101209 | 0.266211 | 0.214902 | 0.353776 |
| position_aligned_fusion | 0.184110 | 0.104160 | 0.266961 | 0.205901 | 0.341488 |

## Frozen-interface detail

For A1, the selected Phase6G.2A block11+17 fusion and the selected Phase6G.3 A0 1024→256 projection are frozen. The selected Phase6G.3 A1 Spatial Prior Stem is frozen. Only position-wise 1×1 `Project`, scalar `gamma`, and the 1×1 dense head train; no cross-position attention is present.

Aligned diagnostics: `{'spatial_patch_norm_mean': 8.33453532244418, 'clip_patch_norm_mean': 9.568305636592388, 'aligned_residual_patch_norm_mean': 0.61017117721622, 'residual_to_clip_norm_ratio': 0.06377003415136763, 'gamma': 0.023987168446183205}`

## Paired diagnostics

- stem_vs_block11: `{'iou': {'n': 1106, 'mean_difference': -0.14732201613287524, 'median_difference': -0.045198738030154194, 'bootstrap_95_ci': [-0.1593548834904106, -0.13525460010658535], 'wins': 40, 'ties': 334, 'losses': 732, 'wilcoxon_statistic': 6261.0, 'wilcoxon_pvalue': 1.1661083269313243e-117}, 'f1': {'n': 1106, 'mean_difference': -0.210567028147736, 'median_difference': -0.08646240671455764, 'bootstrap_95_ci': [-0.2261717878763813, -0.19496459421943566], 'wins': 40, 'ties': 334, 'losses': 732, 'wilcoxon_statistic': 6212.0, 'wilcoxon_pvalue': 9.71410554884395e-118}}`
- aligned_vs_global_spatial_attention: `{'iou': {'n': 1106, 'mean_difference': -0.001847306669639071, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.004644111158523521, 0.000947845224650198], 'wins': 487, 'ties': 216, 'losses': 403, 'wilcoxon_statistic': 195625.0, 'wilcoxon_pvalue': 0.7324520978631351}, 'f1': {'n': 1106, 'mean_difference': 0.0007508912762969971, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.0026316608560180924, 0.004180812713197105], 'wins': 487, 'ties': 216, 'losses': 403, 'wilcoxon_statistic': 189301.0, 'wilcoxon_pvalue': 0.24351195465655195}}`
- aligned_vs_attention_fusion_probe: `{'iou': {'n': 1106, 'mean_difference': -0.002631350851915205, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.004867031090069917, -0.00043610150347680174], 'wins': 443, 'ties': 204, 'losses': 459, 'wilcoxon_statistic': 190410.0, 'wilcoxon_pvalue': 0.091288990817434}, 'f1': {'n': 1106, 'mean_difference': -0.0026733164249295774, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.005614292120559424, 0.00019617335491849946], 'wins': 443, 'ties': 204, 'losses': 459, 'wilcoxon_statistic': 192653.0, 'wilcoxon_pvalue': 0.16089883181527775}}`
- aligned_vs_stem: `{'iou': {'n': 1106, 'mean_difference': 0.17956301295140245, 'median_difference': 0.10261379780614592, 'bootstrap_95_ci': [0.16777646869370333, 0.19173324166797615], 'wins': 865, 'ties': 213, 'losses': 28, 'wilcoxon_statistic': 4778.0, 'wilcoxon_pvalue': 7.35774613816128e-141}, 'f1': {'n': 1106, 'mean_difference': 0.25863852898064554, 'median_difference': 0.18457789619913006, 'bootstrap_95_ci': [0.24328307915340996, 0.2741858873697176], 'wins': 865, 'ties': 213, 'losses': 28, 'wilcoxon_statistic': 4640.0, 'wilcoxon_pvalue': 4.6768905209897896e-141}}`

```text
SPATIAL_PRIOR_STEM_WEAK
```

Raw predictions, checkpoints, histories, provenance, and paired statistics are under `outputs/phase6g3a_spatial_prior_stem_diagnostic/`.
