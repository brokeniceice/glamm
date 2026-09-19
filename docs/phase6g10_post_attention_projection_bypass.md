# Phase 6G.10 — Post-Attention Projection Bypass Ablation

Status: **COMPLETE STOP**. Only the Rectifier stage and fresh R2/Delta_F probes were trained. Utility, joint R1, internal test, Official1000, and OOD were not used.

## Rectifier objective and Delta_F decodability

| Arm | Rectifier mean IoU | R2 probe mean IoU | Delta_F probe mean IoU | Delta_F/R2 |
|---|---:|---:|---:|---:|
| A0 | 0.181244 | 0.147642 | 0.118960 | 0.805734 |
| A1 | 0.179407 | 0.137850 | 0.118242 | 0.857758 |
| A2 | 0.173795 | 0.150160 | 0.128590 | 0.856350 |

## Primary paired comparisons

| Comparison | Metric | Delta | 95% CI | W/T/L | Wilcoxon p |
|---|---|---:|---|---:|---:|
| A1_minus_A0 | rectifier_objective | -0.001837 | [-0.004186, 0.000377] | 395/270/441 | 0.1134 |
| A2_minus_A0 | rectifier_objective | -0.007448 | [-0.010258, -0.004777] | 334/270/502 | 1.773e-08 |
| A2_minus_A1 | rectifier_objective | -0.005612 | [-0.008159, -0.003048] | 359/280/467 | 2.265e-05 |
| A1_minus_A0 | R2_probe | -0.009792 | [-0.013953, -0.005659] | 275/455/376 | 2.041e-06 |
| A2_minus_A0 | R2_probe | +0.002518 | [-0.000909, 0.005955] | 359/453/294 | 0.06802 |
| A2_minus_A1 | R2_probe | +0.012311 | [0.008055, 0.016559] | 383/444/279 | 1.522e-07 |
| A1_minus_A0 | Delta_F_probe | -0.000719 | [-0.004973, 0.003734] | 267/579/260 | 0.3134 |
| A2_minus_A0 | Delta_F_probe | +0.009630 | [0.005411, 0.013902] | 361/530/215 | 1.699e-09 |
| A2_minus_A1 | Delta_F_probe | +0.010348 | [0.005556, 0.015147] | 373/522/211 | 7.066e-11 |

## Post-attention mapping spectra

| Arm | Mapping | Effective rank (Shannon) | Condition number |
|---|---|---:|---:|
| A0 | out_proj | 144.506 | 7.21e+03 |
| A0 | projection | 147.497 | 1.72e+03 |
| A0 | composition | 42.935 | 4.56e+05 |
| A1 | out_proj | N/A | N/A |
| A1 | projection | 144.923 | 2.67e+03 |
| A1 | composition | 144.923 | 2.67e+03 |
| A2 | out_proj | N/A | N/A |
| A2 | projection | N/A | N/A |
| A2 | composition | N/A | N/A |

## Gate

- `A1_vs_A0_supported`: `False`
- `A2_vs_A0_supported`: `True`
- `A2_above_A1_supported`: `True`
- `A1_above_A2_supported`: `True`

## Metric-direction audit

- Rectifier objective order: `A0 > A1 > A2`
- Delta_F probe order: `A2 > A0 > A1`
- R2 probe order: `A2 > A0 > A1`
- metric conflict: `True`
- decision note: `The single-label gate is metric-dependent: A2 is stably better on Delta_F decodability but stably worse on the Rectifier segmentation objective. Do not adopt A2 for task performance.`

```text
DIRECT_R2_RESIDUAL_BYPASS_SUPPORTED
```

Secondary task-level interpretation: `LOW_EFFECTIVE_RANK_PROJECTION_NOT_CAUSALLY_HARMFUL`
