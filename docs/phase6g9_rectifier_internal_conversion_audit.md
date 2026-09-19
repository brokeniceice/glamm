# Phase 6G.9 — Rectifier Internal Conversion Audit

Status: **COMPLETE STOP**. Only frozen forward, intermediate tensor export, fresh matched 1x1 Conv probes, and diagnostics were used. No architecture, Rectifier, Adapter, Utility, or R1 weights were modified.

## Actual forward graph

```text
F_forensic [B,256,24,24]
        ↓ flatten + forensic_norm + coordinate_sincos
forensic K/V input [B,576,256]
        ↓ k_proj / v_proj
R1_k / R1_v [B,256,24,24]
SAM semantic [B,256,64,64]
        ↓ flatten + semantic_norm + coordinate_sincos
SAM query [B,4096,256]
        ↓ geometry-aware cross-attention (8 heads, 4096x576)
R2 raw attention output [B,256,64,64]
        ↓ out_proj
R3 output projected [B,256,64,64]
        ↓ projection + gamma + support mask
R4 residual pre-support [B,256,64,64]
        ↓ semantic + residual
R5 post-fusion rectified [B,256,64,64]
        ↓ support-masked residual
R6 Delta_F [B,256,64,64]
```

The graph is taken from the actual `GeometryAwareSAMRectifier.forward`, `CrossAttentiveSemanticRectification.forward`, and `GeometryAwareCrossAttention.forward` implementations, not from documentation.

## Core table (full 1106-sample validation)

| Stage | A0 mean IoU | A1 mean IoU | A1-A0 Delta | IoU 95% CI | W/T/L | Wilcoxon p | A0 median IoU | A1 median IoU | A0 F1 | A1 F1 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| R0 evidence input | 0.164403 | 0.179338 | +0.014935 | [0.007983, 0.021897] | 495/254/357 | 2.659e-07 | 0.070047 | 0.084946 | 0.236466 | 0.255991 |
| R1 K projected | 0.170793 | 0.184366 | +0.013573 | [0.006727, 0.020527] | 513/220/373 | 2.395e-06 | 0.085936 | 0.097870 | 0.247056 | 0.264617 |
| R1 V projected | 0.165111 | 0.178491 | +0.013380 | [0.006471, 0.020415] | 491/251/364 | 5.787e-06 | 0.072510 | 0.083299 | 0.237992 | 0.254721 |
| R2 raw cross-attn output | 0.134598 | 0.169290 | +0.034692 | [0.024972, 0.044205] | 488/315/303 | 1.573e-12 | 0.005460 | 0.063853 | 0.190639 | 0.239236 |
| R3 output projected | 0.151838 | 0.166079 | +0.014240 | [0.004956, 0.023355] | 446/278/382 | 0.00402 | 0.044768 | 0.055672 | 0.218419 | 0.234185 |
| R4 residual pre-support | 0.131250 | 0.139374 | +0.008124 | [-0.001303, 0.017654] | 332/423/351 | 0.8405 | 0.000820 | 0.001335 | 0.185284 | 0.193927 |
| R5 post-fusion (rectified) | 0.149737 | 0.148921 | -0.000815 | [-0.009239, 0.007594] | 370/311/425 | 0.1511 | 0.034600 | 0.023607 | 0.213834 | 0.209280 |
| R6 Delta_F (residual) | 0.126298 | 0.138800 | +0.012502 | [0.003561, 0.021621] | 369/397/340 | 0.08647 | 0.000830 | 0.007310 | 0.179389 | 0.195927 |

## Gain retention

| Transition | Gain previous | Gain current | Retention | Interpretable |
|---|---:|---:|---:|---|
| R0 -> R1_v | +0.014935 | +0.013380 | 0.8958 | True |
| R1_v -> R2 | +0.013380 | +0.034692 | 2.5928 | True |
| R2 -> R3 | +0.034692 | +0.014240 | 0.4105 | True |
| R3 -> R4 | +0.014240 | +0.008124 | not interpretable | False |
| R4 -> R5 | +0.008124 | -0.000815 | not interpretable | False |
| R5 -> R6 | -0.000815 | +0.012502 | not interpretable | False |

## Cross-attention diagnostics

| Arm | Entropy mean | Max weight mean | Effective tokens mean | Spatial variance mean | GT-related mass mean | Non-GT mass mean |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 4.315418 | 0.116752 | 111.337363 | 0.066930 | 0.043692 | 0.956308 |
| A1 | 4.284974 | 0.137720 | 114.065743 | 0.063949 | 0.087181 | 0.912819 |

## Information transformation diagnostics

Per-stage feature L2, channel variance, and spatial variance are stored in `transformation_diagnostics.json`. Key pairwise cosine and norm ratios:

| Arm | R0-R1_v cos | R1_v-R2 cos | R2-R3 cos | R3-R6 cos | R6-S_base cos | ||R6||/||S_base|| |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0.051393 | 0.871579 | 0.017420 | -0.020175 | 0.012714 | 8.305330 |
| A1 | 0.060248 | 0.859155 | -0.019832 | 0.060455 | 0.009573 | 8.053558 |

## Decision

- decision: `POST_ATTENTION_PROJECTION_IS_PRIMARY_RECTIFIER_BOTTLENECK`
- first break: `R4/R5`
- stable rule: positive requires mean delta > 0, bootstrap 95% CI lower > 0, Wilcoxon p < 0.05.

```text
POST_ATTENTION_PROJECTION_IS_PRIMARY_RECTIFIER_BOTTLENECK
```

No internal test, Official1000, or OOD data were accessed.
