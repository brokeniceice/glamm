# Phase6M0 — R1 Intervention Manifold Audit

Status: **COMPLETE STOP**. Diagnostic only; no training, architecture change, checkpoint selection, Official1000, or OOD run.

## Provenance and replay

Checkpoint: `outputs/phase6e2_c1_specific_r1/selected_checkpoint.pt` (SHA256 `c067eb2240af9d5fd61fc5dc367338ed585fcc6a5d5d0f982df0fa886302e603`).
This is Phase6E.2 C1-conditioned R1, initialized from Phase4H-C A2 epoch 3 and Phase4F epoch 9. Phase6E.3 is a distinct from-C1 staged R1.
TRAIN: 8836 total, 8741 valid; DEV: 1106 total, 1090 valid.
Fixed-subset exact replay gate: PASS; max intervention algebra error 0.0078125 from bfloat16 rounding; repeated Rectifier, Utility and SAM logits exact.
Historical intermediate tensor snapshots were not saved, so the gate verifies checkpoint/state hashes, authoritative inputs/forward, within-run exact repeatability and intervention algebra. The complete DEV replay is additionally checked against the historical selected mean IoU.

## Q1. Channel subspace rank

Primary: uncentered, image-balanced second moment on C1-valid images and Rectifier support pixels; each image contributes equally. C is Rectifier proposal; ΔS is the actual post-gate SAM intervention.

| Tensor | K90 | K95 | K99 |
|---|---:|---:|---:|
| C uncentered | 1 | 1 | 2 |
| ΔS uncentered | 1 | 1 | 2 |
| C direction | 1 | 1 | 2 |
| ΔS direction | 1 | 1 | 2 |
| ΔS centered | 1 | 2 | 2 |

ΔS uncentered top-one energy share = 0.9842; centered top-one variation share = 0.9295. The centered result shows the concentration is not solely a shared mean vector.
Pixel-pooled ΔS ranks: K90/K95/K99 = 1/1/2.
TRAIN active thresholds (image-balanced r P25/P50/P75) = top75: 3.790420 / top50: 5.245817 / top25: 6.442823.
Active-only ΔS K95 = top75: 1 / top50: 1 / top25: 1.
Per-image ΔS K90/K95 medians = 1.0 / 1.0.
Half-split mean canonical cosines (K4/8/16/32) = 0.9998 / 0.9999 / 0.9904 / 0.9968.
TRAIN ΔS relative reconstruction errors (image-balanced mean, K1/2/4/8/16/32/64/128) = 0.1259 / 0.0164 / 0.0099 / 0.0052 / 0.0041 / 0.0035 / 0.0028 / 0.0017.
Frozen TRAIN ΔS basis → DEV mean projection energy at TRAIN K90/K95/K99 = 0.9819 / 0.9819 / 0.9997.

## Q2. Intervention magnitude

r = ||ΔS||₂ / (||S64||₂ + ε), on the same model support. Image-balanced TRAIN quantiles:

| P50 | P75 | P90 | P95 | P99 | max |
|---:|---:|---:|---:|---:|---:|
| 5.245817 | 6.442823 | 7.404860 | 7.908327 | 8.789943 | 14.037375 |

C/S64 median (image-balanced) = 9.739785; ΔS/C median = 0.523483.
cos(S64,Sadapt), per-image mean = 0.207592.

## Q3. Evidence counterfactuals

A holds the matched Rectifier C fixed and changes only Utility; this matches the historical Utility ranking semantics. B changes both Rectifier and Utility evidence; it is a new diagnostic, not the historical training contract.

TRAIN IoU/F1 use the Phase4F SAM-canvas target after removing padding and mapping to the original-normalized 256 grid. The frozen SAM logits use the same geometry normalization path. DEV replay IoU below uses original-image masks.

| Condition | mean U | mean intervention ratio | mean ΔIoU vs SAM | mean mask change |
|---|---:|---:|---:|---:|
| matched | 0.505040 | 5.081998 | 0.095077 | 0.048540 |
| cross Utility | 0.196652 | 2.029927 | 0.044561 | 0.037892 |
| shuffle Utility | 0.139712 | 1.462707 | 0.028456 | 0.030291 |
| cross full | 0.196652 | 1.979378 | -0.004728 | 0.045023 |
| shuffle full | 0.139712 | 1.391922 | 0.019214 | 0.029992 |

Projection energy onto the frozen matched ΔS TRAIN K95 basis: matched: 0.9801 / cross_utility: 0.9825 / shuffle_utility: 0.9839 / cross_full: 0.9813 / shuffle_full: 0.9835.

Paired A comparisons (matched minus mismatch):

| Comparison | ΔU mean [95% CI] | Δr mean [95% CI] | ΔIoU mean [95% CI] | ΔIoU W/T/L | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| cross Utility | 0.308387 [0.306817, 0.309968] | 3.052072 [3.031755, 3.071471] | 0.050517 [0.047362, 0.053477] | 5092/949/2700 | 1.05e-256 |
| shuffle Utility | 0.365327 [0.364084, 0.366468] | 3.619292 [3.601077, 3.636833] | 0.066621 [0.063129, 0.070243] | 5213/915/2613 | 6.88e-292 |

## Interpretation

Q1: The deployed Phase6E.2 R1 intervention is concentrated in a low-dimensional channel subspace on internal TRAIN. This survives centered, active-only, image-balanced/pixel-pooled and fixed half-split checks, and the frozen basis transfers to DEV. A low-dimensional channel basis alone does not establish that a new model can reproduce the spatial coefficients or the final SAM masks.

Q2: The actual R1 intervention is large relative to S64 on model support; it is not a small perturbation in this measured norm. This magnitude distribution is descriptive, not a chosen R2 residual budget.

Q3: Under the historical Utility-only mismatch contract, matched forensic context produces a larger gate and a larger mean TRAIN segmentation benefit than deterministic cross/shuffle evidence. Paired intervals exclude zero, while the per-image gate-difference/benefit-difference correlations are weak. Full evidence mismatch is a separate diagnostic. Similar matched-basis projection energy across conditions means out-of-basis movement is not the dominant contrast captured by this basis test; amplitude and spatial coefficients require separate interpretation.

Historical selected DEV mean IoU = 0.202732101; replay = 0.202732101; absolute drift = 0.000000000.

The rank and magnitude findings describe this deployed R1 on internal TRAIN. They do not establish an intrinsic SAM dimension or a chosen R2 hyperparameter. All hypothesis failures remain in the diagnostic JSON files.
