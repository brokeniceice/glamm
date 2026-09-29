# Phase6N0 — Low-Rank Evidence-Authorized R2

Status: **COMPLETE STOP AFTER INTERNAL DEV**.

## Frozen protocol

Frozen Phase6M0 TRAIN ΔS uncentered top-two basis: file SHA256 `9872acf654d34168d73e2abfb7bc1ce245752250b7308070eae88b1e2ac8734f`; tensor SHA256 `3629de78ecf82906bfe1cfdad97608c7464ecd0a433272c654be47fe5e82f8ab`. Basis orthonormality max error 2.38e-07. No DEV basis fitting or basis rotation.

Preflight G1–G8: PASS. Full canonical DEV epoch0 C2-G0 exact per-sample parity: PASS. Trainable parameters: 784,243. Ten epochs and 11050 optimizer updates; selected epoch 10 by DEV Mean FG IoU only.

## Canonical internal DEV

| Model | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|
| C2-G0 | 0.155826 | 0.219360 | 0.149181 | 0.259631 |
| R2-v0 selected | 0.200544 | 0.277964 | 0.204791 | 0.339962 |
| Phase6N0 selected epoch 10 | 0.205636 | 0.287740 | 0.207584 | 0.343801 |

| Paired contrast | Mean IoU Δ | 95% CI | W/T/L | Wilcoxon p | Global IoU Δ [95% CI] |
|---|---:|---|---:|---:|---|
| N0 − C2-G0 | 0.049810 | [0.040337, 0.059308] | 624/182/300 | 1.12e-28 | 0.058403 [0.042462, 0.073517] |
| N0 − R2-v0 | 0.005092 | [-0.001382, 0.011523] | 476/191/439 | 0.159 | 0.002793 [-0.007224, 0.012489] |

Mean/global F1, confusion counts and complete paired statistics are saved in `final_internal_dev.json`. The selected epoch was chosen on this DEV population, so the intervals are descriptive for selection, not independent confirmation.

## Mechanism diagnostics

Selected |a1|/|a2| means: 8.700895/1.480177; a2/a1 magnitude ratio: 0.174395. Per-image basis intervention energy fractions b1/b2: 0.966429/0.033571. Coefficient covariance mean: `[[14.712542625150997, -3.1265609520700126], [-3.1265609520700126, 2.2017676662584953]]`.

Intervention ||ΔS||/||S64|| mean: 2.435318; cosine(S64,Sadapt) mean: 0.406855; outside-support max |ΔS|: 0.000000.

Authority g matched/cross/shuffle means: 0.484645/0.087729/0.039765; matched−cross/shuffle: 0.396916/0.444879. Cross uses the next canonical ID for F24/A/E/r_prime with current geometry; shuffle uses one fixed seed-3407 576-patch permutation for F24/A/E and keeps r_prime.

Each epoch's coefficient quantiles, basis use, authority differences, gradient norms and clipping frequency are retained in the diagnostic JSON files and training history. None was a selector.

## Required questions

Q1 — C2-G0: selected Phase6N0 is higher in DEV Mean FG IoU; paired mean Δ = 0.049810, CI [0.040337, 0.059308].

Q2 — R2-v0: Phase6N0 is numerically higher in DEV Mean FG IoU by 0.005092, but the paired 95% CI [-0.001382, 0.011523] includes zero and Wilcoxon p = 0.159. This DEV result does not establish a stable Mean IoU improvement over v0.

Q3 — historical C1 R1: Phase6E.2 mean/global IoU = 0.202732/0.229309; Phase6N0 mean/global IoU = 0.205636/0.207584. N0 is numerically higher in mean by 0.002904 and lower in global by -0.021725. Different backbones and validity populations make this contextual, not a matched causal contrast.

Q4 — second basis direction: measured mean b2 energy fraction = 0.033571, with |a2|/|a1| mean ratio = 0.174395. No binary mechanism threshold was predeclared; these are the observed usage measures.

Q5 — matched authority: mean g_match−g_cross = 0.396916, g_match−g_shuffle = 0.444879; both positive on selected DEV. This describes the fixed mismatch protocol, not calibrated confidence.

No K sweep, extra arm, internal test, Official1000, localization OOD or follow-on R2 was run.
