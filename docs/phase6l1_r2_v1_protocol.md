# Phase6L1 — R2-v1 pre-registered joint revision

Status: implementation and preflight pending. Phase6L0 is complete and immutable.

## Fixed inputs and changes

- Reuse Phase6L0 same-forward C2 TRAIN/DEV caches, canonical spatial references, C2 checkpoint, C2-native frozen SAM, populations, geometry, loss, seed 3407, batch 8, AdamW 1e-4/1e-4, clip 1.0, 10 epochs, and DEV Mean FG IoU earliest-tie selector. Revalidate index and shard SHA, ordered IDs, tensor validity, and geometry. No new generation is used for training.
- Change 1: initialize each bridge head's four offsets at Euclidean SAM-grid radii 0.5, 1.5, 2.5, 3.5 along unit directions `(cos(2πh/8), sin(2πh/8))`. Offset weights remain zero; bias uses `atanh(offset/4)`.
- Change 2: zero-initialized `W_out` produces `delta_raw`; unconstrained per-channel LayerScale starts at 0.01, and `deltaS = alpha * delta_raw`. This is a residual-scale hypothesis, not a guarantee of a smaller learned residual. No other model component or loss changes.
- Fresh v1 initialization only. Preserve Phase6L0 source, checkpoints, metrics, and predictions.

## Gates before the first formal optimizer update

G1 cache/source provenance; G2 Euclidean radial initialization; G3 four distinct initial points per head; G4 full canonical DEV 1,106-sample tensor-exact step-0 `delta_raw=deltaS=0`, `S_adapt=S64`, cached/direct C2-G0 SAM low logits and threshold masks, IDs/order/SEG validity, per-sample metrics and summary; G5 two real-mask backward passes including `alpha` and all R2 modules; G6 frozen C2/SAM hashes and no SAM gradients; G7 finite full-batch forward/backward. Audit optimizer steps are restored before G4. The final `preflight_gates.json` is written only after all gates pass.

A fixed-sample B=1/2/4 C2 generation-grouping diagnostic records token, A/E/r_prime/q_seg/mask differences. It neither changes the immutable training cache nor selects a checkpoint.

## Evaluation and interpretation

Score all 1,106 canonical DEV IDs each epoch, retaining 22 C2-invalid samples at zero. Select by Mean FG IoU only. Report mean/global IoU and F1, TP/FP/FN/TN, paired per-image differences, 2,000 bootstrap draws, Wilcoxon, and full training/gradient/LayerScale/bridge diagnostics. Compare v1 with the matched C2-G0 epoch0 and frozen Phase6L0 R2-v0 selected epoch10. Historical C1-native R1 Phase6E.2 epoch8 is contextual only because C1/C2 and valid populations differ.

The two v1 changes are joint: any gain cannot be partitioned between them. Distinct four-point sampling is claimed only if the selected model's minimum pairwise offset distance remains above the predeclared 0.05 SAM-cell tolerance for every sampled query. A smaller effective residual is claimed only if selected mean `||deltaS||/||S64||` is below Phase6L0's selected value. Gradient clipping frequency or alpha alone cannot establish stabilization. Selected-epoch DEV confidence intervals are conditional on choosing the epoch on that same DEV population.

Stop after 10 epochs, internal DEV selection, paired analysis, and report. Do not access internal test, Official1000, localization OOD, additional arms, or parameter sweeps.
