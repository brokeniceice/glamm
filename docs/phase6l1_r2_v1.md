# Phase6L1 — R2-v1 joint revision

Status: **COMPLETE STOP AFTER INTERNAL DEV**.

## Protocol

R2-v1 changes only the bridge offset initialization and adds an unconstrained per-channel LayerScale after the zero-initialized residual head. LayerScale is a residual-scale hypothesis, not a guaranteed norm bound. Frozen C2/SAM, immutable Phase6L0 cache, geometry, loss, optimizer, seed, ten epochs, and DEV Mean FG IoU selector were retained.

Preflight G1–G7: PASS. Epoch0 exact per-sample C2-G0 parity: PASS. Selected epoch: 9; updates: 11050. Selected checkpoint SHA256: `b41b3a92b6d81ed19ef04a1f1d49f4fb688c51bf00a616adc4db4de42fe0bc26`.

## Canonical internal DEV results

| Model | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
| --- | ---: | ---: | ---: | ---: |
| C2-G0 epoch0 | 0.155826 | 0.219360 | 0.149181 | 0.259631 |
| R2-v0 selected | 0.200544 | 0.277964 | 0.204791 | 0.339962 |
| R2-v1 selected epoch 9 | 0.198522 | 0.275720 | 0.194617 | 0.325824 |

| Paired contrast | Mean IoU delta | Median delta | 95% CI | W/T/L | Wilcoxon p | Global IoU delta [95% CI] |
| --- | ---: | ---: | --- | --- | ---: | --- |
| v1 − C2-G0 | 0.042696 | 0.002926 | [0.033520, 0.051818] | 585/225/296 | 3.732e-27 | 0.045436 [0.028538, 0.061426] |
| v1 − v0 | -0.002022 | 0.000000 | [-0.007925, 0.003697] | 433/226/447 | 0.7033 | -0.010174 [-0.016225, -0.004303] |

Mean/global F1 paired statistics and full confusion counts are in `outputs/phase6l1_r2_v1/final_internal_dev.json`. Both v0 and v1 selected epochs were chosen using this DEV population, so these intervals do not represent independent held-out confirmation.

## Mechanism and stability

Selected v1: minimum pair distance 0.000000 SAM cells; four-point unique fraction 0.998381; valid-query attention standard deviation 0.152646.

Effective residual ratio v0 → v1: 4.296278 → 1.767273. Cosine(S64,S_adapt) v0 → v1: 0.295965 → 0.503811. Raw residual ratio v1: 73.459985; alpha mean/min/max: 0.013482/-0.002298/0.036762.

Gradient clipping frequency is reported in `training_history.csv`; its value alone cannot prove residual stabilization because LayerScale changes gradient units. Point-specific radii, per-head drift, support behavior, offset/attention distributions, and all epochs are preserved in diagnostics files.

## Historical C1-native R1 context

Phase6E.3 C1-native staged R1 epoch-8 internal DEV Mean FG IoU: 0.204904. R2-v1 does not exceed it. The Phase6E.3 selected selector does not preserve a DEV global-IoU result, so none is inferred here. C1 and C2 use different backbones and valid populations; this is contextual, not a matched causal comparison.

## Conclusion

The joint R2-v1 revision does not establish improved final localization over R2-v0 on canonical internal DEV.

No separate gain attribution to the two joint revisions is possible. No internal test, Official1000, localization OOD, hyperparameter search, or additional R2 arm was run.
