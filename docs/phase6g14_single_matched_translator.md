# Phase 6G.14 — Single Matched SAM-Compatible Translator

## 1. Exact scientific question

Under frozen block11+17 evidence, C1, and SAM, does an exactly function-matched single affine translator optimize as well as or better than the factorized two-affine translator?

## 2. A0/A1 architecture

- A0: `R2 -> out_proj -> rectification.projection -> gamma -> support -> Delta`.
- A1: `R2 -> single_projection -> gamma -> support -> Delta`.
- Q/K/V, geometry, evidence, C1, SAM, data, seed, optimizer, schedule, precision and selector are matched. No side, Utility, or nonlinearity.

## 3. Function-equivalent initialization derivation

`W_single=W2@W1`, `b_single=W2@b1+b2`, with common `gamma=0.01`.

## 4. Step-0 equivalence numerical check

```json
{
  "schema": "phase6g14_step0_equivalence_v1",
  "definition": "A0 W2(W1 R2+b1)+b2 versus A1 W_single R2+b_single before common gamma/support",
  "fp32_primary_gate": {
    "max_abs_error": 1.7881393432617188e-06,
    "mean_abs_error": 1.3287498745739867e-07,
    "relative_l2_error": 5.161080594007217e-07
  },
  "bf16_training_path_diagnostic": {
    "max_abs_error": 0.0078125,
    "mean_abs_error": 0.0008380950894206762,
    "relative_l2_error": 0.003836846211925149
  },
  "tolerance": {
    "max_abs_error": 2e-06,
    "relative_l2_error": 2e-06
  },
  "passed": true,
  "note": "BF16 is diagnostic only because factorized and merged evaluation have different rounding points."
}
```

The FP32 analytic/function gate passed. BF16 differences are reported, not hidden: factorized and direct paths have different rounding points.

## 5. Parameter / trainable ownership audit

- A0 trainable: 329,985
- A1 trainable: 264,193
- Complete names/shapes are stored in `initialization_audit.json`.

## 6. Training recipe

Phase6G.10 matched recipe: seed 3407, batch 8, 10 epochs, AdamW lr 5e-5, weight decay .05, betas (.9,.999), 5% warmup+cosine, gradient clip 1.0, BF16 forward, canonical internal TRAIN/DEV, independent DEV mean-IoU selection with earlier-epoch tie break.

## 7. DEV results

| Arm | selected epoch | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 |
|---|---:|---:|---:|---:|---:|---:|
| A0 fresh factorized | 8 | 0.179351 | 0.083773 | 0.254328 | 0.202512 | 0.336815 |
| A1 fresh single | 9 | 0.181284 | 0.088324 | 0.256372 | 0.202926 | 0.337388 |

## 8. Paired statistics

- IoU A1-A0: delta=+0.001933, CI=[0.000382821391740264, 0.0034450235879303365], W/T/L=430/278/398, Wilcoxon p=0.0144076.
- F1 A1-A0: delta=+0.002044, CI=[9.909146963248059e-05, 0.00394682342461348], W/T/L=430/278/398, Wilcoxon p=0.0164333.

## 9. Training dynamics

Full per-epoch loss, DEV IoU, total/translator gradient norm, effective-map norm and gamma trajectory are in each arm's `training_curve.json`. A0 rows=10; A1 rows=10.

## 10. Effective-map analysis

```json
{
  "A0": {
    "weight": {
      "shape": [
        256,
        256
      ],
      "rank": 256,
      "effective_rank_shannon": 41.97061136191982,
      "effective_rank_participation": 10.921190248136115,
      "stable_rank": 3.883111164515989,
      "condition_number": 116932.79310360935,
      "singular_values_top10": [
        3.410742182918972,
        2.5546614795109392,
        0.8796113915373338,
        0.8298535409217528,
        0.8158911292410305,
        0.797086054886011,
        0.7795041535234372,
        0.7703197233890283,
        0.7397382234138298,
        0.7370428881134201
      ]
    },
    "bias_l2": 0.6886029059788175,
    "weight_frobenius": 6.721075967903896
  },
  "A1": {
    "weight": {
      "shape": [
        256,
        256
      ],
      "rank": 256,
      "effective_rank_shannon": 90.8533934576104,
      "effective_rank_participation": 47.71645316334329,
      "stable_rank": 10.8567866792875,
      "condition_number": 14822.576983794248,
      "singular_values_top10": [
        1.741308877264337,
        1.439162640155697,
        0.8594464094275247,
        0.8451224914842012,
        0.7992581790247485,
        0.792543607502906,
        0.7823160126961466,
        0.7705527493086471,
        0.7523637868706696,
        0.7389222915821536
      ]
    },
    "bias_l2": 0.6587899377516657,
    "weight_frobenius": 5.737549777558802
  },
  "comparison": {
    "weight_cosine": 0.9068845636752482,
    "normalized_matrix_similarity": 0.9068845636752334,
    "weight_difference_frobenius": 2.854619636562273,
    "relative_weight_difference": 0.424726583986603,
    "bias_cosine": 0.9791818348553584,
    "bias_difference_l2": 0.14063040995448173
  }
}
```

Actual DEV R2/Delta distribution comparison:

```json
{
  "r2_cosine": {
    "n": 1090,
    "mean": 0.9478699217148877,
    "std": 0.0030408476072958527,
    "median": 0.9475815296173096,
    "p5": 0.9430500745773316,
    "p95": 0.9532210052013397
  },
  "r2_A1_over_A0_norm": {
    "n": 1090,
    "mean": 1.2380925983463953,
    "std": 0.018213624004122287,
    "median": 1.2364872694015503,
    "p5": 1.2100356638431549,
    "p95": 1.2700316488742829
  },
  "delta_cosine": {
    "n": 1090,
    "mean": 0.9218617288891329,
    "std": 0.0013724090140655669,
    "median": 0.9214969873428345,
    "p5": 0.92050501704216,
    "p95": 0.9243497639894486
  },
  "delta_A1_over_A0_norm": {
    "n": 1090,
    "mean": 0.9468765126455815,
    "std": 0.013839243624268189,
    "median": 0.946032702922821,
    "p5": 0.9264328122138977,
    "p95": 0.970584836602211
  },
  "delta_relative_difference": {
    "n": 1090,
    "mean": 0.3885516296012686,
    "std": 0.0032594086571822263,
    "median": 0.3893597424030304,
    "p5": 0.38299387097358706,
    "p95": 0.39162802547216413
  }
}
```

## 11. Auxiliary R2/Delta diagnostics

Stored under `arms/A0/probe` and `arms/A1/probe`. These probes were trained only after final-localization checkpoint selection and never selected an arm/checkpoint.

## 12. Historical G10/G11 references

| Reference | mean FG IoU | Role |
|---|---:|---|
| historical Phase6G.10 A0 | 0.181244 | non-primary reference |
| historical Phase6G.11 frozen-main+side | 0.185546 | secondary, not fresh paired control |

## 13. Supported interpretation

Decision: **DIRECT_SINGLE_TRANSLATOR_OPTIMIZATION_SUPPORTED**. The primary claim is about optimization geometry/implicit regularization of equivalent affine function classes, including precision and AdamW parameterization effects.

## 14. Unsupported interpretation

This experiment cannot claim different nonlinear expressive capacity, cannot select from probes, cannot attribute any result to Utility/side routing, and cannot treat historical references as fresh paired controls.

## 15. Decision

`DIRECT_SINGLE_TRANSLATOR_OPTIMIZATION_SUPPORTED`

Status: COMPLETE STOP. No nonlinear translator experiment was started.
