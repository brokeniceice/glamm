# Phase 6G.0 — Multi-level Dense CLIP Forensic Evidence Audit

Status: **COMPLETE STOP**. Frozen CLIP only; four matched linear probes were trained on internal TRAIN and selected on internal validation. No test, Official1000, OOD, C1, SAM, Rectifier or Utility parameters were accessed for training.

## CLIP audit

ViT-L/14@336 has 24 transformer blocks and 25 hidden states including embeddings. `hidden[-2] = hidden_states[23]` is the output of zero-based block 22, before CLIP final post-layernorm. Patch tokens are `[B,576,1024]`, reshaped to `[B,1024,24,24]`. Preprocessing is ResizeShortest336 + CenterCrop336 + CLIP rescale/normalization.

## Matched probes

| Layer | Block | Mean IoU | Median IoU | Mean F1 | ΔIoU vs current | IoU 95% CI |
|---|---:|---:|---:|---:|---:|---|
| early | 5 | 0.094461 | 0.011007 | 0.144991 | -0.049382 | [-0.058110435620440354, -0.0407742801952188] |
| middle | 11 | 0.151869 | 0.053200 | 0.218890 | 0.008027 | [0.0013834175996456683, 0.014728010243784957] |
| late | 17 | 0.154149 | 0.060752 | 0.222925 | 0.010307 | [0.005417684909808812, 0.015316529251126856] |
| current_hidden_minus2 | 22 | 0.143842 | 0.048468 | 0.209971 | 0.000000 | — |

## Complementarity

Stable superior single layers: `['middle', 'late']`.
Complementarity gate: `True`.

```text
PROCEED_SINGLE_LAYER_REPLACEMENT_TEST
```

All raw predictions, selected logits, paired statistics, rescue sample IDs and provenance are under `outputs/phase6g0_multilevel_dense_clip_audit/`.
