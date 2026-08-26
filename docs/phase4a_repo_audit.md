# Phase 4A repository audit

## Outcome

The repository already contains the canonical data, target, geometry, loss, metric, paired-bootstrap and matched frozen CLIP artifacts required for Phase 4A. FEPN-v0 therefore needs a new standalone lightweight encoder and training loop, but does not need a new split, target generator, evaluator semantics, or baseline sweep.

## Reuse map

| Requirement | Canonical implementation/artifact | Phase 4A decision |
|---|---|---|
| Train/validation manifests | `outputs/data_audits/unified_forensics_split_v1/{train,val}_combined.jsonl` | Reuse exactly |
| Real/Fake labels | `forensics_domain` plus `class_label` validation in `UnifiedForensicsDataset` | Reuse exactly |
| Fake dense target | `UnifiedForensicsDataset._fake_union_mask` | Reuse exactly; Fake only |
| CLIP geometry | `tools.phase3c1.geometry_for('clip', ...)` and `transform_mask`/`inverse_logits` | Reuse exactly |
| Dense loss | `tools.phase3c1.probe_loss` | Reuse `2.0*BCE+0.5*Dice` |
| Binary metric | `tools.phase3c1.binary_metrics` at logit threshold zero | Reuse exactly |
| Summary/statistics | `tools.phase3c1.summarize` and `paired_statistics` | Reuse and extend with micro/global metrics |
| Matched CLIP probe | Phase 3C.1 selected CLIP probe and per-image validation records/logits | Reuse after provenance audit |
| Residual utilities | NPR/SRM expert code exists, but is tied to external expert representations | Do not reuse as FEPN; implement a fixed deterministic high-pass input operator |
| Lightweight encoder | Existing SAM/GLaMM vision modules are large and interface-coupled | Implement minimal standalone convolutional FPN-style encoder |

## Frozen populations

- Train: 17,672 images, exactly 8,836 Real and 8,836 Fake.
- Internal validation: 2,212 images, exactly 1,106 Real and 1,106 Fake.
- Internal test: 2,208 images, sealed.
- Train/validation/test sample-ID intersections are all zero.
- No new dataset is introduced.

Manifest SHA256:

- train: `3a5cd7040ee9ae5b7c8e6cf6cc5f225441d95849984c9ce6ed9abfa07f9f63bc`;
- validation: `deb751692b7f13b681e4c5ad9476ed4ed3b7354229bb80af9e673ba1c47b8889`.

## Target provenance

For every Fake image, `_fake_union_mask` rasterizes the official annotation polygons for every evidence reference and takes their per-image union. Empty or missing references raise an error. Phase 4A does not regenerate, relabel, or paraphrase this target. Real images participate only in global classification and receive no dense target.

The scientifically valid description is visible/explainable forensic evidence region, not all forged pixels.

## Matched CLIP baseline audit

The Phase 3C.1 CLIP probe is reusable because it uses:

- the same 8,836 train Fake and 1,106 validation Fake IDs;
- the same union-mask constructor;
- `ResizeShortest336 + CenterCrop336` geometry;
- inverse mapping to original image space with unseen crop exterior forced to background;
- bilinear logit resize and fixed zero threshold;
- per-image foreground IoU/F1 under the same metric helper.

Phase 4A freezes the same image geometry, target mapping, inverse mapping and threshold. The baseline will not be retrained or retuned.

CLIP selected probe SHA256: `28d1fff92c6870b157b3f22e8efc7ad4c46a7cdcca3c76f120b9ce61a08b694b`.

CLIP validation predictions SHA256: `540e1bf76b32dd312fb9819fcffa7821757b5d75f1f7289e6fa186f459dff510`.

Historical matched validation Fake metrics are mean FG IoU `0.143842`, median FG IoU `0.048468`, and mean FG F1 `0.209971`. They are reused as a frozen baseline, not as an external confirmatory result.

## Architecture decision

FEPN-v0 uses two independent learnable stems over matched 336×336 views: normalized RGB and a deterministic depthwise high-pass residual derived from the same cropped RGB tensor. Concatenation plus a learnable 1×1 projection feeds a small multi-scale convolutional encoder with standard nearest-neighbor FPN top-down fusion. The resulting dense feature drives a small dense head and a global-average-pooled binary head.

This deliberately avoids pretrained forensic experts, transformer/fusion searches, P1, LLM, SAM and external large backbones. The exact parameter count and feature shape must pass the architecture audit before training.

## Evaluation and statistical boundary

Formal dense evaluation uses validation Fake images only, returns logits to original image space, and fixes threshold zero. The selector uses mean FG IoU then FG F1. Global classification is secondary and evaluated on the balanced full validation population. The safety definition is frozen before training as Accuracy ≥ 0.60 and ROC-AUC ≥ 0.60.

Selected FEPN versus CLIP uses matched per-image paired bootstrap with 10,000 repeats and seed 3407. Internal test and official1000 remain sealed.

The user-frozen training budget is ten complete passes: 11,050 optimizer steps and 176,720 image exposures, with formal candidates at epoch 0–10. There is no validation-performance early stopping. Safety stopping is limited to NaN/non-finite state, gradient explosion, implementation failure, or severe simultaneous feature/output collapse.
