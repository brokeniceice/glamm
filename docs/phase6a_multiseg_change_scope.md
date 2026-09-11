# Phase 6A — Minimal MultiSEG-R1 Change Scope

## Proposed target and loss

Real behavior remains unchanged:

```text
[REAL] No identifiable synthetic artifact evidence is detected.
```

Fake behavior becomes:

```text
[FAKE] explanation ...
<p>artifact phrase 1</p>[SEG]
...
<p>artifact phrase K</p>[SEG]
```

Every SEG uses the same P1 projection, R1 utility, rectifier and SAM parameters. No phrase-specific module is introduced. The mask objective is:

```text
L_mask(image) = mean_i [w_BCE L_BCE(mask_pred_i, mask_gt_i)
                        + w_Dice L_Dice(mask_pred_i, mask_gt_i)]
L_mask(batch) = mean over valid images
```

Weights remain undecided in this audit.

## Required changes

| Component | Concrete change | Scope |
|---|---|---|
| raw SynthScars files | none; `refs[i]` already carries phrase and polygons | NO CHANGE |
| frozen split manifest | retain existing IDs/annotation IDs; no resplit | NO CHANGE |
| unified dataset | replace union-before-training with ordered per-ref masks; keep union as derived view | MEDIUM |
| target builder | emit K `<p>...</p>[SEG]` units in matching ref order | HIGH |
| long-sequence policy | truncate/drop only complete phrase-mask pairs and persist selected indices | HIGH |
| collator | carry variable K masks and phrase metadata; existing list container largely reusable | LOW |
| core SEG extraction | already extracts all SEG predictor states | NO CHANGE |
| shared text projection | applies independently to every SEG | NO CHANGE |
| native SAM decoder | already decodes K prompts for one image | NO CHANGE |
| mask count checks | assert exact post-truncation SEG/mask alignment | LOW |
| mask loss | change total-mask global mean to per-image mean of per-phrase losses | MEDIUM |
| R1 cache schema | replace fixed q-seg `[N,256]`/one union target with flattened values plus image offsets and ordered ref IDs | HIGH |
| Phase4/R1 utility | repeat image-side tensors per SEG and run shared module on `[total_SEG,...]` | MEDIUM |
| image rectifier | compute image-only `S64_rect` once; repeat/gate per SEG as required | LOW |
| `FrozenP1SAMPath` wrapper | flatten variable prompts, repeat embeddings, group outputs by offsets | MEDIUM |
| generation parser | preserve ordered phrase text, SEG indices and K masks | MEDIUM |
| image evaluator | continue union of K binary masks at original resolution, threshold `>0` | LOW |
| phrase evaluator/cache | add phrase-level mask artifacts; do not affect image metrics | MEDIUM |
| checkpoint merge/load | load P1/R1 existing weights strictly for unchanged modules; initialize no K-specific parameters | LOW |

Overall scope: **MODERATE**. The GLaMM extraction, projection and SAM core are already multi-SEG, and R1 weights are shared rather than K-specific. The material work is data/target alignment, variable-length caching, per-image loss normalization, and Phase4 tensor grouping. This is more than a shape-only patch at pipeline level, but it does not require a new backbone or a new localization algorithm.

## R1 vectorization contract

Let image `b` have `K_b` valid phrase/SEG pairs and `T=Σ_b K_b`.

```text
q_seg_flat       [T,256]
image_index      [T]
S64_image        [B,256,64,64]  -- compute once
F24_image        [B,256,24,24]  -- compute once
S64/F24 repeated by image_index
preliminary z_L  [T,1,256,256]
CSCU/R1 shared forward over T
adapted S64      [T,256,64,64]
SAM masks        [T,1,256,256] -> grouped [K_b,H_b,W_b]
```

`GeometryAwareSAMRectifier` is already batch-generic and independent of q-seg. `CSCULF` is batch-generic but assumes one q-seg and one z_L for each batch row; treating each phrase as a flattened row is a shape/generalized batching change. Every existing R1 utility/rectifier tensor has the same shape and semantics, so existing R1 weights are a **complete initialization** for these modules. New cache/parser metadata has no checkpoint weights.

The proposed loss normalization is an intentional training-semantics change relative to official GLaMM/LEGION global per-mask averaging. It is required by the user-specified equal-image weighting and must be isolated in the 2×2 ablation.

## Backward-compatible inference/evaluation

- retain each `mask_i` and its generated phrase/SEG ordinal;
- form `M_union = OR_i(mask_logit_i > 0)` only for existing image-level evaluation;
- preserve original resolution, threshold `>0`, full-N accounting and failure=0;
- `[REAL]` or Fake generation with no valid SEG yields no masks and the existing full-N failure policy;
- single-SEG checkpoints remain valid because K=1 is a special case of the grouped representation.

## Future 2×2 architecture ablation

| Arm | Initialization | SEG supervision | R1 image/utility path |
|---|---|---|---|
| L0 | frozen P1 | current single union | no |
| L1 | frozen R1/P1 | current single union | yes |
| L2 | P1 | native per-phrase multi-SEG | no |
| L3 | R1/P1 | native per-phrase multi-SEG | yes, shared weights |

For each frozen TRAIN/validation protocol and metric:

```text
R1 effect at single     = L1 - L0
MultiSEG effect on P1   = L2 - L0
MultiSEG effect on R1   = L3 - L1
interaction             = (L3 - L2) - (L1 - L0)
```

L2/L3 must share data order, target selection, optimizer exposure, prompt grammar and selector. Image-level union metrics provide backward comparison; phrase-level metrics are additional and cannot replace them.

## Recommendation

Proceed to a separately authorized implementation/preflight for L2/L3: **GO**. Before training, require synthetic K={1,2,variable} unit tests, exact phrase-mask count/order assertions, K=1 backward parity, R1 state-dict strict load, and one TRAIN-only batch forward/backward. This document does not authorize those actions.
