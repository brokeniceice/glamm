# Phase 4D-1 — Minimal Position-Aware Evidence Utilization Test

## Status

`DESIGN_ONLY — NOT AUTHORIZED FOR EXECUTION`

This proposal is the smallest discriminative experiment recommended by Phase 4D-0. It does not authorize training, evaluation, checkpoint creation, cache regeneration, or use of sealed populations.

## Primary hypothesis

`H1`: Phase 4C-A's forensic representation contains transferable spatial information, but the Phase 4C-B Reader cannot use it because its evidence-token interaction is permutation-invariant and has no 2D positional encoding.

## Competing hypothesis

`H2`: The Phase 4C-A probe gain is decodable by a standalone dense head but does not transfer meaningfully into the frozen P1/SAM grounding path even when spatial position is exposed.

## Why this experiment has high information gain

Observed: Phase 4C-C gives `matched≈cross-image` and `matched≈spatial-shuffle`, while `matched>zero`.

Therefore: the current Reader consumes a nonzero visual feature distribution but does not demonstrate correct-image or spatial-layout dependence.

Required falsification: after the sole interface change, matched correct-image evidence must outperform both wrong-image evidence and content-shuffled evidence. A raw G0 increase without those causal sensitivities is not evidence retrieval.

## Architecture delta

Keep the selected Phase 4C-B single-layer Reader and additive zero-initialized `beta`. Add one fixed, non-trainable 2D sine/cosine positional code for the `24×24` lattice to the 576 evidence tokens before key/value projection. No deeper Reader, extra layer, new decoder, new backbone, or token-count sweep.

Two matched feature arms are required:

1. `POS-CLIP`: Phase 4C-A projection feature `F0`, shape `256×24×24`.
2. `POS-FORENSIC`: Phase 4C-A adapted feature `F_forensic`, shape `256×24×24`.

The feature-source comparison is necessary to separate general spatial re-access from forensic specialization. P1 remains the frozen no-Reader reference.

## Frozen modules and artifacts

- P1 checkpoint, LLM/LoRA, CLIP tower/projector, embeddings, LM head, classification head.
- Phase 4C-A projection and forensic adapter checkpoints.
- `text_hidden_fcs`, SAM image encoder, prompt encoder, mask decoder, postprocess.
- Canonical prompt, P1 rollout/query cache, train/validation manifests, target construction, image transforms, threshold 0, metrics, bootstrap seed.
- Internal test and official1000 remain sealed.

## Trainable modules

- One Phase 4C-B-sized Reader per arm.
- The existing scalar `beta` per arm.
- The 2D positional code is fixed and non-trainable.

No P1, adapter, CLIP, SAM, projection-head, language, or mask-decoder parameter is trainable.

## Training population and budget

- Deterministic 2,048-sample subset of the existing eligible train Fake population, selected before model execution from the frozen manifest with seed 3407 and mask-area quartile balance.
- Effective batch 4; exactly 512 optimizer steps, one exposure per selected image.
- Same frozen `2×BCE + 0.5×Dice` objective and optimizer hyperparameters as Phase 4C-B unless an implementation-only incompatibility is discovered before execution.
- No validation-based checkpoint selector. Step 0 is an invariance check; step 512 is the sole formal trained endpoint. Step 256 may be retained only for safety diagnostics and cannot replace step 512 post hoc.
- No threshold sweep, LR sweep, layer/head sweep, positional-code sweep, or extra seed.

## Evaluation population and primary endpoint

Use the full frozen internal-validation Fake population under canonical G0 queries. For each arm evaluate:

- matched correct-image evidence;
- the Phase 4C-C frozen cross-image derangement;
- spatial-content shuffle with the 2D position lattice held fixed;
- zero evidence.

Primary endpoint:

```text
POS-FORENSIC matched minus cross-image
paired per-image FG IoU difference, 95% bootstrap CI
```

This directly tests image-specific utilization. The most important negative control is spatial-content shuffle with positions held fixed; it tests whether the learned interface uses content at the correct 2D coordinates rather than merely detecting the presence/distribution of tokens.

Secondary, pre-registered endpoints are matched−shuffle, POS-FORENSIC−POS-CLIP under matched evidence, and matched POS-FORENSIC−P1 G0 IoU. They cannot replace the primary endpoint.

## Expected diagnostic patterns

| Pattern | Interpretation | Decision |
|---|---|---|
| matched>cross and matched>shuffle; forensic>CLIP | Supports H1 and forensic-specific spatial transfer | Eligible for a separately authorized matched full experiment |
| matched>cross/shuffle, but forensic≈CLIP | Spatial interface works, forensic specialization not transferred | Do not claim forensic Method 2; reassess feature source |
| matched≈cross or matched≈shuffle | Position code did not establish causal evidence use | Support H2/current route failure; stop Reader expansion |
| matched gain but zero/cross/shuffle also gain equally | Generic query compensation persists | Stop; do not promote on raw IoU |
| correct-image use appears but matched G0 degrades materially | Mechanism exists but is not useful under the current decoder | Stop method scaling; retain as analysis only |

## Stop rule

Stop the route after the fixed endpoint if either:

1. the primary matched−cross 95% CI lower bound is `≤0`; or
2. matched−shuffle 95% CI lower bound is `≤0`; or
3. matched POS-FORENSIC degrades P1 mean FG IoU by more than `0.005`.

Do not respond by adding layers, unfreezing P1/adapter/SAM, increasing steps, changing the loss, tuning the threshold, or selecting a favorable intermediate checkpoint.

## Go rule

A proposal for a full matched experiment is permitted only if:

- primary matched−cross lower CI `>0`;
- matched−shuffle lower CI `>0`;
- matched POS-FORENSIC does not violate the `−0.005` non-regression bound; and
- POS-FORENSIC−POS-CLIP is directionally positive without a severe classification/language regression.

Passing this rule authorizes only a new proposal, not automatic full training or held-out evaluation.

## Estimated compute

Training uses 512 steps per arm versus 21,730 steps per arm in Phase 4C-B, about `2.36%` of its training steps/exposures. With two arms and four frozen-feature intervention evaluations, expected total compute is approximately `0.05–0.15×` Phase 4C-B, depending on evaluation/cache I/O. No language regeneration should be necessary if the frozen canonical P1 query artifacts pass exact provenance checks.

## Target semantics boundary

The dense supervision is the official-annotation-derived union of visible/explainable artifact regions. It is neither a weak pseudo-mask nor exhaustive pixel-perfect forgery ground truth. Success would mean better grounding to annotated forensic evidence, not complete manipulation segmentation.
