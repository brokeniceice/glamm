# Phase 6A — Classification Attribution and Future Controls

## LEGION Stage-2 source audit

At official commit `d21535dd45f6fea509337a83095966f0b86ac924`, `external/LEGION_official/model/Legion.py:314-381` implements:

```text
CLIP ViT-L/14@336 hidden_states[-2] [B,577,1024]
→ token 0 CLS [B,1024]
→ Linear(1024,2048)
→ ReLU
→ Linear(2048,2)
→ Real/Fake
```

The head has exactly **2,103,298 parameters**. `scripts/cls/train.py:196-219` freezes every pretrained parameter and re-enables only `prediction_head`; CE is the only classifier loss. `LegionClsDataset` decodes BGR→RGB and uses the official `CLIPImageProcessor` for the same vision tower, producing `[3,336,336]`. The Stage-2 route never invokes the LLM, explanation, `[SEG]`, SAM, or Stage-1 mask decoder during classification.

## Feature-level comparison

| Property | P1/R1 current classifier | LEGION Stage-2 |
|---|---|---|
| Feature | final LLM hidden at fixed `[CLS]` | CLIP penultimate-layer CLS |
| Dimension | 4096 | 1024 |
| Head | Linear(4096,2), 8,194 params | MLP 1024→2048→2, 2,103,298 params |
| Prompt dependency | canonical prompt required | none |
| Inference backbone | CLIP + mm projector + 7B LLM | CLIP only |
| Backbone during classifier training | LoRA/embeddings jointly trainable in P1 | entirely frozen |
| Localization dependency | none | none |
| Classification-safe | yes | yes |

LEGION Stage-2 may generalize better OOD because it isolates a pretrained image-global representation, freezes the feature extractor, and gives the classifier nonlinear capacity without entangling the feature with generation grammar. This is a mechanistic hypothesis, not a conclusion from benchmark tuning: head capacity, training recipe/data ordering, label convention and feature choice are all confounded until a matched control is run.

## Classification-safe R1 representations

The machine-readable audit is `outputs/phase6a_architecture_audit/r1_feature_accessibility.csv`.

- **Safe:** raw CLIP grid, `F24 [B,256,24,24]`, `z_F24 [B,1,24,24]`, SAM `S64 [B,256,64,64]`, R1 rectified `S64_rect/residual [B,256,64,64]`, and the isolated `Fctx64` branch.
- **Unsafe for a primary classifier:** q-seg, preliminary `z_L`, joint CSCU language context, utility `U_F`, gated adapted embedding, final mask. These require the generated/teacher-forced phrase and `[SEG]` path; feeding them back creates a “classify after deciding/generating Fake” loop.

The strongest low-intrusion R1-specific safe candidates are pooled `F24` and pooled rectifier residual. The former avoids SAM compute; the latter uses the R1-selected rectifier weights but costs a SAM image-encoder pass.

## Proposed matched experiments

| Arm | Definition | New parameters | Frozen modules | Causal validity | Checkpoint compatibility | Added inference cost |
|---|---|---:|---|---|---|---|
| C0 | current P1/R1 `[CLS]` hidden → Linear | 0 (existing 8,194) | current frozen contract | valid | exact P1/R1 | none |
| C1 | LEGION-style frozen CLIP CLS 1024→2048→2 | 2,103,298 | freeze P1/R1 and CLIP | valid | additive head; P1/R1 untouched | lower than C0 if run standalone; no LLM needed |
| C2 | GAP(`F24`) 256→512→2 | 132,610 | freeze CLIP and Phase4C-A forensic arm | valid | additive head; exact reuse of frozen forensic source | CLIP already required; three local blocks + small MLP |
| C3 | current CLS 4096 ⊕ GAP(`F24`)256 → MLP(4352→256→2) | 1,114,882 | freeze P1/R1 and forensic source | valid | additive fusion; initialize without altering existing head | C0 plus forensic adapter |

Optional C2-R uses GAP of the R1 rectifier residual with the same 132,610-parameter MLP. It is causally valid but materially more expensive because it requires both CLIP-forensic and SAM image encoders; it should not be the first control.

## Priority recommendation

Run **C1 and C2 first**, under a future separately frozen TRAIN/validation-only protocol:

1. C1 directly tests whether LEGION's feature/head choice explains the classifier gap with minimal confounding.
2. C2 tests whether an already available image-only forensic representation adds value without the post-verdict R1 loop.

C3 is justified only if C2 independently carries classification signal; otherwise fusion adds capacity without an attribution basis. No hyperparameter is selected from existing final benchmark results.

## Decision

- classification feature reuse: **GO**, restricted to image-only `F24/z_F24` or rectifier residual;
- LEGION-style classifier control: **GO**;
- complete R1 utility/mask feature feedback: **NO-GO**;
- R1-safe-feature fusion: **CONDITIONAL GO after C2**, not first-line.

No implementation or training was performed.
