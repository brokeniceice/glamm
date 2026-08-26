# Phase 4D-1R — Corrected Minimal Rerun Proposal

Status: **DESIGN ONLY — NOT AUTHORIZED**

Phase 4D-1B identifies zero/tiny gate gradient suppression. The only proposed intervention is:

```text
beta initialization: 0.0 -> 0.03
```

`0.03` is the smallest pre-registered inference scale with at least 80% POS-FORENSIC BF16 survival versus baseline q on the frozen audit population. It was not selected by validation IoU, matched-vs-shuffle performance, binary masks, or threshold tuning.

Everything else remains matched to Phase 4D-1: architecture, fixed 2D positional encoding, two feature arms, frozen 2,048-sample subset and order, 512 steps, optimizer, schedule, loss, seed, validation controls, threshold, and frozen P1/LLM/LoRA/CLIP/adapter/SAM modules.

No Phase 4D-1R training or evaluation is authorized or started by this proposal. Its purpose is a single-variable causal test of whether correcting the gate enables useful localization learning.
