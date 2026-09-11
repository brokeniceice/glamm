# Phase 6D.3 — C1 RINE-Conditioned P1

Status: **RUNNING** on physical GPU 1 under `phase6d3-c1.service`.

- Initialization: `checkpoints/GLaMM-FullScope`, seed 3407; no trained-P1 checkpoint is loaded.
- RINE: Phase6B.6 selected checkpoint, SHA256 `5286b05c82416e3a11d067b1f449b566c39360ffe7133499d09aecd0fcb3b562`.
- H2: `4096 -> 512 -> ReLU -> 2`, newly initialized and jointly trained.
- Projector: `1024 -> 4096 -> GELU -> 4096`; one direct `[FRC]` embedding after visual tokens.
- Selector: minimum internal-validation total loss, unchanged from P1.
- C0 contract was frozen in `configs/phase6d3_c1_rine_conditioned_p1.yaml` before C1 results.

The preflight audit and training trajectory are stored under `outputs/phase6d3_c1/training/`.
This report is replaced by the frozen post-training report only after internal validation/test and DEV-OOD-2560 evaluation finish. No external localization benchmark is scheduled.
