# Phase 6D.3 — C1 RINE-Conditioned P1

Status: **COMPLETE**. C1 was retrained from the same pre-P1 base initialization; no trained P1 checkpoint was used. RINE Q2 and its CLIP tower stayed frozen/eval. H2 and the prescribed two-layer GELU projector were trained jointly.

- Selected checkpoint: `/data/yz/myLISA_storage/checkpoints/phase6d3_c1/best/checkpoint/mp_rank_00_model_states.pt` (epoch 5, step 2500, SHA256 `85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff`)
- RINE checkpoint SHA256: `5286b05c82416e3a11d067b1f449b566c39360ffe7133499d09aecd0fcb3b562`
- Selector: original P1 minimum internal-validation total loss
- DEV-OOD manifest SHA256: `56c30931a4f1c67985e59a5b2f45b2fc5229cbae4eb96a256a36592554580348`

## Classification

| Split | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|
| internal validation | 0.994575 | 0.999786 | 0.994575 | 0.994575 | 0.005425 | 0.994575 |
| internal test | 0.995924 | 0.999551 | 0.996377 | 0.995471 | 0.004529 | 0.995926 |
| DEV-OOD-2560 | 0.702734 | 0.927041 | 0.410156 | 0.995313 | 0.004687 | 0.579790 |

## Localization and generation diagnostics

Full G0 generations, phrase parsing, TF predictions, spatial metrics, and failure records are under `outputs/phase6d3_c1/evaluation/val` and `test`. No external localization benchmark was run.

C0 remained frozen to the preregistered matched definition in the config. Causal attribution of RINE injection is reserved for future C1-vs-C0 comparison.
