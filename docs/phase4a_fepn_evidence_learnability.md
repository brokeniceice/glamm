# Phase 4A — Standalone Task-Aligned Forensic Evidence Learnability

## Outcome

Selected FEPN-v0 checkpoint: epoch **6**. Terminal gate: `GATE_GLOBAL_ONLY_EVIDENCE_LEARNED`. Phase 4B was not started; internal test and official1000 remained sealed.

## Architecture and protocol

FEPN-v0 has 2,107,458 trainable parameters and receives matched normalized RGB plus a deterministic fixed Laplacian residual view at 336×336. Independent learnable stems, channel concatenation, a 1×1 projection and a lightweight convolutional FPN produce `F_dense` at 128×84×84. P1, LLM, SAM and the SEG predictor are absent.

Training used the frozen 17,672-image train population for 10 complete passes (176,720 exposures; 11,050 optimizer steps), with each full batch balanced between Real and Fake. Formal candidates were epoch 0–10; poor validation performance did not trigger early stopping. Real images received only global BCE; Fake images received global BCE plus `2×BCE+0.5×Dice` dense loss on the official-annotation-derived per-image all-reference union evidence target. Dense threshold remained zero.

## Results

| Model | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|
| Matched frozen CLIP probe | 0.143842 | 0.048468 | 0.209971 | 0.178934 | 0.303553 |
| FEPN-v0 epoch 6 | 0.068252 | 0.000000 | 0.108358 | 0.051901 | 0.098681 |

FEPN−CLIP paired FG IoU delta is `-0.075590`, 95% bootstrap CI `[-0.085346, -0.065996]`, win/tie/loss `226/271/609`. FG F1 delta is `-0.101613`, CI `[-0.114166, -0.089172]`.

Global validation classification: Accuracy `0.930832`, Precision `0.935160`, Recall `0.925859`, F1 `0.930486`, ROC-AUC `0.981889`. The preregistered safety condition was `PASS`.

`F_dense` non-collapse diagnostic: channel std mean `0.448304`, spatial variance mean `0.163497`, status `PASS`.

## Final questions and interpretation

1. Non-collapsed feature learning: `supported` by finite channel and spatial variance.
2. Dense localization is reported above under the same original-space target and zero-threshold evaluator as CLIP.
3. Significant superiority to CLIP: `no` under the preregistered paired-bootstrap rule.
4. Global authenticity information: `meaningfully above chance`.
5. RGB+Residual FEPN continuation: governed by `GATE_GLOBAL_ONLY_EVIDENCE_LEARNED`; this phase alone does not prove that residual input is causally necessary because RGB-only was not run.
6. Evidence supported: `global only`.
7. Phase 4A gate: `GATE_GLOBAL_ONLY_EVIDENCE_LEARNED`.
8. The FEPN living route is updated without rewriting its initial hypothesis.
9. Phase 4B remains not authorized and requires an explicit new instruction even if Phase 4A passed.

The result concerns visible/explainable forensic evidence localization, not all forged pixels. It does not establish end-to-end autonomous G0 improvement, because no MLLM integration occurred.
