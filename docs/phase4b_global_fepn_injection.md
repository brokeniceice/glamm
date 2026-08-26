# Phase 4B-G — Global Forensic Evidence Injection into P1

## Outcome

Final gate: `GATE_GLOBAL_FEPN_NOT_USEFUL_TO_P1`.

Phase 4A 成功学习了强 global authenticity representation，但 dense evidence 显著弱于 matched CLIP，因此本阶段只授权 global interface，不把 dense FEPN 接入 SAM/grounding path。

## Frozen interface

FEPN 使用 epoch 6，特征是 `dense_features.mean((2,3))` 的 128D classification-MLP input；它不是 scalar logit 或 P(fake)，GAP 后没有额外 activation/normalization。固定 projector 为 `128→256→4×4096`，共 4,243,712 参数。四个 continuous tokens 位于 576 个原视觉 embedding 后、canonical text 前，真实进入 autonomous generation 初始上下文与 KV cache；tokenizer、raw token IDs、embedding table 和 LM head 均未修改。

## Selected checkpoints

- PROJ-ONLY: step 1000.
- PROJ-LORA: step 500.
- P1 remains the original step 3500 / epoch 7 baseline.

## Validation results

| arm | Detection Acc | Detection F1 | G0 mean FG IoU | G0 mean FG F1 | phrase semantic | structure | SEG-valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0.985986 | 0.985979 | 0.148233 | 0.210697 | 0.638102 | 0.987342 | 0.974684 |
| PROJ-ONLY | 0.988698 | 0.988652 | 0.123058 | 0.181936 | 0.641603 | 0.983725 | 0.972875 |
| PROJ-LORA | 0.985081 | 0.984952 | 0.120279 | 0.178377 | 0.639250 | 0.988246 | 0.974684 |

PROJ-LORA vs P1 paired G0 IoU delta is -0.027954, 95% CI [-0.039136, -0.016378]. Phrase-Only and TF-Full are recorded in `phrase_only_metrics.json` and `tf_full_metrics.json`; they are diagnostics rather than selector inputs.

## Attribution and boundaries

PROJ-ONLY answers whether original P1 can consume a mapped representation without language-path adaptation; PROJ-LORA tests whether LoRA is needed. Classification, phrase, structure and G0 are reported separately so a binary authenticity shortcut is not mislabeled as rich forensic reasoning. Matched SFT trigger: False; completed: False. Evidence-specific attribution supported: False.

All selector and statistics use internal validation, canonical prompt, direct batch=1 and fixed mask-logit threshold 0. Internal test and official1000 remained sealed. No K/depth/LR sweep, FEPN fine-tuning, dense integration, SAM fusion or threshold tuning was performed.

## Route decision

The next FEPN decision must follow `GATE_GLOBAL_FEPN_NOT_USEFUL_TO_P1`. This phase does not support claims that dense FEPN is established, that the global feature precisely localizes artifacts, or that FEPN resolves the full localization bottleneck. Per the hard stop, no Phase 4C or further variant starts automatically.
