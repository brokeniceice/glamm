# Primary freeze proposal

## Proposed freeze

`PRIMARY_CANDIDATE = CSCU-LF`：Cross-Source Conditional Utility Late Fusion。

Frozen components：P1 LLM/LoRA/text_hidden_fcs、SAM image/prompt encoder、original SAM mask decoder、CLIP backbone、Phase4C-A forensic adapter+dense head、canonical G0 policy、original-normalized geometry、outside-support vacuous rule、exact P1 fallback。

Future trainable candidates（尚未授权）：

- L/F context projections；
- CMX-style joint channel/spatial interaction in旁路；
- selective local context exchange；
- explicit `U_F` head；
- optional, separately switchable cross-source refinement heads。

Official ECoLaF kernel remains frozen/parity-tested. `d_F^adapt=d_F^ECoLaF*U_F` 必须标记为 adapted extension。absence/vacuous/off 不进入该路径，bit-exact dispatch `z_L`。

## Freeze 尚未完成的量

本阶段故意不冻结最终 utility target/loss、`tau`、projection width、window/head count、refinement inclusion、loss weights、optimizer、epoch 或 selector。`07_conditional_utility_target_study.md` 已说明，凭 intuition 选择这些量会把 design study 伪装成 protocol。

## Rejected designs

- original intrinsic-uncertainty PCERF 作为 Primary；
- `U_F=sigmoid(MLP(F))`；
- hard winner-take-all router；
- standalone cross-attention 被命名为 utility-aware；
- 修改 ECoLaF discount 后仍称 native ECoLaF；
- Phase4F fixed gamma、epoch9 warm-start；
- Phase4E new decoder；
- TF/GT/condition identity gate；
- 未经 dominance evidence 自动加入 OGM-GE；
- over-composed CMX+TokenFusion+MAG+MoE。

`SECONDARY_CANDIDATE=AHBFR`，保持设计储备，不实施、不训练。
