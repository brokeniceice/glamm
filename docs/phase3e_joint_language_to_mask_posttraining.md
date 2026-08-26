# Phase 3E — Joint Language-to-Mask Grounding Post-training

## 最终状态与结论

本阶段由用户在 selector 完成后终止，随后仅授权完成三模型 canonical TF-PHRASE validation 诊断。最终 validation route gate 为：

`GATE_JOINT_LANGUAGE_MASK_POSTTRAINING_NOT_SUPPORTED_ON_VALIDATION`

JOINT 的十个非零 checkpoint 均未超过 P1，正式 selector 回退 P1 step 0。SFT-CONT 选择 step 900，出现很小的 G0 validation 增益，但 paired bootstrap CI 跨 0；其 canonical TF-PHRASE 则出现很小但 CI 不跨 0 的下降。当前联合 mask objective 因此没有显示 autonomous G0 或 teacher-forced spatial gain。

## 协议

- 数据：固定 validation；Detection 2212 张，G0/TF-PHRASE 为其中 1106 张 Fake。
- Detection、G0、TF-PHRASE User prompt 均显式 canonical；G0/Detection direct batch=1。
- mask threshold 固定为 0.0；未调 threshold。
- 未使用 internal test 或 official1000；本报告不支持 held-out 泛化结论。
- mask target 是官方 polygons 派生的 per-image all-ref union，不是 pseudo mask。

## Selector 结果

| Arm | checkpoint role | G0 mean IoU | G0 mean F1 | Cls Acc | Phrase semantic | Structure valid |
|---|---|---:|---:|---:|---:|---:|
| P1 | frozen step 0 | 0.148233 | 0.210697 | 0.985986 | 0.638102 | 0.987342 |
| SFT-CONT | selected step 900 | 0.150260 | 0.212555 | 0.986438 | 0.643496 | 0.993671 |
| JOINT | best trained step 300, diagnostic | 0.145738 | 0.207758 | 0.986438 | 0.638784 | 0.991863 |
| JOINT formal | fallback P1 step 0 | 0.148233 | 0.210697 | 0.985986 | 0.638102 | 0.987342 |

G0 IoU paired delta（new−base，95% bootstrap CI）：SFT−P1 `+0.002027 [-0.003147, +0.007188]`；trained JOINT−P1 `-0.002495 [-0.008604, +0.003501]`；trained JOINT−SFT `-0.004522 [-0.010536, +0.001552]`。SFT checkpoint 是在同一 validation 上按 G0 选择，故其 CI 是 selection-conditioned 描述，不能作为独立确认性显著性。

## Canonical TF-PHRASE

| Arm | checkpoint role | Mean IoU | Mean F1 | Global IoU | Global F1 |
|---|---|---:|---:|---:|---:|
| P1 | frozen baseline | 0.342928 | 0.452020 | 0.389014 | 0.560130 |
| SFT-CONT | selected step 900 | 0.342390 | 0.451305 | 0.388583 | 0.559683 |
| JOINT | trained step 300 diagnostic | 0.338669 | 0.448001 | 0.381126 | 0.551906 |

TF IoU paired delta：SFT−P1 `-0.000537 [-0.000991, -0.000115]`；trained JOINT−P1 `-0.004259 [-0.009643, +0.001300]`；trained JOINT−SFT `-0.003721 [-0.009166, +0.001653]`。

## 机制解释与边界

训练公平性审计 PASS；两臂各 1000 optimizer steps、4000 image exposures，样本顺序、标签、text-token 分母和 LoRA LR 完全一致。mask-only backward 对 LoRA、text_hidden_fcs、mask_decoder 的梯度均非零，因此不能把负结果解释成“mask loss 没有进入模型”。

允许的结论是：在 P1 初始化、当前 4000-exposure budget、既定 loss 和可训练模块下，联合 mask post-training 没有产生 validation G0 或 TF-PHRASE 增益。结果更符合当前优化目标/接口未能改善 autonomous language-to-mask grounding，而不是实现断路。

不能声称 mask decoder 是唯一瓶颈、所有 direct spatial supervision 都无效，或 SFT-CONT 已获得 held-out 泛化提升。若论文需要最终有效性结论，应冻结本报告中的 checkpoint 与协议后，再一次性解封 held-out test；本阶段没有执行该步骤。
