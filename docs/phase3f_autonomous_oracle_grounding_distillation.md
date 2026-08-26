# Phase 3F — Autonomous–Oracle Grounding Distillation

## 结论

Phase 3F 完成。最终 gate：`GATE_AUTONOMOUS_ORACLE_DISTILLATION_NOT_LEARNABLE`。AOGD 未形成可部署的显著 canonical G0 增益，因此第一创新冻结为 **Stage-I Phrase-Aligned Forensic SFT only**；停止继续搜索 post-training 方法。FEPN 是下一条主要研究路线，但本阶段不自动启动。

## Validation 主结果

| 模型 | G0 IoU/F1 | Phrase-Only IoU/F1 | TF-Full IoU/F1 |
|---|---:|---:|---:|
| P1 | 0.148233 / 0.210697 | 0.245798 / 0.343492 | 0.342928 / 0.452020 |
| AOGD step1000 | 0.152015 / 0.213847 | 0.245402 / 0.342996 | 0.341346 / 0.449995 |
| matched P3F-SFT | N/A | N/A | N/A |

Selector 的算术最优为 step 1000。G0 IoU 相对 P1 的 paired difference 为 +0.003782，95% CI [-0.002090, +0.009897]，不支持正增益。matched 4500-step SFT 按用户冻结的条件协议未触发；Phase 3E SFT 仅为历史参考，不能称为 matched control。

## Representation 与训练通路

32-Fake preflight 证明 256D autonomous–oracle gap 存在，且 `L_repr` 对 LoRA 的梯度非零。正式训练仅 LoRA 改变；`text_hidden_fcs`、mask decoder 和其他参数 byte-hash 不变。validation matched eligible n=1072；256D gap AOGD−P1=-0.002053，95% CI [-0.007859, +0.003687]。

逐图 `Δgap_i` 与 `ΔIoU_i` 的相关性仅作预注册外诊断，不参与 selector/gate。256D projected：Pearson r=-0.3477，Spearman ρ=-0.3025；4096D raw：Pearson r=-0.4543，Spearman ρ=-0.3338。其中 `Δgap<0` 表示对齐改善，`ΔIoU>0` 表示定位改善，因此若 representation alignment 与定位收益方向一致，预期为负相关。

## Rollout matched control

Phase 3F 复用了 Phase 3B 的同一 canonical batch=1 P1 autonomous cache，SHA256 `6ab4c428748e426132edaf6570ee38f12bbc8c1a2e0d44b089152563fd0990ed`；step 0/500/1000/4500 hash invariant PASS，无 refresh、筛选或混用。因而 Phase 3B 与 Phase 3F 的差异是固定 trajectory 上的 supervision target，而不是 generated trajectory exposure。

## 边界

所有 Detection/G0 使用 canonical prompt 与 direct batch=1；TF-full 使用 canonical user prompt；mask threshold 固定为 logit 0。internal test 与 official1000 未使用。不得据此声称所有 representation distillation 无效；结论仅限本次 frozen P1 trajectory、cosine 256D target 和 LoRA-only 实现。
