# Phase 2D.1：Grounding Gap 精确 Trace 与 Replay 诊断报告

## 1. 执行摘要

Phase 2D.1 已完成。它是一个**只读诊断阶段**：没有训练、没有改权重、没有重新选择 checkpoint、没有调阈值，也没有把 Phase 2D 的分析类别当作训练标签。实验在冻结的 Phase 2A primary checkpoint 上，对 SynthScars official test 的 64 个确定性选样建立 token、hidden、projection、SAM prompt 和 mask 的逐级 trace，并执行固定 token replay 与受控替换。

核心结论是：G0 与 TF 的最终定位都采用生成结束后的**无 cache 全序列 forward**来取得 `[SEG]` 前一位 predictor hidden，因此当前证据不支持“G0 最终 mask 使用 cache hidden、TF 使用 full-forward hidden”这一实现分叉。把同一图像的 TF predictor hidden、投影或 SAM prompt 注入 G0 下游，均精确重建 TF mask；这确定了 TF 表征是下游差异的充分中介，但不能单独证明 predictor state 的构造存在 bug。短语上下文在 severe 子集平均恢复 53.05% 的 TF–G0 gap，支持后续优先处理自回归语言 grounding / exposure gap；不授权 NPR/SRM 等 forensic fusion。

## 2. 继承结论与本阶段目标

Phase 2B 已排除指标口径造成的“约 40 点 LEGION 差距”，Phase 2D 则确认内部 TF→G0 mean FG IoU gap 是真实的 generated-context grounding generalization 问题。Phase 2D.1 的任务不是再比较模型优劣，而是回答：推理 trace 是否改变结果、历史 token 能否精确回放、cache stepwise 与 canonical full-forward 是否一致、TF 优势在语言/hidden/projection/SAM 哪一级被传递，以及 Phase 2D 的 13 个 F 候选是否具有独特证据。

## 3. 冻结配置与数据

- checkpoint：`checkpoints/phase2a_unified_baseline/single/best/checkpoint/mp_rank_00_model_states.pt`
- SHA256：`07250fe4e82dee3b1a69c2c3b65311404757e6a7ca4e12adb17b4a845304c072`
- 训练状态：step 2500 / epoch 5，永久保持 Phase 2A primary baseline
- 数据：SynthScars official test；本阶段仅使用其中 64 个确定性诊断样本，不用于模型选择
- mask 阈值：固定 logit `> 0`
- historical G0 replay batch size：8，与 Phase 2B 正式评测一致
- 推理精度与模型配置：沿用冻结 baseline；未下载或引入新权重

## 4. 推理路径审计

canonical G0 先以 greedy generation（`num_beams=1`、`use_cache=true`）产生 token；生成完成后，`GLaMMForCausalLM.evaluate` 会对完整生成序列再执行一次无 cache forward。定位 predictor 是该 full-forward 中 `[SEG]` 之前位置的 4096 维 causal hidden，随后经过冻结的 `text_hidden_fcs[0]` 投影为 256 维 embedding，再进入 SAM prompt encoder、mask decoder、postprocess，最后按固定零阈值二值化。TF 也使用同一套 full-forward predictor 抽取与 SAM 下游。

因此 generation cache 影响 token 生成过程，但 canonical G0 的最终 mask predictor 并非直接取自 generation cache。精确路径见 `audit/inference_trace_path.json`。

## 5. Trace 实现与保存内容

新增 trace 层只注册/读取推理中间量，不改变 forward 输入或模型状态。每例保存或索引：prompt 与输出 token、token 文本、`[SEG]` 位置、predictor hidden、投影 embedding、SAM sparse/dense prompt、低分辨率及原分辨率 mask logits、二值 mask、GT 与 IoU，以及 checkpoint、数据、阈值、执行语义等 provenance。大 tensor 使用 `.pt`，结构化索引使用 JSON/JSONL。

## 6. Trace invariance

预先指定的 3 个样本全部 PASS。对 G0 与 TF，trace 开关前后 token/`[SEG]` 位置一致，mask-logit 最大绝对差均为 0，二值 mask 完全一致。一个样本的“当前原始生成 token”与历史 Phase 2B token 不同，但当前无 trace 与当前有 trace 完全一致；这说明 trace 没有改变输出，同时揭示 BF16 generation 可能存在跨运行变化，二者已分开记录。

## 7. 样本选择与资格门控

64 个样本由固定规则确定性选择：包含全部 13 个 Phase 2D F 候选、32 个 severe 成员、18 个 negative-gap 成员和 14 个 matched controls；成员关系允许重叠。最终 60 例状态为 `COMPLETE`；2 例因 historical full-forward IoU 与冻结记录差异超过预设 5e-4 容差而标记 `TRACE_FORWARD_REPRODUCIBILITY_MISMATCH`；2 例没有可用 G0 `[SEG]` predictor，标记 `G0_SEG_PREDICTOR_UNAVAILABLE`。这 4 例均在干预前排除，未被悄悄补值。

## 8. Canonical G0 / TF 与 Exact Replay

对历史 G0 token 的 Replay-FullForward 使用原始 padded token batch、batch size 8、无 cache，目的是复现 Phase 2B 的实际定位语义。62/64 例通过历史复现门槛；其中 2 例无 G0 predictor，故 60 例进入干预。Replay-Stepwise 则固定相同 token，先完整处理 prompt，之后逐 token 更新 KV cache，用于比较执行语义，而不是替代 canonical 路径。

在 16 个预选可用样本上，full-forward 对 stepwise 的平均 hidden cosine 为 0.999538、hidden relative L2 为 0.030261、projection cosine 为 0.999950、mask-logit cosine 为 0.999957；二值 mask IoU 平均 0.932666，0/16 完全相同。说明两种语义存在小而可见的数值/掩码差异；但因 canonical G0 与 TF 最终定位都使用 full-forward，该差异不是当前 TF–G0 gap 的直接来源。

## 9. 干预定义

- H1a（Oracle full context）：使用 TF 完整上下文及受控 `[SEG]` endpoint。它同时改变文本、长度、位置与 hidden，是多因素诊断，不是单因素因果实验。
- H1b（Position-matched oracle）：未执行，状态为 `H1B_NOT_IDENTIFIABLE`。padding/placeholder 会进入 causal attention 或改变语义，无法构造可信的“只改语义且严格同位置”条件。
- H2a：同一 sample、同一 image，仅把 TF predictor hidden 送入冻结 G0 下游。
- H2b：同一 sample、同一 image，仅替换为 TF projected embedding。
- H2c：同一 sample、同一 image，仅替换 TF sparse/dense SAM prompt，mask decoder 不变。
- H3（Phrase-only）：提供缺陷短语上下文；同时改变文本、长度、`[SEG]` 位置和 hidden trajectory，属于多因素诊断。
- H4：离线计算 G0/TF token 的精确最长公共前缀与最早分叉；未执行 continuation replacement，因为那会同时改变多个变量。

所有 H2 swap 均通过同图像、shape/dtype/device、冻结模块和零阈值完整性校验；60/60 eligible 干预有效。

## 10. 总体结果

| 模式 | n | mean FG IoU | 相对 G0 平均恢复 |
|---|---:|---:|---:|
| G0 | 60 | 0.329287 | — |
| TF | 60 | 0.657629 | 0.328342 |
| H1a | 60 | 0.657629 | 0.328342 |
| H2a hidden swap | 60 | 0.657629 | 0.328342 |
| H2b projection swap | 60 | 0.657629 | 0.328342 |
| H2c SAM prompt swap | 60 | 0.657629 | 0.328342 |
| H3 phrase-only | 60 | 0.424110 | 0.094823 |

H2a/b/c 在正 TF gap 的 41 例中均恢复 100% gap。三者相等是冻结 deterministic 下游中的必然中介链结果：TF hidden 经相同 projection 和 SAM 会产生 TF projection、TF prompt 与 TF mask。它证明注入的表征足以决定下游差异，不证明某一级模块自身损坏，也不代表可部署模型。H2 平均恢复的 95% bootstrap CI 为 [0.188554, 0.465991]；H3 为 [-0.018811, 0.214090]。

## 11. Severe 子集

32 个 selected severe 样本中 29 个 eligible。G0/TF mean IoU 分别为 0.035341/0.870627。H2a/b/c 均恢复至 0.870627，即平均恢复 0.835287。H3 达到 0.478846，平均恢复 0.443505，在 29 例正 gap 中平均恢复比例 53.05%，且 93.10% 样本得到正恢复。该结果支持短语级 grounding 信息对 severe failure 有实质诊断价值，但 H3 不是纯语义单因素干预。

## 12. Phase 2D 的 13 个 F 候选

13/13 均 eligible。G0/TF mean IoU 为 0.058601/0.851733；H2a 平均恢复 0.793132，H3 mean IoU 为 0.598074、平均恢复约 66.75% 的正 gap。由于 H2 在全部 eligible 样本中都按相同机制精确重建 TF，F 子集没有显示独有的 state-swap 异常；`F_PROXY_FALSE_POSITIVE_confirmed=0`，结论为 `STATE_MEDIATOR_SUPPORTED_BUT_ANOMALY_SPECIFICITY_UNRESOLVED`。不能把 F 类解释成已证实的 bug 类或训练标签。

## 13. Negative-gap 子集

18/18 eligible，G0/TF mean IoU 为 0.499116/0.147187；将 TF 中介注入自然会降至 TF 水平。H3 mean IoU 为 0.216987，平均优于 TF 但低于 G0。G0/TF 平均 token 长度为 182.72/220.78。探索性地看，完整 GT context 并不总是对分割最优，较长上下文也可能伴随定位退化；这不是长度或语义的因果证明。

## 14. 表征关联分析

在 60 例中，G0–TF hidden cosine 与 gap 的 Pearson/Spearman 为 -0.521/-0.561；hidden relative L2 与 gap 为 0.501/0.544。projection cosine 与 gap 为 -0.586/-0.611，projection relative L2 为 0.537/0.551。差异更大的表征通常伴随更大的定位 gap，但这些是观察性相关，不可解释为因果效应。

## 15. H4 最早分叉

每例的 token 最长公共前缀、最早不同 token 和后续 `[SEG]` 位置已写入 `replay/h4_prefix_divergence.jsonl`。H4 只定位语言轨迹从何处分叉；未把 TF continuation 拼入 G0，因为这会连带改变内容、长度、位置与后续 hidden，无法声称单因素因果。

## 16. 定性可视化

`qualitative/index.html` 覆盖全部 64 个 selected case，展示真实图像/GT 以及可用的 G0、TF、H2、H3 mask，并标注 subgroup、状态、token/`[SEG]` 与 IoU。缺失或被 gate 排除的结果显式显示状态，不使用占位预测伪装成有效输出。

## 17. 结论等级

### CERTAIN

1. Trace instrumentation 对抽查的 G0/TF 输出严格不变。
2. checkpoint 与运行时 projection、SAM prompt encoder、SAM mask decoder 在实验前后 hash 不变。
3. canonical G0 和 TF 最终 predictor 都来自无 cache full-sequence forward。
4. 同图像下，TF hidden/projection/prompt 经冻结下游足以精确重建 TF mask。
5. 本阶段没有训练、调阈值、checkpoint/test-set selection 或 proxy 标签构造。

### SUPPORTED BUT NON-CAUSAL

1. phrase-only 在 severe 与 F 子集有明显平均恢复，支持语言 grounding / exposure-gap 假设。
2. G0–TF hidden、projection 和 mask 表征距离与 IoU gap 中度相关。
3. cache-stepwise 与 full-forward 有小幅执行差异，但不是 canonical TF–G0 最终路径差异。

### UNRESOLVED

1. H1b 无法在不引入新混杂的前提下识别。
2. 无法将语言内容、序列长度、`[SEG]` 位置和 hidden trajectory 完全解耦。
3. F 候选是否构成独立异常机制仍不可识别。
4. 两个 historical replay mismatch 的底层数值来源未被唯一定位。
5. 本阶段不能推出任何新训练方法一定改善 official test。

## 18. Phase 3 Decision Gate

- `GATE_A_LANGUAGE_GROUNDING_SUPPORTED`
- `GATE_B_PREDICTOR_STATE_MEDIATOR_SUPPORTED`
- `GATE_F_FORENSIC_FUSION_NOT_AUTHORIZED`

Phase 3 训练方向获得**有限授权**：可以围绕 phrase-level/autoregressive grounding 与 train–inference exposure-gap mitigation 设计训练；不得把本结果扩张为 NPR/SRM forensic fusion 的授权，也不得把 H2 oracle swap 当成可部署方案。任何 Phase 3 仍须重新冻结数据、指标、选择器和消融边界。

## 19. 产物清单

- `selection/`：64 例确定性选择与分层摘要
- `traces/`：逐例 trace manifest 与 tensor artifacts
- `replay/`：full-forward/stepwise replay、invariance、H4
- `controlled_modes/`：干预规格、逐例结果与 integrity validation
- `representations/`：hidden/projection/SAM 表征指标
- `metrics/`：总体、分组、bootstrap、F 与 negative-gap 分析
- `qualitative/`：64 例真实 mask HTML
- `audit/`：路径、语义与 no-weight-mutation 证据
- `manifest.json`：顶层冻结配置、状态、gate 与纪律声明

## 20. 实验纪律与最终判断

本阶段正式报告位于 `docs/phase2d1_trace_replay.md`，其余新产物位于 `outputs/phase2d1_trace_replay/`；未修改历史 Phase 2A/2B/2D artifacts。checkpoint 及关键运行时模块 hash 前后一致。最稳妥的归因是：TF 与 G0 的差异主要由进入共同 full-forward 定位链的上游序列/上下文包造成，并通过 predictor representation 传递到冻结 SAM 下游；现有证据不足以宣称存在 SEG state 构造 bug，也不支持 forensic feature fusion。

## 21. 测试与回归

- Phase 2D.1 专项：`11 passed`，1 个依赖弃用 warning。
- 完整 `tests/`：`129 passed, 3 skipped`，5 个预期 warning。
- warning 均来自既有依赖弃用提示、既有 Fake segmentation 防御性提示及 Transformers generation 配置提示；没有测试失败。
