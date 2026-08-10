# Phase 1C：Autoregressive `[SEG]` Activation Diagnosis & Minimal Repair

## 结论

Phase 1B 的 GT-Fake trigger `0.34375` 主要不是普通 text CE 对 `[SEG]` 监督不足，也不是 mask decoder ceiling。根因优先级为：

1. **partial assistant prefix 实现 bug**：旧 G2 的 `[CLS] [FAKE]` 被 conversation template 当作完整 assistant turn，在 `[FAKE]` 后追加 terminal EOS；修正为真正 continuation 后，同一个旧 prompt 的 trigger 从 `0.34375` 恢复到 `0.8125`。
2. **user-prompt distribution shift**：将修正后的旧 synthetic-artifact/known-Fake-style prompt 换回训练时 Unified prompt 后，G1 trigger 从 `0.8125` 进一步恢复到 `0.96875`。
3. **剩余 autoregressive drift**：Unified G0/G1 各只剩 1 个 repetition-loop failure；旧 prompt 下 explanation exact match 和 prefix overlap 明显下降。

因此诊断不满足实现 `L_seg_token` 的许可门槛：G0/G1 已接近完整触发且 generated/prefilled `[FAKE]` continuation 数值完全对齐。本阶段没有实现或运行 lambda 0/5 SEG-loss ablation，也没有改变训练 grammar、loss 权重或正式 optimizer policy。Phase 1A 默认 GT-conditioned localization 已最小修复为“训练时 Unified prompt + structural GT `[FAKE]` continuation prefix”；corrected old-prompt 和 terminal-EOS legacy 路径均保留。

## 1. Phase 1B 问题摘要

Phase 1B checkpoint 在 64-sample subset 上已达到 classification/LM verdict/agreement 1.0，TF-FullContext Mean IoU 0.4210，但原 GT-Fake Generate Mean IoU 0.04398、trigger 0.34375；Joint Mean IoU 0.24532、trigger 0.75、classification pass 1.0。故 Phase 1C 冻结 `outputs/phase1b_overfit/final_checkpoint.pt`，先隔离 prompt、prefix、autoregressive drift、stop reason 与 mask ceiling。

## 2. G0/G1/G2/G3 严格定义

所有第一轮结果使用同一 Phase 1B 320-step checkpoint、同一 32 Fake、seed 3407、greedy/1 beam、400 `max_new_tokens`、同一 preprocess 和 mask metric。

| Mode | User prompt | Assistant input | 后续 | Classification gate |
|---|---|---|---|---|
| G0 Unified Free | 训练时 Unified prompt | `[CLS]` | 自由生成 verdict/explanation/`[SEG]` | 否 |
| G1 Unified + GT Fake | 训练时 Unified prompt | `[CLS] [FAKE]` continuation | 自由生成 explanation/`[SEG]` | 否 |
| G2 corrected old prompt | 原 synthetic-artifact prompt | `[CLS] [FAKE]` continuation | 自由生成 explanation/`[SEG]` | 否 |
| G3 TF Full | 原 G3 prompt | `[CLS] [FAKE] <GT explanation> [SEG]` | causal forward，不 generate | 否 |

另保留两条诊断：Joint 是原 prompt + `[CLS]` free generation + classification gate；legacy G2 精确复现 `[FAKE]` 后带 terminal EOS 的 Phase 1B 路径。

## 3. 四模式结果

| Mode | Mean IoU | Global IoU | Mean F1 | Global F1 | Trigger | Mean/Median gen tokens |
|---|---:|---:|---:|---:|---:|---:|
| G0 | 0.39888 | 0.51442 | 0.52082 | 0.67936 | 0.96875 | 132.31 / 106.5 |
| G1 | 0.39574 | 0.51249 | 0.51742 | 0.67768 | 0.96875 | 132.91 / 105.5 |
| G2 corrected | 0.27281 | 0.29517 | 0.36149 | 0.45581 | 0.81250 | 162.72 / 110.0 |
| G3 TF Full | 0.42102 | 0.51028 | 0.54783 | 0.67574 | 1.0（GT `[SEG]`） | N/A |
| Joint | 0.24532 | 0.28699 | 0.32320 | 0.44598 | 0.75000 | 189.41 / 101.5 |
| legacy G2 | 0.04398 | 0.10106 | 0.07002 | 0.18356 | 0.34375 | 204.78 / 122.5 |

G0 与 G3 Mean IoU 只差 0.02214；训练分布内 free generation 已接近 teacher-forced upper bound。原先 0.377 的巨大差距主要由 evaluator prefix/prompt 路径造成。

## 4. Joint、G0、G1、G2 关系与 decision tree

- Joint -> G0：trigger `0.75 -> 0.96875`，IoU `0.2453 -> 0.3989`。Joint classification pass 已是 1.0，所以提升来自 Unified prompt 和更稳定的 explanation trajectory，不是去掉 classification gate。
- G0 vs G1：trigger 完全相同，IoU 仅差 0.00314。正确 prefill `[FAKE]` 不破坏 trajectory。
- G1 vs corrected G2：trigger 相差 0.15625，IoU 相差 0.12292，确认旧 user prompt 存在实质 distribution shift。
- corrected G2 vs legacy G2：只去掉 prefix 后 terminal EOS，trigger 增加 0.46875、IoU 增加 0.22883，证明 Phase 1B 低 trigger 的最大单一原因是 implementation bug。

结果整体落在 Case A，但在 legacy baseline 中还叠加了 terminal-EOS prefix bug。最小修复顺序因此是先修 prefix，再将默认 GT-conditioned protocol 改为 G1；不新增训练任务。

## 5. Generated `[FAKE]` vs prefilled `[FAKE]` alignment

按 human/animal/object/scene 各 2 个样本，共 8 个样本比较：

- 8/8 的 G0 first-next token 都是 token ID 32009 (`[FAKE]`)；
- Path A（模型生成 `[FAKE]`）与 Path B（正确 prefill `[FAKE]`）raw token IDs 8/8 完全相同；
- raw sequence length、position ID、attention-mask length、multimodal-expanded length一致；
- `[FAKE]` 后 predictor hidden 最大绝对差 0.0；
- first-next-token vocabulary logits 最大绝对差 0.0；
- prefilled prompt 8/8 均不以 EOS 结束。

结论：修复后的 generated/prefilled continuation 完全对齐，没有 KV cache、position、attention mask 或 multimodal expansion 偏移。逐样本 hidden 前 16 维、top-10 logits、长度与 position metadata 见 `baseline_diagnosis/prompt_prefix_alignment.json`。

## 6. Missing `[SEG]` failure taxonomy

| Mode | Triggered | Repetition | Invalid special | EOS before SEG | Max-token | Other |
|---|---:|---:|---:|---:|---:|---:|
| G0 | 31 | 1 | 0 | 0 | 0 | 0 |
| G1 | 31 | 1 | 0 | 0 | 0 | 0 |
| G2 corrected | 26 | 6 | 0 | 0 | 0 | 0 |
| Joint | 24 | 8 | 0 | 0 | 0 | 0 |
| legacy G2 | 11 | 10 | 8 | 3 | 0 | 0 |

分类为互斥主因；达到 400-token 上限且检测到 loop 的样本优先记为 `REPETITION_LOOP`。当前正确 G0/G1 最大失败原因均是单个 repetition loop；旧实现的最大类别也是 repetition，其次是错误 special-token trajectory，再次是 EOS-before-SEG。

每个失败样本均记录 sample ID、完整 text/token IDs、generated length、stop reason、special-token flags、EOS/SEG position、max tokens、repetition statistics；逐 mode JSONL/summary 已保存。

## 7. `[SEG]` probability/rank 与 EOS 竞争

| Mode | Trigger success max PSEG mean | Failure max PSEG mean | EOS-before rate |
|---|---:|---:|---:|
| G0 | 0.89278 | 0.01406 | 0 |
| G1 | 0.90242 | 0.01591 | 0 |
| G2 corrected | 0.89193 | 0.24419 | 0 |
| Joint | 0.89318 | 0.19097 | 0 |
| legacy G2 | 0.79523 | 0.02911 | 0.09375 |

G0/G1 唯一失败样本的 `[SEG]` 最好 rank 为 2，但概率只有约 0.014-0.016，属于 repetition trajectory 而不是 EOS 抢占。corrected G2 的 6 个 loop failure 中，3 个样本 max PSEG 约 0.448-0.497，说明局部 step 曾接近/达到 top rank，但 prompt-induced trajectory 最终进入 loop。

legacy G2 的 3 个 EOS failure 在 EOS step 的 `(PSEG, PEOS)` 分别约为 `(0.000002, 0.5574)`、`(0.00627, 0.7838)`、`(0.02247, 0.2532)`；EOS 明显占优。所有逐 step `p_seg/rank_seg/p_eos/rank_eos` trace 均在 `baseline_diagnosis/seg_probability_traces/`，没有保存完整 vocabulary logits。

## 8. Explanation drift 与 premature `[SEG]`

| Mode | Exact explanation match | Mean exact-prefix overlap / GT | `<50% GT length` |
|---|---:|---:|---:|
| G0 | 24/32 | 0.85325 | 2/32 |
| G1 | 25/32 | 0.85873 | 2/32 |
| G2 corrected | 10/32 | 0.43647 | 2/32 |
| Joint | 9/32 | 0.42816 | 1/32 |
| legacy G2 | 0/32 | 0 | 14/32 |

G0/G1 的 median explanation length ratio 为 1.0，绝大多数在已过拟合 subset 上精确复现 GT；G2/Joint exact match 明显降低且 failure 增加。因此 explanation drift 是旧 prompt 剩余 trigger gap 的主要伴随因素。G0/G1 各有 2 个 `<50% GT length` 样本，但当前没有额外 SEG up-weight，不能归因于 auxiliary-loss-induced premature SEG；总体 SEG position median 为 G0 100、G1 99、G2 88、legacy 47，legacy 明显更早且更异常。

forced-continuation diagnostic 只比较了实际轨迹中的 SEG/EOS logits，没有 force `[SEG]` 后把 mask计入正常 localization metric。

## 9. Implementation bug 与最小修复

旧 evaluator 用 `conv.get_prompt()` 构造非空 assistant `[FAKE]`，template 自动追加 `sep2=</s>`。这等价于告诉模型 assistant turn 已结束，再从 EOS 后继续 generation；G0/Joint 的空 assistant path 没有该 EOS，因而形成隐藏的不公平条件。

修复后 partial assistant prefix 会移除末尾 assistant terminator，保证 prompt 精确停在 `[CLS] [FAKE]`。`legacy_gt_fake_generate` 保留旧行为复现 Phase 1B 数字；corrected old-prompt G2 单独保留。Phase 1A 默认 `gt_fake_generate` 现使用训练时 Unified user prompt + GT `[FAKE]` continuation，GT explanation 从不进入 generation input。

诊断记录器另修复了二维/三维 hidden tensor 的兼容索引；这是 instrumentation bug，不影响模型预测或已写盘的 G0-G3 metric。

## 10. 是否需要 `L_seg_token` 与 lambda 0/5

不需要，且按阶段 gate **未实现、未训练**：

- G0/G1 trigger 均为 0.96875；
- G0 IoU 已接近 G3；
- prefix alignment 无实现差异；
- G0/G1 没有 EOS-before-SEG，普通 L_text 已能在训练分布内稳定激活 `[SEG]`；
- 主要剩余 failure 是 prompt-induced/repetition trajectory，不是普遍低 PSEG。

因此 lambda 0 vs 5 ablation 被标记为 `not_run_by_diagnostic_gate`，见 `seg_loss_ablation/lambda_0/status.json` 与 `lambda_5/status.json`。没有 `seg_token_loss`、premature-SEG 或 explanation-degradation实验数字可报告；强行运行会违反“一次只改一个有诊断支持的变量”和“只有 trigger 本身不足才允许加入”的约束。对应 L_seg-specific tests 也不适用；既有 predictor-position/causal tests 继续通过。

## 11. 8-Fake TF mask-only overfit

确定性选择 human/animal/object/scene 各 2 个 Fake，从 Phase 1B checkpoint 开始，只让 SAM mask decoder + `text_hidden_fcs` 接收 mask loss gradient；不改变 architecture，最大 1000 steps，每 100 steps评估，达到 Mean IoU 0.95 后停止。

| Step | TF Mean IoU | TF Mean F1 | Mean mask loss（前一窗口） |
|---:|---:|---:|---:|
| 0 | 0.49123 | 0.61155 | - |
| 100 | 0.87402 | 0.93065 | 0.11798 |
| 200 | 0.92229 | 0.95864 | 0.04258 |
| 300 | 0.92968 | 0.96194 | 0.03013 |
| 500 | 0.94187 | 0.96896 | 0.01850 |
| 700 | 0.95865 | 0.97837 | 0.01475 |

step 700 达到 early-stop threshold；Global IoU/F1 为 0.96475/0.98206。四类最终 Mean IoU：human 0.89766、animal 0.98747、object 0.97667、scene 0.97281。结论：mask branch 能真正记忆极小数据；Phase 1B TF 0.421 主要反映 64-sample/320-step 未充分 mask overfit，而非明显的 resize/SAM/text-projection ceiling。本阶段未改 SAM architecture。

## 12. Region encoder participation audit

真实 mixed Real/Fake forward/backward 上：

- Unified sample 的 `bboxes=None`；
- `region_encoder` forward hook count = 0；
- 299,142,912 个 region encoder 参数中，获得 gradient 的 tensor 数 = 0；
- gradient norm = 0；
- 它不参与当前 Unified Forensics loss graph。

建议正式 Unified training 默认 freeze region encoder，可减少约 299M trainable 参数。该建议尚未写入 full-training config，以免与本阶段 diagnosis 同时改变 optimizer variable，等待人工确认。

## 13. Embedding/lm_head tying 与 token-row策略

`embed_tokens.weight` 与 `lm_head.weight`：parameter identity=false、data pointer 不同、`config.tie_word_embeddings=false`，二者未 tied，各为 `[32010,4096]`。

mixed backward 的 embedding-row grad norm：`[CLS]=0.13624`、`[REAL]=1.79e-5`、`[FAKE]=0.10570`、已有 `[SEG]=4.92e-6`。若采用 row masking，必须保留这四行，不能把 `[SEG]` 当旧词错误冻结。

参数组建议：优先 Option A，将完整 embedding/lm_head 从 3e-4 下调到 3e-5（必要时 1e-5）并单独 preflight；它对旧 vocabulary explanation token 仍有可控更新。Option B 的 gradient-row mask helper和数学测试已实现，但不建议未经独立对照直接作为默认，因为只更新新增行会同时限制旧 vocabulary 的 explanation/output adaptation。Phase 1C 没有改变实际 optimizer group。

## 14. Real/Fake LM token supervision imbalance

full frozen train split 恰为 Real 8,836 + Fake 8,836，但当前 token-mean L_text denominator 为：

- Real supervised tokens：106,032，均值 12/sample，占 8.2155%；
- Fake supervised tokens：1,184,601，均值 134.065/sample，占 91.7845%；
- Fake/Real token contribution ratio：11.1721:1。

所以 sample 1:1 远不等于 text supervision 1:1。该现象没有影响 fixed `[CLS]` 的 causal leakage结论，但会影响 LM objective weighting。

`per_sample_text_loss_normalization` 已实现并有精确数学 unit test：先对每样本有效 target token 求 mean CE，再对有监督样本求 mean；默认值保持 false，未用于任何 Phase 1C 数字，等待正式 ablation 决策。

## 15. Checkpoint selection

本阶段继续只保留 last/诊断 checkpoint，没有设计 composite score，没有使用 LOKI，也没有用旧 OOD/terminal-EOS GT-Fake metric选择 checkpoint。正式 baseline 可保留 best validation total loss，但 generation protocol 已修复后仍需人工确认最终 selection policy。

## 16. 输出 artifacts

`outputs/phase1c_seg_activation/` 包含：

- `baseline_diagnosis/G0...G3`、Joint、corrected/legacy G2 predictions 和 metrics；
- 每个 autoregressive mode 的 failure JSONL/summary；
- 每样本 SEG/EOS probability traces；
- `prompt_prefix_alignment.json`；
- gated SEG-loss status；
- 8-Fake subset、每 100-step TF evaluation、metrics、summary 和 final diagnostic checkpoint；
- region/tying/token-row parameter audit；
- full-train Real/Fake text-token audit；
- config、git commit、console logs 与完整 unittest log。

主要新增/修改文件：`configs/phase1c_seg_activation.yaml`、`scripts/phase1c_seg_activation.py`、`eval/phase1c_seg_diagnosis.py`、`eval/forensics.py`、`eval/forensics_eval.py`、`model/GLaMM.py`、`train.py`、`scripts/phase1b_preflight.py`、两份 forensic tests、Phase 1A 文档和本文档。

## 17. Tests

执行：

```bash
/home/yz/miniconda3/envs/glamm_official/bin/python -m unittest discover -v tests
```

结果：**Ran 63 tests，OK**。新增覆盖：generated/prefilled prefix终止语义、G0/G1 无 classification gate、legacy G2复现、SEG probability/rank、EOS-vs-SEG、failure taxonomy、region participation helper、weight tying、old-vocab gradient masking、per-sample text normalization。Phase 1A/1B 的 Detection、五种 localization protocol、causal SEG predictor、mixed loss、token/leakage/save-load 等全部回归通过。

L_seg-token 的 Fake-only/predictor/lambda-zero tests 未添加，因为 gate 判定明确禁止进入该实现；这不是测试失败或遗漏运行，而是 diagnosis-first 边界的执行结果。

## 18. 必答诊断问题

1. **低 trigger 是否主要来自 Known-Fake prompt OOD？** 不完全是。最大原因是 terminal-EOS prefix implementation bug（+0.46875 trigger）；prompt shift 是次要但显著原因（再 +0.15625）。
2. **Unified prompt + GT `[FAKE]` 后是否恢复？** 是，trigger 0.96875、Mean IoU 0.39574。
3. **prefilled/generated `[FAKE]` 是否对齐？** 是，8/8 IDs相同，hidden/logits max diff均为0。
4. **missing `[SEG]` 最大原因？** 正确 G0/G1 是单个 repetition loop；corrected G2 6 个全是 repetition；legacy 依次是 repetition、invalid special trajectory、EOS。
5. **失败样本 PSEG 是否接近 EOS？** G0/G1 failure 没有 EOS且 max PSEG很低；legacy 3 个 EOS failure 中 EOS 均明显大于 PSEG；corrected G2部分 loop step PSEG曾接近0.5/top rank，但未形成稳定终止。
6. **explanation drift 是否主要原因？** 对 old prompt 剩余 gap 是；G0/G1 exact match 24/25，G2仅10，legacy为0。
7. **普通 L_text 对 `[SEG]` 是否不足？** 对训练分布内路径没有证据支持；G0/G1 trigger 96.875%。
8. **L_seg_token 是否显著提高 trigger？** 未运行；诊断 gate 不支持加入该变量。
9. **是否损害 explanation/导致 premature SEG？** 不适用；未实现 L_seg。G0/G1 无 auxiliary loss时只有2/32明显短 explanation。
10. **8-Fake mask branch 能否继续 overfit？** 能，TF Mean IoU 0.4912 -> 0.95865。
11. **region encoder 是否需要训练？** 当前不需要；hook=0、grad=0，建议 freeze。
12. **embedding/lm_head 是否 tied？** 否。
13. **降低 LR 还是只训练新 rows？** 建议先独立验证较低全矩阵 LR（3e-5/1e-5）；row-only可行但不作为默认，并必须保留 `[SEG]`。
14. **text-token imbalance 多严重？** Fake占91.78% token denominator，是 Real 的11.17倍。
15. **是否具备启动 full Unified baseline 条件？** 核心 generation protocol、prefix实现和mask capacity已通过；但仍需人工确认 region freeze、embedding/lm_head LR 与 checkpoint-selection policy。Phase 1C 不授权自动启动 full training。

## 19. 停止点

Phase 1C 到此停止。未启动正式长训练，未接入 NPR/SRM/fusion/consistency/artifact head/evidence query/Known-Fake auxiliary/prompt-diversity training，未修改 frozen split 或 Fake grammar顺序，未使用 LOKI。等待人工检查。

## 20. Phase 1D 最终 protocol 状态（不改写 Phase 1C 实验数字）

Phase 1D 已落实 Phase 1C 的 region encoder 结论：当前 Unified graph 中该模块固定冻结且不进入 optimizer。正式 G1 仍为 canonical Unified prompt + `[CLS] [FAKE]` structural continuation；正式 Joint 改为与 G0 完全相同的 canonical prompt + `[CLS]` 自由生成，仅在 scoring 时增加 classification gate。旧 prompt 的 legacy/corrected G2 路径继续可复现，但不作为默认 protocol。

在 classification gate 全通过的 32 Fake controlled smoke 中，G0/Joint 的 prompt hash、raw prompt、generated token IDs、SEG position、mask intersection/union 与 aggregate 全部相等。该状态是 Phase 1D 的 protocol finalization，不追溯修改本文件的 Phase 1C 历史数字。
