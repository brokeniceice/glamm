# Phase 2D：自回归 Grounding Gap 诊断

## 1. 执行摘要

Phase 2D 只诊断冻结 Phase 2A step-2500 checkpoint 在 teacher forcing（TF）与真实自回归 G0 推理之间的定位差异，没有训练、修改权重、重选 checkpoint 或使用 official test 做模型选择。

在 official SynthScars-1000 上，逐图 `TF foreground IoU - G0 foreground IoU` 的均值为 **0.221020**，中位数为 **0.159767**；79.3% 的样本差值为正，19.9% 为负。预注册 severe subset（TF IoU ≥ 0.70 且 G0 IoU ≤ 0.30）包含 **119/1000（11.9%）**，其平均 G0/TF IoU 为 0.067323/0.813124，平均差值为 0.745801。

现有产物支持“自回归路径没有稳定调用 TF 条件下已经存在的定位能力”，但不能把原因干净地分离为语言内容错误或 `[SEG]` 隐状态构造错误。历史 artifact 没有保存 TF token、预测 mask 或隐藏状态；当前接口也不能在只改变一个因素的前提下实现 H1/H2。因此根因状态是：**ROOT_CAUSE_PARTIALLY_UNRESOLVED**。

## 2. 冻结的既有结论

- Phase 2A primary checkpoint 永久保持 step 2500 / epoch 5，SHA256 为 `07250fe4e82dee3b1a69c2c3b65311404757e6a7ca4e12adb17b4a845304c072`。
- Phase 2B 已澄清 Phase 2A foreground IoU 与 LEGION 文中 fg/bg mIoU 不是同一指标；Phase 2D 不重开该比较。
- Phase 2B.1 的 category parity 状态保持 `CATEGORY_PARITY_UNRESOLVED`，本阶段没有创建 category proxy。
- Phase 2C 的 B3 仍是 classification-only；本阶段仅把其 frozen prediction 作为分析变量，没有接入定位路径。

## 3. 科学问题

本阶段的问题是：为什么同一 frozen checkpoint 在 TF 完整 GT 解释条件下有较强定位能力，而 G0 自回归解释路径明显下降？检查链路包括生成文本、`[SEG]` 触发、`[SEG]` predictor hidden、projection、SAM prompt 与 mask decoder。

## 4. 仓库与推理实现审计

- G0：canonical unified prompt 后自由自回归生成，实现在 `eval/forensics.py`。
- G1：相同 prompt，加结构性 `[FAKE]` continuation prefix。
- TF：`[FAKE] + 完整 GT explanation + [SEG]` 经过 causal forward。
- `[SEG]` predictor state：取 `[SEG]` 前一位置的 causal hidden state，再经过 `text_hidden_fcs[0]`。
- SAM：projection 结果进入 SAM prompt encoder、mask decoder，再 postprocess 到原图大小。
- official manifest、G0/G1/TF 各 1000 条 prediction 均存在；G0/G1 保存了文本与 token ID。
- 历史 artifact **未保存** TF token ID、逐图预测 mask、G0/TF `[SEG]` hidden 或 projection tensor。因此不能从混淆计数反推出预测 mask，也不能离线计算 hidden distance。

完整路径与证据见 `audit/repository_audit.json` 和 `audit/inference_mode_audit.json`。

## 5. 可复现性检查

本阶段没有重复昂贵推理，而是按“已有 artifact 足够时直接复用”的纪律，从 1000 条历史逐图 TP/FP/FN 重新聚合。G0、G1、TF 的样本数、global foreground IoU 和 global foreground F1 与历史 metrics 的绝对误差均为 0，状态为 `MATCH`。

这证明 Phase 2D 的数值语义与历史 artifact 一致；它不是一次新的模型前向复跑。详情见 `metrics/reproducibility.json`。

## 6. official-1000 逐图差值分布

逐图 foreground IoU 差值统计如下：

| 统计量 | 数值 |
|---|---:|
| N | 1000 |
| 均值 | 0.221020 |
| 中位数 | 0.159767 |
| 标准差（总体） | 0.281774 |
| Q1 / Q3 | 0.012946 / 0.396258 |
| P90 / P95 | 0.637928 / 0.772857 |
| 最小 / 最大 | -0.808648 / 0.965788 |
| 正差比例 | 79.3% |
| `abs(gap) ≤ 0.01` | 7.0% |
| 负差比例 | 19.9% |

global-pixel 指标单独聚合：G0/G1/TF foreground IoU 为 0.173136/0.174037/0.475917，foreground F1 为 0.295168/0.296476/0.644910。global 指标不是逐图贡献，报告没有把 global delta 冒充 per-image delta。

## 7. TF-high / G0-low 子集

预注册条件 `TF ≥ 0.70 且 G0 ≤ 0.30` 命中 119 张图：

- 平均 G0 IoU：0.067323；
- 平均 TF IoU：0.813124；
- 平均差值：0.745801；
- 差值中位数：0.753347。

这 119 张图直接说明：对这些相同图像和 union-mask GT，当前 SAM/定位链在 TF 上具有明显能力，但 G0 没有稳定调用该能力。它不能单独说明问题发生在语言还是 hidden/state。

## 8. Controlled hybrid 模式

本阶段逐项审计了 H1–H4，并把定义、固定变量、改变变量和不可执行原因写入 `controlled_modes/mode_specs.json`。

- H1：现有 generate 接口不能在保持历史终止语义的同时，只从 oracle prefix 自回归生成 `[SEG]`。自由续写还会改变 suffix 长度与 SEG 位置，故未作为单因素实验执行。
- H2：把已生成 G0 全序列重新 teacher-force 会同时改变 KV/cache 轨迹、位置处状态构造和执行路径；当前接口不能只替换 SEG state，状态为 `H2_NOT_IDENTIFIABLE`。
- H3：annotation 的 `refs.phrase` 可用，但 phrase-only TF 相比 TF-full 同时改变文本、序列长度、SEG 位置和 state construction，只能是 `MULTI_FACTOR_DIAGNOSTIC_ONLY`，本阶段没有把它冒充因果实验。
- H4：G0 token 已保存，TF token 未保存；现有 prefix replay 同样不能隔离一个变量，因此仅保留为后续 instrumentation 设计。

因此，本阶段没有生成虚假的 `per_image_controlled_results.jsonl`，而是在 manifest 中明确记录未执行原因。

## 9. Grounding 文本分析

采用可复查的 deterministic lexical overlap v1：从 authoritative annotation 的 `refs.phrase` 提取 target terms，与 G0 生成文本做词项重叠。1000 张图中：

- `TARGET_PRESENT`：68；
- `PARTIAL_TARGET`：797；
- `MISSING_TARGET`：135。

该方法不是语义裁判：它不处理同义词、词形、指代和空间关系，所以只能作为保守 proxy。`TARGET_PRESENT` 组与其余组的平均 gap 差为 0.017584，bootstrap 95% CI 为 [-0.058886, 0.100273]，没有显示出稳定组间差异。由此不能断言语言错误是唯一或确定根因。

## 10. Failure taxonomy

在 119 个 severe 样本上，deterministic proxy 分类为：

| 类别 | 数量 | 比例 |
|---|---:|---:|
| B Partial grounding | 80 | 67.2% |
| D Hallucinated grounding candidate | 24 | 20.2% |
| F SEG/state anomaly candidate | 13 | 10.9% |
| E Classification-semantic contradiction | 2 | 1.7% |

其中 F 只表示“词面 target 看起来存在，但 G0 mask 差且 TF mask 强”，不是对 SEG 根因的证明。所有逐样本依据和代表样本见 `text_analysis/failure_taxonomy_samples.jsonl`；状态明确为 `DETERMINISTIC_PROXY_REQUIRES_HUMAN_REVIEW`。

## 11. `[SEG]`、hidden state 与 SAM 诊断

G0 的 `[SEG]` 触发率历史值为 98.2%，因此“未生成 `[SEG]`”不是主要总体解释。

隐藏状态、projection 输出与 SAM prompt 距离在本阶段保持 **UNRESOLVED**。原因不是算力不足，而是历史 official1000 artifact 没有保存所需 tensor，且当前接口不能构造严格单因素 H1/H2。为避免新增 instrumentation 改变历史推理或运行含义混杂的 proxy，本阶段没有补跑 selected-subset hidden inference，也没有声称 SEG state 是根因。

## 12. Phase 2C 分类耦合

Phase 2C B0/B3 与 Phase 2D official1000 通过 exact sample_id 全集匹配，未连接 internal-1104。

- B0 正确 973 张：平均 gap 0.218343；B0 错误 27 张：0.317483。正确减错误的 bootstrap 均值差为 -0.099140，95% CI [-0.209887, 0.004987]。方向上分类错误与较大 gap 共现，但区间包含 0，只能视为探索性非因果证据。
- B3 修正 B0 的样本共有 11 张，其中 9 张仍满足 G0 IoU ≤ 0.30；其平均 G0/TF IoU 为 0.101178/0.418300。
- B3 正确但 G0 IoU ≤ 0.30 的样本有 731 张。这个结果清楚表明：更强的 classification prediction 不会自动修复 frozen G0 grounding。
- B3 与 frozen LM verdict 不一致的样本有 15 张，其中 14 张 G0 IoU ≤ 0.30；该组平均 gap 为 0.222432，与总体 0.221020 接近，不能据此建立额外因果结论。

B3 没有进入 localization inference，因此这些均为 coupling observation，不是 B3 intervention。

## 13. 定性审计

`qualitative/index.html` 收录 28 个代表案例，覆盖最大正差、最大负差、TF-high/G0-low、TF-high/G0-high、TF-low/G0-low 和 B3-correct/G0-bad。页面展示原图、GT polygon union 叠加、G0/TF 指标、文本与分类结果。

历史 artifact 没有保存预测 mask，页面明确标注 G0/TF mask 不可重建，没有用 TP/FP/FN 伪造空间形状。

## 14. 统计分析

统计量和 10,000 次固定 seed bootstrap 已落盘。所有 subgroup 比较均是 post-hoc exploratory，不用于阈值选择或 checkpoint 选择，也不把 p-value/CI 当因果证据。

## 15. CERTAIN

1. official1000 逐图 TF−G0 foreground IoU 的均值为 0.221020，中位数为 0.159767；119 张图满足预注册 severe 条件。
2. G0/G1/TF 历史逐图计数重新聚合与历史 aggregate 完全一致。
3. B3 修正 B0 的 11 张图中有 9 张仍是 G0 IoU ≤ 0.30；B3 classification 没有自动修复 frozen G0 grounding。
4. 本阶段没有训练、修改权重、改变 checkpoint selector、修改阈值或创建 proxy category。

## 16. SUPPORTED BUT NON-CAUSAL

1. 分类错误组的平均 gap 较大，但 bootstrap CI 略跨 0，证据方向性存在而不充分。
2. severe subset 中大多数样本被 lexical proxy 标为 partial/missing/hallucinated target；这支持语言质量值得继续研究，但 proxy 不能作为权威语义判断。
3. 有 13 个 severe 样本在词面 target 存在时仍表现为强 TF/弱 G0，说明只看表面文本不足以解释全部失败，但不能定位到 SEG state。

## 17. UNRESOLVED

1. teacher forcing 的收益中，语言语义上下文和 SEG hidden/state construction 各占多少，当前接口无法干净分离。
2. hidden、projection 与 SAM prompt 距离是否和 gap 相关，没有可用 persisted tensor。
3. phrase-only oracle 是否足够恢复定位，尚未通过严格单因素实验验证。
4. B3 的 NPR/SRM 信号是否对 grounding 提供互补信息，没有 intervention 证据。

## 18. Phase 3 建议

Phase 3 不应直接把 NPR/SRM 注入 LLM 或 SAM。优先级建议如下：

1. 先补充只读、可验证不改变输出的 inference tracing 接口，保存 selected subset 的 G0/TF token-aligned predictor hidden、projection 和 mask；同时为 H1/H2 设计严格的 prefix/state replay 语义。
2. 在该接口能实现单因素对照后，再决定是 phrase-level/autoregressive grounding enhancement，还是 SEG representation/projection redesign。
3. B3 继续保持 classification-only；其 classification 增益没有自动迁移到 grounding。
4. 鉴于 Phase 2C 仍有 source/compression shortcut 风险，若未来要做 forensic-grounding fusion，应先进行 source/compression controlled validation。

当前最准确的 gate 是：**优先解决并可识别化 autoregressive grounding/exposure gap；暂不授权 Phase 3 训练。**

## 19. Artifact 清单

- `manifest.json`：阶段边界、hash、状态和纪律声明。
- `audit/`：仓库与推理语义审计。
- `per_image/`：1000 张 paired 表及 119 张 severe subset。
- `metrics/`：paired 分布、subgroup、bootstrap 与可复现性。
- `controlled_modes/`：H1–H4 machine-readable specification。
- `classification_coupling/`：Phase 2C exact-scope 联结。
- `text_analysis/`：deterministic 文本指标与 failure taxonomy。
- `qualitative/`：28 个案例的 HTML 审计包。

## 20. 实验纪律

```text
training_started: false
model_weights_modified: false
checkpoint_selection_performed: false
test_set_model_selection_performed: false
proxy_category_labels_created: false
```

## 21. 测试结果

- Phase 2D 专项：`8 passed`。
- 项目 `tests/` 完整回归：`118 passed, 3 skipped, 5 warnings`。
- 直接从仓库根执行无路径限制的 pytest 会额外收集 vendored CenterNet2 自带测试，并使其 `tools` 包遮蔽项目的 `tools.utils`；这不是项目测试口径，已明确记录而未修改第三方目录。
