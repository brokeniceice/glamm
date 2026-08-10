# Phase 1A：可解释 AIGC 取证的分层 Localization Evaluation Protocol

## 1. 阶段结论与边界

Phase 1A 已实现相互独立的 Detection、GT-Fake Generate、TF-FullContext、TF-MinimalContext 和 Joint 五种 evaluation mode，以及共享的 causal `[SEG]` predictor extraction、mask metrics、JSONL/JSON 输出和内容类别分组汇总。

本阶段没有开始正式长训练，没有加入 NPR、SRM、forensic evidence fusion 或 consistency loss，没有修改 frozen train/val/test split，也没有改变 Unified Fake/Real training grammar。本文不指定最终论文主 Localization 指标。

## 2. Unified Training 与分任务 Evaluation 为什么不冲突

Unified training 学习的是一条联合自回归链：

```text
image -> fixed [CLS] query -> verdict -> explanation -> [SEG] -> mask
```

训练 grammar 保持：

```text
Fake: [CLS] [FAKE] <artifact explanation> [SEG]
Real: [CLS] [REAL] No identifiable synthetic artifact evidence is detected.
```

Evaluation 则回答不同的科学问题。Detection 衡量真实性判断；teacher forcing 隔离 grounding/mask decoder 能力；GT-Fake Generate 隔离 authenticity classification error，但保留 explanation 和 trigger 的生成难度；Joint 让 classification error 完整传播到部署结果。它们使用同一个模型和训练任务，只改变评测时可见的 GT 信息与记分规则，因此不存在目标冲突。

当前 GLaMM 不能把所有 localization 都绑定 classification gate，因为这会把两类失败混成一个数值：

- authenticity 判断错误；
- 已知图像为 Fake 后的 evidence reasoning / `[SEG]` activation / mask decoding 错误。

分层协议既保留严格的 Joint 指标，也提供不受 classification gate 影响的 Fake-only localization 诊断。

## 3. 五种 mode 的信息边界

| Mode | GT Fake provided | GT explanation provided | Verdict generated | Explanation generated | SEG generated | Classification gates localization |
|---|---|---|---|---|---|---|
| Detection | No | No | Evaluated from `[CLS]` next-token logits | N/A | N/A | N/A |
| GT-Fake Generate | Yes，prefix 提供 `[FAKE]` | No | Provided `[FAKE]` | Yes | Yes | No |
| TF-FullContext | Yes | Yes | Teacher-forced | Teacher-forced | Teacher-forced in input sequence | No |
| TF-MinimalContext | Yes | No | Teacher-forced | No GT explanation；使用固定模板 | Teacher-forced in input sequence | No |
| Joint | No | No | Yes | Yes | Yes | Yes，以 classification head 为 gate |

五种 mode 对应独立 evaluator 函数，不共享隐式 classification/filter 分支：

- `evaluate_detection`
- `evaluate_gt_fake_generation_localization`
- `evaluate_teacher_forced_full_context`
- `evaluate_teacher_forced_minimal_context`
- `evaluate_joint_localization`

CLI 接口：

```text
--forensics_eval_mode detection
--forensics_eval_mode gt_fake_generate
--forensics_eval_mode tf_full_context
--forensics_eval_mode tf_minimal_context
--forensics_eval_mode joint
--forensics_eval_mode all
```

## 4. Mode A：Detection

样本为 frozen test split 的全部 Real + Fake。输入正常 forensic prompt，assistant prefix 只有固定 `[CLS]`，不提供 GT `[REAL]`/`[FAKE]`，也不运行 localization 所需的 SAM path。

Classification head：

```text
h_cls -> classification_head -> p_cls(real, fake)
```

报告 Accuracy、Precision、Recall、F1、ROC-AUC、Fake Precision 和 Fake Recall。

LM verdict：

```text
LM logits at h_cls -> select only [REAL], [FAKE] -> binary softmax
```

报告 LM Accuracy、LM F1、LM Fake Recall，并记录 CLS-LM Agreement 与四类数量：

- CLS correct / LM correct
- CLS correct / LM wrong
- CLS wrong / LM correct
- CLS wrong / LM wrong

这里仅统计一致性，不计算 consistency loss。

## 5. Mode B：GT-Fake Autoregressive Localization

样本选择严格由 GT 决定：

```text
gt_label == Fake AND seg_valid == True
```

绝不使用 `cls_pred` 过滤。Phase 1C diagnosis 后，默认 user prompt 固定为训练时的 Unified forensic prompt：

```text
Determine whether this image is authentic and explain the forensic evidence.
```

assistant continuation prefix 精确为：

```text
[CLS] [FAKE]
```

从 `[FAKE]` 后开始 generation。模型必须自行生成 explanation 和 `[SEG]`；backend 的该路径不会读取 `manifest_row.explanation`，并有专门的无 GT explanation 泄漏测试。

Phase 1C 在同一 Phase 1B checkpoint/32 Fake 上确认：Unified prompt + GT `[FAKE]` 的 trigger 为 0.96875；旧 known-Fake-style prompt 在修正 continuation 后为 0.8125；旧实现把 partial `[FAKE]` 当完整 assistant turn并追加 terminal EOS 时仅为 0.34375。因而默认协议使用训练分布内 prompt，并以结构化 `[FAKE]` 表达 GT authenticity condition。corrected old prompt 与 terminal-EOS legacy mode 均保留为 diagnostic/reproducibility 路径，不作为默认 checkpoint metric。

即使 classification head 预测 Real，只要生成有效 mask，仍正常计算 localization。没有生成 `[SEG]` 或没有有效 pred mask 时，该 GT Fake 样本保留在 denominator 中，IoU/F1 记 0，并设置 `seg_trigger_failure=true`。

该指标表示：真实性条件已知，但 forensic language reasoning、grounding activation 和 mask prediction 均由模型自主完成。

## 6. Mode C：TF-FullContext

样本仍为 GT Fake 且 `seg_valid=True`。输入完整 GT sequence：

```text
[CLS] [FAKE] <GT artifact explanation> [SEG]
```

此 mode 不调用 `generate()`，只做普通 causal forward，再将预测 `[SEG]` 的前一位置 hidden state送入 `text_hidden_fcs` 和 SAM。

TF-FullContext 使用的 GT 信息包括：

- GT authenticity `[FAKE]`；
- 完整 GT artifact explanation；
- teacher-forced sequence 中的 `[SEG]` token position；
- GT mask仅用于计算 metric，不进入模型输入。

它是 grounding/mask decoder 的 upper-bound / diagnostic metric，不是 fully autonomous localization。输出应称为 TF-FullContext mIoU、TF-FullContext Pixel-F1 或 Teacher-Forced Localization，而不是不带语义限定的普通 IoU。

## 7. Mode D：TF-MinimalContext

样本为 GT Fake，不使用 GT explanation。canonical template 固定在 `configs/forensics_eval_phase1a.json`：

```text
[CLS] [FAKE] Artifact evidence: [SEG]
```

此 mode 用于测量移除 GT explanation context 后 localization 的变化。由于固定模板与训练时的自然 explanation 存在 distribution shift，它只作为 diagnostic，不替代 TF-FullContext 或 GT-Fake Generate。

## 8. Mode E：Joint End-to-End Localization

输入未知图像和正常 forensic prompt，assistant prefix 只有 `[CLS]`。模型自行生成 verdict、explanation 和 `[SEG]`。Localization 仍只在 GT Fake 上统计，但 classification head 是显式 gate：

- `cls_pred=Fake` 且产生有效 `[SEG]`/mask：正常计算；
- `cls_pred=Real`：即使 LM verdict 为 Fake 或内部产生 perfect mask，Joint IoU/F1 仍为 0；
- `cls_pred=Fake` 但缺失 `[SEG]`/mask：Joint IoU/F1 为 0，并记录 `grounding_activation_failure=true`。

Joint 回答完整系统从未知图像开始，能否最终正确识别并定位 synthetic evidence。

## 9. 正确的 causal `[SEG]` predictor hidden

GLaMM segmentation representation 不是 `[SEG]` token 自身的 hidden state，而是：

> the causal hidden state used to predict `[SEG]`

若 text token `[SEG]` 位于位置 `s`，其前面有 `n` 个会各净扩展 575 个位置的 image token，则展开后的 predictor 位置为：

```text
p_seg_predictor = s + 575 * n - 1
```

统一 helper `extract_seg_predictor_hidden(...)` 同时服务 teacher-forced 和 generated full-sequence path。它返回 `p_seg_predictor` 的 hidden，不返回 `[SEG]` 自身位置，也不使用 generate hidden-state 的嵌套 tuple 做隐式 step 猜测。

Generation 的流程为：

```text
generate complete sequence ... X [SEG]
-> causal full forward over generated sequence
-> shared helper locates X position
-> X hidden -> text_hidden_fcs -> SAM
```

### Teacher forcing 为什么没有 future leak

完整 input 中存在 `[SEG]` 不代表 predictor hidden 看见了 `[SEG]`。Causal attention 在位置 `k` 只能访问 `<=k` 的 token；`[SEG]` 位于 `k+1`，因此位置 `k` 的 hidden 与只 forward 到 `k` 所得最后 hidden 在 eval mode 下数值一致。它可以看见前面的 GT `[FAKE]` 和 GT explanation，这正是 TF-FullContext 被定义为 upper-bound/diagnostic 的原因。

测试使用随机 tiny causal Llama 验证：

- `prefix + X + [SEG]` 的 `X` hidden 与 `prefix + X` 的最后 hidden数值对齐；
- 强制 generation 得到 `X [SEG]` 时，生成 `[SEG]` 那一步的 predictor hidden 与 shared helper 从完整 generated sequence 取出的 `X` hidden数值对齐。

## 10. 统一 mask metric

所有 localization mode 只调用 `compute_binary_mask_metrics`。固定阈值来自 protocol config：

```text
pred_binary = mask_logit > 0
```

它与 `sigmoid(mask_logit) > 0.5` 完全等价，并有单元测试覆盖。若模型产生多个 mask，evaluation 先做 pixelwise logit max，再按固定阈值形成 union prediction。

逐样本统计：

```text
intersection_i = TP_i
union_i = TP_i + FP_i + FN_i
IoU_i = intersection_i / union_i
PixelF1_i = 2 TP_i / (2 TP_i + FP_i + FN_i)
```

同时报告：

```text
Mean Image IoU = mean(IoU_i)
Global IoU = sum(intersection_i) / sum(union_i)

Mean Image Pixel-F1 = mean(PixelF1_i)
Global Pixel-F1 = 2 sum(TP_i) / (2 sum(TP_i) + sum(FP_i) + sum(FN_i))
```

Mean-image 对每张图等权；global-pixel 对像素统计聚合，大 mask 会有更高权重。二者不可用同一个模糊的“IoU”字段表示。

## 11. Trigger、coverage、missing `[SEG]`

GT-Fake Generate 和 Joint 都报告：

```text
seg_trigger_rate = 成功生成 [SEG] 且得到有效 mask 的 GT Fake 数 / GT Fake 总数
seg_trigger_failure_count = GT Fake 总数 - 成功 trigger 数
```

没有 `[SEG]` 的 Fake 不会被静默丢弃，而是用 empty prediction 与 GT mask 计算，得到 IoU=0、Pixel-F1=0。Joint 另报：

```text
classification_pass_rate = GT Fake 中 classification head 预测 Fake 的比例
```

## 12. Classification error 在不同 mode 中的处理

| 情况：GT Fake、`cls_pred=Real` | Localization 处理 |
|---|---|
| GT-Fake Generate | 不 gate；有效 mask 正常计分 |
| TF-FullContext | 不 gate；正常 teacher-forced 计分 |
| TF-MinimalContext | 不 gate；正常 diagnostic 计分 |
| Joint | empty prediction 语义，IoU/F1=0 |

这使 classification failure 只在完整 end-to-end Joint 指标中传播，同时保留独立 localization 能力分析。

## 13. Real 排除规则

GT-Fake Generate、TF-FullContext、TF-MinimalContext 和 Joint 的 localization records 都只由：

```text
gt_label == Fake AND seg_valid == True
```

创建。Real 的 `mask=None`，完全不进入 localization denominator，不能出现 `Real zero mask vs pred zero mask -> IoU=1`。

## 14. 内容类别分组

所有 Fake localization 汇总同时输出：

```text
overall
human
animal
object
scene
```

每组使用相同 mean/global metric 和 trigger 逻辑。该分组只分析 frozen split，不改变样本比例。

## 15. 输出文件与字段

每种 mode 独立输出：

```text
forensics_predictions_detection.jsonl
forensics_predictions_gt_fake_generate.jsonl
forensics_predictions_tf_full_context.jsonl
forensics_predictions_tf_minimal_context.jsonl
forensics_predictions_joint.jsonl

forensics_metrics_detection.json
forensics_metrics_gt_fake_generate.json
forensics_metrics_tf_full_context.json
forensics_metrics_tf_minimal_context.json
forensics_metrics_joint.json
```

Localization JSONL 包含 sample/image/source/content/GT、两套 verdict prediction/probability、generated explanation、trigger/mask 状态、GT information flags、classification gate 状态、`intersection/union/tp/fp/fn`、image IoU 和 image Pixel-F1。

示例命令：

```bash
python eval/forensics_eval.py \
  --version <phase0.5-checkpoint> \
  --forensics_manifest_dir outputs/data_audits/unified_forensics_split_v1 \
  --dataset_dir datasets \
  --synthscars_root datasets/SynthScars \
  --output_dir <evaluation-output-dir> \
  --forensics_eval_mode all
```

`--max_samples` 只用于 bounded smoke；缺省时评估完整 frozen test split。

## 16. 修改文件列表

- `configs/forensics_eval_phase1a.json`：固定 minimal-context template、mask threshold 和 content groups。
- `eval/forensics.py`：五种独立 protocol、共享 mask metrics、aggregation、records 和输出 writer。
- `eval/forensics_eval.py`：GLaMM backend、CLI、prompt/context isolation 和 frozen test runner。
- `model/GLaMM.py`：统一 causal SEG predictor helper；teacher/generation 共用 indexing；无 SEG 时跳过不必要 SAM path。
- `dataset/dataset.py`：partial-assistant inference 不再进入训练 labels parser。
- `dataset/forensics/unified.py`：向 evaluation backend保留 frozen manifest row；训练 target 不变。
- `tests/test_forensics_evaluation_protocol.py`：Phase 1A protocol、metrics 与 causal alignment tests。
- `tests/test_unified_forensics_pipeline.py`：partial-assistant inference regression test。

## 17. 单元测试结果

完整回归：

```text
python -m unittest discover -v tests
Ran 49 tests in 0.265s
OK
```

Phase 1A 专项 14 tests 全部通过，覆盖：

1. Teacher-forced `[SEG]` indexing 使用前一位置；
2. causal no-future-leak；
3. GT-Fake Generate 忽略 classification error；
4. Joint 将 classification error 传播为 0；
5. GT-Fake missing `[SEG]` 为 0 且计数；
6. Joint missing `[SEG]` 为 0 且记录 activation failure；
7. TF-FullContext 不调用 generation；
8. GT explanation usage flags 与真实 backend 无泄漏 prefix；
9. Real 排除；
10. threshold equivalence；
11. mean-image/global-pixel aggregation；
12. generation hidden-state alignment；
13. Detection 双 head、agreement quadrants 与输出文件；
14. minimal-context canonical config。

另有 partial-assistant inference regression test，防止 evaluation prompt被训练 label parser拒绝。

## 18. 真实 checkpoint bounded smoke

使用本机 `checkpoints/GLaMM-FullScope`，在 frozen test split 前 4 张（2 Real + 2 Fake）、`max_new_tokens=16` 范围完成五模式真实链路 smoke：

```text
Detection records: 4
GT-Fake Generate records: 2
TF-FullContext records: 2
TF-MinimalContext records: 2
Joint records: 2
```

五组 JSONL 和五组 metrics JSON 均成功生成并重新解析，字段完整；输出位于 `outputs/phase1a_protocol_smoke/`。

重要限制：该基础 checkpoint 不包含训练后的 Phase 0.5 classification head，加载时 classification head 为新初始化。因此本次 smoke 只证明 checkpoint loading、frozen preprocessing、五模式 forward/generation、mask shape处理和文件输出能跑通；绝不将其中数值作为模型性能或论文实验结果。

## 19. Phase 1A 十项验收回答

1. GT Fake、`cls_pred=Real` 时，GT-Fake Generate 仍正常计算 localization：是，classification 不 gate、不参与筛选。
2. Joint 是否将该 classification error 计为 0：是，即使存在 perfect mask。
3. Teacher-forced 完整序列是否使用 `[SEG]` 前一位置 hidden：是，统一 helper明确取 `s-1`（含 multimodal expansion offset）。
4. 是否验证未来 `[SEG]` 不泄漏：是，full/prefix causal hidden 数值对齐测试通过。
5. GT-Fake Generate 是否完全没有 GT explanation：是，prompt精确停在 `[CLS] [FAKE]`，真实 backend无泄漏测试通过。
6. TF-FullContext 是否标记 GT explanation：是，`uses_gt_explanation=true`。
7. Real 是否不参与 Fake-only IoU/F1：是，所有 localization denominator 由 GT Fake + `seg_valid` 决定。
8. Mean IoU 与 Global IoU 是否分开：是，Pixel-F1 也分别报告 mean/global。
9. 没有 `[SEG]` 的 Fake 是否明确记 0：是，不跳过，并进入 trigger failure count。
10. generation 与 teacher-forced indexing 是否经过测试：是，分别有直接 indexing、causal leak 和 generation-step alignment 测试。

Phase 1A 到此停止。后续需在 Unified baseline checkpoint 完成后，由人工结合 prior-work comparability、GT information amount、generation robustness 和实验结果决定论文主表协议；本阶段不进入 Phase 1B。

## Phase 1D 最终 protocol 状态（不改写历史数字）

Phase 1D 已将正式默认定义冻结如下：G0 使用 canonical Unified user prompt + `[CLS]` 并自由生成；G1 使用同一 canonical prompt + `[CLS] [FAKE]` structural continuation，继续作为默认 GT-authenticity-conditioned localization；Joint 与 G0 使用完全相同的生成输入和轨迹，唯一差别是 classification-head gate。legacy G2 与 corrected old-prompt G2 仅保留用于历史复现，不再作为默认 protocol。

canonical prompt id 为 `unified_forensics_v1`，SHA256 为 `32f0856f718fb3e19f34b552892636fb6c2613adbae4afb8826dea89f670654d`；evaluation JSONL 持久化 prompt id/hash、raw prompt、assistant prefix 和 generated token IDs。
