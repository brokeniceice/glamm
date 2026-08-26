# Phase 3D.2-A — Spatial Optimization Attribution Audit

## 1. Scientific question and final status

本阶段试图区分 Phase 3D.2 的负结果来自 did-not-learn、learned-but-did-not-generalize、evaluation/interface mismatch，还是稳定的局部信号。按预注册顺序，必须先完成 train/eval consistency audit。

最终状态：**`COMPLETE — STOPPED AFTER AUDIT A`**。Primary gate：**`GATE_PHASE3D2_IMPLEMENTATION_MISMATCH_FOUND`**。

## 2. No-training boundary

本阶段只执行只读 checkpoint/model inspection；所有 model run 均使用 `model.eval()` 与 `torch.no_grad()`，并在运行后验证未产生 parameter gradients。没有 backward、optimizer、scheduler、parameter update、checkpoint creation/averaging、额外 optimizer step、loss/LR/reward/threshold tuning、语言或分类训练、架构修改、FEPN、NPR、SRM 或 FOCAL。

## 3. Test-set isolation

internal test 与 official1000 未读取、未评估，也未用于 protocol、selector、subgroup 或路线决策。本阶段仅访问 frozen train/validation manifests。形式化 evaluation 在 Audit A mismatch 后未启动。

## 4. Checkpoint provenance

冻结只读 checkpoint 为 step 0、50、100、250。step 0 是原始 P1，SHA256 `fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326`；其余为既有 Phase 3D.2 diagnostic checkpoints。Phase 3D.2 formal selected checkpoint 仍是 step 0；本阶段没有重选或提升 step 250。完整路径和 SHA256 见 `outputs/phase3d2a_spatial_attribution_audit/checkpoint_audit_manifest.json`。

## 5. Audit A — Train/eval consistency

Audit A 发现 material language-prompt mismatch。

Phase 3D.2 training dataset 使用 canonical unified user question：

```text
Determine whether this image is authentic and explain the forensic evidence.
```

TF-PHRASE evaluator 使用 legacy localization question：

```text
Analyze the synthetic artifacts in this image, explain the forensic evidence, and localize the corresponding artifact regions.
```

具体调用链为：`teacher_forced_localization()` 构造正确的 authoritative assistant content，但随后调用 `_causal_forward()`；后者调用 `_batch()` 时没有覆盖其 `FORENSICS_QUESTION` 默认参数。Training 侧则直接消费 `UnifiedForensicsDataset._conversation()` 产生的 canonical unified prompt。

两侧 authoritative assistant target 均为：

```text
[FAKE] explanation
Target regions: <authoritative phrase> [SEG]
```

image preprocessing、per-image all-ref union GT mask、SAM postprocess 与 frozen logit threshold 也一致；错位位于 authoritative assistant target 之前的 user prompt。

## 6. Fixed-sample numerical evidence

使用 Phase 3D.2 frozen schedule 的前 8 个样本，同时运行实际 training-style frozen-language forward 与 evaluation-style TF forward：

- raw input token sequence：`0/8` exact equal；
- `[SEG]` count：两侧均为每样本 1 个；
- GT mask：`8/8` pixel exact equal，IoU 均为 1；
- frozen `[SEG]` hidden max-absolute difference：`0.40625–1.3125`；
- hidden cosine similarity：`0.990938–0.997114`；
- postprocessed mask-logit max-absolute difference：`1.093749–2.256592`。

因此该 prompt mismatch 已实际传播到 language-conditioned spatial representation 与最终 mask logits，不是仅有字符串差异的无影响实现细节。逐样本证据见 `train_eval_consistency_audit.json`。

## 7. Population construction

由于 manifest preparation 与 Audit A 的 GPU forward 并行开始，在 Audit A 返回前已完成以下只读冻结：

- SEEN-TRAIN：从真实 metrics log 与 frozen schedule 恢复 1,000 exposures，对应 1,000 unique identities，无重复 exposure；
- TRAIN-HOLDOUT：同一 Fake train split 中固定 seed `3407` 抽取 1,000 unique images，与 SEEN-TRAIN 的 sample ID 和 canonical image identity 交集均为空；
- VALIDATION-FAKE：确认冻结 1,106 images。

这些 population 在 mismatch 发现后没有进入 checkpoint evaluation；其 manifests 不构成科学结果。

## 8. Checkpoint × population results and paired bootstrap

**`NOT_RUN_STOPPED_BY_AUDIT_A`**。没有生成 checkpoint×population localization measurement，也没有执行 paired bootstrap。对应 JSON 是明确的 stop-status record，不是零差值或缺失值插补。

## 9. Training-loss vs metric trajectory

**`NOT_RUN_STOPPED_BY_AUDIT_A`**。既有 training loss 未与不匹配的 population metric 对齐。`loss_metric_trajectory.png` 只是停止状态图，不包含实验曲线。

## 10. Soft-mask diagnostics

**`NOT_RUN_STOPPED_BY_AUDIT_A`**。没有 threshold sweep、best-threshold search、post-hoc calibration 或替代正式 FG IoU 的 soft result。

## 11. Subgroup analysis

**`NOT_RUN_STOPPED_BY_AUDIT_A`**。没有观察结果后创建 subgroup，也没有 subgroup selector 或 cherry-picking。

## 12. Module-swap attribution

**`NOT_RUN_STOPPED_BY_AUDIT_A`**。没有构造或评估 FC/Decoder swap model，没有保存新 checkpoint，也没有进行 ablation training。

## 13. Representation-change audit

除 Audit A 为判定 materiality 所需的 frozen `[SEG]` hidden 与 mask logits 对照外，step0/step250 representation-change attribution **未执行**；没有训练 probe。

## 14. Failure attribution and gate

唯一 primary gate 为：

> **`GATE_PHASE3D2_IMPLEMENTATION_MISMATCH_FOUND`**

这意味着 Phase 3D.2 当前机制解释暂停。先前 step0/50/100/150/200/250 validation 数值与 step0 formal selection 仍是历史记录事实，但该 evaluator 并未复现 training user prompt，所以这些数值不能回答 matched protocol 下是否拟合、是否过拟合或是否泛化。

## 15. Engineering trainability vs metric-relevant learnability

Phase 3D.2 的 parameter/gradient audit 已证明 `text_hidden_fcs` 与 `mask_decoder` 在工程意义上收到梯度并更新；这一结论不变。由于 Audit A mismatch，metric-relevant localization learnability/generalization 仍未判定。二者不得混写。

## 16. Conclusion boundaries

本阶段不支持以下推论：corrected evaluator 会显示正增益；seen samples 已被有效拟合；模型发生 overfitting；mask decoder、projection 或 frozen representation 是首要瓶颈；需要 FEPN/NPR/SRM/FOCAL；或所有 direct spatial supervision 均无效。

## 17. Next-route authorization status

Phase 3D.2-A 到此停止。若要解决 protocol mismatch，需要新的明确授权来冻结 matched evaluator/protocol 与其统计解释。当前未授权自动修 evaluator、重评 checkpoint、重选 selector、重训、增加 steps、调 BCE/Dice/LR/threshold、FC-only/decoder-only training、joint language-spatial optimization、架构修改、external baseline 或 held-out test evaluation。

## 18. Artifact index

主 artifact 位于 `outputs/phase3d2a_spatial_attribution_audit/`：`experiment_manifest.json`、`checkpoint_audit_manifest.json`、`train_eval_consistency_audit.{json,md}`、population manifests、`failure_attribution.json`、`route_gate.json`、`final_attribution_report.md` 与 `completion_manifest.json`。所有被停止的 downstream outputs 均显式标为 `NOT_RUN_STOPPED_BY_AUDIT_A`。
