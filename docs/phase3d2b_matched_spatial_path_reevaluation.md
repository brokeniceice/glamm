# Phase 3D.2-B — Matched Spatial-Path Re-evaluation

## 1. Scientific question

本阶段只回答：当 TF-PHRASE evaluator 的 user prompt 与 Phase 3D.2 training 完全一致时，已有 spatial-training checkpoints 是否改善 internal-validation Fake localization？这是 read-only matched-protocol retrospective evaluation，不是新训练，也不静默修正原实验历史。

## 2. Strict boundary and isolation

所有 inference 均使用 `model.eval()` 与 `torch.no_grad()`。没有 backward、optimizer、scheduler、parameter update、新 checkpoint、checkpoint averaging、额外 steps、BCE/Dice/LR/reward/threshold tuning、SFT、classification training、G0、seen-train、train-holdout、subgroup、soft-mask、module-swap、FEPN、NPR、SRM、FOCAL 或架构修改。

唯一 protocol change 是将 TF-PHRASE user prompt 从 legacy localization question 改为 Phase 3D.2 training 的 canonical authenticity question。internal test 与 official1000 保持封存。

## 3. Checkpoints and historical preservation

只读评估原 selector 全部候选 step 0/50/100/150/200/250；路径、SHA256、stored optimizer step 与历史 selector status 见 `outputs/phase3d2b_matched_spatial_reevaluation/checkpoint_manifest.json`。

两个 selector 概念严格分开：

- historical selector：legacy mismatched TF evaluator，step 0 selected；
- matched-protocol retrospective selector：本阶段 canonical matched evaluator 的只读回顾选择。

后者不是 corrected original selector，也不授权 checkpoint promotion。

## 4. Matched protocol consistency hard gate

8 个 Phase 3D.2-A 固定样本全部通过：

- raw/effective token sequence：8/8 exact equal；
- `[SEG]` count 与 predictor index：8/8 exact equal；
- GT union mask：8/8 pixel exact equal；
- frozen `[SEG]` hidden：max/mean absolute difference 全为 0，cosine similarity 全为 1；
- `text_hidden_fcs` 256D output：difference 全为 0；
- native mask logits：difference 全为 0；
- postprocessed mask logits：difference 全为 0。

Consistency status：**PASS**。因此允许进入完整 1,106 validation Fake evaluation。

## 5. Matched validation metrics

| Step | Mean FG IoU | Median FG IoU | Δ mean IoU vs 0 | Mean FG F1 | Median FG F1 | Δ mean F1 vs 0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.342928 | 0.304517 | +0.000000 | 0.452020 | 0.466866 | +0.000000 |
| 50 | 0.340327 | 0.308696 | -0.002601 | 0.449079 | 0.471760 | -0.002941 |
| 100 | 0.334423 | 0.306211 | -0.008505 | 0.440435 | 0.468854 | -0.011585 |
| 150 | 0.339438 | 0.309005 | -0.003490 | 0.446847 | 0.472122 | -0.005172 |
| 200 | 0.339065 | 0.306359 | -0.003863 | 0.445653 | 0.469028 | -0.006367 |
| 250 | 0.339935 | 0.304916 | -0.002993 | 0.446843 | 0.467334 | -0.005177 |

所有训练 checkpoint 的 mean FG IoU 与 mean FG F1 均低于 step 0。

原 Phase 3D.2 legacy-mismatched evaluator 表保留如下，不参与 matched retrospective selector：

| Step | Legacy TF FG IoU | Legacy TF FG F1 |
|---:|---:|---:|
| 0 | 0.340823 | 0.449934 |
| 50 | 0.338428 | 0.447210 |
| 100 | 0.332482 | 0.438323 |
| 150 | 0.337848 | 0.445243 |
| 200 | 0.338030 | 0.444637 |
| 250 | 0.338544 | 0.445345 |

## 6. Paired bootstrap

所有比较使用同一 1,106 Fake population、10,000 repeats、seed 3407。

| Step−0 | IoU mean Δ | IoU 95% CI | IoU W/T/L | F1 mean Δ | F1 95% CI |
|---:|---:|---:|---:|---:|---:|
| 50 | -0.002601 | [-0.007516, +0.002171] | 494/70/542 | -0.002941 | [-0.008282, +0.002295] |
| 100 | -0.008505 | [-0.014516, -0.002637] | 458/79/569 | -0.011585 | [-0.017910, -0.005261] |
| 150 | -0.003490 | [-0.008391, +0.001416] | 476/77/553 | -0.005172 | [-0.010564, +0.000261] |
| 200 | -0.003863 | [-0.009424, +0.001635] | 492/77/537 | -0.006367 | [-0.012487, -0.000298] |
| 250 | -0.002993 | [-0.008372, +0.002255] | 497/78/531 | -0.005177 | [-0.011058, +0.000632] |

没有训练 checkpoint 满足 primary positive criterion。step 100 的 FG IoU/F1 显著下降，step 200 的 FG F1 也显著下降；其余 primary IoU CI 跨 0。这些 secondary harmful points 不意味着所有训练 step 都分别达到显著 harmful。

## 7. Matched-protocol retrospective selector

按冻结规则，以 matched mean FG IoU 为 primary、mean FG F1 为 tie-breaker，回顾选择仍为 **step 0**：

```text
historical_selected_step = 0
matched_retrospective_selected_step = 0
whether_selection_changes = false
```

没有 checkpoint 被提升或替换 P1。

## 8. Legacy versus matched protocol

| Checkpoint | Legacy IoU | Matched IoU | Matched−Legacy IoU | IoU 95% CI | Legacy F1 | Matched F1 | Matched−Legacy F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| step 0 | 0.340823 | 0.342928 | +0.002105 | [+0.000767, +0.003460] | 0.449934 | 0.452020 | +0.002086 |
| step 250 | 0.338544 | 0.339935 | +0.001391 | [-0.000352, +0.003143] | 0.445345 | 0.446843 | +0.001498 |

step 0 的 matched prompt 对同一模型产生小幅正向 measurement shift；step 250 的 IoU shift CI 跨 0。它们只是 same-model evaluation-context diagnostic，不能表述为模型改进。结合 Phase 3D.2-A 的 hidden/logit audit，证据支持 forensic grounding 对完整 conversational trajectory 敏感；这不是未经更多干预即可外推的严格因果定律。

## 9. Phase 3D.2 interpretation

Matched evaluator 没有推翻负方向：全部训练 checkpoint mean 低于 step 0，retrospective selector 不变，且没有显著正向 IoU gain。Phase 3D.2 的 negative validation direction 因此获得 matched-protocol 支持。

边界仍然成立：这不证明所有 direct spatial supervision 无效，不证明 mask decoder 或 projection 是唯一/首要瓶颈，也不判定 seen-sample fitting、overfitting 或 train/validation generalization mechanism。

## 10. Final route gate

Primary gate：

> **`GATE_MATCHED_SPATIAL_OPTIMIZATION_NO_VALIDATION_GAIN`**

选择该 gate 是因为所有训练 checkpoint 均未显著优于 step 0，且 retrospective selector 仍为 step 0。step 100 的显著下降作为 secondary harmful evidence 单独报告，不扩展成所有 checkpoints 都显著有害。

## 11. Next-route authorization

按照 route tree，现在可以提出恢复 Phase 3D.2-A downstream attribution（seen-train、train-holdout、loss-vs-metric、soft-mask、subgroup、module-swap）的新申请，但本阶段不自动授权或执行。当前也不授权 G0、test、official1000、checkpoint promotion、继续训练、增加 steps、loss/LR/threshold tuning、joint optimization 或架构路线。

Phase 3D.2-B 到此停止并等待下一阶段明确指令。

## 12. Artifacts

完整 machine-readable outputs 位于 `outputs/phase3d2b_matched_spatial_reevaluation/`，包括 experiment/checkpoint/protocol manifests、consistency audit、matched metrics、paired bootstrap、retrospective selector、legacy-matched comparison、route gate、final report 与 completion manifest。
