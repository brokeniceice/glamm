# Phase 6B.1a — Exact LEGION Stage-2 Recipe Control

## 结论

Phase 6B.1a 已完成，且只读取冻结 internal train / validation。`C1-Exact` 保持 C1-L 的分类结构不变，并严格复用了当前 LEGION-retrained Stage-2 的训练路径。三轮 validation Accuracy 与原 LEGION-retrained Stage-2 逐值一致，说明该控制实验已确定性复现当前 Stage-2 contract。

recipe 对 internal validation 的净影响很小：相对 C1-L 三随机种子均值，C1-Exact 的 Accuracy 增加 `0.000452`（`+0.0452` 个百分点），F1 增加 `0.000487`，ROC-AUC 降低 `0.000166`。主要变化是阈值为 0.5 时 Fake recall 增加 `0.003315`，同时 TNR 降低 `0.002411`。因此 internal validation 不支持“C1-L 与 LEGION-retrained 的剩余差距主要由 recipe 带来”的强结论；recipe 更像是在固定阈值处移动 Real/Fake 权衡，而不是整体提高排序能力。

结论：`ALLOW_EXTERNAL_CONFIRMATION = YES`。理由是结构、训练集、验证集和训练 contract 均已通过；但本阶段未访问 external benchmark，外部确认必须作为后续显式授权阶段执行。

## 固定结构与训练防火墙

```text
Frozen CLIP ViT-L/14@336 penultimate CLS [1024]
→ Linear(1024, 2048)
→ ReLU
→ Linear(2048, 2)
```

- trainable parameters：`2,103,298`，严格只有 prediction head 的 4 个参数张量。
- CLIP、LLM、SAM、mask decoder 和其余 LEGION 参数全部冻结。
- classification forward 只经过在线 CLIP penultimate CLS 和 prediction head；不经过 explanation、`[SEG]` 或 SAM。
- loader 只读取 internal train `17,672` 和 internal validation `2,212`。
- internal test、official1000、AIGI-Holmes、GenImage、LOKI、RAISE 均未读取。

Preflight：PASS；batch 64 forward/backward loss=`0.53515625`，4/4 trainable tensors 获得非零梯度，峰值显存 `19,707,344,384` bytes。

## Exact Stage-2 contract

| 项目 | C1-Exact | 当前 LEGION-retrained Stage-2 |
|---|---:|---:|
| batch size | 64 | 64 |
| gradient accumulation | 1 | 1 |
| effective global batch | 64 | 64 |
| learning rate | 1e-3 | 1e-3 |
| weight decay | 0 | 0 |
| epochs | 3 | 3 |
| scheduler | cosine | cosine |
| selector | max internal-val Accuracy | max internal-val Accuracy |
| threshold | 0.5 | 0.5 |
| seed | 3407 | 3407 |
| labels in training | Real=1, Fake=0 | Real=1, Fake=0 |
| feature path | online BF16 CLIP CLS | online BF16 CLIP CLS |
| trainer / optimizer path | same LEGION wrapper and Transformers Trainer | same |

相对当前 LEGION-retrained Stage-2 的已知差异：**无**。C1-Exact 使用完整官方 `LegionForCls` 容器，以保证 head 初始化时的 RNG 消耗和 Trainer 行为也一致；容器中的语言/定位组件不参与分类 forward 且不可训练，因此不构成额外分类架构。

## Validation selector

| Epoch | Step | Eval loss | Selector Accuracy |
|---:|---:|---:|---:|
| 1 | 277 | 0.044902 | 0.985081 |
| **2** | **554** | 0.047273 | **0.987794** |
| 3 | 831 | 0.046065 | 0.987794 |

Epoch 2 与 epoch 3 Accuracy 相同；`load_best_model_at_end` 保留首次达到最大值的 epoch 2 / step 554。以上三行与 Phase 5A-3 中原 LEGION-retrained Stage-2 完全相同。

## Selected C1-Exact internal-validation metrics

论文侧统一把 Fake 作为正类，并固定 `prob_fake >= 0.5`：

| N | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 2,212 | 0.987342 | 0.998822 | 0.991863 | 0.982821 | 0.017179 | 0.987399 |

Confusion counts：TP=`1097`，TN=`1087`，FP=`19`，FN=`9`。

注意：Trainer selector 使用官方 `argmax(logits)`，其 Accuracy=`0.987794`；统一报告在重新载入 selected checkpoint 后使用冻结规则 `prob_fake >= 0.5`，Accuracy=`0.987342`。两者相差恰好 1/2212 个样本；可能来源包括二分类边界上的 `argmax` / `>= 0.5` 语义及 BF16 重放数值差异，现有证据不进一步归因。样本集合没有变化，checkpoint selection 仍严格按训练时的官方 argmax Accuracy 执行。

## C1-L vs C1-Exact

C1-L 是三个预注册 seed 的均值；C1-Exact 是本阶段固定 seed=3407 的单次 exact-recipe control，因此该表是 protocol attribution，不是配对显著性检验。

| Metric | C1-L mean | C1-Exact | Exact − C1-L |
|---|---:|---:|---:|
| Accuracy | 0.986890 | 0.987342 | +0.000452 |
| ROC-AUC | 0.998988 | 0.998822 | -0.000166 |
| Fake recall | 0.988547 | 0.991863 | +0.003315 |
| TNR | 0.985232 | 0.982821 | -0.002411 |
| FPR | 0.014768 | 0.017179 | +0.002411 |
| F1 | 0.986911 | 0.987399 | +0.000487 |

### 问题回答

1. **recipe 对 internal validation 有多大影响？** 很小。Accuracy/F1 各提高约 `0.05` 个百分点，ROC-AUC 略降；更明显的是 Fake recall/TNR 的阈值权衡移动。
2. **是否已与 LEGION Stage-2 training contract 基本对齐？** 是，而且对当前可控项目已逐项完全对齐；三轮 selector 轨迹也与原训练完全一致。
3. **是否允许进入 external confirmation？** 允许。`ALLOW_EXTERNAL_CONFIRMATION = YES`，但本阶段严格 STOP，未自动运行任何 external benchmark。

## Artifacts

- machine result：`outputs/phase6b1a_exact_stage2_control/results.json`
- preflight：`outputs/phase6b1a_exact_stage2_control/preflight_batch64.json`
- training summary：`outputs/phase6b1a_exact_stage2_control/training_summary.json`
- selected checkpoint：`outputs/phase6b1a_exact_stage2_control/checkpoints/checkpoint-554`
- loaded-best final model：`outputs/phase6b1a_exact_stage2_control/final_model`
- training log：`outputs/phase6b1a_exact_stage2_control/training.log`

阶段状态：`COMPLETE_AND_STOPPED_BEFORE_EXTERNAL_CONFIRMATION`。
