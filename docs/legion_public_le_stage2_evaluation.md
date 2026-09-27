# Public LEGION-LE + aligned Stage-2：分类测试与外部 OOD

状态：**COMPLETE**。本轮只评测冻结的第二阶段分类头；公开 LE 的定位/解释权重不变。

| 数据集 | N | Accuracy | F1 | ROC-AUC | Fake recall | TNR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Internal2208 | 2208 | 0.987772 | 0.987821 | 0.999283 | 0.991848 | 0.983696 | 0.016304 |
| AIGI-Holmes TestSet | 99999 | 0.883919 | 0.869728 | 0.974483 | 0.774995 | 0.992840 | 0.007160 |
| GenImage held-out | 100000 | 0.705230 | 0.591431 | 0.926150 | 0.426700 | 0.983760 | 0.016240 |
| LOKI classification | 2217 | 0.576906 | 0.547297 | 0.679072 | 0.430524 | 0.791111 | 0.208889 |
| RAISE998 | 998 | 0.991984 | - | - | - | 0.991984 | 0.008016 |

RAISE998 只有 Real；F1、ROC-AUC 与 Fake recall 不作双类解释。GenImage 的八个生成器分组指标见结果 JSON。

## 评测口径

- 模型：作者公开 `legion_LE` 第一阶段权重 + 本项目按 `legion-retrained` 第二阶段配方训练的分类头；冻结 checkpoint SHA256：`274aafcc01a81cf1848ecf99db65578c42b2b758b9a9417eb625e8d007cc658b`。
- 两张卡并行推理：卡 1 跑 Internal2208、GenImage；卡 2 跑 AIGI-Holmes、LOKI、RAISE998。分类阈值固定为 Fake probability `>= 0.5`，Real=1/Fake=0 为模型内部标签，汇总中 Fake 为正类。
- 所有结果已核对冻结 manifest SHA256、样本数、唯一性、逐样本 ID 顺序、GT 标签、预测阈值和重新计算的总体指标；没有用测试集或外部 OOD 调阈值、改权重或选模。
- 冻结泄漏审计为 **BLOCKED_OVERLAP**，继续评测 override 为 **ACTIVE**。AIGI-Holmes 与 internal-TRAIN 有 779 个 exact SHA256 重叠，保留在完整 TestSet 中。外部清单合计 793 对 pHash 近重叠，AIGI-Holmes 与 GenImage 之间另有 35 对 pHash 近重叠。

## 复核入口

- [汇总 JSON](../outputs/legion_public_le_stage2_evaluation/results.json)
- [冻结评测协议](../outputs/legion_public_le_stage2_evaluation/protocol.json)
- [训练记录](legion_public_le_stage2_aligned.md)
- [最终模型](../checkpoints/legion_public_le_stage2_aligned/final_model)
