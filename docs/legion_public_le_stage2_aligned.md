# 公开 LEGION LE 权重上的第二阶段分类训练

状态：**COMPLETE**（2026-09-25）。从作者公开的 `khr0516/legion_LE@f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa` 初始化官方 `LegionForCls`，只训练新分类头。未重训第一阶段，也未使用测试集、Official1000 或外部 OOD 选模。

## 与 legion-retrained 对齐的训练口径

| 项目 | 本轮 | legion-retrained |
|---|---|---|
| 第二阶段初始化 | 作者公开 `legion_LE` | 自训 Stage 1 merged LE |
| 训练 / 验证 | 相同冻结清单，17,672 / 2,212 | 相同 |
| 标签 | Real=1、Fake=0 | 相同 |
| 可训练参数 | `prediction_head` 2,103,298 个 | 相同 |
| 轮数、学习率、调度器 | 3、`1e-3`、cosine | 相同 |
| 有效 batch | 卡 1、2 各 32，DDP 合计 64 | 单卡 64 |
| 选模 | internal validation Accuracy 最大，`load_best_model_at_end` | 相同 |

两卡 DDP 与单卡 64 的有效 batch 一致，但随机数、样本分片和数值执行路径不同，不宣称逐步或逐权重 bitwise 复现。预检确认两个 rank、有效 batch 64、有限 loss 和非零梯度；仅有分类头可训练。预检峰值 CUDA allocated 为 17,946,174,976 bytes（rank 0）。

## 训练结果

| Epoch | Step | Internal validation Accuracy |
|---:|---:|---:|
| 1 | 277 | 0.985986 |
| 2 | 554 | 0.986890 |
| 3 | 831 | **0.988246** |

选择 epoch 3 / step 831，最佳 Accuracy 精确值为 `0.9882459312839059`。`legion-retrained` 的既有最佳值为 `0.9877938517179023`，本轮高 `0.0004520795660036`，在 2,212 张验证图像上相当于 1 张；这只是验证集点差，不构成 OOD 或统计显著性结论。

最终模型：[final_model](../checkpoints/legion_public_le_stage2_aligned/final_model)；目录规范 SHA256：`274aafcc01a81cf1848ecf99db65578c42b2b758b9a9417eb625e8d007cc658b`。模型索引含四个 `prediction_head` 张量；原公开 LE 索引不含该分类头。训练结束时服务正常退出，卡 1、2 已释放。

## 结果边界和复核入口

此模型应标注为 **public LEGION-LE + aligned Stage-2 head**。它沿用公开 LE 的定位/解释权重；第二阶段只训练 CLIP CLS 特征上的分类头，因此不能把它写成 `legion-retrained`、`legion-retrained-match` 或论文丢失的最终权重。当前仅有 internal validation 训练期指标，尚未进行 internal test 或外部 OOD 分类评测。

- [冻结协议](../checkpoints/legion_public_le_stage2_aligned/protocol.json)
- [预检](../checkpoints/legion_public_le_stage2_aligned/preflight_batch32.json)
- [训练汇总](../checkpoints/legion_public_le_stage2_aligned/training_summary.json)
- [最终模型文件身份](../checkpoints/legion_public_le_stage2_aligned/final_model_identity.json)
- [服务状态](../checkpoints/legion_public_le_stage2_aligned/status.json)
- [双卡自动接续脚本](../scripts/legion_public_le_stage2_aligned.py)
