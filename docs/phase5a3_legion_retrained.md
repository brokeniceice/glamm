# Phase 5A-3 — LEGION on Frozen Internal Training Data

## 结论

两阶段训练已完成。Stage 1 从官方指定的 `GLaMM-GranD-Pretrained@a2513f97c9404065cfd5849325e61d5d53456441` + SAM 初始化；Stage 2 只从本次 Stage 1 merged LE 初始化。公开 intermediate `legion_LE` 未参与初始化。

## 固定身份与数据防火墙

- LEGION source commit: `d21535dd45f6fea509337a83095966f0b86ac924`；完成后 clean: `True`
- Stage 1 base: `/home/yz/groundingLMM_official/checkpoints/phase5a3_legion_retrained/base/GLaMM-GranD-Pretrained`
- SAM: `/home/yz/groundingLMM_official/checkpoints/phase5b0_fakeshield/models/sam_7790786db131bcdc639f24a915d9f2c331d843ee/checkpoints/sam_vit_h_4b8939.pth`
- CLIP fixed local revision: `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`
- train/val adapter summary SHA256: `4bd835b2b24613bf7df5c67388e38b0333132304a034b2b25471dd753c250169`
- official1000 leakage = 0
- internal test leakage = 0
- external benchmark leakage = 0
- public intermediate `legion_LE` initialization = NO

Loader 只接收冻结的 train/val manifest；训练及选模未传入 internal test、official1000、LOKI、RAISE 或其他 external benchmark 路径。

## Stage 1 — Localization + Explanation

官方原始 sample unit 被保留：一条原始 annotation/caption 为一个样本，每个 ref 保留独立 phrase、独立 `<p>…</p> [SEG]` 与独立 polygon mask；不使用 R1 的 combined phrase，不做 union mask。

- frozen Fake images: 8836
- candidate original annotation samples: 8971
- usable LE samples: 8971
- excluded: 0 (`{}`)
- validation annotation samples: 1128
- epochs=3, LoRA r=8, lr=1e-4, CE=1.0, Dice=0.2, BCE=0.4
- micro batch/GPU=1, GPUs=2, grad accumulation=8, effective global batch=16
- steps/epoch=561; zero-gradient padding does not duplicate training contribution
- preflight loss=4.215424; peak GPU memory=21.76 GiB
- trainable parameters=587,427,044; exact list: `/home/yz/groundingLMM_official/checkpoints/phase5a3_legion_retrained/stage1_run/preflight.json`
- selector: minimum frozen internal-validation teacher-forced total loss; all three epochs retained
- selected checkpoint: `epoch3_global_step1683` (val total `1.018036`)

| Epoch | Train total | Train CE | Train BCE | Train Dice | Val total |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.360326 | 1.186429 | 0.035110 | 0.138787 | 1.065668 |
| 2 | 0.978151 | 0.833618 | 0.025210 | 0.119323 | 1.026487 |
| 3 | 0.889878 | 0.756459 | 0.022571 | 0.110848 | 1.018036 |

Merged `LEGION-retrained-LE`: `/home/yz/groundingLMM_official/checkpoints/phase5a3_legion_retrained/stage1_le`  
Canonical checkpoint SHA256: `6b66fd51f8ea0b26a1930084f858010efc99faa52c04c0802e1666801e304844`

## Stage 2 — Detection Classification

- initialization: `/home/yz/groundingLMM_official/checkpoints/phase5a3_legion_retrained/stage1_le`
- train: 17,672 = 8,836 Real (1) + 8,836 Fake (0)
- validation: 2,212 internal validation only
- epochs=3, lr=1e-3, cosine scheduler, one GPU, per-device/effective global batch=64
- trainable: prediction_head only (2,103,298 parameters)
- preflight loss=0.535156; peak GPU memory=18.35 GiB

| Epoch | Internal val accuracy | Internal val loss |
|---:|---:|---:|
| 1 | 0.985081 | 0.044902 |
| 2 | 0.987794 | 0.047273 |
| 3 | 0.987794 | 0.046065 |

Best internal-validation checkpoint: `/home/yz/groundingLMM_official/checkpoints/phase5a3_legion_retrained/stage2_cls/checkpoints/checkpoint-554`  
Best accuracy: `0.987794`  
`LEGION-retrained-CLS`: `/home/yz/groundingLMM_official/checkpoints/phase5a3_legion_retrained/stage2_cls/final_model`  
Canonical checkpoint SHA256: `f33fda9ddf0e22bcd9bca8dcc998d8a9c0c6421f6bd6a46573044c6cc7365739`

## Paper–Code Training Configuration Discrepancy

本实验属于 **same-data controlled retraining**，不是对论文原数据与八卡硬件规模的 exact reproduction。

1. Stage 1：论文为 8 GPUs × 2/GPU = global batch 16；此前已完成的 2 GPUs × 1 × accumulation 2 = gb4 run 已标记为 `stage1_gb4_pilot`，不进入论文主比较。本次最终重跑使用 2 GPUs × 1 × accumulation 8，等效 global batch 16。
2. Stage 2：论文 nominal batch 为 8 × 64 = 512，但论文 ProGAN train 约 720k 样本；本项目 controlled train 仅 17,672。机械使用 512 会使每 epoch 仅约 35 次更新，因此保留官方 per-device batch 64、lr=1e-3、3 epochs、cosine、prediction-head-only 和 accuracy selector，采用单卡 effective global batch 64，共 277 steps/epoch、831 updates。
3. 优先冻结相同训练数据与 supervision、无泄漏、官方架构与 trainable modules，并采用与本数据规模相称的 architecture-specific optimization。

## Dependency / launcher deviations

官方仓库保持未修改。外部 wrapper 仅修复：Stage 1 单 epoch hard break、随机有放回 HybridSegDataset、validation first-1000 截断、未使用数据集的缺失 prompt 常量与官方 merge 旧方法名；Stage 2 保持官方 Real=1/Fake=0，并修复 meta-tensor loader 与图像 collator。模型核心、prompt、LoRA/loss/optimizer/scheduler recipe 未修改。

本阶段到此 STOP；未运行 official1000、LOKI、鲁棒性或 R1 对比。
