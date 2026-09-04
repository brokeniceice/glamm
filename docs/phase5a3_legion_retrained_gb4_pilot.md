# Phase 5A-3 — GB4 PILOT ONLY (Superseded)

> **协议修正后降级：** 本文记录的 Stage 1 effective global batch=4，与论文 Stage 1 global batch=16 不一致。该 Stage 1 现标记为 `stage1_gb4_pilot`；由它初始化的 Stage 2 标记为 `stage2_from_gb4_pilot`。两者均不得进入最终论文主比较，历史结果仅保留作审计。

# LEGION on Frozen Internal Training Data — Historical Pilot Record

## 结论

两阶段训练已完成。Stage 1 从官方指定的 `GLaMM-GranD-Pretrained@a2513f97c9404065cfd5849325e61d5d53456441` + SAM 初始化；Stage 2 只从本次 Stage 1 merged LE 初始化。公开 intermediate `legion_LE` 未参与初始化。

## 固定身份与数据防火墙

- LEGION source commit: `d21535dd45f6fea509337a83095966f0b86ac924`；完成后 clean: `True`
- Stage 1 base: `/home/yz/groundingLMM_official/checkpoints/phase5a3_legion_retrained/base/GLaMM-GranD-Pretrained`
- SAM: `/home/yz/groundingLMM_official/checkpoints/phase5b0_fakeshield/models/sam_7790786db131bcdc639f24a915d9f2c331d843ee/checkpoints/sam_vit_h_4b8939.pth`
- CLIP fixed local revision: `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`
- train/val adapter summary SHA256: `1c9d4d94934024882f56d5169043da966223c7b32e2ccd86b4b1bb7a533856f1`
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
- micro batch/GPU=1, GPUs=2, grad accumulation=2, nominal global batch=4
- steps/epoch=2243; zero-gradient padding does not duplicate training contribution
- preflight loss=4.215424; peak GPU memory=21.76 GiB
- trainable parameters=587,427,044; exact list: `/home/yz/groundingLMM_official/checkpoints/phase5a3_legion_retrained/stage1_run/preflight.json`
- selector: no new free-generation selector; all three epochs retained, epoch 3 merged per frozen recipe

| Epoch | Train total | Train CE | Train BCE | Train Dice | Val total |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.227503 | 1.061510 | 0.031436 | 0.134557 | 1.100045 |
| 2 | 0.991655 | 0.850607 | 0.024633 | 0.116415 | 1.073763 |
| 3 | 0.883930 | 0.755873 | 0.021479 | 0.106578 | 1.068994 |

Merged `LEGION-retrained-LE`: `/home/yz/groundingLMM_official/checkpoints/phase5a3_legion_retrained/stage1_le`  
Canonical checkpoint SHA256: `350f83783d19ff44624a337f473c239951564bc83f81ce212e8313fb26621201`

## Stage 2 — Detection Classification

- initialization: `/home/yz/groundingLMM_official/checkpoints/phase5a3_legion_retrained/stage1_le`
- train: 17,672 = 8,836 Real (0) + 8,836 Fake (1)
- validation: 2,212 internal validation only
- epochs=3, lr=1e-3, cosine scheduler, batch=64
- trainable: prediction_head only (2,103,298 parameters)
- preflight loss=0.894531; peak GPU memory=18.35 GiB

| Epoch | Internal val accuracy | Internal val loss |
|---:|---:|---:|
| 1 | 0.986890 | 0.043604 |
| 2 | 0.988246 | 0.046938 |
| 3 | 0.988698 | 0.045128 |

Best internal-validation checkpoint: `/home/yz/groundingLMM_official/checkpoints/phase5a3_legion_retrained/stage2_cls/checkpoints/checkpoint-831`  
Best accuracy: `0.988698`  
`LEGION-retrained-CLS`: `/home/yz/groundingLMM_official/checkpoints/phase5a3_legion_retrained/stage2_cls/final_model`  
Canonical checkpoint SHA256: `1e2f6d6db4f98aab7f710daa4f38feff3595ac0a1cd1362335d3f49dbb9483cc`

## Dependency / launcher deviations

官方仓库保持未修改。外部 wrapper 仅修复：Stage 1 单 epoch hard break、随机有放回 HybridSegDataset、validation first-1000 截断、未使用数据集的缺失 prompt 常量；Stage 2 显式采用冻结标签 Real=0/Fake=1。模型核心、prompt、LoRA/loss/optimizer/scheduler recipe 未修改。

本阶段到此 STOP；未运行 official1000、LOKI、鲁棒性或 R1 对比。
