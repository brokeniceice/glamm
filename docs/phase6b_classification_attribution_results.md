# Phase 6B — Classification Attribution / Minimal Improvement

## Outcome

Phase 6B 已在冻结 internal TRAIN/validation 上完成，所有新模型都只是 frozen-feature classification head。没有更新 CLIP、P1/R1、LLM/LoRA、forensic arm、SAM、rectifier 或现有 classifier；没有访问 Internal test、Official1000 或 external benchmark，也没有训练 C3。

结论：**CLIP CLS 是主要有效 representation；C1-L 相对 C1-S 的大头容量增益很小。F24 GAP 明确含有独立 Real/Fake signal，但显著弱于 C0 与 capacity-matched CLIP CLS，不支持进入正式 C3 fusion。下一步候选冻结为 C1-L，但本阶段不自动做 external confirmation。**

## Protocol

- data：冻结 TRAIN 17,672（8,836 Real + 8,836 Fake）；validation 2,212（1,106 + 1,106），原 manifest 顺序、无 resplit；Fake=1。
- feature：`openai/clip-vit-large-patch14-336` revision `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`，BF16 frozen penultimate hidden token 0；同一次 forward 的 patch tokens 经 frozen Phase4C-A selected forensic adapter 后 GAP 得 F24 `[256]`。
- integrity：TRAIN/validation extraction failure 均为 0；两 feature 共用 sample ID、label、image 与 order。CLIP parameter hash `ea25ce94579902eb0a94c9638c0277360f2b93cc156eb20f2fc4a481653afc17`；forensic checkpoint SHA256 `725dd44e0c867360e7a963eb13d8312187fff185849bd52de6b78d4b27a078f7`；R1 SHA256 `9b38ef1e62c86c74287895e681ec8ebb96b724e3bae52622d52b61bca7ddf5a5`。
- heads：C1-L `1024→2048→2`（2,103,298）；C1-S `1024→128→2`（131,458）；C2 `256→512→2`（132,610）。
- optimizer：AdamW，LR candidates `1e-4/3e-4/1e-3`，weight decay `1e-4`，10 epochs，batch 512，5% linear warmup + cosine decay。
- seeds：3407/3408/3409；每 seed 以 validation ROC-AUC 最大选择，tie 依次为 Accuracy、earlier epoch、smaller LR。
- 所有 operating-point metrics 固定 threshold 0.5；ECE 为 15 个 equal-width bins。完整预注册协议见 `outputs/phase6b_classification_attribution/protocol.json`。

## Main internal-validation results

多 seed 数值为 mean ± sample std。C0 是历史 generation-derived architecture reference，只保留 Accuracy，不能当作同 regime 单变量控制。

| Arm | Feature | Head params | Acc | Balanced Acc | ROC-AUC | Fake Recall | TNR | FPR | F1 | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | LLM `[CLS]` | existing 8,194 | 0.986438 (historical) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| C1-L | CLIP CLS | 2,103,298 | 0.986890 ± 0.000452 | 0.986890 ± 0.000452 | 0.998988 ± 0.000017 | 0.988547 ± 0.001381 | 0.985232 ± 0.001044 | 0.014768 ± 0.001044 | 0.986911 ± 0.000457 | 0.010203 ± 0.000018 | 0.005375 ± 0.000679 |
| C1-S | CLIP CLS | 131,458 | 0.985986 ± 0.001196 | 0.985986 ± 0.001196 | 0.998865 ± 0.000051 | 0.985835 ± 0.003764 | 0.986136 ± 0.001381 | 0.013864 ± 0.001381 | 0.985981 ± 0.001231 | 0.010809 ± 0.000201 | 0.007237 ± 0.000513 |
| C2 | F24 GAP | 132,610 | 0.900995 ± 0.003588 | 0.900995 ± 0.003588 | 0.963895 ± 0.000834 | 0.881555 ± 0.003132 | 0.920434 ± 0.004143 | 0.079566 ± 0.004143 | 0.899033 ± 0.003607 | 0.073420 ± 0.000971 | 0.030838 ± 0.000539 |

Selected hyperparameters:

| Arm | Per-seed selected LR/epoch |
|---|---|
| C1-L | seed 3407: lr=0.0003, epoch=4; seed 3408: lr=0.0003, epoch=5; seed 3409: lr=0.0003, epoch=6 |
| C1-S | seed 3407: lr=0.001, epoch=4; seed 3408: lr=0.001, epoch=4; seed 3409: lr=0.001, epoch=6 |
| C2 | seed 3407: lr=0.001, epoch=10; seed 3408: lr=0.001, epoch=10; seed 3409: lr=0.001, epoch=10 |
| LP-CLIP | seed 3407: lr=0.001, epoch=10; seed 3408: lr=0.001, epoch=10; seed 3409: lr=0.001, epoch=9 |
| LP-F24 | seed 3407: lr=0.001, epoch=10; seed 3408: lr=0.001, epoch=10; seed 3409: lr=0.001, epoch=10 |

## Attribution

| Contrast | Δ Accuracy | Δ ROC-AUC | Interpretation |
|---|---:|---:|---|
| C1-L − C1-S | +0.000904 | +0.000123 | 同 CLIP feature；大 head 只有很小正增益，不是主要来源 |
| C1-S − C2 | +0.084991 | +0.034970 | 近参数量下 CLIP global CLS 明显优于 F24 GAP |
| C1-L − C0 | +0.000452 | N/A | C1-L internal Accuracy 略高，但 C0 regime 不同，仅架构参考 |
| C1-S − C0 | -0.000452 | N/A | 一次错误量级的轻微降低，仍显示强 CLIP signal |
| C2 − C0 | -0.085443 | N/A | F24 不接近现有正式 classifier |

## Feature diagnostics and linear probes

| Feature | Normalized centroid L2 | Within cosine Real | Within cosine Fake | Between cosine | Within−between Real/Fake |
|---|---:|---:|---:|---:|---:|
| CLIP CLS | 0.247398 | 0.598144 | 0.597521 | 0.567593 | 0.030551 / 0.029927 |
| F24 GAP | 0.141959 | 0.907526 | 0.929186 | 0.908353 | -0.000828 / 0.020832 |

| Diagnostic probe | Acc | ROC-AUC | Interpretation |
|---|---:|---:|---|
| CLIP CLS → Linear | 0.979355 ± 0.001453 | 0.997995 ± 0.000238 | 线性可分性很强；非线性大头不是 CLIP 有效的必要条件 |
| F24 GAP → Linear | 0.811935 ± 0.010972 | 0.890521 ± 0.010124 | 明显高于随机，证明有独立 signal；但远弱于 CLIP |

Centroid/cosine 是 representation geometry diagnostics，不参与 checkpoint selector。

## Required decisions

1. **CLIP CLS 是否值得替代 current LLM classifier？** `GO_FOR_FREEZE_CONFIRMATION`：C1-L internal Accuracy/ROC-AUC 强且三 seed 稳定，但尚未做 external confirmation，不能宣布最终替代。
2. **C1-L 优势有多少由 head capacity 解释？** 很少：相对 C1-S 仅 ΔAcc `+0.000904`、ΔAUC `+0.000123`；主要 signal 已存在于 CLIP CLS。
3. **F24 是否包含独立 Real/Fake signal？** 是；C2 AUC `0.963895`，linear probe AUC `0.890521`。但作为正式 classifier 为 `NO-GO / WEAK`，因为明显低于 C0/C1。
4. **是否值得进入 C3 fusion？** **NO-GO at Phase6B exit**。C2 未达到 C0 附近，额外 fusion 容量会掩盖 attribution。
5. **下一步冻结哪个 candidate？** **C1-L**，作为后续用户授权下 external generalization confirmation 的唯一首选；C1-S 保留 capacity ablation，不替代主候选。

C2 和 LP-F24 的三个 seed 均在预注册 epoch 10 边界被选中，因此这里的 NO-GO 严格限定为“在本阶段冻结预算下不进入 C3/正式 classifier”。按 hyperparameter firewall，不因看到边界结果而单独延长 F24 训练。

## Firewall and STOP

`results.json` 明确记录：Internal test / Official1000 / external access 均为 false，threshold sweep=false，C3 trained=false。Phase 6B 到此 **STOP**，不自动进入 external classification evaluation。
