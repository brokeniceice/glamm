# Phase 6B.6 — RINE-on-C1 Controlled Training

## 结论

在冻结的 internal validation 上，`RINE-on-C1` 明确优于 `C1-Exact`：

- Accuracy：`0.987342 → 0.994123`（`+0.006781`，即 `+0.6781` 个百分点）；
- ROC-AUC：`0.998822 → 0.999845`（`+0.001024`）；
- Fake recall 与 TNR 同时提高，FPR 从 `0.017179` 降至 `0.009946`；
- 错误样本数从 28 降到 13；同样本配对中 RINE-only correct 为 21，C1-only correct 为 6，exact McNemar `p=0.005925`。

因此，本阶段支持：

```text
RINE multi-layer representation + official Q1/TIE/Q2/head + SupCon package
在 internal validation 上成立。
```

但不能进一步声称“增益由 TIE 单独造成”：本阶段按要求没有做 intermediate/TIE/SupCon ablation，而且一轮后 TIE 权重相对随机初始化变化很小。当前证据证明的是完整 RINE-on-C1 package 超过 C1-Exact，不是每个子模块的独立因果贡献。

阶段已停止，**没有运行任何 OOD benchmark**。

## 1. 冻结协议

训练前先生成并冻结：

```text
outputs/phase6b6_rine_training/protocol.json
SHA256 768c1473d69c81c5fff0bb72bb2ebdf9d8048d5508c6117c5927f4bbd5f8c809
```

训练完成后复核该 SHA256 未变化。

### 模型与 loss

| 项目 | 冻结值 |
|---|---|
| official RINE commit | `9b7fd5857cc205d0412be6aeee0d7611b95bd620` |
| official repo final status | detached HEAD, clean |
| backbone | current CLIP ViT-L/14@336 |
| CLIP | frozen |
| block hooks | 24 个 `encoder.layers[i].layer_norm2` CLS |
| `q` | 2 |
| `D'` | 1024 |
| TIE | `alpha[1,24,1024]`, layer dimension softmax |
| dropout | 0.5 |
| loss | BCE sum + `0.2 × SupCon` |
| SupCon temperature | 0.07 |
| trainable | Q1 / TIE alpha / Q2 / classifier，仅 6,323,201 参数 |
| frozen/unreachable | LLM / R1 / SAM / localization |

初始化严格载入 Phase 6B.5 seed-3407 snapshot：

```text
outputs/phase6b5_rine_preflight/rine_head_initialized_seed3407.pt
SHA256 6edc6f19bf9f4a61c05f38f53f88acfcdee52194143039435c45483687e8147a
```

### Training recipe

| 项目 | 值 |
|---|---:|
| epochs | 1 |
| per-device batch | 128 |
| world size | 1 |
| gradient accumulation | 1 |
| effective global batch | 128 |
| train samples/epoch | 17,672，恰好完整覆盖一次 |
| optimizer steps | 139 |
| optimizer | `torch.optim.Adam` |
| learning rate | `1e-3` |
| weight decay | 0 |
| scheduler | none |
| shuffle | deterministic, seed 3407 |
| drop last | false |

以上 `batch=128 / Adam / lr=1e-3 / one epoch / no active LR reduction` 沿用官方 released 4-class best recipe。为保持 `C1-Exact single-layer vs RINE multi-layer` 的 controlled interface，预处理继续使用 Phase 6B.5 冻结的 current C1 RGB + `CLIPImageProcessor@336`，没有引入官方 RINE 的 224 random augmentation。该差异、seed 差异和数据替换均在训练前写入 protocol，不是结果后修改。

## 2. 数据与 selector

没有 resplit：

| Split | N | Manifest SHA256 |
|---|---:|---|
| internal TRAIN | 17,672 | `ea46ca7d07c6e585911c757e24f8998c5e67783e4b47866a8feeae8333832f59` |
| internal validation | 2,212 | `15f668ee771fde740003f53e71f6af7f5e5e3bf18edcdba2774276edd8fe98dc` |

RINE 官方训练代码固定 epoch 后报告 validation，并没有可直接复用的单一 internal-validation checkpoint selector；其公开 best-config 过程反而汇总了论文测试生成器，不能移植为本项目 selector。因此训练前预注册：

```text
max validation ROC-AUC
tie → Accuracy
tie → earlier epoch
threshold = 0.5
Fake = positive
```

官方 released recipe 只有一轮，所以唯一候选 epoch 1 被选择，没有 epoch 搜索或结果后改 selector。

## 3. Internal-validation 主结果

| Model | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1-Exact | 0.987342 | 0.998822 | 0.991863 | 0.982821 | 0.017179 | 0.987399 | 1097 | 1087 | 19 | 9 |
| **RINE-on-C1** | **0.994123** | **0.999845** | **0.998192** | **0.990054** | **0.009946** | **0.994147** | **1104** | **1095** | **11** | **2** |
| **RINE − C1** | **+0.006781** | **+0.001024** | **+0.006329** | **+0.007233** | **−0.007233** | **+0.006748** | +7 | +8 | −8 | −7 |

该比较满足：同一个 frozen CLIP、同一个 RGB/336 preprocessing、相同 internal train/validation population、固定 threshold=0.5、相同 Fake-positive metric convention。不同之处是两种预注册的 architecture-specific head 与训练 recipe：C1-Exact 是单层 CLS MLP/LEGION Stage-2 recipe；RINE 是多层 CLS + Q1/TIE/Q2/head + SupCon/official RINE recipe。因此结论应称为**两个完整方法的 controlled comparison**，不能缩写成“只改一个 tensor，其余优化完全相同”。

### Paired comparison

| 状态 | 样本数 |
|---|---:|
| both correct | 2,178 |
| C1-only correct | 6 |
| RINE-only correct | 21 |
| both wrong | 7 |
| prediction disagreement | 27 |

两侧 sample IDs 和 labels exact aligned。对 6 vs 21 个方向性 discordant cases 做 two-sided exact McNemar/binomial test：

```text
p = 0.0059246123
```

这支持 internal validation 上的配对改善；它不是 OOD 泛化证据，也不替代后续冻结 checkpoint 后的独立 external confirmation。

## 4. Loss

官方 RINE 的 BCE 是 batch sum，而 SupCon 是 batch mean，所以同时报告原始官方组合量和可解释的归一化分量：

| Split | BCE sum | BCE/image | SupCon sample-weighted mean | official total mean/batch | Batches |
|---|---:|---:|---:|---:|---:|
| TRAIN | 1685.248643 | 0.095363 | 4.532328 | 13.026471 | 139 |
| validation | 39.078534 | 0.017667 | 4.889123 | 3.138204 | 18 |

其中每批实际反向目标严格为：

```text
BCEWithLogitsLoss(reduction="sum") + 0.2 × SupCon
```

训练全程 loss 和 15/15 trainable parameter gradients 有限；CLIP gradient count 始终为 0。没有临时切换为 BCE-only。

## 5. TIE layer-weight statistics

TIE 对每个 1024-D channel 分别在 24 层做 softmax，不是每层单一标量。selected epoch 统计：

| 统计 | 值 |
|---|---:|
| weight matrix | `[24,1024]` |
| per-channel sum max abs error | `2.3842e-7` |
| global min / max | `0.000394 / 0.544090` |
| mean entropy across channels | `2.739187` |
| mean effective layer count `exp(entropy)` | `15.6766 / 24` |
| top layers by mean weight | `8, 17, 4, 10, 24` |

### Per-layer mean weight

均匀层权重为 `1/24 = 0.041667`。表中的 winner 是在多少个 feature channels 上该层权重最大。

| Layer | Mean weight | Std | Winning channels |
|---:|---:|---:|---:|
| 1 | 0.040636 | 0.049973 | 45 |
| 2 | 0.039447 | 0.039323 | 41 |
| 3 | 0.042626 | 0.047493 | 51 |
| 4 | 0.043601 | 0.051831 | 44 |
| 5 | 0.042154 | 0.048304 | 35 |
| 6 | 0.041479 | 0.046663 | 52 |
| 7 | 0.039942 | 0.047737 | 42 |
| **8** | **0.044231** | 0.050248 | 47 |
| 9 | 0.037945 | 0.039568 | 31 |
| **10** | **0.043383** | 0.047303 | 44 |
| 11 | 0.042240 | 0.044597 | 46 |
| 12 | 0.042231 | 0.046345 | 50 |
| 13 | 0.041423 | 0.045477 | 41 |
| 14 | 0.040328 | 0.040025 | 33 |
| 15 | 0.041332 | 0.044358 | 33 |
| 16 | 0.040664 | 0.045809 | 47 |
| **17** | **0.044047** | 0.046286 | 44 |
| 18 | 0.040809 | 0.044849 | 44 |
| 19 | 0.042082 | 0.044771 | 37 |
| 20 | 0.042336 | 0.050599 | 53 |
| 21 | 0.041170 | 0.044281 | 37 |
| 22 | 0.041102 | 0.044520 | 41 |
| 23 | 0.041626 | 0.046052 | 41 |
| **24** | **0.043166** | 0.051296 | 45 |

### TIE 训练前后变化

| 统计 | Initial | Selected epoch 1 |
|---|---:|---:|
| mean entropy | 2.738879 | 2.739187 |
| mean effective layers | 15.6725 | 15.6766 |
| top-5 layer order | 8,17,4,10,24 | 8,17,4,10,24 |

- `alpha` mean absolute change：`0.006882`
- `alpha` max absolute change：`0.051015`
- TIE weight mean absolute change：`0.0002746`
- winner layer 改变的 channels：`13 / 1024`

解释：权重是 channel-specific，个别 channel 已高度偏向某层（最大 0.544），但 layer-level mean 仍接近均匀；而且大部分分布特征来自随机初始化，一轮训练只轻微调整 alpha。不能仅凭 top-5 顺序声称第 8/17 层具有稳定的取证因果重要性，也不能把本阶段增益主要归因于 TIE 学出了强层选择。

## 6. Checkpoint selection 与身份

| Artifact | SHA256 |
|---|---|
| epoch-1 checkpoint file | `26b33e3f9c25f2b3c33798a0513a5763216a18299d8871b87fe8fbd5c7a77557` |
| selected checkpoint file | `5286b05c82416e3a11d067b1f449b566c39360ffe7133499d09aecd0fcb3b562` |
| RINE tensor state（两者相同） | `f59124fa6716da5487f251d258d7e49bfba16fd1fcc443bed53397108fb9fada` |
| validation predictions | `fe4ba3a43b7a521c5635dbef62037cff053820eb0676833fe2b7474748673973` |
| results JSON | `8ed606ebfa168a2be15572936dc3e961ff014af625d3a2c51731f7cbb449c5d1` |

两个 checkpoint 文件 SHA 不同是因为 `selected_checkpoint.pt` 被重新序列化；逐 key tensor state exact equal，且 tensor-state SHA 完全相同。selected epoch 为 `1`。

## 7. Runtime 与 invariance

| 项目 | 值 |
|---|---:|
| GPU | NVIDIA RTX A6000, `cuda:1` |
| train + validation + finalization | 179.32 s |
| peak CUDA allocation | 11,249,029,632 bytes |
| CLIP gradient count | 0 |
| localization source hashes before/after | exact equal |
| protocol hash before/after | exact equal |
| official RINE repo after run | fixed commit, clean |

以下路径在训练前后 SHA256 exact equal：

- `model/GLaMM.py`
- `model/SAM/build_sam.py`
- `model/sam_forensic_rectifier.py`
- `external/LEGION_official/model/Legion.py`

RINE training process没有加载 LLM、R1、SAM 或 localization model；optimizer 只收到 6,323,201 个 RINE-head 参数。

## 8. Firewall

```text
internal TRAIN read             YES
internal validation read        YES
resplit                         NO
internal test                   NO
OOD benchmark                   NO
threshold tuning                NO
layer/q/D'/xi change            NO
BCE-only switch                 NO
AIDE                            NO
LLM fusion                      NO
localization change             NO
```

## 9. Artifacts

```text
outputs/phase6b6_rine_training/
├── protocol.json
├── training.log
├── checkpoint_epoch1.pt
├── selected_checkpoint.pt
├── validation_predictions_epoch1.pt
├── results.json
├── verification.json
└── status.json
```

最终判断：

```text
RINE_ON_C1_INTERNAL_VALIDATION_GAIN = YES
TIE_SPECIFIC_CAUSAL_ATTRIBUTION = NOT_TESTED
ALLOW_OOD_AUTOMATICALLY = NO
```

阶段状态：`COMPLETE_AND_STOPPED_BEFORE_OOD`。
