# Phase 6B.5 — RINE-on-C1 Implementation & Preflight

## 结论

`READY_FOR_TRAINING = YES`

RINE-on-C1 已按官方 RINE 的完整 Q1/TIE/Q2/classifier 与 `BCE + ξ·SupCon` 路径实现，并在当前冻结 CLIP ViT-L/14@336 上通过一个 balanced internal-TRAIN batch 的 forward/backward preflight。

本阶段没有 optimizer step、没有正式多 epoch 训练、没有读取 internal test 或任何 OOD benchmark、没有运行 AIDE 或 LLM fusion。internal validation 只做 manifest identity/count/hash 审计，没有读取 validation image。

## 1. 官方 RINE 身份与固定配置

| 项目 | 固定值 |
|---|---|
| official repo | `external/RINE_official` |
| remote | `https://github.com/mever-team/rine.git` |
| commit | `9b7fd5857cc205d0412be6aeee0d7611b95bd620` |
| checkout | detached HEAD |
| git status | clean |
| backbone family | `ViT-L/14`, hidden width 1024 |
| official configuration basis | released best 4-class ProGAN configuration |
| `q / nproj` | `2` |
| `D' / proj_dim` | `1024` |
| `ξ / factor` | `0.2` |
| dropout | `0.5`（PyTorch `nn.Dropout()` default，与官方一致） |
| loss | `BCEWithLogitsLoss(reduction="sum") + 0.2 × SupCon` |
| SupCon temperature/base temperature | `0.07 / 0.07` |

配置不是依据本项目 OOD 数值挑选。官方 [`scripts/best.py`](../external/RINE_official/scripts/best.py)调用 `best_configs` 重训其发布 best 配置；官方提交的 grid result `results/grid/ViT-L-14_4_0.2_2_1024_128_0.001_1.pickle` 与 [`src/utils.py`](../external/RINE_official/src/utils.py#L324-L343)共同固定 4-class released model 为 `q=2, D'=1024, ξ=0.2`。选择 4-class official setting 的原因是本项目 internal train 是多来源、异质内容训练集，而不是 RINE 的单一/两个 ProGAN object-class setting；这是训练前的 source-based preregistration，不涉及本项目 OOD 选择。

官方仓库在审计 import 时曾由 Python 自动生成两个 `__pycache__` 目录；它们由本阶段删除后再 detached checkout，最终 status clean。未修改任何官方源码。

## 2. 实现边界

新增项目侧文件：

- [`model/rine_on_c1.py`](../model/rine_on_c1.py)：HF CLIP hook adapter、官方 RINE head、官方 SupCon 与组合 loss。
- [`scripts/phase6b5_rine_preflight.py`](../scripts/phase6b5_rine_preflight.py)：固定数据/权重、防火墙、单 batch preflight 与证据落盘。

实际路径：

```text
Frozen current HF CLIP ViT-L/14@336
→ hook all 24 encoder.layers[i].layer_norm2
→ token index 0 from every block
→ [B,24,1024]
→ Q1: Dropout + 2 × (Linear 1024→1024 + ReLU + Dropout)
→ TIE: alpha [1,24,1024], softmax over block dimension
→ weighted block sum
→ Q2: Dropout + 2 × (Linear 1024→1024 + ReLU + Dropout)
→ classifier: 1024→1024→1024→1 with ReLU/Dropout
```

RINE head 的 module names、层顺序、默认 dropout、`alpha` 初始化、TIE softmax dimension 和单 binary logit 均逐项转录自官方 [`src/models.py`](../external/RINE_official/src/models.py#L19-L89)。项目 adapter 只负责 OpenAI CLIP 与 HF CLIP 的 tensor layout/模块命名桥接，没有改 RINE 结构。

## 3. HF CLIP 与官方 RINE hook 语义

### 官方行为

官方 RINE 遍历 `clip.visual.named_modules()`，hook 名称包含 `ln_2` 的所有模块；每个 OpenAI CLIP `ln_2` 输出为 sequence-first `[tokens,batch,channels]`。官方：

```python
torch.stack([h.output for h in hooks], dim=2)[0, :, :, :]
```

先取 token 0，再得到 `[batch,blocks,channels]`。

### 当前 HF 对应

HF `CLIPEncoderLayer.layer_norm2` 与 OpenAI CLIP `ResidualAttentionBlock.ln_2` 都位于 attention residual 之后、MLP 之前。HF 输出是 batch-first `[batch,tokens,channels]`，所以 adapter 对每层执行 `value[:,0,:]` 后沿 block 维 stack。

这里**没有**把 `output_hidden_states[-2]` 当成 RINE 的最后一个 hook。后者是 C1-Exact 的现有单层 feature；RINE 官方取得的是每个 block 内部 pre-MLP `ln_2` 输出，两者位置不同。

### 实测 shape 与逐值布局验证

| 检查 | 结果 |
|---|---:|
| Transformer block 数 | 24 |
| 每层完整 hook shape | `[8,577,1024]`，24/24 一致 |
| 每层 CLS shape | `[8,1024]` |
| RINE stacked input | `[8,24,1024]` |
| official sequence-first 重建 | `[8,24,1024]` |
| HF adapter vs official-layout max abs difference | `0.0` |

逐值差异为 0 证明 adapter 只进行了 layout 变换与同一 token 的选取，没有改变 hook 值。完整 24 层名称和 shape 在 `outputs/phase6b5_rine_preflight/block_hook_audit.json`。

## 4. Trainable parameter firewall

| 模块 | 参数张量数 | 参数量 | 状态 |
|---|---:|---:|---|
| TIE `alpha [1,24,1024]` | 1 | 24,576 | trainable |
| Q1（2 个 1024→1024 Linear） | 4 | 2,099,200 | trainable |
| Q2（2 个 1024→1024 Linear） | 4 | 2,099,200 | trainable |
| classifier（1024→1024→1024→1） | 6 | 2,100,225 | trainable |
| **合计** | **15** | **6,323,201** | **trainable** |
| CLIP ViT-L/14@336 | — | 303,507,456 | frozen |
| LLM / R1 / SAM / localization | — | 不载入此 wrapper | frozen / unreachable |

可训练参数量精确等于官方代码中 `q=2, D'=1024, 24 blocks` 的 `6,323,201`。所有 trainable name 都严格位于：

```text
rine.alpha
rine.proj1.*
rine.proj2.*
rine.head.*
```

非法 trainable 参数数为 0；CLIP 训练参数数为 0，backward 后 CLIP gradient 数也为 0。完整逐张量 manifest 在 `outputs/phase6b5_rine_preflight/trainable_parameters.json`。

## 5. 单 batch forward/backward

### 数据与标签

- input：internal TRAIN 的 8 张图，确定性取 4 Fake + 4 Real 并交替排列；这样每一类均有多个 positive pair，可真实验证 SupCon。
- Stage-2 frozen manifest 标签：`Real=1, Fake=0`。
- RINE/BCE fake-positive 标签：`Real=0, Fake=1`。
- 唯一转换：`rine_fake_label = 1 - manifest_label`。
- current C1 `CLIPImageProcessor`、RGB 与 336×336 preprocessing 保持不变。

### 输出和 loss

| 项目 | 实测值 |
|---|---:|
| batch | 8 |
| RINE logits | `[8,1]` |
| Q2 embedding | `[8,1024]` |
| multi-layer CLS | `[8,24,1024]` |
| BCE sum | `5.5380220413` |
| SupCon | `2.0641236305` |
| total = BCE + 0.2×SupCon | `5.9508466721` |
| loss finite | PASS |
| 15/15 trainable tensors gradient present | PASS |
| 15/15 gradients finite | PASS |
| CLIP gradients absent | PASS |
| peak CUDA allocation | `1,781,838,848` bytes（RTX A6000, `cuda:1`） |

本阶段只调用 `loss.backward()` 验证计算图，没有创建 optimizer、没有调用 `optimizer.step()`、没有发生训练更新。初始 RINE head 以 seed 3407 保存为 `outputs/phase6b5_rine_preflight/rine_head_initialized_seed3407.pt`，这只是初始化快照，不是 trained checkpoint。

- initialized-head SHA256：`6edc6f19bf9f4a61c05f38f53f88acfcdee52194143039435c45483687e8147a`
- preflight JSON SHA256：`8dbb36d64c2559a0199f1c818d9f99bc5dc2bfffc3b8356643bc1b6b9ddbbc15`

## 6. BCE 与 supervised contrastive loss 一致性

[`OfficialSupConLoss`](../model/rine_on_c1.py)逐语义转录官方 [`src/utils.py::SupConLoss`](../external/RINE_official/src/utils.py#L513-L608)：

- normalized Q2 embedding；
- 输入 shape `[batch,1,projection_dim]`；
- label-equality positive mask；
- self-contrast mask；
- temperature/base temperature 均为 0.07；
- 无 positive pair 时 denominator guard 与官方一致。

组合方式也与官方 [`train_one_experiment`](../external/RINE_official/src/utils.py#L175-L203)相同：BCE 使用 `reduction="sum"`，再加 `factor × SupCon`。本批 BCE、SupCon、total 均有限且均参与 backward。

## 7. Localization state invariance

RINE wrapper 只拥有：

```text
standalone frozen CLIPVisionModel
+ RINE classification head
```

它没有加载、引用或向 optimizer 暴露 LLM、R1 rectifier、SAM、mask decoder、`text_hidden_fcs`、generation 或 localization module。

preflight 前后对下列关键源码/checkpoint index 做 SHA256、size 与 mtime snapshot：

- `model/GLaMM.py`
- `model/SAM/build_sam.py`
- `model/sam_forensic_rectifier.py`
- `external/LEGION_official/model/Legion.py`
- C1-Exact checkpoint-554 index

before/after snapshot exact equal；localization trainable/optimizer parameters 均为空。证据见 `outputs/phase6b5_rine_preflight/localization_invariance.json`。

结论：**localization implementation、checkpoint identity 和运行路径完全未改变。**

## 8. C1-Exact compatibility

同一批图另外走原有 C1-Exact 路径：

```text
same frozen CLIP
→ output_hidden_states[-2][:,0]
→ original checkpoint-554 prediction head 1024→2048→2
```

| 检查 | 结果 |
|---|---:|
| head checkpoint | `phase6b1a_exact_stage2_control/checkpoints/checkpoint-554` |
| head tensor SHA256 | `9244192be7aa3d8eefefe20732553a21bfab208b0433b1c276545d1f1e4bcafe` |
| output shape | `[8,2]` |
| logits finite | PASS |
| 与冻结 internal-TRAIN cache prediction disagreement | `0` |
| max abs logit difference | `0.125` |

`0.125` 是在线 BF16 在本次 batch=8 与历史 cache batch=64 GEMM replay 下的 logit 数值差；8/8 argmax prediction 一致，且没有改 C1 feature/head/weights。该检查的结论限定为用户要求的“C1-Exact 仍可原样运行”，不把不同 batch shape 下的 BF16 输出声称为 bitwise identical。证据见 `outputs/phase6b5_rine_preflight/c1_exact_compatibility.json`。

## 9. 数据防火墙

| 数据 | 行为 |
|---|---|
| internal TRAIN | manifest 完整审计；只读取 8 张 preflight 图 |
| internal validation | manifest count/hash/uniqueness 审计；未读取图像 |
| internal test | 未访问 |
| AIGI-Holmes | 未访问 |
| GenImage | 未访问 |
| LOKI | 未访问 |
| RAISE / official1000 / 其他 OOD | 未访问 |

冻结 manifest：

| Split | N | SHA256 |
|---|---:|---|
| internal TRAIN | 17,672 | `ea46ca7d07c6e585911c757e24f8998c5e67783e4b47866a8feeae8333832f59` |
| internal validation | 2,212 | `15f668ee771fde740003f53e71f6af7f5e5e3bf18edcdba2774276edd8fe98dc` |

## 10. Launcher deviation

第一次 preflight 命令在 import 阶段因 `scripts/` 启动时项目根目录不在 `sys.path` 而退出：`ModuleNotFoundError: No module named 'model'`。当时尚未加载模型、manifest 或图像，也没有执行 forward/backward。修复仅在 repo-external preflight launcher 中显式加入项目根目录；RINE、CLIP、C1 或 localization 核心代码均未因此修改。修复后一次完整 preflight 通过。

CLIP loader 输出“text-tower weights 未被 `CLIPVisionModel` 使用”的提示，这是从完整 CLIP checkpoint 只实例化 vision 子模型的预期行为；vision weights 已正常加载，不是 missing vision weights。

## 11. Artifacts

```text
outputs/phase6b5_rine_preflight/
├── protocol.json
├── block_hook_audit.json
├── trainable_parameters.json
├── preflight.json
├── localization_invariance.json
├── c1_exact_compatibility.json
├── rine_head_initialized_seed3407.pt
└── status.json
```

最终 gate：

```text
official repo fixed and clean       PASS
24 block shapes                     PASS
HF/official hook semantics          PASS
trainable firewall                  PASS
single-batch forward/backward       PASS
BCE finite                          PASS
SupCon finite                       PASS
CLIP frozen/no gradients            PASS
localization state unchanged        PASS
C1-Exact still executable           PASS
internal-only firewall              PASS
formal training started             NO
OOD accessed                        NO

READY_FOR_TRAINING = YES
```

阶段状态：`COMPLETE_AND_STOPPED_BEFORE_FORMAL_TRAINING`。
