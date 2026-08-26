# Phase 4E-0.5 — TF-FDG Hardened Architecture

## 1. 冻结定义

TF-FDG 是一个 **K=4 coordinated-query、geometry-aware semantic/forensic dual-path decoder**。TF branch 仅训练期存在；P1、SAM image encoder、CLIP 与 selected 4C-A adapter 均冻结。当前实现规格见 `model/tf_fdg.py`，未接入任何正式训练入口。

## 2. Training graph 与 tensor contract

| Tensor | Shape | Coordinate system | Storage/compute dtype | Source | 状态 | Gradient / loss |
|---|---|---|---|---|---|---|
| `h_TF` | `[B,4096]` | causal token sequence；`[SEG]` 前一 state | BF16 input / FP32 LN | frozen P1 canonical TF | frozen input | Stage T 仅向 teacher QG 下游传梯度；P1 无梯度 |
| `h_G0` | `[B,4096]` | canonical autonomous trajectory | BF16 / FP32 LN | frozen P1 G0 | frozen input | Stage S 向 student QG 传梯度；P1 无梯度 |
| `S64` | `[B,256,64,64]` | SAM ResizeLongestSide1024 + right/bottom pad；每 token 有 original-normalized `(x,y)` | frozen cache BF16 | SAM image encoder | frozen | 无 encoder gradient |
| `S32` | `[B,256,32,32]` | 由 `S64` learned 1×1 + deterministic area down-resolution；非独立 scale | BF16，loss FP32 | semantic pyramid | trainable | mask/KD → semantic pyramid |
| `F24` | `[B,256,24,24]` | CLIP shortest336 + center crop；每 token 映射 original-normalized `(x,y)`，含 FOV validity | frozen cache BF16 | selected 4C-A adapter | frozen evidence | 不更新 adapter |
| `F24'` | `[B,256,24,24]` | 与 `F24` 相同 | BF16 | forensic pyramid 1×1 | trainable | mask/KD → forensic pyramid |
| `F32_coord` | `[B,256,32,32]` | 按 original coordinate 四邻域 inverse-distance resample；crop 外为 invalid/zero | BF16 | deterministic coordinate resampler | 无参数 | gradient 回到 forensic pyramid |
| `S_rect` | `[B,1024,256]` | `S32` original-coordinate tokens | BF16 attention / FP32 audit | rectification block | trainable | mask/KD → Q/K/V/out/projection/gamma |
| `Q_T/Q_S` | `[B,4,256]` | exchangeable grounding slots；无固定 ordinal identity | BF16 | QueryGenerator | T trainable / S trainable | mask 与启用的 KD |
| `A_sem^l` | `[B,8,4,1024]` | layer `l` query→S32 | FP32 loss view | decoder semantic attention | trainable | 只记录；Full 不对其做 KD |
| `A_for^l` | `[B,8,4,576]` | layer `l` query→F24 original-coordinate lattice | FP32 loss view | decoder forensic attention | trainable | teacher qualification TRUE 时，layers 3/4 做 KD |
| `D^2,D^4` | each `[B,4,256]` | matched query slot state | BF16 / normalized FP32 loss | decoder layers 2、4 | trainable | `L_feature` |
| `Z_slot` | `[B,4,32,32]` | S32 common decoder geometry | FP32 loss | query mask embedding × fused mask feature | trainable | slot/union mask loss、assignment、KD |
| `Z_union` | `[B,1,32,32]` | 同一 decoder geometry | FP32 | probabilistic OR | deterministic | `L_union`、`L_logit` |

所有 24→32 / 24↔64 交互通过 original-coordinate map；禁止把 `interpolate(24,64)` 当作几何对齐。`48×48` 若未来只由 24×24 upsample 获得，只能称 **multi-resolution forensic decoder feature**，不能称 multi-scale forensic representation。本冻结规格只使用原始 `F24` 和 coordinate-resampled `F32_coord`。

## 3. QueryGenerator 与初始化

```text
LayerNorm(4096)
→ Linear(4096,1024)
→ GELU
→ Linear(1024,4×256)
→ reshape [B,4,256]
```

Stage T 完成、按 validation TF mean FG IoU 选择 teacher 后，冻结：`teacher_QG + teacher_semantic_pyramid + teacher_forensic_pyramid + teacher_rectification + teacher_decoder + teacher_mask_head`。

Student 初始化逐 tensor exact copy：

```text
student QueryGenerator = copy(teacher QueryGenerator)
student pyramids       = copy(teacher pyramids)
student rectification  = copy(teacher rectification)
student decoder/head   = copy(teacher decoder/head)
```

随后 teacher input=`h_TF`，student input=`h_G0`；teacher `requires_grad=False` 且输出 detach。synthetic audit 已验证 copy exact 与 teacher gradient 全空。

## 4. Slot semantics、supervision 与 aggregation

slot 是 **exchangeable evidence-grounding slot**，不绑定“左上/第一个 phrase”等 ordinal identity。

- `M≤4`：official ref polygon mask 与 slot 通过 `DiceCost+BCECost` Hungarian；matched slot 用 BCE+Dice，未匹配 slot 监督 empty；同时用 union loss。
- `M>4`（train 718/8,836）：不创造 region bundle；仅 union supervision + collapse audit。
- union 固定为 probabilistic OR：`p_union=1-Π(1-sigmoid(z_k))`，使用 `log1p/expm1` 稳定实现；禁止 weighted average。

active slot 与 `SLOT_COLLAPSE` population rule 见 `phase4e05_slot_target_availability.md`，训练前冻结，不能 post-hoc 修改。

## 5. Geometry contract

### CLIP token → original

原图 `(H,W)` 先按 shortest edge 336 resize 为 `(Hc,Wc)`，center crop box `(top,left,top+336,left+336)`。24×24 token center `(r,c)` 映射：

```text
x_orig_norm = (left + (c+0.5)*336/24) / Wc
y_orig_norm = (top  + (r+0.5)*336/24) / Hc
```

crop 外没有 forensic token，不补造 evidence。

### SAM token → original

scale=`1024/max(H,W)`，resize `(Hs,Ws)` 后右/下 pad。64×64 token center先映射到 1024 input；落在 `x<Ws,y<Hs` 才 valid：

```text
x_orig_norm = ((c+0.5)*1024/64) / Ws
y_orig_norm = ((r+0.5)*1024/64) / Hs
```

32×32 坐标由相同 original-coordinate rule 产生。position encoding、cross-attention locality bias、resampling 与 attention KD 均使用此 coordinate，不混用 tensor index。

synthetic geometry（左上、右下、中央小区域、横/纵窄条、wide/tall center）通过：

```text
CLIP_TO_ORIGINAL_ERROR: 0.013594 <= 0.05
SAM_TO_ORIGINAL_ERROR: 0.007366 <= 0.02
CLIP_SAM_ALIGNMENT_ERROR: 0.013279 <= 0.05
```

误差为 normalized centroid quantization error；完整 case 见 `outputs/phase4e05_tf_fdg_hardening/geometry_audit.json`。

## 6. Cross-Attentive Semantic Rectification

在 S32/F24 level：

```text
Q = Wq(LN(S) + PE(c_S))
K = Wk(LN(F) + PE(c_F))
V = Wv(F)
B_geo(i,j) = -||c_S_i-c_F_j||² / (2*0.25²)
A = softmax(QKᵀ/sqrt(d_head) + B_geo + validity_mask)
R = Wo(A V)
S_rect = S + support_S * gamma * Projection(R)
```

`support_S=0` 表示该 SAM location 落在 CLIP center-crop FOV 外。gamma 不 zero-init。使用前两条 frozen train-Fake SAM/4C-A feature cache、不看 label/IoU 的 scale rule，候选 `0.01/0.03/0.05/0.1` 中选择最小满足 median residual/base norm `[0.005,0.10]` 且 BF16 survival `≥0.80` 的值：

```text
gamma_init = 0.01
median residual/base norm = 0.018952
BF16 token survival = 1.0
```

mask-only backward 的 gradient norm：Q `5.72e-5`、K `6.06e-5`、V `3.43e-4`、attention out `6.26e-4`、semantic projection `1.09e-3`、gamma `4.93e-3`，全部非零。

## 7. Teacher qualification

Teacher selector 仍仅为 epoch 0–5 的 validation TF mean FG IoU。选定后才做 qualification；cross/shuffle、slot diversity、student、official1000、internal test 均不参与 selector。

### Mask capability

比较 selected Teacher TF 与 frozen P1 canonical TF，validation Fake direct batch=1，报告 paired delta/CI/W-T-L：

- `ADEQUATE`：mean delta `≥-0.010` 且无 collapse/invalid output；
- `WEAK`：`-0.030≤delta<-0.010`；
- `FAILED`：delta `<-0.030`，或 nonfinite/catastrophic mask collapse。

margin 在执行前冻结，仅用于 teacher 是否值得蒸馏，不是 architecture performance selector。

### Forensic utilization

selected teacher 固定后评测 matched/cross-image/spatial-content-shuffle/zero：

- `TRUE`：paired 95% CI lower `>0`；
- `FALSE`：CI upper `≤0`；
- `INCONCLUSIVE`：其余。

分别得到 `TEACHER_IMAGE_SPECIFIC_USE`（matched−cross）与 `TEACHER_SPATIAL_SPECIFIC_USE`（matched−shuffle）。shuffle 只打乱 feature content，original-coordinate lattice 固定。

### Loss enable matrix

| Qualification | Student action |
|---|---|
| mask ADEQUATE + spatial TRUE | 启用全部 KD |
| mask ADEQUATE/WEAK，但 spatial FALSE/INCONCLUSIVE | assignment 不含 attention cost；`L_attention=0`；保留 relation/feature/logit（WEAK 时 logit 降为 0）并记录原因 |
| mask FAILED | 不开始 Full KD student；报告 teacher failure，不搜索新架构；`-teacher entirely` topology 仍可作为已定义 baseline，但不能冒充 TF-FDG Full |

## 8. Teacher–student correspondence

每图每对 slot 的 detached cost：

```text
C_ij = 1.0 * DiceCost(sigmoid(ZS_i), sigmoid(ZT_j))
     + 1.0 * mean_abs_logit(ZS_i, ZT_j)
     + λ_attn * JS(A_for,S_i, A_for,T_j)
```

只有 teacher spatial use=`TRUE` 时 `λ_attn=0.25`，否则为 0。cost 不含 query-relation loss，避免 correspondence 循环定义。Hungarian 得 permutation `π` 后才计算 KD。

## 9. KD losses

```text
L_total = 1.0 L_mask
        + 0.20 L_relation
        + w_attn L_attention
        + 0.25 L_feature
        + w_logit L_logit
```

- `L_mask`：M≤4 时 `L_slot + L_union`；overflow 时只有 `L_union`。
- `L_relation`：permutation 后 Q L2-normalize，`G=QQᵀ [B,4,4]`，SmoothL1；不做 pointwise Q cosine。
- `L_attention`：仅 decoder layers 3/4 的 forensic attention；先按 original coordinate 核对同 grid，再取 `log(clamp(P_attn))` 作为等价 pre-softmax score，`softmax(score/T)` 后 KL，`T=2`。禁止对已归一化 probability 直接再次 softmax。
- `L_feature`：layers 2/4 同构 query feature `[B,4,256]` channel/L2 normalize 后 L1；维度相同，不加 1×1 adapter。
- `L_logit`：同一 32×32 geometry 的 continuous union logits，temperature soft BCE；不 threshold、不 inverse-to-original。

一次 train-only numerical correction 后，weighted gradient/mask-gradient：relation `1.66e-4`、attention `2.34e-3`、feature `4.69e-2`、logit `3.21e-4`，均位于 `[1e-5,1e3]`。正式训练后不再调整系数。

## 10. Inference graph

```text
image
 ├─ frozen P1 canonical G0 → h_G0 [B,4096]
 │                              └─ student QG → Q [B,4,256]
 ├─ frozen SAM → S64 → semantic pyramid → S32
 └─ frozen CLIP + frozen 4C-A → F24 → forensic pyramid
              original-coordinate rectification + dual-path decoder
                         → Z_slot → probabilistic OR → Z_union → canonical inverse geometry
```

inference 不出现 TF teacher、authoritative phrase、GT mask、teacher feature 或 oracle evidence。

## 11. 参数与复杂度审计

isolated specification 的精确 trainable parameter：

| Group | Params |
|---|---:|
| QueryGenerator | 5,253,120 |
| semantic pyramid | 65,792 |
| forensic pyramid | 65,792 |
| rectification | 329,985 |
| four-layer decoder | 5,268,480 |
| mask feature/head | 262,912 |
| **Student total** | **11,246,081** |

Stage S 同时持有一个同规模 frozen teacher snapshot，但 optimizer 只注册 student。正式集成后必须重新报告真实 params、FLOPs、peak memory；这里不是性能结果。

