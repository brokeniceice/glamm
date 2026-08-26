# Phase 4D-1R — Corrected Minimal Position-Aware Evidence Rerun

## 1. Executive Summary

- Image-specific utilization: **INCONCLUSIVE**.
- Spatial-specific utilization: **INCONCLUSIVE**.
- Forensic-specific utilization: **INCONCLUSIVE**.
- Reader optimization: **HEALTHY**.
- Position-aware transfer: **NOT_SUPPORTED**.
- Phase 4D-2 proposal gate: **NO**; Phase 4D-2 was not started.

修正 beta initialization 后，Reader 本体获得了稳定的非零梯度，且 image-specific residual 能以非零比例穿过 BF16 并改变 SAM continuous logits。因此 Phase 4D-1 的 optimization defect 已被修复。尽管如此，正式 train 与 validation IoU intervention 均没有建立 matched 优于 cross-image 或 spatial-shuffle 的正 CI，属于预注册 Outcome C：corrected minimal Reader 仍未学出 position-sensitive evidence use。

## 2. Single-Variable Protocol Verification

唯一改变变量为 `beta initialization: 0.0 -> 0.03`。`0.03` 来自 Phase 4D-1B 预注册 numerical viability rule，而非 validation IoU、matched-vs-shuffle、binary mask 或性能选择。

- Parent Reader init hash：`0eecf17b0140c78833493de7928de4492245462ff3093ba6d2cab6a659ab28df`
- Corrected two-arm init hash：`056cfdc19c95c27bcd71ffdccc4fa67e5e57bd852d7f2b5745f5c22dda588367`
- Frozen 2,048 train order hash：`0ad5c03a7c216bc54ab9f5c938581ab21a3d44ed0a046ee17371e8bdd3a98f3f`
- Formal endpoint：step512；无 checkpoint selector

架构、fixed 2D sine/cos position、P1/SAM/CLIP/adapter、query/SAM/evidence cache、训练样本与顺序、optimizer、LR、scheduler、loss、batch size、seed、evaluator、inverse geometry、threshold 0、validation N=1106、cross-image mapping 和 spatial-content permutation 均与 Phase 4D-1 相同。

## 3. Beta / Gradient Diagnostics

两臂均记录 step 0/64/128/256/384/512。Reader-body 与 query/key/value/output projection 梯度在训练阶段保持非零。

| Arm | Step512 Reader grad | Q grad | K grad | V grad | Output grad | Final beta |
|---|---:|---:|---:|---:|---:|---:|
| POS-CLIP-R | `4.718e-4` | `3.086e-5` | `3.404e-5` | `2.867e-4` | `3.712e-4` | `0.03135472` |
| POS-FORENSIC-R | `4.809e-4` | `2.022e-5` | `2.310e-5` | `2.950e-4` | `3.778e-4` | `0.03203794` |

结论：`READER_OPTIMIZATION: HEALTHY`。

## 4. Train Mechanism Diagnostics

POS-FORENSIC-R 在完整 2,048 train subset 上：

- matched−cross：mean `+0.000003143`，median `0`，95% CI `[-0.000005010, +0.000011586]`，W/T/L `118/1829/101`，Wilcoxon p `0.405444`。
- matched−shuffle：mean `-0.000000383`，median `0`，95% CI `[-0.000002577, +0.000001360]`，W/T/L `26/2005/17`，Wilcoxon p `0.491281`。

连续信号并非完全消失：

| Comparison | BF16 survival | Mean SAM logit absolute difference |
|---|---:|---:|
| matched vs cross-image | 64.31% | `0.0038202` |
| matched vs spatial-shuffle | 23.63% | `0.0008453` |
| matched vs zero | 86.43% | `0.0086125` |

这说明接口传输了微小的 image/spatial-specific continuous signal，但训练目标没有把它转化为可检测的 matched IoU 优势。

## 5. Validation Main Results

| Arm / condition | N | Mean FG IoU | Median FG IoU | Mean FG F1 |
|---|---:|---:|---:|---:|
| P1 / no Reader | 1106 | 0.148233 | 0.037966 | 0.210697 |
| POS-CLIP-R / matched | 1106 | 0.148227 | 0.037922 | 0.210687 |
| POS-CLIP-R / cross-image | 1106 | 0.148236 | 0.037916 | 0.210699 |
| POS-CLIP-R / spatial-shuffle | 1106 | 0.148233 | 0.037922 | 0.210694 |
| POS-CLIP-R / zero | 1106 | 0.148229 | 0.037861 | 0.210693 |
| POS-FORENSIC-R / matched | 1106 | 0.148239 | 0.037916 | 0.210702 |
| POS-FORENSIC-R / cross-image | 1106 | 0.148239 | 0.037677 | 0.210695 |
| POS-FORENSIC-R / spatial-shuffle | 1106 | 0.148238 | 0.037916 | 0.210699 |
| POS-FORENSIC-R / zero | 1106 | 0.148232 | 0.037861 | 0.210696 |

## 6. Matched vs Cross

POS-FORENSIC-R matched−cross-image：mean `+0.000000376`，median `0`，95% CI `[-0.000015785, +0.000015800]`，W/T/L `60/993/53`，Wilcoxon p `0.719193`。

CI lower bound 不大于 0，因此没有建立 image-specific utilization。

## 7. Matched vs Shuffle

POS-FORENSIC-R matched−spatial-content-shuffle：mean `+0.000001269`，median `0`，95% CI `[-0.000009099, +0.000012960]`，W/T/L `5/1091/10`，Wilcoxon p `0.649563`。

Shuffle 严格为 `F_shuffle = F[perm]` 后再加固定 `P_2D`；从未联合置换 `(F+P)`。CI lower bound 不大于 0，因此没有建立 spatial-specific utilization。

## 8. Forensic vs CLIP

POS-FORENSIC-R matched−POS-CLIP-R matched：mean `+0.000012442`，median `0`，95% CI `[-0.000007856, +0.000033667]`，W/T/L `89/931/86`，Wilcoxon p `0.430584`。

方向为正但 CI 跨 0，只能标记 `INCONCLUSIVE`，不能声称 forensic-specific advantage。

## 9. P1 Non-regression

POS-FORENSIC-R matched−P1 G0：mean `+0.000006080`，median `0`，95% CI `[-0.000018214, +0.000031557]`，W/T/L `122/844/140`，Wilcoxon p `0.673383`。

均值未低于 `-0.005` non-regression floor，因此不触发 `MECHANISM_EXISTS_BUT_NOT_USEFUL`；但这也不构成相对 P1 的有效提升。

## 10. Train-to-Validation Generalization

```text
CORRECTED_READER_DID_NOT_LEARN_POSITION_SENSITIVE_USE
```

Reader optimization 已正常，但 train matched≈cross/shuffle，validation 同样 matched≈cross/shuffle。因此不是“train 学会但未泛化”，而是这个 corrected minimal Reader 在当前目标和预算下没有学出 position-sensitive evidence utilization。

该结论只适用于这一 matched minimal Reader，不能外推为 forensic information 无法迁移，也不能证明所有 position-aware 或 forensic spatial architecture 无效。

## 11. Final Decision

```text
IMAGE_SPECIFIC_UTILIZATION: INCONCLUSIVE
SPATIAL_SPECIFIC_UTILIZATION: INCONCLUSIVE
FORENSIC_SPECIFIC_UTILIZATION: INCONCLUSIVE
READER_OPTIMIZATION: HEALTHY
POSITION_AWARE_TRANSFER: NOT_SUPPORTED
FORENSIC_TRANSFER: INCONCLUSIVE
PROCEED_TO_PHASE_4D_2: NO
```

正式 endpoint 为 step512，没有 checkpoint selection。Internal test 与 official1000 保持封存，Phase 4D-2 未启动。
