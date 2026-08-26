# Phase 4D-1 — Minimal Position-Aware Forensic Evidence Test

## 1. Executive Summary

- 正确图像利用：**INCONCLUSIVE**。
- 正确空间位置利用：**INCONCLUSIVE**。
- forensic feature 相对 CLIP：**INCONCLUSIVE**。
- minimal recipe optimization：**INCONCLUSIVE**。
- position-aware transfer：**INCONCLUSIVE**。
- 是否建议 Phase 4D-2：**NO**。

本实验只检验固定二维位置编码、2,048 train Fake、512 steps、单一 seed 的最小 Reader。失败不能外推为 forensic information 不可迁移。

## 2. Preflight

- Feature shape：`256×24×24`；position：固定不可训练 `576×256` 2D sine/cosine。
- Shared Reader init hash：`0eecf17b0140c78833493de7928de4492245462ff3093ba6d2cab6a659ab28df`。
- 两 arm 均从同一个 `reader_init.pt` 加载；sample subset、顺序、optimizer、scheduler 与 step 数一致。
- Train subset：2,048；hash `0ad5c03a7c216bc54ab9f5c938581ab21a3d44ed0a046ee17371e8bdd3a98f3f`；mask-area quartile 每组 512。
- Scale gate：`PASS`。
- Internal test 与 official1000 未访问。

## 3. Training Diagnostics

训练日志位于 `phase4d1_training_log.csv`。两个 arm 都固定运行到 step 512；step 256 仅为诊断，不参与 endpoint 选择。最终 beta：POS-CLIP `-0.003036`，POS-FORENSIC `-0.002370`。

只读 BF16 interface audit 显示：POS-CLIP/POS-FORENSIC 的 float32 residual norm 均值分别为 `0.026260` / `0.018242`；matched-vs-shuffle q 差异 norm 均值仅 `0.000582` / `0.000475`。转为冻结 SAM 接口使用的 BF16 后，只有 `49/2048` 与 `37/2048` 样本保留任一元素差异；最终二值 mask 仍全部相同。这支持“minimal recipe 未建立足够强的位置敏感交互”，不改变预注册 gate。

## 4. Main Results

| Arm / condition | N | Mean FG IoU | Median FG IoU | Mean FG F1 |
|---|---:|---:|---:|---:|
| P1 | 1106 | 0.148233 | 0.037966 | 0.210697 |
| pos_clip / matched | 1106 | 0.148235 | 0.037966 | 0.210698 |
| pos_clip / cross_image | 1106 | 0.148234 | 0.037966 | 0.210698 |
| pos_clip / spatial_shuffle | 1106 | 0.148235 | 0.037966 | 0.210698 |
| pos_clip / zero | 1106 | 0.148234 | 0.037966 | 0.210697 |
| pos_forensic / matched | 1106 | 0.148234 | 0.037966 | 0.210697 |
| pos_forensic / cross_image | 1106 | 0.148233 | 0.037966 | 0.210697 |
| pos_forensic / spatial_shuffle | 1106 | 0.148234 | 0.037966 | 0.210697 |
| pos_forensic / zero | 1106 | 0.148234 | 0.037966 | 0.210697 |

## 5. Image-Specific Test

POS-FORENSIC matched−cross-image：mean `+0.000000`，median `+0.000000`，95% CI `[-0.000000,+0.000001]`，W/T/L `1/1104/1`，Wilcoxon p `0.654721`。

## 6. Spatial-Specific Test

POS-FORENSIC matched−spatial-content-shuffle：mean `+0.000000`，median `+0.000000`，95% CI `[+0.000000,+0.000000]`，W/T/L `0/1106/0`，Wilcoxon p `1`。Shuffle 只置换 `F` content；固定 `P_2D` lattice 未被置换。

## 7. Forensic-Specific Test

POS-FORENSIC matched−POS-CLIP matched：mean `-0.000001`，median `+0.000000`，95% CI `[-0.000004,+0.000000]`，W/T/L `0/1103/3`，Wilcoxon p `0.108809`。

## 8. Train vs Validation Diagnostic

- Train POS-FORENSIC matched−cross：mean `-0.000000`，95% CI `[-0.000000,+0.000000]`，W/T/L `1/2045/2`。
- Train POS-FORENSIC matched−shuffle：mean `0`，95% CI `[0,0]`，W/T/L `0/2048/0`。
- 归因：`OPTIMIZATION_OR_INTERFACE_INCONCLUSIVE`。
- POS-FORENSIC matched−P1 validation mean IoU：`+0.0000002407`；未触发 `-0.005` non-regression floor。

## 9. Decision

```text
IMAGE_SPECIFIC_UTILIZATION: INCONCLUSIVE
SPATIAL_SPECIFIC_UTILIZATION: INCONCLUSIVE
FORENSIC_SPECIFIC_UTILIZATION: INCONCLUSIVE
MINIMAL_RECIPE_OPTIMIZATION: INCONCLUSIVE
POSITION_AWARE_TRANSFER: INCONCLUSIVE
PROCEED_TO_PHASE_4D_2: NO
```

`MINIMAL_RECIPE_RESULT: OPTIMIZATION_OR_INTERFACE_INCONCLUSIVE`

本阶段到此停止；没有自动启动 Phase 4D-2。Mask target 仍是官方 polygons 派生的 per-image all-reference visible/explainable artifact union，不是 pseudo-mask，也不是完整 pixel-perfect forgery GT。
