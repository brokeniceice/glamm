# CSCU-LF interaction architecture freeze

## Minimum-sufficient faithful configuration

结果前一次性冻结：

| 项 | 冻结值 | 理由 |
|---|---|---|
| projection width `d` | 64 | 延续 G1-R feasibility proposal；足够承载 4-head interaction，避免无依据 width screen |
| local window `w` | 7×7 | 覆盖 64-grid 的局部 correspondence，复杂度远低于 global 4096-token attention |
| heads | 4 | 每 head 16 channels；minimum multi-head configuration |
| blocks | 1 | 保留一次完整 context exchange，避免堆叠造成不必要 optimization/attribution risk |
| normalization | GroupNorm(8)；q projection 用 LayerNorm | batch-size independent；适合 future small-batch preflight |
| activation | GELU | 与已有 context/head lineage一致 |
| CMX residual scale | `lambda_C=lambda_S=0.5` | CMX 原论文默认值；不做 sweep |
| topology | bidirectional channel+spatial rectification → bidirectional local exchange | 完整 A3，不缩成 scalar gate |
| sharing | joint pooled/spatial descriptors；direction-specific gates/Q/K/V/output | 保留 joint dependency 与 asymmetric source roles |
| refinement | `REFINEMENT_INCLUDED=NO` | 下阶段只检验 conditional utility hypothesis |

## Context construction

Language：

```text
S64 -- Conv1x1 256→64 + GN + GELU --┐
q_seg -- Linear 256→64 + LN + GELU -- broadcast 64x64 --┼ concat 144ch
z_L -- deterministic bilinear 256→64 -- Conv1x1 1→16 + GN + GELU --┘
concat -- Conv3x3 144→64 + GN + GELU --> L64 [B,64,64,64]
```

Forensic：`F24,z_F24` 先通过 Phase4G-0.5 已验证的 original-normalized cell-center mapping 得到 `F64,z_F64,M64`。`F64→64ch`、`z_F64→16ch`，再与 `M64` concat 为81ch，经 `3×3→64 + GN + GELU` 得 `Fctx64`。outside support 乘 `M64=0`，不能成为 background evidence。

所有 projection/interaction/U head 用 FP32 且 future trainable；输入 source tensors detached。P1/SAM/CLIP/4C-A、G1-C source evidential heads与 temperature全部 frozen。

## CMX-style interaction

Joint average/max pooled L/F descriptors产生两个 direction-specific 64-channel gates；joint channel-average/max maps 经7×7 conv产生两个 spatial gates：

\[
L_r=L+0.5W^C_L\odot F+0.5W^S_L\odot F,
\]
\[
F_r=F+0.5W^C_F\odot L+0.5W^S_F\odot L.
\]

随后一层7×7 local cross-attention双向交换 context。L-query→F-key/value 严格 support-mask；F-query output也乘 support。该机制的 spatial-shuffle sensitivity来自同坐标/邻域 key-value真实变化，不依赖 condition label。

冻结实现见 `model/csculf.py`；配置与参数量见 `outputs/phase4g1p/architecture_manifest.json`。
