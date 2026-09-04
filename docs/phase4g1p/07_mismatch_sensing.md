# Synthetic mismatch sensing

本测试只问 forward graph 是否具备 sensing capacity，不要求随机初始化的 `U_F` 方向正确，不构成 performance evidence。

## Spatial shuffle

固定 L，把同一 synthetic F 的24×24 cells按 seed 3407 permutation。以下 comparison paths全部非恒定：

| Tensor | mean absolute difference |
|---|---:|
| `Lr*Fr` | 0.689595 |
| `|Lr-Fr|` | 0.352383 |
| local cosine | 0.0551671 |
| context-exchange output | 0.689801 |
| U pre-head comparison input | 0.503831 |

因此 spatial correspondence、local attention key/value与explicit comparison都真实依赖位置，`SPATIAL_MISMATCH_SENSING_CAPACITY=YES`。

## Cross image

在B=2 synthetic pair中固定 `S64/q_seg/z_L`，交换 `F24/z_F24`。Language `L64` bit-exact不变；`Fr`、comparison、utility pre-head分别产生0.691069、0.504558、0.354852 mean absolute difference。

因此 joint graph在结构上能感知 `L_i,F_j` 与 `L_i,F_i` 不同，`CROSS_IMAGE_MISMATCH_SENSING_CAPACITY=YES`。这不预言 future learned utility magnitude或方向；formal direction只能由新 UTILITY-AUDIT gate判定。

机器结果见 `outputs/phase4g1p/mismatch_sensing.json`。
