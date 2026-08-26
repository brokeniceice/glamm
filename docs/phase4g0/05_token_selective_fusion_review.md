# Token-Level Selective Fusion 综述

## TokenFusion 的原始机制

TokenFusion 为每层 token feature `e_m^l ∈ R^{N×C}` 学习 importance score：

`s^l(e_m^l) = MLP(e_m^l) ∈ [0,1]^N`。

训练同时包含各模态 task loss 与 token score 的 L1 pruning loss。低于阈值的 token 不继续保留原值，而由同位置的另一模态投影 token 替换：

`e_m^l = e_m^l I[s>=θ] + Proj_{m'}(e_m^l) I[s<θ]`。

Residual Positional Alignment 保留被替换 token 的原 positional embedding，并在首层后停止位置嵌入梯度。论文在图像转换、RGB-D segmentation 与 image/point-cloud detection 上验证，且单模态 transformer 主体大体保留。

## 对本项目的适配

自然映射是：

- SAM spatial tokens：`S64 → [B,4096,256]`；
- forensic tokens：`F24 → [B,576,256]`，先按原图归一化坐标投影/重采样到 4096 个位置；
- token score 输入需包含 SAM token、`q_seg` broadcast 与 P1 coarse mask logit，才能从“视觉 token importance”适配为“language-conditioned spatial reliability”。

完整图：

```text
h_G0 [B,4096] --frozen text_hidden_fcs--> q_seg [B,256]
S64 [B,256,64,64] ---------------------> S [B,4096,256]
F24 [B,256,24,24] --coord projection---> F64 [B,4096,256]
q_seg + S + P1 coarse logit --score--> s [B,4096,1]
S/F64 --position-preserving selective substitution--> S_sel [B,256,64,64]
S_sel + q_seg --original frozen SAM decoder--> mask
```

## 优点

- token 粒度能保留局部冲突，而不是整图 scalar。
- 位置对应有原论文专门机制。
- 有明确的 score/pruning auxiliary loss，gate collapse 不是完全无约束。
- 保留原 SAM decoder，因果消融清晰。

## 关键不匹配

- TokenFusion 的“uninformative token”由单流任务重要性定义，不等于 language localization reliability。
- 原机制是硬替换，可能在可靠 P1 token 上造成不可逆破坏；不具 MAG 式位移上界。
- P1 的 S64 本身是 image embedding，不直接携带 q_seg；language condition 只在 decoder 交互，因此 score 必须大幅适配。
- 24×24 CLIP center-crop 与 64×64 SAM resize/pad 的支持范围不同，边缘 token 不能伪造对应。

## 结论

TokenFusion 是有成熟依据的 **Secondary/备选机制**，但不选为 Phase 4G-1 Primary。原因不是复杂度，而是核心 reliability 语义不匹配和硬替换风险。若未来使用，必须单独预注册：hard substitution、soft bounded enhancement、score auxiliary loss 和 coordinate support mask，不能用一个弱化版 scalar gate代表该方向。

