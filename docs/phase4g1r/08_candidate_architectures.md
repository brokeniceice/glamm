# Candidate architectures

## Candidate A / Primary proposal：CSCU-LF

全名：**Cross-Source Conditional Utility Late Fusion**。它采用 CMX 的联合 interaction 血统、保留 ECoLaF official conflict kernel，并把 `U_F(L,F)` 作为独立的 adapted discount variable。

### Tensor graph

```text
q_seg [B,256] ---- Linear/proj --------┐
S64 [B,256,64,64] -- 1x1 proj --------┼--> L64 [B,d,64,64]
z_L [B,1,256,256] -- down/proj -------┘

F24 [B,256,24,24] -- frozen geometry --> original-normalized F64 [B,256,64,64]
z_F24 [B,1,24,24] -- same geometry ----> z_F64 [B,1,64,64]
support --------------------------------> M64 [B,1,64,64]
F64,z_F64,M64 -- proj -----------------> Fctx64 [B,d,64,64]

L64,Fctx64
  -> CMX-style joint channel + spatial rectification in context branch only
  -> local/windowed context exchange
  -> [Lr,Fr,Lr*Fr,|Lr-Fr|,local cosine,p_L,p_F,conflict]
  -> explicit U_F [B,1,64,64] -> 256 grid, support-masked

optional separated R_L/R_F refinement heads -> source masses m_L,m_F
m_L,m_F -> official ECoLaF conflict d_E -> adapted d_F=d_E,F*U_F
         -> committed-mass discount -> official Dempster + DSmP -> fused mask

forensic absent/vacuous/off OR outside support -> exact z_L dispatch
invalid canonical G0 -> formal invalid policy unchanged
```

`U_F` 同时依赖 L/F interaction，绝不是 `MLP(F)`。局部同坐标 product/difference/cosine 使 spatial shuffle 在 forward 中必然换掉 correspondence；`q_seg/S64/z_L` 与 F 的 joint context 使 cross-image mismatch 可被感知。模型不保证一定学会正确响应，因此仍需 future corruption gate，但 sensing capacity 是结构性存在的。

### A1/A2/A3 判断

- A1（channel+spatial rectification）有成熟 CMX 支持，但 long-range semantic mismatch 能力有限；
- A2（cross-attention only）能交换 context，却不天然产生 identifiable utility，也可能受 global noisy correlation；
- A3（rectification + selective context exchange + late fusion）同时覆盖 local mismatch、semantic mismatch 和 P1-preserving late fusion。

因此 Primary 选 A3 的**完整机制族**，但 interaction 只作用于旁路 context/utility（以及独立可消融 refinement），不覆盖冻结 P1 representation。

### Strong/weak behavior

连续 `U_F` 加上 soft evidential fusion而非硬 router：strong-strong 可联合增强；L strong/F weak 由低 utility保护；L weak/F strong 由高 utility补偿；weak-weak 可由独立 `R_L/R_F` refinement 探索，不宣称 utility 本身能凭空产生正确像素。

### Complexity proposal（未冻结实现）

若 `d=64`、utility grid=64×64、window=7，projection + CM-FRM-like interaction + 一层 window context exchange + utility/refinement heads 预计约 `0.35–0.8M` trainable parameters。局部 attention 的主要量级约 `O(HW*w²*d)`，每方向约 `4096×49×64≈12.8M` dot-product operations，避免 4096-token global quadratic attention。该估计只用于可行性，不因参数少而选择 Primary；具体 d/window/heads 必须由 future authorization 先冻结。

## Candidate B / Secondary：AHBFR

AHBFR 在 `S64` 内使用 CMX-style channel/spatial calibration 生成 forensic-conditioned shift，再用 MAG norm cap：

\[
S'_{64}=S_{64}+\alpha\frac{\|S_{64}\|}{\|\Delta S\|+\epsilon}\Delta S,
\]

并继续通过原 SAM prompt/mask decoder。它直接回答“如何安全修改 representation”，空间 mismatch sensing成熟，但 utility 不独立、causal identifiability 弱，且重新引入 representation override 风险。因此保持 Secondary，不训练。

## Candidate C

Standalone Cross-Attention Conditional Fusion 不单列正式候选。cross-attention 可作为 A3 的 context exchange component，但若没有独立 `U_F` 与 adapted discount mapping，就无法回答 reliability/utility-aware claim，也无法做 clean intervention。
