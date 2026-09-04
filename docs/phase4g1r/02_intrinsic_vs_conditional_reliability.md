# Intrinsic confidence 与 conditional utility

## 概念分离

Intrinsic confidence 是单源函数：

\[
R_F=R(F_{24},z_{F24}).
\]

它描述 forensic branch 自身的 evidence strength/confidence。G1-C 已表明，这个量不能稳定代理 localization error，也不能感知 matched、cross-image、spatial-shuffle 与当前语言轨迹的关系。

Conditional utility 必须是联合函数：

\[
U_F(x,y)=U(S_{64},q_{seg},z_L,F_{24},z_{F24},M_F;x,y),
\]

其中 `M_F` 是 geometry-derived CLIP support。它回答的是：在当前 P1/SAM context 下，此位置的 forensic evidence 是否值得进入融合。

## 必须由 forward graph 保证的差异

1. `U_F` 同时消费 L 与 F；禁止 `sigmoid(MLP(F))`。
2. 空间对应通过 original-normalized alignment 和局部同坐标 interaction 显式进入；shuffle 会改变成对的 L-F 输入。
3. `q_seg`、`S64` 与 `z_L` 共同定义当前 language-conditioned object/spatial context；cross-image F 无法仅凭自身保持高 compatibility。
4. `U_F` 是独立 tensor node，source predictions 在 utility intervention 前完成，因此未来可固定 prediction/feature，只打乱 `U_F`。
5. outside support 强制 `U_F=0` 且 forensic mass vacuous；不是 background evidence。

## 不是 winner-take-all

`U_F` 是连续的 forensic contribution control，不输出硬专家 ID。强-强时允许 Dempster/late fusion 联合增强；L 强/F 弱时压低 F；L 弱/F 强时允许补偿；弱-弱仍可由明确分离的 cross-source refinement branch 探索联合修正。refinement 与 utility 必须分别消融，不能把所有增益都归为 routing。

因此 `PCERF_EVIDENTIAL_REFINEMENT_DIRECTION=OPEN`：G1-C 否定的是 intrinsic uncertainty 作为主要 utility proxy，不是否定所有 evidential refinement。
