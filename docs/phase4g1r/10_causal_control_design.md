# Causal control design

## G1-C permutation 为什么无效

Dirichlet/evidential masses同时编码 class belief direction 与 evidence strength。G1-C 用 `replace_uncertainty` 重分配 ignorance 时，需要重标定 singleton mass；在 committed mass 极低或 tie 附近，数值 fallback/renormalization 可改变 argmax。因此所谓“只换 reliability”没有保持 source prediction 这个前提，正式 sensitivity statistics 在提交前失败关闭。

这不证明 reliability causal contribution 为 false，只证明原 intervention 不可识别。

## 新 intervention node

CSCU-LF 在 source posterior/refinement materialize 后另行输出：

\[
U_F\in[0,1]^{B\times1\times256\times256}.
\]

它只进入 `d_F^adapt=d_F^ECoLaF*U_F`。未来干预直接替换 `U_F` tensor，不重构 evidence、alpha、belief 或 source probability。

Formal utility permutation 的执行顺序必须是：

1. 固定并 hash `p_L,p_F,m_L,m_F` 与 aligned features；
2. 计算 matched `U_F`；
3. 用 seed 3407 做 image/spatial utility permutation；
4. bit-exact assert 所有 source predictions/features 不变；
5. assert outside support仍为 `U_F=0`、mass vacuous；
6. 仅重算 adapted discount/fusion；
7. 若任一 identity assert 失败，control fail-closed，不提交 causal sensitivity、不换 seed。

`UTILITY_INTERVENTION_IDENTIFIABLE=YES` 是 tensor-graph property，不是 causal contribution 已得到支持。

## Required ablations

Full 必须对比 no-cross-source、Intrinsic-Uncertainty PCERF、static、no-utility、no-conflict、utility permutation、no-F、no-L、matched-CLIP、cross-image、spatial-shuffle、vacuous/off exact P1。Full vs Intrinsic-Uncertainty PCERF 是 failure-driven redesign 的关键 arm。

Refinement、utility、conflict fusion三者分别有开关：只有 source prediction/refinement 固定后干预 utility，才可写 utility causal claim。完整 machine contract 见 `outputs/phase4g1r/causal_controls.json`。
