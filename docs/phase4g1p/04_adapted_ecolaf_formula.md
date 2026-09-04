# Adapted ECoLaF formula freeze

Official ECoLaF conflict computation、discount kernel、Dempster combination与DSmP保持不改。CSCU-LF 明确标记为 **ADAPTED EXTENSION**。

令 official discounts 为 `d_L^E,d_F^E`，explicit conditional utility为 `U_F`：

\[
d_F^A=U_Fd_F^E,
\qquad
d_L^A=1-U_F(1-d_L^E).
\]

第二式是 zero semantics companion：当 `U_F=0` 时，不仅 forensic singleton归零，也撤销由 forensic conflict 对 language造成的间接 discount，保证 strict language-source-only；当 `U_F=1` 时两路 discounts与 official ECoLaF 完全相同。

对每个 expert `m`：

\[
b'_{m,k}=d_m^A b_{m,k},qquad
u'_m=1-\sum_k b'_{m,k}.
\]

再将 adapted masses送入不变的 official Dempster/DSmP。实现对 `U=0`、`U=1` 使用显式 identity selection，避免浮点代数破坏 bit parity。

`d_F^A=d_F^E U_F`满足指令；companion language restoration使“utility=0”真正表示完全关闭 forensic-induced effect，而不是残留 conflict-discounted language mass。reference implementation为 `model/csculf.py::adapted_ecolaf_fuse`，parity结果见 `outputs/phase4g1p/adapted_ecolaf_parity.json`。
