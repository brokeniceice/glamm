# ECoLaF extension review

Primary source：[ECoLaF, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Deregnaucourt_A_Conflict-Guided_Evidential_Multimodal_Fusion_for_Semantic_Segmentation_WACV_2025_paper.html)，[official code](https://github.com/deregnaucourtlucas/ECoLaF)。

## 保留的原生成熟机制

ECoLaF 让各 modality 先产生 pixel evidential masses，再根据专家间 conflict 做 adaptive discount，最后用 Dempster-Shafer late fusion；它直接面向 dense semantic segmentation，并报告 missing/sensor-failure robustness。G1-C 已完成 official kernel parity，因此以下保持：mass convention、conflict computation、adaptive discount、Dempster combination、DSmP、vacuous modality handling。

## 明确标记的 adapted extension

Primary 提议不改 official conflict kernel，而在其输出之后加入独立 conditional utility：

\[
d_F^{adapt}(x,y)=d_F^{ECoLaF}(x,y)\,U_F(x,y),\qquad 0\le U_F\le1.
\]

然后仅对 forensic singleton committed mass 施加 `d_F^adapt`，余量回到 ignorance mass，再进入原 Dempster/DSmP。Language 作为 P1 anchor 保持 `U_L=1`；其 ECoLaF conflict discount 是否保留是 future protocol 必须固定的公式 arm，不能由 validation 选择。

该组合必须称为 **ECoLaF-style adapted conditional discount**，不能称为原生 ECoLaF。原论文支持 conflict-guided late fusion，不直接支持 language-conditioned utility。

## Identity 与四种强弱情形

- absent/vacuous/off：不进入 adapted kernel，bit-exact dispatch `z_L`；
- outside CLIP support：forensic mass `[0,0,1]`，`U_F=0`；
- strong-strong：`U_F` 高且 conflict 可管理时两源共同形成 fused mass；
- L强/F弱：低 utility/conflict discount 保护 P1；
- L弱/F强：高 utility 允许 forensic compensation；
- weak-weak：融合不强制选边，独立 refinement arm仍可探索，但不得与 utility claim 混合。

G1-C 的 forensic QMF PASS 是保留 effective weight 审计的理由，不是保留 intrinsic uncertainty gate 的理由。
