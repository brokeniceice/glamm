# Conditional utility target freeze

## Primary target

Native unit固定为 original-normalized 64×64 CLIP-support cell。使用 frozen、calibrated G1-C source posterior 的 binary pixel NLL/BCE：

\[
\ell_m=-y\log p_m-(1-y)\log(1-p_m),
\qquad
t_F=\sigma((\ell_L-\ell_F)/\tau).
\]

`UTILITY_TARGET_FROZEN=YES`。Dice仅作 future region diagnostic；hard winner不是 sole objective；GT仅在 target construction，CSCU-LF forward signature无 GT。

Future U supervision固定为 pre-sigmoid utility logit对 soft `t_F` 的 `BCEWithLogits`。先在每图 support cells内取 mean，再对 images等权 mean；不做 class/foreground/target reweighting，不加入Dice、fused segmentation或source refinement loss。

## Tau one-shot rule

只读取新 `UTILITY-FIT` 的 5,171 valid-G0 images，在 support 内统计 20,064,384 个 cell。结果前已固定：

\[
\tau=\frac{P90(|\ell_L-\ell_F|)}{\operatorname{logit}(0.95)}.
\]

得到：

`tau = 0.041720069924898906`

没有 tau candidates、sweep、utility correlation 或 IoU selection。delta 的 P10/P50/P90 分别为 -0.0398501 / 0.00140974 / 0.0643585；target P5/P50/P95 为 0.0799769 / 0.508447 / 0.968082。`t≤0.01` 为2.9858%，`t≥0.99` 为3.8850%，未退化为几乎全0/1；`0.45≤t≤0.55` 为50.447%，但其余近半 cells保留明确连续偏好。foreground/background target mean为0.554672/0.524588。

四态只作 diagnostic：both-correct 19,157,732；L-correct/F-wrong 86,958；L-wrong/F-correct 120,719；both-wrong 698,975。future 必须分别报告 U distribution，不能把 both-wrong解释为任一 source可靠。

完整 histogram、percentiles、source checkpoint/temperature hashes见 `outputs/phase4g1p/utility_target_manifest.json`。
