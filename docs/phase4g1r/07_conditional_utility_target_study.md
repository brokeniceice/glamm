# Conditional utility target study（只设计）

GT 只允许出现在 future training target 侧，绝不进入 inference graph。当前不冻结最终 loss、temperature、weight 或 threshold。

## Option 1：soft relative pixel loss

令两源用同一 pixel unit 与 frozen definition 计算 `ell_L,ell_F`：

\[
t_F=\sigma((\ell_L-\ell_F)/\tau).
\]

相比 hard winner，它保留相对差异；但 `tau`、source calibration 与 loss family 会改变 target。BCE 对 region size直接，Dice 是 image/region coupling，不自然产生独立 pixel utility；二者不可互换后择优。该 option 最接近首选研究方向，但仍是 `PROMISING_NOT_FROZEN`。

## Option 2：source correctness pair

四态 target 适合诊断：

- L correct / F wrong：应保护 language；
- L wrong / F correct：应允许 forensic compensation；
- both correct：不应强迫二选一，可鼓励联合低损失或保持中高 utility；
- both wrong：没有正确专家可选，utility 不能伪装成 correctness，应该交给 refinement/joint-failure auxiliary analysis。

硬 correctness 依赖 classification threshold，边界像素易抖动，不能作为唯一 target。

## Option 3：counterfactual marginal gain

定义加入 F 前后的固定 reference-fusion loss difference，最贴近“F 是否帮助最终结果”；但 target 依赖 reference fusion，容易形成 circular supervision。只有先冻结 reference formula，才能构造。

## Loss family 判断

- pixel BCE/NLL：与 QMF source loss及 conditional comparison最直接；
- Dice：适合作为 region-level secondary diagnostic，不宜单独定义 pixel utility；
- calibration loss：约束 `U_F` 数值含义，但不是 localization utility 本身；
- hard BCE winner：会鼓励 winner-take-all，应拒绝为 sole objective。

未来 freeze review 应选择一个 primary continuous target、一个四态 diagnostic，并预先决定 both-correct/both-wrong 的处理。不得依据 development validation 在 BCE/Dice/temperature/loss weight间选择。

机器可读选项见 `outputs/phase4g1r/conditional_utility_options.json`。
