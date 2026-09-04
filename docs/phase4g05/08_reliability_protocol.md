# Reliability calibration 与 validity protocol

## 状态

本文件冻结未来 G1-C protocol，不执行 fitting。`RELIABILITY_CALIBRATION_PROTOCOL_FROZEN = YES`。

## Deployment-time inputs

允许：`q_seg,z_L,S64,valid-SEG,F24,z_F24,support,prediction disagreement,conflict`。禁止：GT、TF/Phrase identity、authoritative phrase、GT mask/polygon、evaluation condition label。GT只可在 TRAIN-FIT loss或TRAIN-CAL/AUDIT target侧出现，绝不进入 forward/routing。

## Calibration

1. TRAIN-FIT拟合 head 后冻结其参数。
2. TRAIN-CAL分别优化 language/forensic 的正值 scalar concentration temperature：`e'_m=e_m/T_m`，`T_m=softplus(tau_m)+1e-6`。
3. 只用预注册 NLL objective与 deterministic LBFGS；不做 temperature grid/sweep，不看 IoU选择 temperature。
4. calibration 前后均报告 NLL/Brier/ECE；若 NLL变差，不以 validation另选方案，报告 failure。
5. ECE固定15个 `[0,1]` 等宽 bin；high-confidence error固定为 predicted correctness probability ≥0.9。

## Pixel audit

分别报告 all、foreground、background与boundary。boundary固定为 original-normalized 256 mask的3×3 morphological gradient。每个 source报告 posterior NLL、Brier、ECE、reliability diagram、high-confidence error rate；uncertainty target为 source pixel error。

## Image/QMF audit

每图聚合 mean uncertainty/evidence strength/discount，并报告其与 FG IoU及error的 Spearman、Pearson。QMF criterion固定为 `corr(weight_m, per-image source NLL_m) < 0`；同时给 image-level bootstrap 95% CI，不用单个 p-value替代 effect。

## Corruption 与 causal controls

固定 matched、cross-image、spatial-shuffle、zero/vacuous。cross/shuffle不得改变 P1；outside crop始终vacuous。报告 forensic uncertainty、discount、conflict与routing decision相对matched的 paired变化。

reliability permutation只在 TRAIN-AUDIT内按固定 seed 3407打乱 image-level与spatial reliability，保持 source predictions不变。必须显著改变 fusion/routing行为；future Full还必须优于 Static matched fusion或Constant reliability至少一项，否则 `RELIABILITY_MECHANISM_SUPPORTED=NO`。

## G1-C stopping rule

G1-C只回答 reliability是否携带与 source error/fusion utility稳定相关的信息，不回答 full PCERF性能。若 source calibration失效、QMF负相关不成立、corruption response方向不稳或permutation不改变decision，则停止，不进入 full training，不改用 validation挑 signal。

