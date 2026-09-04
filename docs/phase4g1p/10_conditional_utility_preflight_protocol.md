# Frozen Conditional-Utility Validity Preflight protocol

本文件只冻结下一阶段；当前未执行 fitting/calibration/audit。

## Populations and fitting boundary

- UTILITY-FIT：5,258 total / 5,171 valid；仅 valid可训练 U branch。
- UTILITY-CAL：1,127 / 1,108 valid；只允许对预冻结 utility calibration scalar/formula操作，具体优化器必须在未来授权中明确，否则不得校准。
- UTILITY-AUDIT：1,126 / 1,108 valid；唯一 formal read，一旦进入不得修改 model、tau、loss、formula、split、seed或threshold。
- 旧 G1-C AUDIT、development validation、internal test、official1000禁止。

## Frozen future FIT/CAL execution semantics

本阶段只冻结、未执行：

- trainable：371,803-parameter context/interaction/U branch；source heads与全部 experts frozen；
- loss：soft-target `BCEWithLogits(a_U,t_F)`，每图support mean后images等权；无target/class weighting；
- optimizer：AdamW，lr=1e-4，weight decay=1e-4，batch=8，10 epochs，grad clip=1，seed3407，无scheduler/early stopping；
- checkpoint：epoch10 final only；intermediate仅crash recovery，禁止按指标选epoch；
- UTILITY-CAL：冻结model，只拟合一个正 scalar `T_U=softplus(tau_U)+1e-6`，作用于utility logits；init1，deterministic LBFGS，同一image-balanced soft BCE；无grid/sweep；
- UTILITY-AUDIT：冻结branch与T_U，formal read exactly once。

这些数值沿用已验证的G1-C小头preflight budget，以minimum-sufficient protocol一次冻结；不依据 utility correlation或localization结果选择。执行仍需新的人工授权。

## Primary validity gate

Primary unit固定为 **image-clustered pixel relation**：pool support pixels计算 Spearman(`U_F`,`ell_L-ell_F`)，95% CI用 image-cluster bootstrap（整图有放回，再保留图内全部 support cells）。要求 point>0 且 CI lower>0。Pixel-level relation是primary；同时报告每图 mean U vs mean relative loss的image aggregation，但不得切换 primary。

`UTILITY_RELATIVE_LOSS_RELATION`：方向正确但 CI跨0为 INCONCLUSIVE；point≤0为 FAIL。

四态分别报告 U mean/distribution；hard correctness不参与 primary target。

## Corruption gates

Matched、cyclic cross-image、seed3407 spatial shuffle。P1 language不变，outside support vacuous。对每图 mean U做 paired image bootstrap 10,000 repeats/seed3407：

- cross−matched mean<0 且95% CI upper<0；
- shuffle−matched mean<0 且95% CI upper<0。

两者都必须 PASS；方向正确但 CI跨0为 INCONCLUSIVE，point≥0为 FAIL。

## QMF conditional gate

Forensic effective weight vs calibrated forensic source NLL：Pearson和Spearman都必须 point<0 且各自95% CI upper<0。QMF是 validity criterion，不要求 intrinsic uncertainty。

## Utility causal control

先要求 source predictions/masses/features bit-exact。之后 seed3407 image/spatial permutation只替换U。在 L/F disagreement pixels：两种 permutation mean absolute fused-FG probability change均≥0.01，且至少一种 source-dominance assignment change≥5%。Identity失败则 fail-closed，不计算sensitivity。

## Overall stop

还必须 `VACUOUS_EXACT_P1/NO_ORACLE_LEAKAGE/INVALID_G0_POLICY=PASS`。任一 FAIL则 preflight FAIL；无FAIL但有INCONCLUSIVE则INCONCLUSIVE；全部PASS也只能申请 formal training，不能自动进入。禁止用 fused IoU、G0/Phrase/TF、threshold/tau/width/window/head/loss/checkpoint sweep选择结果。

Formal Intrinsic-Uncertainty PCERF必须作为matched reference，核心比较是 CSCU-LF vs Intrinsic-PCERF 的 utility validity、cross/shuffle response和causal identifiability。

`CONDITIONAL_UTILITY_PREFLIGHT_PROTOCOL_FROZEN=YES`。
