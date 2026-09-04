# Future conditional-utility preflight proposal（未授权）

## Stage 0：implementation/invariance preflight

只有新授权后才可实现。先做 synthetic tensor graph，不训练：shape/dtype/finite、geometry/support、official ECoLaF parity、adapted discount formula parity、gradient isolation、parameter/FLOP manifest、vacuous/off exact P1、invalid-G0 policy、utility intervention identity。任何失败即停止。

## Population firewall

G1-C TRAIN-AUDIT 已消费且本阶段用于 redesign，未来不得再作为 architecture/utility audit population。新 manifest 必须在授权前冻结，并只从原 G1-C TRAIN-FIT/TRAIN-CAL pool 构造 mutually exclusive conditional-utility FIT/CAL/AUDIT；原 G1-C AUDIT、development validation、internal test、official1000 全排除。具体 count、hash、valid-G0 数必须在任何 fitting 前固定。

## Conditional-utility validity（future protocol must freeze target first）

建议预注册以下 gate；它们不是本阶段执行结果：

1. `UTILITY_RELATIVE_LOSS_RELATION`：以预冻结相对 utility target `loss_L-loss_F` 为方向，Spearman(`U_F`, target) point>0 且 image-cluster bootstrap 95% CI lower>0；pixel 与 image aggregation全部报告，primary unit须预先指定。
2. `QMF_CONDITIONAL_RELATION`：forensic effective weight 与 calibrated forensic source NLL 的 Pearson、Spearman均 point<0 且 95% CI upper<0。
3. `CROSS_IMAGE_UTILITY_RESPONSE`：cross-image `U_F-matched U_F` image-paired mean<0 且 95% CI upper<0。
4. `SPATIAL_SHUFFLE_UTILITY_RESPONSE`：spatial-shuffle 同样 mean<0 且 upper<0。
5. `UTILITY_INTERVENTION_IDENTITY`：source feature/prediction hashes bit-exact；否则 fail-closed。
6. `UTILITY_PERMUTATION_SENSITIVITY`：在 L/F disagreement pixels，image 与 spatial 两种 permutation 的 mean absolute fused-FG probability change 都≥0.01，且至少一种 source-dominance change≥5%。只有 identity gate先 PASS 才计算。
7. `VACUOUS_EXACT_RECOVERY`、`NO_ORACLE_LEAKAGE`、`INVALID_G0_POLICY` 全 PASS。

有 FAIL 则 next-stage conditional-utility mechanism FAIL；无 FAIL 但 CI 跨零则 INCONCLUSIVE。只有全部 PASS 才可申请 formal training，仍不得自动进入。不得使用 IoU、Phrase/TF、development validation、threshold/loss sweep重新选择 signal。

## Training balance

保持 `TRAINING_BALANCE_AUDIT_REQUIRED=YES`、`INTERVENTION_REQUIRED=UNRESOLVED`。G1-C gradient norm约10倍差异，但 loss/strength/saturation不支持 optimization dominance；未来只记录 gradient norm/cosine与各分支 contribution，不自动加入 OGM-GE。

当前 `NEXT_PREFLIGHT_JUSTIFIED=YES` 只批准提出上述 freeze package；`FORMAL_TRAINING_JUSTIFIED=NO`。
