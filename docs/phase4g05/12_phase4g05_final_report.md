# Phase 4G-0.5 最终报告

## Outcome

PCERF 已从 Phase 4G-0 的 proposed graph硬化为可执行的 expert-conditioned evidential refinement + source reliability estimation + conflict-aware late fusion。关键修正是分离 TMC Dirichlet opinion与ECoLaF `K+1` mass kernel，并把空/vacuous source identity放在官方 kernel外的source-availability dispatch，解决64→256不可逆导致的P1 fallback矛盾。

所有 implementation hard gates通过；frozen expert关系为 MODERATE complementarity，frozen oracle headroom为 HIGH。后两项均只作诊断，不是full-training gate，也不是PCERF upper bound。

## Formal gates

```yaml
FORMULA_PARITY: PASS
P1_LANGUAGE_ONLY_EXACT_RECOVERY: PASS
FORENSIC_VACUOUS_EXACT_RECOVERY: PASS
FORENSIC_OFF_EXACT_RECOVERY: PASS
GEOMETRY_VALIDATED: YES
VACUOUS_SUPPORT_VALIDATED: YES
GRADIENT_ISOLATION: PASS
NO_ORACLE_LEAKAGE: PASS
INVALID_G0_POLICY: PASS
HEAD_REFINEMENT_ALLOWED: YES
SOURCE_CLASS_DIRECTION_PRESERVATION_REQUIRED: NO
SOURCE_INFORMATION_UTILIZATION_AUDIT_REQUIRED: YES
EXPERT_COMPLEMENTARITY: MODERATE  # diagnostic only
FROZEN_EXPERT_ORACLE_HEADROOM: HIGH  # diagnostic only
RELIABILITY_SPLIT_FROZEN: YES
RELIABILITY_CALIBRATION_PROTOCOL_FROZEN: YES
TRAINING_BALANCE_AUDIT_REQUIRED: YES
TRAINING_BALANCE_INTERVENTION_REQUIRED: UNRESOLVED
PCERF_ARCHITECTURE_HARDENED: YES
G1_C_RELIABILITY_PREFLIGHT_JUSTIFIED: YES
PHASE4G1_FULL_TRAINING_JUSTIFIED: NO
```

## Evidence summary

- 10类ECoLaF parity case所有中间量与官方实现最大绝对误差0；TMC parity为0。
- 7类高频/稀疏synthetic pattern在language-only/vacuous/off三种条件下均bit-exact恢复P1。
- non-square center-crop外为exact vacuous；geometry/support通过。
- 真实冻结source的单次backward中两个head均有finite gradient；P1 SAM与Phase4C-A grad none，module/checkpoint hash不变。
- TRAIN-FIT/CAL/AUDIT为6185/1326/1325，image ID互斥并精确覆盖train Fake。
- TRAIN-AUDIT valid-G0上foreground双向rescue为0.2013/0.1633；pixel oracle相对最佳single source为+0.17164 mean FG IoU。

## Scientific boundary

本阶段没有证明learned reliability与localization error稳定相关，因此只允许下一步单独申请 G1-C Reliability Calibration Preflight。`G1_C_RELIABILITY_PREFLIGHT_JUSTIFIED=YES` 表示公式、split、harness与hard invariants足以开始该受限preflight；它绝不等价于性能训练有依据。

没有运行 full PCERF training、1-epoch probe、512-step screening、development validation architecture search、threshold/temperature/gamma sweep、checkpoint selection、AHBFR、Teacher/KD、internal test或official1000。

## STOP

Phase 4G-0.5 到此停止。等待人工批准 G1-C；不得自动进入 fitting、Phase4G-1、AHBFR或任何 held-out evaluation。

