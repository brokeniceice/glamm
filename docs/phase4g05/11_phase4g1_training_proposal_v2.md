# Phase 4G-1 Training Proposal v2（只设计，未执行）

## 阶段顺序

1. `G1-C Reliability Calibration Preflight`：只有新人工授权后，按 frozen TRAIN-FIT/CAL/AUDIT 与 `08_reliability_protocol.md`执行。
2. `G1-F Full Training`：G1-C通过后仍需另一份明确授权、预算与 stopping rule；不能自动进入。

本文件不授权 optimizer、epoch、checkpoint selector或evaluation。

## Mandatory arms

1. Full PCERF。
2. Static matched equal-probability fusion。
3. Constant/no reliability。
4. No conflict discount。
5. Sample-only reliability。
6. Reliability permutation。
7. No language stream。
8. No forensic stream。
9. `no-z_L` 与 `no-z_F24`，并配套 `z_L-only/z_F24-only`。
10. forensic expert→matched CLIP expert。
11. matched/cross-image/spatial-shuffle/zero-vacuous/off。
12. frozen P1 reference。
13. frozen forensic-only reference。

不得把所有 arms先小预算筛选再挑公式。Head输入消融还必须覆盖 no-q_seg、no-S64与no-F24。

## Loss 与 optimization audit

source evidential loss使用 TMC expected CE+annealed KL公式；fused segmentation loss及各项数值权重必须在正式 G1-F authorization中冻结，不从 validation回推。每 step记录 source head loss、fusion loss、gradient norm/cosine、evidence strength、uncertainty、conflict、discount、saturation与per-stream contribution。

```yaml
TRAINING_BALANCE_AUDIT_REQUIRED: YES
TRAINING_BALANCE_INTERVENTION_REQUIRED: UNRESOLVED
```

不默认添加 OGM-GE、Pareto gradient manipulation、modality dropout、load balancing或entropy regularization。只有 dominance audit触发且获得新授权，才能增加 matched intervention arm。

## Future selector

沿用 Pareto，不使用 `G0+TF` scalar。坐标为 canonical G0、Phrase/TF recovery与oracle-gap retention，并用matched-cross、matched-shuffle作mechanism constraints。正式 threshold-boundary分类指标使用direct batch=1。non-inferiority margin、预算、checkpoint cadence与lexicographic tie-break必须在结果前由未来协议冻结；本阶段不做selector。

## Claim mapping

- reliability消融强、refinement弱：ROUTING_DOMINATED。
- refinement强、reliability弱但非零：REFINEMENT_DOMINATED。
- 两者独立贡献：HYBRID。
- reliability/conflict消融无影响：不得写reliability-aware主贡献，只能定位 expert-conditioned evidential fusion/refinement。
- Full几乎由E_F决定：改写为 forensic evidential segmentation with language auxiliary context。

## Seals

development validation不得用于G1-C signal/architecture selection。internal test与official1000继续封存；AHBFR、Teacher/KD与held-out evaluation不自动启动。

