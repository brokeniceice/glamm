# 候选比较矩阵

评分 1–5；复杂度维度中 5 表示更易实现/更低风险。

| 维度 | A PCERF | B AHBFR | C TSST |
|---|---:|---:|---:|
| 1. Literature maturity | 5 | 5 | 4 |
| 2. Phase4F problem match | 5 | 5 | 3 |
| 3. P1 compatibility | 5 | 4 | 3 |
| 4. SAM decoder preservation | 5 | 5 | 5 |
| 5. Spatial geometry preservation | 4 | 5 | 4 |
| 6. Reliability adaptivity | 5 | 4 | 3 |
| 7. Conflict handling | 5 | 3 | 2 |
| 8. Modality dominance protection | 5 | 4 | 2 |
| 9. Training stability | 4 | 3 | 2 |
| 10. Implementation complexity | 4 | 3 | 2 |
| 11. Parameter efficiency | 5 | 4 | 4 |
| 12. Causal ablation clarity | 5 | 4 | 3 |
| 13. Deployment availability | 5 | 5 | 5 |
| **总分 / 65** | **62** | **54** | **42** |

## 评分依据

### A PCERF

ECoLaF 已在 dense segmentation 直接验证 conflict-guided discount；TMC/QMF补充 uncertainty 与 weight–loss criterion。冻结双 expert、只训练 reliability/fusion，使 P1 compatibility、dominance protection 和 causal ablation 最强。空间几何为 4 而非 5，是因为 forensic expert 原生只有 24×24，late resize 有边界上限。训练稳定性为 4，是因为 evidential conflict 仍需数值与 calibration preflight。

### B AHBFR

CMX 与 MAG 都成熟，且与 Phase 4F fixed residual 高度匹配；但把 uncertainty、channel/spatial calibration 和 norm cap迁移到非对称 SAM–forensic setting属于跨任务组合。feature injection 仍可能重现 shortcut，causal attribution不如双 expert late fusion。

### C TSST

TokenFusion 本身成熟且位置机制清楚，但本项目的 token importance 必须从“视觉信息量”改成“language-conditioned correctness”；硬替换与 24→64 对应造成稳定性和 P1 compatibility 风险。拒绝是路线优先级判断，不是负结果。

## Formal selection

```yaml
PRIMARY_CANDIDATE: PCERF
SECONDARY_CANDIDATE: AHBFR
REJECTED_CANDIDATES:
  - TSST
  - Phase4F_epoch9_warm_start
  - uncalibrated_scalar_sigmoid_gate
  - over_composed_CMX_TokenFusion_MAG_OGM_MoE
```

选择不基于 validation performance；依据是内部证据对齐、原论文成熟度、因果可识别性、P1 exact recovery 与 dominance control。

