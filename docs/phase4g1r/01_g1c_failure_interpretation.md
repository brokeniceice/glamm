# G1-C failure interpretation（冻结）

## 不可改写的结论

G1-C 的 `RELIABILITY_PREFLIGHT=FAIL` 保持不变。本阶段不重跑 G1-C，不用新命名稀释 preregistered failure，也不把未完成的 permutation control 解释成因果否定。

| 类别 | 冻结陈述 | 证据 |
|---|---|---|
| supported negative | `FORENSIC_INTRINSIC_UNCERTAINTY_AS_LOCALIZATION_RELIABILITY=NOT_SUPPORTED` | uncertainty 与 `1-FGIoU` Spearman=-0.212329，95% CI 全负，方向与预注册要求相反 |
| supported negative | `CURRENT_SPATIAL_MISMATCH_AWARENESS=NOT_SUPPORTED` | spatial shuffle 后 forensic effective weight 反而增加 +0.011045，95% CI [+0.010649,+0.011451] |
| inconclusive | `LANGUAGE_INTRINSIC_RELIABILITY=INCONCLUSIVE` | Spearman=+0.020673，但 95% CI [-0.033526,+0.074773] 跨零 |
| inconclusive | `CROSS_IMAGE_RELIABILITY_RESPONSE=INCONCLUSIVE` | weight delta=-0.000274，95% CI [-0.001349,+0.000795] 跨零 |
| positive partial | `FORENSIC_EFFECTIVE_WEIGHT_QUALITY_RELATION=PARTIALLY_SUPPORTED` | forensic weight 与 source NLL 的 Pearson/Spearman 均为负且 CI 全负 |
| untested causal | `RELIABILITY_CAUSAL_CONTRIBUTION=NOT_VALIDLY_TESTED` | 原 permutation 在 formal statistics 前改变 source argmax；不能写成 `FALSE` |

Calibration PASS 只说明 concentration scaling 对 NLL 非退化；它不修复 uncertainty 与 localization error 的错误关系。Vacuous exact recovery PASS 说明 identity dispatch 仍可保留，也不构成 reliability validity。

因此失败指向控制变量错配，而不是直接证明 forensic expert 无用：source-intrinsic confidence 不能独自回答“此 forensic evidence 对当前 language-conditioned spatial context 是否有用”。机器可读映射见 `outputs/phase4g1r/failure_to_mechanism_map.json`。

本阶段未访问 development validation、internal test 或 official1000。
