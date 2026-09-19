# Phase 6G.8A — Matched Adapter Input/Output Probe

Status: **COMPLETE STOP**. Only four fresh matched 1x1 Conv probes were trained. CLIP, block11+17 fusion, Adapter, Rectifier, Utility, SAM, and C1 were frozen; Utility/Rectifier/SAM/C1 were not instantiated because they are not required to score these evidence representations.

## Core table (full 1106-sample Phase6G.8 population)

| Arm | Adapter input | Adapter output | Delta | IoU 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---:|---|---|---:|
| A0 | 0.144113 | 0.164454 | +0.020340 | [0.015160523991750408, 0.025728443551230683] | 496/285/325 | 1.521e-14 |
| A1 | 0.178937 | 0.179411 | +0.000475 | [-0.00263310137824725, 0.0034127555710306906] | 386/253/467 | 0.522 |

## Paired per-sample statistics

- `A0` IoU: `{'n': 1106, 'mean_difference': 0.020340226523693553, 'median_difference': 0.0, 'bootstrap_95_ci': [0.015160523991750408, 0.025728443551230683], 'wins': 496, 'ties': 285, 'losses': 325, 'wilcoxon_statistic': 116475.0, 'wilcoxon_pvalue': 1.52094376277415e-14}`
- `A0` F1: `{'n': 1106, 'mean_difference': 0.02526455282151908, 'median_difference': 0.0, 'bootstrap_95_ci': [0.018526256849515102, 0.0322266334698032], 'wins': 496, 'ties': 285, 'losses': 325, 'wilcoxon_statistic': 117881.0, 'wilcoxon_pvalue': 7.494548941696577e-14}`
- `A1` IoU: `{'n': 1106, 'mean_difference': 0.00047454497797842174, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.00263310137824725, 0.0034127555710306906], 'wins': 386, 'ties': 253, 'losses': 467, 'wilcoxon_statistic': 177507.0, 'wilcoxon_pvalue': 0.5220144758612673}`
- `A1` F1: `{'n': 1106, 'mean_difference': -0.002046629426333561, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.005994911952822449, 0.0017048256920647712], 'wins': 386, 'ties': 253, 'losses': 467, 'wilcoxon_statistic': 172311.0, 'wilcoxon_pvalue': 0.17316448033669374}`

- A1 adapter gain minus A0 adapter gain: `{'n': 1106, 'mean_difference': -0.01986568154571513, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.025813885921006592, -0.014094575350629267], 'wins': 374, 'ties': 194, 'losses': 538, 'wilcoxon_statistic': 156446.0, 'wilcoxon_pvalue': 8.056213580613114e-11}`
- stable-positive rule: `paired mean delta > 0, bootstrap 95% CI lower > 0, Wilcoxon p < 0.05`
- historical references (not used in statistics): `{'historical_block22_source_probe': 0.143842, 'historical_block11_17_fusion_probe': 0.186741, 'historical_phase6g3_a0_adapter_output': 0.184992}`

```text
ADAPTER_SPECIALIZATION_REDUNDANT_FOR_MULTILEVEL_FUSION
```

No test, Official1000, or OOD data were accessed.
