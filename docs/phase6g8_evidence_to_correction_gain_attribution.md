# Phase 6G.8 — Evidence-to-Correction Gain Attribution

Status: **COMPLETE STOP**. A0/A1 Adapter, Rectifier, Utility, SAM, C1 and CLIP-side sources were frozen. Only matched 1x1 spatial probes were trained on internal TRAIN and selected on internal validation.

## Core gain-retention table

| Stage | block22 A0 | block11+17 A1 | Delta | IoU 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---:|---|---|---:|
| F_forensic | 0.164564 | 0.179360 | +0.014796 | [0.007885652888947188, 0.021742249950633372] | 495/255/356 | 3.003e-07 |
| Delta_F | 0.125253 | 0.128361 | +0.003107 | [-0.006215394733820279, 0.012657962175796763] | 327/421/358 | 0.6496 |
| U_F * Delta_F | 0.041156 | 0.030999 | -0.010157 | [-0.017172784758774488, -0.003399410613428423] | 174/764/168 | 0.2364 |
| S_adapt | 0.135894 | 0.157839 | +0.021945 | [0.015042660250281017, 0.02899742181098073] | 522/285/299 | 2.03e-14 |
| final mask | 0.202488 | 0.210203 | +0.007715 | [-0.0006194742124850484, 0.01610629445051865] | 487/185/434 | 0.02909 |

## Retention and Utility attribution

- Gain retention: `{'rectifier': {'value': 0.21002505366870197, 'interpretable': True}, 'utility': {'value': -3.2686589493377856, 'interpretable': True}, 'sam_space': {'value': None, 'interpretable': False}}`
- Utility distributions (valid SEG only): `{'A0': {'per_sample_mean_distribution': {'n': 1090, 'mean': 0.5062317021259474, 'std': 0.05331471745298817, 'median': 0.5014073252677917, 'p5': 0.43014034777879717, 'p25': 0.4697197824716568, 'p75': 0.5413036495447159, 'p95': 0.5947666823863983, 'min': 0.2493438571691513, 'max': 0.6902651786804199}, 'per_sample_median_distribution': {'n': 1090, 'mean': 0.5238734687960476, 'std': 0.03894646644659113, 'median': 0.5268704891204834, 'p5': 0.462890625, 'p25': 0.5078125, 'p75': 0.54296875, 'p95': 0.578125, 'min': 0.2494029402732849, 'max': 0.7398826479911804}}, 'A1': {'per_sample_mean_distribution': {'n': 1090, 'mean': 0.5701122298426584, 'std': 0.0540874336759292, 'median': 0.56258225440979, 'p5': 0.49881298542022706, 'p25': 0.530833899974823, 'p75': 0.6040960997343063, 'p95': 0.6669566869735718, 'min': 0.36161571741104126, 'max': 0.7743579745292664}, 'per_sample_median_distribution': {'n': 1090, 'mean': 0.5510301513409396, 'std': 0.05091633658682762, 'median': 0.54296875, 'p5': 0.490234375, 'p25': 0.515625, 'p75': 0.57421875, 'p95': 0.64453125, 'min': 0.375, 'max': 0.82421875}}}`
- Utility attribution: `{'A1_gate_minus_A0_paired': {'n': 1090, 'mean_difference': 0.06388052771671103, 'median_difference': 0.058747828006744385, 'bootstrap_95_ci': [0.06155881685230437, 0.0661782814965609], 'wins': 1062, 'ties': 0, 'losses': 28, 'wilcoxon_statistic': 4141.0, 'wilcoxon_pvalue': 5.819922036201669e-175}, 'A1_gate_on_delta_advantage_locations_proxy': {'definition': 'samples with A1-A0 Delta_F probe IoU > 0.05', 'n': 195, 'A1_mean_gate': 0.5772651822139055, 'A1_mean_gate_elsewhere': 0.5685537653595376}, 'correlations_valid': {'gain_F_vs_A1_gate_spearman': 0.018967795183681895, 'gain_Delta_vs_A1_gate_spearman': 0.015432905698053558, 'gain_Delta_vs_A1_gate_pearson': -0.003952208337662764}, 'position_proxy': {'definition': 'A1 Delta probe positive and A0 negative at aligned 64x64 locations', 'n_positions': 55576, 'A1_U_mean': 0.5590048795881006, 'other_positions_A1_U_mean': 0.4757002291892471}}`

The 1106-row table preserves the historical comparison population. Utility interpretation is restricted to the 1090 deployable exactly-one-SEG samples; the corresponding probe metrics are saved separately in `results.json`.

```text
RECTIFIER_IS_PRIMARY_CONVERSION_BOTTLENECK
```

No test, Official1000, or OOD data were accessed.
