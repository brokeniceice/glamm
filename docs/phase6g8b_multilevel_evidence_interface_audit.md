# Phase 6G.8B — Multi-Level Evidence Interface / Channel Audit

Status: **COMPLETE STOP**. Only the two projection interfaces and four fresh matched probes were trained. CLIP caches, block11+17 fusion, the Phase6G.3 full Adapter, C1, and SAM were frozen; Rectifier/Utility/SAM/C1 were not instantiated.

## Core table (full 1106-sample Phase6G.8 population)

| Representation | Mean FG IoU | Median FG IoU | Mean FG F1 | Delta vs source | IoU 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---:|---:|---|---:|---:|
| block11+17 fused 1024 source | 0.178801 | 0.093790 | 0.258043 | - | - | - | - |
| full Adapter -> 256 | 0.179349 | 0.085441 | 0.256015 | +0.000548 | [-0.002569, 0.003501] | 390/252/464 | 0.5456 |
| projection-only -> 256 | 0.178235 | 0.099577 | 0.258752 | -0.000566 | [-0.002645, 0.001583] | 417/232/457 | 0.331 |
| projection-only -> 512 | 0.178817 | 0.099471 | 0.259120 | +0.000016 | [-0.002444, 0.002526] | 468/219/419 | 0.5323 |

## Projection retention and width comparisons

| Comparison | Delta mean FG IoU | IoU 95% CI | W/T/L | Wilcoxon p | Delta mean FG F1 | F1 95% CI |
|---|---:|---|---:|---:|---:|---|
| full Adapter 256 - source 1024 | +0.000548 | [-0.002569, 0.003501] | 390/252/464 | 0.5456 | -0.002028 | [-0.006010, 0.001787] |
| projection 256 - source 1024 | -0.000566 | [-0.002645, 0.001583] | 417/232/457 | 0.331 | +0.000709 | [-0.001938, 0.003451] |
| projection 512 - source 1024 | +0.000016 | [-0.002444, 0.002526] | 468/219/419 | 0.5323 | +0.001077 | [-0.002163, 0.004489] |
| projection 512 - projection 256 | +0.000581 | [-0.002145, 0.003335] | 465/217/424 | 0.1888 | +0.000368 | [-0.003214, 0.003956] |
| full Adapter 256 - projection 256 | +0.001114 | [-0.002345, 0.004530] | 414/243/449 | 0.62 | -0.002737 | [-0.007185, 0.001572] |
| full Adapter 256 - projection 512 | +0.000532 | [-0.003221, 0.004112] | 406/232/468 | 0.723 | -0.003105 | [-0.007898, 0.001463] |

## Valid-SEG sensitivity (1090 samples)

The primary table uses the full 1106-sample Phase6G.8 validation population. The valid-SEG-only table below is a sensitivity check, not a replacement for the primary population.

| Representation | Mean FG IoU | Median FG IoU | Mean FG F1 | Delta vs source | IoU 95% CI | W/T/L | Wilcoxon p |
|---|---:|---:|---:|---:|---|---:|---:|
| block11+17 fused 1024 source | 0.181426 | 0.098363 | 0.261831 | - | - | - | - |
| full Adapter -> 256 | 0.181982 | 0.087924 | 0.259773 | +0.000556 | [-0.002616, 0.003625] | 390/236/464 | 0.5456 |
| projection-only -> 256 | 0.180852 | 0.103124 | 0.262550 | -0.000574 | [-0.002689, 0.001541] | 417/216/457 | 0.331 |
| projection-only -> 512 | 0.181442 | 0.103985 | 0.262923 | +0.000016 | [-0.002480, 0.002540] | 468/203/419 | 0.5323 |

## Decision rationale

- `projection_256 - source`: `{'n': 1106, 'mean_difference': -0.0005655640494867116, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.0026450101933865582, 0.0015827215429752217], 'wins': 417, 'ties': 232, 'losses': 457, 'wilcoxon_statistic': 183930.0, 'wilcoxon_pvalue': 0.33097065606051157}`
- `full_adapter - projection_256`: `{'n': 1106, 'mean_difference': 0.0011137505863805836, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.002345415013848719, 0.004529868902737662], 'wins': 414, 'ties': 243, 'losses': 449, 'wilcoxon_statistic': 182776.0, 'wilcoxon_pvalue': 0.6200061216559247}`
- `projection_512 - projection_256`: `{'n': 1106, 'mean_difference': 0.0005812685559265394, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.0021452912652111943, 0.003334788958201987], 'wins': 465, 'ties': 217, 'losses': 424, 'wilcoxon_statistic': 187739.0, 'wilcoxon_pvalue': 0.18882029334985428}`
- `full_adapter - source`: `{'n': 1106, 'mean_difference': 0.0005481865368938717, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.002569008293856749, 0.0035009567223017717], 'wins': 390, 'ties': 252, 'losses': 464, 'wilcoxon_statistic': 178185.0, 'wilcoxon_pvalue': 0.5456373633493071}`
- `projection_512 - source`: `{'n': 1106, 'mean_difference': 1.5704506439827636e-05, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.002444161188906963, 0.0025262875416550688], 'wins': 468, 'ties': 219, 'losses': 419, 'wilcoxon_statistic': 192148.0, 'wilcoxon_pvalue': 0.5323372728168184}`
- `full_adapter - projection_512`: `{'n': 1106, 'mean_difference': 0.0005324820304540439, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.00322084872684225, 0.004112465530075165], 'wins': 406, 'ties': 232, 'losses': 468, 'wilcoxon_statistic': 188541.0, 'wilcoxon_pvalue': 0.7229601476686511}`

The first decision branch is taken because `projection_256` is not stably different from `source` and `full_adapter` is not stably different from `projection_256`. The 512-wide projection and the three-block Adapter also show no stable gain. The valid-SEG sensitivity table gives the same pattern.

## Interpretation

- The 1024->256 interface is not a detectable information bottleneck on this frozen block11+17 fusion source.
- The three LocalForensicBlocks do not add a stable gain over a single 1x1 projection to 256 for this evidence interface.
- The result is an interface-level attribution from internal validation only; it does not authorize deleting the Adapter, changing Rectifier input dimensions, retraining R1, or accessing test/Official1000/OOD.

- stable rule: `{'stable_positive': 'paired mean delta > 0, bootstrap 95% CI lower > 0, Wilcoxon p < 0.05', 'stable_negative': 'paired mean delta < 0, bootstrap 95% CI upper < 0, Wilcoxon p < 0.05', 'equivalent': 'neither stable positive nor stable negative (95% CI includes zero)'}`
- historical references (not used in statistics): `{'phase6g8a_source_block11_17_fusion': 0.1789369539062384, 'phase6g8a_full_adapter_output': 0.17941149888421679, 'historical_block11_17_fusion_probe': 0.186741, 'historical_phase6g3_a0_adapter_output': 0.184992}`

```text
256_INTERFACE_SUFFICIENT_LOCAL_ADAPTER_REDUNDANT
```

No test, Official1000, or OOD data were accessed.
