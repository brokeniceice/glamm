# Phase 6G.11 — Zero-Initialized Correction Side Path

Status: **COMPLETE STOP**. Only the zero-initialized side path was trained; the Phase6G.10 A0 Rectifier and all other modules were frozen.

## Rectifier objective

| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 |
|---|---:|---:|---:|
| A0 frozen double projection | 0.181244 | 0.086192 | 0.256231 |
| A1 A0 + zero-init side | 0.185546 | 0.094848 | 0.263737 |

- paired A1 - A0: `{'n': 1106, 'mean_difference': 0.004301626757551539, 'median_difference': 0.0, 'bootstrap_95_ci': [0.0009165758923755294, 0.007822999296083115], 'wins': 420, 'ties': 263, 'losses': 423, 'wilcoxon_statistic': 170607.0, 'wilcoxon_pvalue': 0.30421110480545643}`
- selected epoch: `4`
- side update norm: `0.593043`

## Mechanism diagnostics

- residual norms/cosines: `{'main': {'mean': 358.3511962890625, 'median': 366.8279113769531, 'std': 88.4048080444336, 'min': 138.69300842285156, 'max': 533.4304809570312}, 'side': {'mean': 631.1195068359375, 'median': 651.2745361328125, 'std': 155.27197265625, 'min': 245.3942108154297, 'max': 939.5465087890625}, 'ratio': {'mean': 1.761583924293518, 'median': 1.7617675065994263, 'std': 0.014779017306864262, 'min': 1.6968648433685303, 'max': 1.7978852987289429}, 'cosine': {'mean': 0.40575286746025085, 'median': 0.4045068025588989, 'std': 0.007222577463835478, 'min': 0.3921276330947876, 'max': 0.44557496905326843}, 'n': 256, 'side_spectrum': {'shape': [256, 256], 'rank': 256, 'effective_rank_shannon': 8.426974880418106, 'effective_rank_participation': 5.4493421241798075, 'stable_rank': 3.0918705674242375, 'condition_number': 184999.9296903414, 'singular_values_top10': [0.3372683561868348, 0.269660832457335, 0.22619895363756215, 0.1774323122507067, 0.13974749561281652, 0.11832791623297792, 0.11368531256890778, 0.08248602574504064, 0.0707129320806465, 0.0628089768032546]}, 'side_subspace': {'side_input_energy_in_main_top8': 0.46319632180152714, 'side_input_energy_in_main_bottom8': 0.014466331829953242, 'side_input_energy_outside_main_top8': 0.5368036781984729}}`
- side spectrum: `{'shape': [256, 256], 'rank': 256, 'effective_rank_shannon': 8.426974880418106, 'effective_rank_participation': 5.4493421241798075, 'stable_rank': 3.0918705674242375, 'condition_number': 184999.9296903414, 'singular_values_top10': [0.3372683561868348, 0.269660832457335, 0.22619895363756215, 0.1774323122507067, 0.13974749561281652, 0.11832791623297792, 0.11368531256890778, 0.08248602574504064, 0.0707129320806465, 0.0628089768032546]}`
- side subspace: `{'side_input_energy_in_main_top8': 0.46319632180152714, 'side_input_energy_in_main_bottom8': 0.014466331829953242, 'side_input_energy_outside_main_top8': 0.5368036781984729}`

```text
CONTROLLED_CORRECTION_SIDE_PATH_SUPPORTED
```

No Utility, joint R1, internal test, Official1000, or OOD was accessed.
