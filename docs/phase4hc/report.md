# Phase 4H-C Direct U_F Fine-tuning — A1/A2 Parallel Arms

## Protocol

A1从Phase4G-1S mismatch-aware utility初始化，A2以seed3407随机初始化同一371,803参数CSCU utility branch。除此之外架构、数据顺序、cross/shuffle corruption、loss、optimizer和selector完全一致。两臂公式均为`S64_adapt=S64+U_F*Delta_F`，无mapper、late fusion、refinement或新decoder。Loss为`L_seg+L_relative+L_ranking`，权重1:1:1；tau=0.0417200699，margin=.1。两臂在GPU1/2并行训练，各自只用DEV G0 selector。

## Results

| Arm | G0 | Phrase | TF |
|---|---:|---:|---:|
| A0 | 0.172797 | 0.222843 | 0.255661 |
| B | 0.173483 | 0.220989 | 0.251102 |
| A1 | 0.180234 | 0.224565 | 0.250310 |
| A2 | 0.179532 | 0.224736 | 0.252063 |

- A1 vs A0 G0: delta=+0.007437, CI=[+0.005071,+0.009991], W/T/L=470/311/325, p=1.309e-09
- A1 vs B G0: delta=+0.006751, CI=[+0.004704,+0.008984], W/T/L=482/315/309, p=2.766e-12
- A2 vs A0 G0: delta=+0.006734, CI=[+0.004344,+0.009329], W/T/L=483/307/316, p=2.364e-09
- A1−A2 G0: delta=+0.000703, CI=[-0.000304,+0.001697], W/T/L=395/306/405, p=0.6851

## Arm diagnostics

A1 selected epoch=3；language order=YES；matched>cross/shuffle=YES；gate collapse=NO。U_F matched diagnostics：`{"scope": "Phase4F semantic support only", "U_F": {"n": 3697394, "mean": 0.5603707432746887, "std": 0.14789561927318573, "percentiles": {"0": 0.0091552734375, "1": 0.18359375, "5": 0.33203125, "25": 0.490234375, "50": 0.52734375, "75": 0.62890625, "95": 0.859375, "99": 0.94140625, "100": 1.0}}, "low_saturation_fraction_U_le_0.01": 1.0818430494559142e-06, "high_saturation_fraction_U_ge_0.99": 7.140164126409033e-05}`。

A2 selected epoch=3；language order=YES；matched>cross/shuffle=YES；gate collapse=NO。random initialization hash=`2a8871c53f2cac81d9fc3c76a8c916c71ddc8be88582ab1d3c679a7cb32fc2a1`。U_F matched diagnostics：`{"scope": "Phase4F semantic support only", "U_F": {"n": 3697394, "mean": 0.5638870000839233, "std": 0.10205142199993134, "percentiles": {"0": 0.09521484375, "1": 0.322265625, "5": 0.412109375, "25": 0.5078125, "50": 0.546875, "75": 0.61328125, "95": 0.76171875, "99": 0.85546875, "100": 0.98828125}}, "low_saturation_fraction_U_le_0.01": 0.0, "high_saturation_fraction_U_ge_0.99": 0.0}`。

## Interpretation and gates

```json
{
  "schema": "phase4hc_gate_summary_v1",
  "A1_COMPLETE": "YES",
  "A2_COMPLETE": "YES",
  "IDENTICAL_DATA_ORDER": "PASS",
  "A1_LANGUAGE_ORDER_PRESERVED": "YES",
  "A2_LANGUAGE_ORDER_PRESERVED": "YES",
  "A1_MATCHED_GT_CROSS_SHUFFLE": "YES",
  "A2_MATCHED_GT_CROSS_SHUFFLE": "YES",
  "A1_GATE_COLLAPSE": "NO",
  "A2_GATE_COLLAPSE": "NO",
  "A1_SOURCE_HASH_INTEGRITY": "PASS",
  "A2_SOURCE_HASH_INTEGRITY": "PASS",
  "UTILITY_PRETRAINING_HELPFUL": "NOT_SUPPORTED",
  "UTILITY_PRETRAINING_NECESSARY": "NOT_SUPPORTED",
  "A1_SIGNIFICANTLY_BEATS_PHASE4HB": "YES",
  "A1_FORENSIC_DOMINANCE": "NO",
  "A2_FORENSIC_DOMINANCE": "NO",
  "INTERNAL_TEST_ACCESSED": "NO",
  "OFFICIAL1000_ACCESSED": "NO"
}
```

结论必须同时结合G0 CI、Phrase/TF和language order；不得只按G0宣告成功。到此严格STOP，未访问internal test或official1000。
