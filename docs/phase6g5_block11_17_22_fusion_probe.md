# Phase 6G.5 — Block11+17+22 Cross-Layer Fusion Probe

Status: **COMPLETE STOP**. Only fusion and a linear spatial probe were trained on internal TRAIN/validation.

| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 |
|---|---:|---:|---:|
| A0 block11+17 | 0.186741 | 0.109707 | 0.269635 |
| A1 block11+17+22 | 0.183542 | 0.107603 | 0.266878 |

- IoU paired: `{'n': 1106, 'mean_difference': -0.003199089478896734, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.00757878079039422, 0.0012253412185983447], 'wins': 439, 'ties': 195, 'losses': 472, 'wilcoxon_statistic': 199429.0, 'wilcoxon_pvalue': 0.29733843759684087}`
- F1 paired: `{'n': 1106, 'mean_difference': -0.00275659039267631, 'median_difference': 0.0, 'bootstrap_95_ci': [-0.008411907170718241, 0.002815348485238183], 'wins': 439, 'ties': 195, 'losses': 472, 'wilcoxon_statistic': 202017.0, 'wilcoxon_pvalue': 0.47375591429939223}`
- attention: `{'overall': {'block11': 0.43505895137786865, 'block17': 0.11965777724981308, 'block22': 0.44528335332870483}, 'per_head': {'block11': [0.3894822895526886, 0.8566427230834961, 0.8327670693397522, 0.14707620441913605, 0.09504832327365875, 0.5296062231063843, 0.5051395297050476, 0.12470885366201401], 'block17': [0.21562203764915466, 0.07084707915782928, 0.08603379130363464, 0.0848553404211998, 0.04937242344021797, 0.06290393322706223, 0.3601696491241455, 0.02745797485113144], 'block22': [0.39489561319351196, 0.07251019775867462, 0.08119914680719376, 0.7680684328079224, 0.8555791974067688, 0.4074898362159729, 0.13469088077545166, 0.8478331565856934]}, 'gamma': 0.009912229143083096}`
- rescue samples: `19`

```text
BLOCK22_ADDITIONAL_VALUE_NOT_SUPPORTED
```

Future multi-level dense evidence source is fixed to **block11 + block17**.
