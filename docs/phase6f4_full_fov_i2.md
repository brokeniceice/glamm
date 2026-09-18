# Phase 6F.4 — Full-FOV I2 Joint Training

Status: **COMPLETE STOP**. C1, CLIP, Phase4C-A adapter, SAM and source heads remained frozen. Utility and Rectifier used the exact Phase6E.2 I2 seed-3407 random initialization; no selected new-R1 state was loaded.

- selected epoch: `8` by internal-validation canonical-G0 mean FG IoU
- checkpoint SHA256: `78afb3d50f4fd502f899405fb22ddebda5fe59dde6f9dac92a2500bf4fd28525`
- current new R1 mean IoU: `0.202488`
- Full-FOV I2 mean IoU: `0.204031`
- paired IoU: `{'mean_delta': 0.0015432562207698052, 'median_delta': 0.0, 'bootstrap_95_ci': [-0.0037109783050276648, 0.006715865291102356], 'wins': 469, 'ties': 197, 'losses': 440, 'wilcoxon_statistic': 202811.0, 'wilcoxon_pvalue': 0.6146299523901122}`
- current new R1 mean F1: `0.283836`
- Full-FOV I2 mean F1: `0.285173`
- paired F1: `{'mean_delta': 0.0013368732890640946, 'median_delta': 0.0, 'bootstrap_95_ci': [-0.005192352825036605, 0.007806278888673745], 'wins': 469, 'ties': 197, 'losses': 440, 'wilcoxon_statistic': 202651.0, 'wilcoxon_pvalue': 0.6004991863772418}`

Coverage/aspect/tile-count results are saved in `outputs/phase6f4_full_fov_i2/comparison.json`.

```text
FULL_FOV_I2_NOT_STABLY_BETTER
```

No test, Official1000, or OOD data was accessed. No staged Full-FOV pretraining was started.
