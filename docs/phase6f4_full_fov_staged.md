# Phase 6F.4 — Full-FOV I2 Joint Training

Status: **COMPLETE STOP**. C1, CLIP, Phase4C-A adapter, SAM and source heads remained frozen. Utility and Rectifier used the exact Phase6E.2 I2 seed-3407 random initialization; no selected new-R1 state was loaded.

- selected epoch: `8` by internal-validation canonical-G0 mean FG IoU
- checkpoint SHA256: `2afa37cea98830fafbdf645d47df93be25a9fadd823fe2fd88476d9d53bf3466`
- current new R1 mean IoU: `0.202488`
- Full-FOV I2 mean IoU: `0.203822`
- paired IoU: `{'mean_delta': 0.0013342218936732205, 'median_delta': 0.0, 'bootstrap_95_ci': [-0.003119013827400196, 0.0059438843047558195], 'wins': 454, 'ties': 195, 'losses': 457, 'wilcoxon_statistic': 204060.0, 'wilcoxon_pvalue': 0.6460839990853908}`
- current new R1 mean F1: `0.283836`
- Full-FOV I2 mean F1: `0.285474`
- paired F1: `{'mean_delta': 0.0016385953760418048, 'median_delta': 0.0, 'bootstrap_95_ci': [-0.003799160950016067, 0.007247381108481075], 'wins': 454, 'ties': 195, 'losses': 457, 'wilcoxon_statistic': 205189.0, 'wilcoxon_pvalue': 0.7511745921061507}`

Coverage/aspect/tile-count results are saved in `outputs/phase6f4_full_fov_i2/comparison.json`.

```text
FULL_FOV_I2_NOT_STABLY_BETTER
```

No test, Official1000, or OOD data was accessed. No staged Full-FOV pretraining was started.
