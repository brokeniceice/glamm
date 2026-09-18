# Phase 6F.1 — C1 + new R1 Forensic Coverage Attribution Diagnostic

## Protocol

Read-only internal-validation Fake diagnostic (`n=1106`). C1 and selected new R1 were frozen; optimizer count was 0. No training, checkpoint selection, threshold tuning, internal test, Official1000, or external OOD access occurred. Both arms reused the identical cached C1 canonical-G0 trajectory, original-space inverse geometry, and `mask logit > 0`; the only model-path difference was selected new R1 disabled/enabled.

## Overall

- C1 mean IoU: `0.134698`
- C1 + new R1 mean IoU: `0.202488`
- mean delta IoU: `0.067790`, bootstrap CI `[0.05693541111708953, 0.0787026356363619]`
- mean delta F1: `0.089465`, bootstrap CI `[0.0760576492812962, 0.10277869418120203]`

## Exact-crop coverage bins

| coverage | n | C1 mean IoU | C1 median | new R1 mean IoU | new R1 median | delta IoU | IoU CI | delta F1 | F1 CI | newR1 FN hull | FN boundary | FN outside |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
| 1.0 | 1018 | 0.142092 | 0.042319 | 0.214599 | 0.122365 | 0.072507 | [0.06101369096268884, 0.08398143516533503] | 0.095417 | [0.08127938194863035, 0.10953124439758544] | 10590056 | 326700 | 0 |
| [0.9,1.0) | 16 | 0.070902 | 0.029159 | 0.154201 | 0.091623 | 0.083299 | [0.012987995378679056, 0.1635187103888541] | 0.107641 | [0.01807699190192725, 0.20618830378539066] | 151638 | 9830 | 10722 |
| [0.5,0.9) | 34 | 0.062855 | 0.001254 | 0.068942 | 0.014584 | 0.006087 | [-0.018728689640344277, 0.03290261784714551] | 0.012422 | [-0.023400414178014774, 0.052269667770088506] | 422490 | 37717 | 215057 |
| (0,0.5) | 23 | 0.038624 | 0.000000 | 0.025037 | 0.000000 | -0.013587 | [-0.06737309246755256, 0.02181110726901321] | -0.010239 | [-0.07820942329567043, 0.03903458678276781] | 77338 | 21102 | 207806 |
| 0 | 15 | 0.011118 | 0.000000 | 0.006882 | 0.000000 | -0.004236 | [-0.01701226540775763, 0.004564812785649176] | -0.006321 | [-0.027328923339736747, 0.00888724919210773] | 0 | 0 | 44387 |

## Attribution

- Spearman exact crop coverage vs delta IoU: `{'rho': 0.11527981580216419, 'pvalue': 0.00012189006624870346}`
- Spearman current hull coverage vs delta IoU: `{'rho': 0.08815838380849385, 'pvalue': 0.0033435965704383705}`
- gain difference, coverage<0.9 minus coverage=1.0: `{'difference_left_minus_right': -0.07485482966646942, 'bootstrap_95_ci': [-0.09785388080988923, -0.053119759418947275]}`
- adjusted exploratory OLS: `{'formula': 'delta_iou ~ exact_crop_coverage + gt_area_fraction + aspect_ratio', 'coefficients': {'intercept': 0.08313698346787808, 'coverage': 0.06399574288366004, 'gt_area_fraction': -0.08306224887248768, 'aspect_ratio': -0.06691433265591498}, 'r_squared': 0.013946914640416086}`
- aspect groups: `{'aspect_ratio_lt_1.2': {'n': 896, 'c1_mean_iou': 0.15184121688877714, 'c1_median_iou': 0.052769619206957497, 'new_r1_mean_iou': 0.22919942787102834, 'new_r1_median_iou': 0.14221574011384228, 'mean_delta_iou': 0.07735821098225118, 'delta_iou_bootstrap_95_ci': [0.06492417647411615, 0.08978002068723732], 'mean_delta_f1': 0.10138485520808356, 'delta_f1_bootstrap_95_ci': [0.08606350591307853, 0.11683962121874218], 'c1_fn_hull': 11229311, 'c1_fn_boundary': 290923, 'c1_fn_outside': 516, 'new_r1_fn_hull': 10247398, 'new_r1_fn_boundary': 322246, 'new_r1_fn_outside': 516}, 'aspect_ratio_ge_1.2': {'n': 210, 'c1_mean_iou': 0.06155549983375995, 'c1_median_iou': 0.0, 'new_r1_mean_iou': 0.08851914171101954, 'new_r1_median_iou': 0.0038297135300840906, 'mean_delta_iou': 0.026963641877259623, 'delta_iou_bootstrap_95_ci': [0.011463230246627851, 0.043332476917492015], 'mean_delta_f1': 0.038609160770444385, 'delta_f1_bootstrap_95_ci': [0.01838451106962811, 0.05980893913871617], 'c1_fn_hull': 1028025, 'c1_fn_boundary': 69429, 'c1_fn_outside': 462918, 'new_r1_fn_hull': 994124, 'new_r1_fn_boundary': 73103, 'new_r1_fn_outside': 477456}}`

## Support boundary

- boundary-affected images: `316`; unaffected: `790`
- maximum GT access added by expanding center hull to exact crop footprint: `431289` pixels, `0.023431` of aggregate GT
- affected-minus-unaffected new-R1 gain: `{'difference_left_minus_right': -0.012776833599242365, 'bootstrap_95_ci': [-0.03670694766041098, 0.011329545254211232]}`
- FN densities: `{'c1': {'hull': 0.700887873708465, 'boundary': 0.8355232802134995, 'outside': 0.9505558518275423}, 'new_r1': {'hull': 0.642802518575564, 'boundary': 0.9166684056398378, 'outside': 0.9803749435943717}}`
- FP densities: `{'c1': {'hull': 0.04147666118923344, 'boundary': 0.009123910819618351, 'outside': 0.013093831781106095}, 'new_r1': {'hull': 0.025112141416427737, 'boundary': 0.0027681051887956387, 'outside': 0.006685087250404252}}`

Embedding-level assertions passed: rectifier residual and utility gate were exactly zero outside semantic support. The decoded mask was not forced to match outside support; observed decoded outside-hull changes are recorded in `statistics.json`.

## Decision

`COVERAGE_BOTTLENECK_SUPPORTED`

`SUPPORT_BOUNDARY_EFFECT_PRESENT = NO`

This result is an attribution diagnostic, not evidence that a full-FOV method improves performance. Phase 6F.1 stops here; no multi-tile or geometry correction was implemented.
