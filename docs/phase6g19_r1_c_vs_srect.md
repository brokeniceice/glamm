# Phase 6G.19 — R1 Structured Forensic Context: C vs Srect

Status: **COMPLETE STOP**.

## Matched contract

Trainable parameters per arm: `462299`. Initial trainable hash exact match: `True`.
Both arms use the same frozen C1, CLIP block11+17 fusion, projection, collapsed Translator/Rectifier, gamma, support, SAM and evidential heads. Final injection is always `S64 + U_F*C`.

## Results

A1 C: `{"status": "COMPLETE", "arm": "A1_C", "selected_epoch": 10, "selected_checkpoint_sha256": "e9e43bdbd7c0e695e2dd3e65aba3394fcfd9c2a9228019bf0b7c688205fa6c23", "metrics": {"n": 1106, "mean_foreground_iou": 0.19047791385260807, "median_foreground_iou": 0.08994126846498625, "mean_foreground_f1": 0.26755701959646966, "global_foreground_iou": 0.20205650338855397, "global_foreground_f1": 0.336184701499412, "threshold_logit": 0.0, "aggregate_counts": {"tp": 5017910, "fp": 6427065, "fn": 13389217}}, "gate_stats": {"pixels": {"n": 3748038, "mean": 0.5637624738970366, "median": 0.5390625, "std": 0.12237101125086682, "min": 0.03955078125, "max": 1.0, "fraction_lt_0.1": 8.057549042992628e-05, "fraction_gt_0.9": 0.0205318089090879, "binary_entropy_mean": 0.6517464611720498}, "per_image": {"n": 1106, "mean": 0.5550814369595719, "median": 0.5544570684432983, "std": 0.08249622344776127, "min": 0.0, "max": 0.7318772077560425, "fraction_lt_0.1": 0.014466546112115732, "fraction_gt_0.9": 0.0, "binary_entropy_mean": 0.6705289412050967}}, "trainable_parameters": 462299, "initial_trainable_sha256": "c577fef362130db06164f07ae0908c905e8c25c789c13a5fab44c147ee63eef6"}`

A2 Srect: `{"status": "COMPLETE", "arm": "A2_Srect", "selected_epoch": 10, "selected_checkpoint_sha256": "3482ee7d35266f5247b2c3b869d181eda20664a50190baa346b14eccd29063af", "metrics": {"n": 1106, "mean_foreground_iou": 0.19024423140459257, "median_foreground_iou": 0.08934876973406179, "mean_foreground_f1": 0.26743419182153033, "global_foreground_iou": 0.2013394133168649, "global_foreground_f1": 0.3351915555004931, "threshold_logit": 0.0, "aggregate_counts": {"tp": 4999854, "fp": 6425835, "fn": 13407273}}, "gate_stats": {"pixels": {"n": 3748038, "mean": 0.5662548537559001, "median": 0.5390625, "std": 0.12150995421153515, "min": 0.0439453125, "max": 1.0, "fraction_lt_0.1": 6.349988980901474e-05, "fraction_gt_0.9": 0.02361982455887587, "binary_entropy_mean": 0.6512146930820327}, "per_image": {"n": 1106, "mean": 0.5578008887187913, "median": 0.558116227388382, "std": 0.08722122064957287, "min": 0.0, "max": 0.787871241569519, "fraction_lt_0.1": 0.014466546112115732, "fraction_gt_0.9": 0.0, "binary_entropy_mean": 0.668242670278594}}, "trainable_parameters": 462299, "initial_trainable_sha256": "c577fef362130db06164f07ae0908c905e8c25c789c13a5fab44c147ee63eef6"}`

## Paired A1-A2

`{"n": 1106, "mean_difference": 0.0002336827755289626, "median_difference": 0.0, "bootstrap_95_ci": [-0.0006686928838907832, 0.0011377178679312816], "wins": 410, "ties": 274, "losses": 422, "wilcoxon_statistic": 170322.0, "wilcoxon_pvalue": 0.6713584557812892, "wins_ties_losses": [410, 274, 422], "f1": {"n": 1106, "mean_difference": 0.00012282804469071414, "median_difference": 0.0, "bootstrap_95_ci": [-0.0010060022199982175, 0.001268619556603954], "wins": 410, "ties": 274, "losses": 422, "wilcoxon_statistic": 171006.0, "wilcoxon_pvalue": 0.7446964442313327}}`

## Decision

```text
C_AND_SRECT_NO_STABLE_DIFFERENCE
```

No zero-X arm, late logit residual, modulation, new objective, test, Official1000, or OOD was used.
