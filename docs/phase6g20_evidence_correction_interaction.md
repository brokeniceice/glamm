# Phase 6G.20 — Evidence–Correction Interaction Audit

Status: **COMPLETE STOP**.

## Arms

A0 is exact reuse of G19 A1_C. A1 replaces only concat/3x3 merge with heads=4, window=7 local Evidence-query/Correction-key-value attention.

A0: `{"status": "COMPLETE", "arm": "A1_C", "selected_epoch": 10, "selected_checkpoint_sha256": "e9e43bdbd7c0e695e2dd3e65aba3394fcfd9c2a9228019bf0b7c688205fa6c23", "metrics": {"n": 1106, "mean_foreground_iou": 0.19047791385260807, "median_foreground_iou": 0.08994126846498625, "mean_foreground_f1": 0.26755701959646966, "global_foreground_iou": 0.20205650338855397, "global_foreground_f1": 0.336184701499412, "threshold_logit": 0.0, "aggregate_counts": {"tp": 5017910, "fp": 6427065, "fn": 13389217}}, "gate_stats": {"pixels": {"n": 3748038, "mean": 0.5637624738970366, "median": 0.5390625, "std": 0.12237101125086682, "min": 0.03955078125, "max": 1.0, "fraction_lt_0.1": 8.057549042992628e-05, "fraction_gt_0.9": 0.0205318089090879, "binary_entropy_mean": 0.6517464611720498}, "per_image": {"n": 1106, "mean": 0.5550814369595719, "median": 0.5544570684432983, "std": 0.08249622344776127, "min": 0.0, "max": 0.7318772077560425, "fraction_lt_0.1": 0.014466546112115732, "fraction_gt_0.9": 0.0, "binary_entropy_mean": 0.6705289412050967}}, "trainable_parameters": 462299, "initial_trainable_sha256": "c577fef362130db06164f07ae0908c905e8c25c789c13a5fab44c147ee63eef6"}`

A1: `{"status": "COMPLETE", "selected_epoch": 8, "selected_checkpoint_sha256": "ead03e2d50e2d0e5c64bc16a5d972a7456f161d056b27389f3c620cd04eb1b5a", "metrics": {"n": 1106, "mean_foreground_iou": 0.1897636477738184, "median_foreground_iou": 0.0931307691333596, "mean_foreground_f1": 0.2670763068660856, "global_foreground_iou": 0.20462312850640724, "global_foreground_f1": 0.3397297024507842, "threshold_logit": 0.0, "aggregate_counts": {"tp": 5134716, "fp": 6686400, "fn": 13272411}}, "gate_stats": {"pixels": {"n": 3748038, "mean": 0.5889198203584023, "median": 0.55859375, "std": 0.11418532434879786, "min": 0.1240234375, "max": 0.99609375, "fraction_lt_0.1": 0.0, "fraction_gt_0.9": 0.02560379590601803, "binary_entropy_mean": 0.6473677213640754}, "per_image": {"n": 1106, "mean": 0.5800746797060233, "median": 0.5824161469936371, "std": 0.08232596257877936, "min": 0.0, "max": 0.7291954755783081, "fraction_lt_0.1": 0.014466546112115732, "fraction_gt_0.9": 0.0, "binary_entropy_mean": 0.6637410638111838}}, "trainable_parameters": 405147, "shared_initialization_exact": true}`

## Paired A1-A0

`{"n": 1106, "mean_difference": -0.0007142665880142215, "median_difference": 0.0, "bootstrap_95_ci": [-0.0021093138257503867, 0.0007130394644615146], "wins": 399, "ties": 271, "losses": 436, "wilcoxon_statistic": 160740.0, "wilcoxon_pvalue": 0.04816730727693936, "wins_ties_losses": [399, 271, 436], "f1": {"n": 1106, "mean_difference": -0.00048071337286147137, "median_difference": 0.0, "bootstrap_95_ci": [-0.0022469583205247793, 0.0013270810922514902], "wins": 399, "ties": 271, "losses": 436, "wilcoxon_statistic": 161531.0, "wilcoxon_pvalue": 0.06254276491432405}}`

## Decision

```text
SIMPLE_STRUCTURED_MERGE_PREFERRED
```

No Srect, zero-X, FiLM, late-logit residual, third arm, new loss, test, Official1000, or OOD was used.
