# Final Frozen Evaluation Report

Generated: `2026-09-08T09:00:59.613123+00:00`

## Data leakage warning

The frozen leakage audit status is **BLOCKED_OVERLAP**. Evaluation proceeded because `leakage_override.json` is **ACTIVE**, under the recorded user authorization. This is an override, not a passed leakage audit.

External benchmark ↔ internal train: **779 exact SHA256 overlaps** and **793 pHash near-overlap pairs** (frozen Hamming threshold from the data audit). No overlapping sample was removed by the evaluation scripts.

Affected-dataset exact counts: `{"AIGI-Holmes-TestSet": 779}`  
Affected-dataset pHash counts: `{"AIGI-Holmes-TestSet": 784, "GenImage-heldout": 7, "LOKI-classification": 1, "PAL4VST-test": 1}`

AIGI-Holmes TestSet ↔ GenImage held-out: **0 exact** and **35 pHash near-overlap pairs**.

## Classification

LEGION public intermediate classification is **N/A: the released legion_LE intermediate checkpoint is LE-only and has no trained prediction_head**.

| Dataset | Model | Execution | N | Real | Fake | Accuracy | F1 | ROC-AUC | TNR | FPR |
|---|---|---|---|---|---|---|---|---|---|---|
| aigi_holmes | legion_intermediate | N/A_NO_CLASSIFICATION_HEAD | 99999 | 50000 | 49999 | N/A | N/A | N/A | N/A | N/A |
| aigi_holmes | legion_retrained | RUN_NEW | 99999 | 50000 | 49999 | 0.888109 | 0.875091 | 0.976110 | 0.992320 | 0.007680 |
| aigi_holmes | r1 | RUN_NEW | 99999 | 50000 | 49999 | 0.831168 | 0.797995 | 0.956480 | 0.995380 | 0.004620 |
| genimage | legion_intermediate | N/A_NO_CLASSIFICATION_HEAD | 100000 | 50000 | 50000 | N/A | N/A | N/A | N/A | N/A |
| genimage | legion_retrained | RUN_NEW | 100000 | 50000 | 50000 | 0.729160 | 0.637779 | 0.931478 | 0.981440 | 0.018560 |
| genimage | r1 | RUN_NEW | 100000 | 50000 | 50000 | 0.641980 | 0.455317 | 0.871324 | 0.984680 | 0.015320 |
| internal | legion_intermediate | N/A_NO_CLASSIFICATION_HEAD | 2208 | 1104 | 1104 | N/A | N/A | N/A | N/A | N/A |
| internal | legion_retrained | REUSED_HISTORICAL | 2208 | 1104 | 1104 | 0.986413 | 0.986486 | 0.999258 | 0.980978 | 0.019022 |
| internal | p1 | REUSED_HISTORICAL | 2208 | 1104 | 1104 | 0.983696 | 0.983621 | 0.998389 | 0.988225 | 0.011775 |
| internal | r1 | EXACT_REUSE_P1 | 2208 | 1104 | 1104 | 0.983696 | 0.983621 | 0.998389 | 0.988225 | 0.011775 |
| loki | legion_intermediate | N/A_NO_CLASSIFICATION_HEAD | 2217 | 900 | 1317 | N/A | N/A | N/A | N/A | N/A |
| loki | legion_retrained | RUN_NEW | 2217 | 900 | 1317 | 0.583672 | 0.559847 | 0.681164 | 0.785556 | 0.214444 |
| loki | r1 | RUN_NEW | 2217 | 900 | 1317 | 0.542625 | 0.472973 | 0.654220 | 0.831111 | 0.168889 |
| raise998 | legion_intermediate | N/A_NO_CLASSIFICATION_HEAD | 998 | 998 | 0 | N/A | N/A | N/A | N/A | N/A |
| raise998 | legion_retrained | RUN_NEW | 998 | 998 | 0 | 0.988978 | 0.000000 | N/A | 0.988978 | 0.011022 |
| raise998 | r1 | RUN_NEW | 998 | 998 | 0 | 0.993988 | 0.000000 | N/A | 0.993988 | 0.006012 |

RAISE998 is a **Real-only OOD false-positive benchmark**. Its meaningful primary quantities are TNR/FPR (and TN/FP); it is not described as a balanced-accuracy benchmark.

## Localization

Inference-condition boundary: SynthScars reuses historical **P1/R1 G0**; LOKI reuses historical **P1/R1 G1**; the new X-AIGD and PAL4VST runs use **P1/R1 G1** as explicitly frozen for this final evaluation. LEGION public-intermediate and LEGION-retrained use official **L-FREE** throughout. Therefore P1/R1-vs-LEGION numbers are cross-protocol comparisons rather than identical-prompt comparisons.

| Dataset | Model | Condition | Execution | N | Empty-GT | GT type | Mean FG IoU | Mean FG IoU (nonempty GT) | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loki | legion_intermediate | L-FREE | REUSED_HISTORICAL | 229 | 0 | bounding_box_union | 0.098816 | 0.098816 | 0.160061 | 0.116530 | 0.208735 |
| loki | legion_retrained | L-FREE | REUSED_HISTORICAL | 229 | 0 | bounding_box_union | 0.079178 | 0.079178 | 0.127911 | 0.124314 | 0.221137 |
| loki | p1 | G1 | REUSED_HISTORICAL | 229 | 0 | bounding_box_union | 0.076893 | 0.076893 | 0.126401 | 0.077554 | 0.143945 |
| loki | r1 | G1 | REUSED_HISTORICAL | 229 | 0 | bounding_box_union | 0.061839 | 0.061839 | 0.103666 | 0.045873 | 0.087722 |
| pal4vst | legion_intermediate | L-FREE | RUN_NEW | 1441 | 313 | official_pixel_artifact_mask | 0.059880 | 0.076495 | 0.095279 | 0.061973 | 0.116714 |
| pal4vst | legion_retrained | L-FREE | RUN_NEW | 1441 | 313 | official_pixel_artifact_mask | 0.050836 | 0.064942 | 0.081036 | 0.045726 | 0.087454 |
| pal4vst | p1 | G1 | RUN_NEW | 1441 | 313 | official_pixel_artifact_mask | 0.061279 | 0.076509 | 0.094792 | 0.052402 | 0.099585 |
| pal4vst | r1 | G1 | RUN_NEW | 1441 | 313 | official_pixel_artifact_mask | 0.097387 | 0.103134 | 0.138952 | 0.094196 | 0.172175 |
| synthscars | legion_intermediate | L-FREE | REUSED_HISTORICAL | 1000 | 0 | official_polygon_union | 0.223234 | 0.223234 | 0.321147 | 0.241450 | 0.388980 |
| synthscars | legion_retrained | L-FREE | REUSED_HISTORICAL | 1000 | 0 | official_polygon_union | 0.196195 | 0.196195 | 0.286687 | 0.200490 | 0.334014 |
| synthscars | p1 | G0 | REUSED_HISTORICAL | 1000 | 0 | official_polygon_union | 0.229544 | 0.229544 | 0.319323 | 0.236949 | 0.383118 |
| synthscars | r1 | G0 | REUSED_HISTORICAL | 1000 | 0 | official_polygon_union | 0.286588 | 0.286588 | 0.393974 | 0.267042 | 0.421521 |
| xaigd | legion_intermediate | L-FREE | RUN_NEW | 2419 | 247 | official_human_artifact_polygon_union | 0.083862 | 0.093399 | 0.129184 | 0.090972 | 0.166773 |
| xaigd | legion_retrained | L-FREE | RUN_NEW | 2419 | 247 | official_human_artifact_polygon_union | 0.077187 | 0.085964 | 0.119954 | 0.079918 | 0.148008 |
| xaigd | p1 | G1 | RUN_NEW | 2419 | 247 | official_human_artifact_polygon_union | 0.071161 | 0.078332 | 0.109794 | 0.070236 | 0.131253 |
| xaigd | r1 | G1 | RUN_NEW | 2419 | 247 | official_human_artifact_polygon_union | 0.080213 | 0.079206 | 0.119900 | 0.059704 | 0.112680 |

### Localization GT semantics

- **SynthScars Official1000:** official per-reference polygons, unioned by the evaluator at original resolution.
- **X-AIGD labeled_test:** official human perceptual-artifact polygons. Official records with `labels=[]` are retained and evaluated as all-zero GT masks, matching the official evaluator. Raw polygons remain authoritative; rasterization follows the official int32/clamping + `cv2.fillPoly` policy. For paper-level X-AIGD comparison, dataset-global FG IoU/F1 are the primary category-agnostic metrics; per-image means are supplementary.
- **PAL4VST test:** official pixel artifact masks.
- **LOKI229:** union of 687 official regional bounding boxes on 229 Fake images. This is box-derived localization GT and is not semantically equivalent to X-AIGD/PAL4VST perceptual-artifact masks.

## GenImage per-generator classification

| Model | Generator | N | Real | Fake | Accuracy | F1 | ROC-AUC | TNR | FPR |
|---|---|---|---|---|---|---|---|---|---|
| legion_retrained | adm | 12000 | 6000 | 6000 | 0.622000 | 0.411673 | 0.875066 | 0.979500 | 0.020500 |
| legion_retrained | biggan | 12000 | 6000 | 6000 | 0.755000 | 0.682231 | 0.962562 | 0.984000 | 0.016000 |
| legion_retrained | glide | 12000 | 6000 | 6000 | 0.760167 | 0.691599 | 0.952912 | 0.982500 | 0.017500 |
| legion_retrained | midjourney | 12000 | 6000 | 6000 | 0.806750 | 0.766771 | 0.950696 | 0.978167 | 0.021833 |
| legion_retrained | sdv4 | 12000 | 6000 | 6000 | 0.787667 | 0.736068 | 0.962765 | 0.983167 | 0.016833 |
| legion_retrained | sdv5 | 16000 | 8000 | 8000 | 0.792125 | 0.743443 | 0.962061 | 0.981875 | 0.018125 |
| legion_retrained | vqdm | 12000 | 6000 | 6000 | 0.578083 | 0.293173 | 0.836111 | 0.981167 | 0.018833 |
| legion_retrained | wukong | 12000 | 6000 | 6000 | 0.710500 | 0.603153 | 0.938608 | 0.981000 | 0.019000 |
| r1 | adm | 12000 | 6000 | 6000 | 0.526667 | 0.129102 | 0.754379 | 0.983167 | 0.016833 |
| r1 | biggan | 12000 | 6000 | 6000 | 0.753500 | 0.678967 | 0.961975 | 0.985667 | 0.014333 |
| r1 | glide | 12000 | 6000 | 6000 | 0.622500 | 0.408925 | 0.889720 | 0.983833 | 0.016167 |
| r1 | midjourney | 12000 | 6000 | 6000 | 0.713250 | 0.605345 | 0.908792 | 0.986667 | 0.013333 |
| r1 | sdv4 | 12000 | 6000 | 6000 | 0.688250 | 0.556911 | 0.915984 | 0.984667 | 0.015333 |
| r1 | sdv5 | 16000 | 8000 | 8000 | 0.688562 | 0.557735 | 0.918437 | 0.984375 | 0.015625 |
| r1 | vqdm | 12000 | 6000 | 6000 | 0.518833 | 0.101183 | 0.766515 | 0.983500 | 0.016500 |
| r1 | wukong | 12000 | 6000 | 6000 | 0.608750 | 0.372074 | 0.839156 | 0.985667 | 0.014333 |

## Benchmark integrity

| Dataset | Task | N | Manifest SHA256 | Revision |
|---|---|---|---|---|
| Internal2208 | classification | 2208 | fbc2d422c45fc871e06ed82d7b4c8fab88f1ed9e83d523a56591c93173ddd174 | "unified_forensics_split_v1" |
| SynthScars | localization | 1000 | b50dfae4a1a8691bb45da986a3648ebddb5f0cac0d55da0ad03aa30ae41fafb3 | "ee1cd553c2403f551fb1da60745cb2cdcb975e74" |
| X-AIGD | localization | 2419 | ab442977c11acb15ee2573b0aa2d2f8e22bb428c2bf2fe5298abd146bffb5c49 | "92180f32030507ab54a40d6f1b88f39d6cec8178" |
| PAL4VST | localization | 1441 | 2fbe8d273f09fa5db473d93c30762de3ca5369f9ee462d26bf383ce5812fa280 | "GitHub 9db472581715024e4fb69af7ffd2d64b5230bbef; Drive file 1h2geaBGrQVNKrjNPUs0oWdE5vXhdwTn_" |
| LOKI-localization | localization | 229 | bf6f9f12660fe1082c000b4760514391a4ab0730bb08beaf8b82cb26132fcf15 | {"huggingface": "314ddacc5080b024d6b8d962b448065cd54c9f42", "git": "9b2dac636e660aa5fd158be7888abaf3dd268140"} |
| LOKI-classification | classification | 2217 | 9376271a34603ff0ebcb7b21d9fc9a9219d7e28e4d350ffff6f8612d2a953c17 | {"huggingface": "314ddacc5080b024d6b8d962b448065cd54c9f42", "git": "9b2dac636e660aa5fd158be7888abaf3dd268140"} |
| AIGI-Holmes | classification | 99999 | 380325bc0cbba1e80044d0d2dbb40ff2f788865288bb47402c1d4b7967431982 | "3e856ce5ed44ac3b578bf36434829ea42953be02" |
| GenImage | classification | 100000 | f7844320a3d358e62cb709a13fc1b10f10786a59c400febd26bd4f02091cbcf7 | "71c983e6262684bc2c6b6af99582e8f568c259a5" |
| RAISE998 | classification | 998 | d792f5e684abc42cbddd8466019576a2b6ee9fd53c7fcbe5354322a94caa4e49 | "RAISE-1k official TIFF selection manifest" |

## Final frozen protocol

Localization: SynthScars Official1000; X-AIGD official `labeled_test`; PAL4VST official `test`; LOKI229. Models: P1, R1, public intermediate LEGION, and LEGION-retrained.

Classification: Internal2208 (historical frozen reuse), AIGI-Holmes official TestSet, GenImage official held-out partition, LOKI classification, and RAISE998 Real-only. The old project-created `AIGI-test` is not used as the main official benchmark.

No threshold tuning, benchmark-dependent sample filtering, test resplitting, or model selection is performed by the finalizer.

