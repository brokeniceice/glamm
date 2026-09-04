# Phase 5A-2 — R1 G0 vs LEGION public intermediate L-FREE

## Status

COMPLETED. This is a reference/diagnostic comparison of the released LEGION intermediate LE checkpoint, not a reproduction of the paper-final checkpoint and not a main paper baseline.

## Frozen protocol

- Shared official1000 manifest: `/data/yz/groundingLMM_official/outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl`
- Manifest SHA256: `fff3c3839e54d831955f8055501882f27a408c885e0524c2d7bf5d3edbe8c700`
- Ordered 1000-ID SHA256: `34b136b24365aedf4878a86b3f12856a974991d3615f1c603213792c99a2e114`
- LEGION source commit: `d21535dd45f6fea509337a83095966f0b86ac924`
- LEGION mode: official L-FREE prompt, free generation, `[SEG]` to SAM, mask logit `> 0`, all returned masks unioned at original resolution.
- R1 mode: existing frozen P1 G0; no retraining, prompt change, or classification gate.
- Primary metric: all 1,000 images. No `[SEG]` / no mask is an all-zero prediction and therefore receives IoU/F1=0 because every official1000 image has a nonempty Fake target.
- Conditional `both_valid_seg_and_mask` rows are coverage-conditioned diagnostics only, never the primary comparison.
- R1 Phrase and R1 TF-PHRASE: N/A for this LEGION comparison; no artificial LEGION phrase/teacher-forcing condition was constructed.

## Primary all-sample results

| Condition | Model | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 | valid `[SEG]`+mask |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | R1 G0 | 0.286588 | 0.232139 | 0.393974 | 0.267042 | 0.421521 | 968/1000 |
| original | LEGION L-FREE | 0.223234 | 0.161411 | 0.321147 | 0.241450 | 0.388980 | 999/1000 |
| jpeg70 | R1 G0 | 0.289210 | 0.243713 | 0.397971 | 0.264150 | 0.417910 | 980/1000 |
| jpeg70 | LEGION L-FREE | 0.225599 | 0.166046 | 0.324528 | 0.236418 | 0.382424 | 998/1000 |
| jpeg80 | R1 G0 | 0.286972 | 0.240894 | 0.395111 | 0.266674 | 0.421062 | 980/1000 |
| jpeg80 | LEGION L-FREE | 0.220686 | 0.160120 | 0.318300 | 0.242709 | 0.390613 | 999/1000 |
| gaussian5 | R1 G0 | 0.277030 | 0.224266 | 0.379402 | 0.248886 | 0.398573 | 909/1000 |
| gaussian5 | LEGION L-FREE | 0.221999 | 0.160563 | 0.319491 | 0.250561 | 0.400718 | 999/1000 |
| gaussian10 | R1 G0 | 0.265002 | 0.217321 | 0.364091 | 0.246811 | 0.395907 | 880/1000 |
| gaussian10 | LEGION L-FREE | 0.216565 | 0.160064 | 0.315025 | 0.235084 | 0.380677 | 999/1000 |

## Paired primary comparison

Values are R1 minus LEGION; a positive value favors R1. CI is deterministic paired bootstrap 95%; p is two-sided Wilcoxon signed-rank.

| Condition | Δ mean FG IoU (R1−LEGION) | Δ mean FG F1 (R1−LEGION) |
| --- | --- | --- |
| original | 0.063354 [0.048736, 0.077773], p=1.19e-18 | 0.072826 [0.055759, 0.089781], p=1.64e-17 |
| jpeg70 | 0.063611 [0.049214, 0.078111], p=8.29e-20 | 0.073442 [0.056546, 0.090197], p=7.41e-19 |
| jpeg80 | 0.066285 [0.051890, 0.081095], p=1.87e-19 | 0.076811 [0.059648, 0.094302], p=9.47e-19 |
| gaussian5 | 0.055031 [0.039459, 0.070808], p=1.06e-14 | 0.059911 [0.041294, 0.078477], p=6.42e-13 |
| gaussian10 | 0.048436 [0.032878, 0.064238], p=3.74e-11 | 0.049066 [0.030571, 0.067713], p=5.13e-09 |

## Robustness: original minus corrupted condition

Positive values denote degradation from original. These are independently paired within each model; they are not a cross-model superiority test.

| Condition | R1 Δ mean FG IoU | LEGION Δ mean FG IoU | R1 Δ mean FG F1 | LEGION Δ mean FG F1 |
| --- | --- | --- | --- | --- |
| jpeg70 | -0.002622 [-0.007633, 0.002186], p=0.154 | -0.002365 [-0.011519, 0.006385], p=0.655 | -0.003997 [-0.010277, 0.002173], p=0.144 | -0.003381 [-0.014299, 0.007184], p=0.551 |
| jpeg80 | -0.000384 [-0.005432, 0.004704], p=0.435 | 0.002547 [-0.005991, 0.011018], p=0.59 | -0.001138 [-0.007377, 0.005222], p=0.521 | 0.002847 [-0.007687, 0.013249], p=0.686 |
| gaussian5 | 0.009557 [0.002547, 0.016822], p=0.573 | 0.001235 [-0.008530, 0.011312], p=0.101 | 0.014572 [0.005667, 0.023811], p=0.481 | 0.001656 [-0.010383, 0.013873], p=0.124 |
| gaussian10 | 0.021586 [0.013444, 0.030050], p=0.000375 | 0.006668 [-0.003727, 0.016990], p=0.0916 | 0.029883 [0.019451, 0.040763], p=0.000393 | 0.006123 [-0.006427, 0.018439], p=0.148 |

## Both-valid `[SEG]` diagnostic

This subset excludes any image without a valid localization output from either model. It describes localization quality conditional on both pipelines emitting a segment, and must not replace the all-sample primary result.

| Condition | N | R1 mean FG IoU | LEGION mean FG IoU | Δ mean FG IoU (R1−LEGION) |
| --- | ---: | ---: | ---: | --- |
| original | 967 | 0.296310 | 0.225383 | 0.070927 [0.056308, 0.085324], p=1.73e-22 |
| jpeg70 | 978 | 0.295271 | 0.226479 | 0.068792 [0.054788, 0.083007], p=1.12e-22 |
| jpeg80 | 979 | 0.293074 | 0.220596 | 0.072478 [0.057967, 0.086752], p=2.66e-22 |
| gaussian5 | 908 | 0.304235 | 0.220532 | 0.083703 [0.068933, 0.098645], p=5.93e-28 |
| gaussian10 | 880 | 0.301138 | 0.216758 | 0.084381 [0.068719, 0.099572], p=4.28e-27 |

## Boundary of interpretation

The released `legion_LE` checkpoint is an intermediate public checkpoint. These results establish its behavior under the frozen shared protocol only. The formal main baseline route remains retraining official LEGION on the shared training data before making paper-level comparative claims.

Machine-readable artifact: `outputs/phase5a2_legion_public_reference/shared_results.json`.
