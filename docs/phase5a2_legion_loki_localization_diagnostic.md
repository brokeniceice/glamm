# Phase 5A-2 — LOKI localization supplement

## Status

COMPLETED — CROSS-PROTOCOL DIAGNOSTIC ONLY.

R1 uses the existing known-Fake G1 condition, while LEGION uses official image-only L-FREE. The comparison does not have strict input-condition parity and is not a main baseline or an exact reproduction of the LEGION paper result.

## Frozen population and evaluator

- Manifest: `/data/yz/LOKI/loki_media_aggregate/legion_localization/manifest.jsonl`
- Manifest SHA256: `c9a29854717867419c2386be75bf3cf4c0366f43230e62b653594ca908628bc8`
- Ordered 229-ID SHA256: `0b9b182f3b1b2d9b747858250360360018d6f56e4176e39bf20488837e609503`
- GT: per-image union of filled regional xywh bounding boxes at original 512x512 resolution.
- No `[SEG]` / empty / invalid output: all-zero prediction and included in full-N primary metrics.
- LEGION: official prompt, free generation, `[SEG]` to SAM, logit `>0`, multiple masks union; no GT label, phrase, description or bbox is model input.

## Full-N=229 descriptive results

| Model | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 | valid `[SEG]`+mask |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 G1 | 0.061839 | 0.018219 | 0.103666 | 0.045873 | 0.087722 | 220/229 |
| LEGION L-FREE | 0.098816 | 0.059680 | 0.160061 | 0.116530 | 0.208735 | 229/229 |

Descriptive paired R1−LEGION mean FG IoU difference: `-0.036977`; bootstrap 95% CI `[-0.054498, -0.019294]`; wins/ties/losses `73/27/129`. Because the conditions differ, this is descriptive rather than a strict model-effect test.

## Both-valid diagnostic

N=220; R1 G1 mean FG IoU `0.064369`; LEGION L-FREE `0.098787`.

Machine-readable result: `outputs/phase5a2_legion_loki_reference/shared_results.json`.
