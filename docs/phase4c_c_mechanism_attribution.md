# Phase 4C-C — Image-Specific Evidence Utilization Attribution

## 1. Executive Summary

- Correct-image evidence dependence: **FALSE**.
- Spatial-organization dependence: **FALSE**.
- Query-compensator mechanism: **PARTIAL**.
- Mechanistic difference between CLIP and forensic Readers: **INCONCLUSIVE**.
- Proceed to Phase 4C-D: **NO**. No next phase was started.

## 2. Protocol

Frozen Phase 4C-B selected Readers and Phase 4C-A sources were evaluated with the unchanged Phase 4C-B decoder, inverse resize, logit threshold 0.0, target, metrics, and paired bootstrap. Full baseline reproduction uses 1,106 validation Fake images. The four-query intervention matrix uses the fixed 1,076-image common legal-query population. Cross-image evidence uses the saved seed-3407 bijective derangement. Spatial shuffle uses one saved 576-token permutation. No model was trained and internal test/official1000 remained sealed.

## 3. Baseline Reproduction

Both Readers reproduced Phase 4C-B G0, Phrase-Only, and TF-Full within absolute mean-IoU tolerance 1e-4. Phrase-Repair uses the historical exact token-span replacement; current and historical G0 projected queries were exact for all 1,078 valid cases.

## 4. Main Intervention Results

| Reader | Query | Evidence | Mean FG IoU | Median FG IoU |
|---|---|---|---:|---:|
| clip_reader | G0 | matched | 0.158637 | 0.055204 |
| clip_reader | G0 | cross_image | 0.158663 | 0.055409 |
| clip_reader | G0 | spatial_shuffle | 0.158637 | 0.055204 |
| clip_reader | G0 | global_repeat | 0.158403 | 0.054633 |
| clip_reader | G0 | zero | 0.153575 | 0.046550 |
| clip_reader | phrase_repair | matched | 0.240541 | 0.181290 |
| clip_reader | phrase_repair | cross_image | 0.240591 | 0.180783 |
| clip_reader | phrase_repair | spatial_shuffle | 0.240542 | 0.181290 |
| clip_reader | phrase_repair | global_repeat | 0.242438 | 0.181288 |
| clip_reader | phrase_repair | zero | 0.259987 | 0.197131 |
| clip_reader | phrase_only | matched | 0.219489 | 0.157908 |
| clip_reader | phrase_only | cross_image | 0.219468 | 0.157596 |
| clip_reader | phrase_only | spatial_shuffle | 0.219492 | 0.157908 |
| clip_reader | phrase_only | global_repeat | 0.221598 | 0.157883 |
| clip_reader | phrase_only | zero | 0.246249 | 0.176306 |
| clip_reader | tf_full_context | matched | 0.333192 | 0.288869 |
| clip_reader | tf_full_context | cross_image | 0.333170 | 0.285315 |
| clip_reader | tf_full_context | spatial_shuffle | 0.333192 | 0.288869 |
| clip_reader | tf_full_context | global_repeat | 0.334170 | 0.288187 |
| clip_reader | tf_full_context | zero | 0.344531 | 0.303662 |
| forensic_reader | G0 | matched | 0.158147 | 0.055779 |
| forensic_reader | G0 | cross_image | 0.158179 | 0.055225 |
| forensic_reader | G0 | spatial_shuffle | 0.158147 | 0.055779 |
| forensic_reader | G0 | global_repeat | 0.158019 | 0.055403 |
| forensic_reader | G0 | zero | 0.153304 | 0.046326 |
| forensic_reader | phrase_repair | matched | 0.243522 | 0.181131 |
| forensic_reader | phrase_repair | cross_image | 0.243362 | 0.180378 |
| forensic_reader | phrase_repair | spatial_shuffle | 0.243521 | 0.181131 |
| forensic_reader | phrase_repair | global_repeat | 0.243808 | 0.180194 |
| forensic_reader | phrase_repair | zero | 0.260377 | 0.195501 |
| forensic_reader | phrase_only | matched | 0.223754 | 0.164863 |
| forensic_reader | phrase_only | cross_image | 0.223675 | 0.164714 |
| forensic_reader | phrase_only | spatial_shuffle | 0.223754 | 0.164863 |
| forensic_reader | phrase_only | global_repeat | 0.224151 | 0.165365 |
| forensic_reader | phrase_only | zero | 0.246178 | 0.178520 |
| forensic_reader | tf_full_context | matched | 0.335180 | 0.289194 |
| forensic_reader | tf_full_context | cross_image | 0.335331 | 0.289276 |
| forensic_reader | tf_full_context | spatial_shuffle | 0.335180 | 0.289194 |
| forensic_reader | tf_full_context | global_repeat | 0.335596 | 0.290735 |
| forensic_reader | tf_full_context | zero | 0.344469 | 0.304349 |

## 5. Image-Specific Evidence Test

- CLIP matched-minus-cross G0: -0.000026 [-0.000123, +0.000066].
- Forensic matched-minus-cross G0: -0.000032 [-0.000163, +0.000109].

## 6. Spatial Evidence Test

- CLIP matched-minus-shuffle/global G0: +0.000000 [+0.000000, +0.000000] / +0.000233 [-0.000008, +0.000482].
- Forensic matched-minus-shuffle/global G0: -0.000000 [-0.000000, +0.000000] / +0.000128 [-0.000094, +0.000374].
- Because the Reader has no positional embedding, fixed token permutation is expected to be permutation-invariant; the intervention directly audits this architectural property.

## 7. Evidence-Absent Test

- CLIP matched-minus-zero G0: +0.005061 [+0.002344, +0.007769].
- Forensic matched-minus-zero G0: +0.004843 [+0.002330, +0.007328].

## 8. Query-Quality Interaction

- CLIP gap-TF vs G0 gain Spearman rho: +0.2806, CI [0.22352419225625145, 0.3366394072273985].
- Forensic gap-TF vs G0 gain Spearman rho: +0.2650, CI [0.20620947704512751, 0.3218061717404309].
- Quartile values and Phrase-Repair gap correlations are recorded in `phase4c_c_statistics.json` and Figures 3–4.

## 9. CLIP vs Forensic Mechanism Comparison

The primary comparison is intervention sensitivity rather than raw matched IoU. Full paired sensitivity results are recorded under `clip_vs_forensic_sensitivity`; forensic-specific downstream use is reported conservatively as **INCONCLUSIVE**.

## 10. Failure Cases

Cases were selected automatically by fixed metric order for matched-over-cross, matched-approximately-cross, G0 improvement, and TF harm. The selection manifest and panels are under `qualitative/`.

## 11. Decision

```text
IMAGE_SPECIFIC_EVIDENCE_UTILIZATION: FALSE
SPATIAL_EVIDENCE_UTILIZATION: FALSE
READER_AS_QUERY_COMPENSATOR: PARTIAL
FORENSIC_SPECIFIC_DOWNSTREAM_UTILIZATION: INCONCLUSIVE
PROCEED_TO_PHASE_4C_D: NO
```

Evidence:

1. Matched-vs-cross paired effects are reported on the identical 1,076 images with a shared derangement.
2. Matched-vs-shuffle/global/zero isolates token order, dense variation, and evidence absence.
3. Reader gain is related to frozen P1 TF/Phrase-Repair gaps without changing query tokens.
4. CLIP/forensic sensitivity is compared per image, not inferred from raw means alone.
5. Phrase-Repair/TF effects bound whether the Reader merely compensates weak autonomous queries.
6. Optional same-class cross, per-image shuffle, and dataset-mean evidence controls were not run and are not implied.
