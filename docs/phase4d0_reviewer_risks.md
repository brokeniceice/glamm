# Phase 4D-0 — Reviewer and Dataset Risk Audit

## Reviewer A — Multimodal LLM

| Attack | Severity | Existing answer | Residual risk |
|---|---|---|---|
| “P1 is only prompt/target engineering with richer labels.” | High | Matched C0/P1 protocol isolates the training-target change; G0 gains `+0.049758`, CI excludes 0; matched TF interaction shows learned phrase-conditioned adaptation | Novelty framing must emphasize autonomous language-to-grounding alignment, not the literal string template |
| “Oracle gains do not establish deployable reasoning.” | High | Paper can lead with deployable G0 and keep Phrase-Only/TF/Phrase Repair explicitly diagnostic | Large residual oracle gap remains and must be presented, not hidden |
| “The Reader is claimed as multimodal evidence retrieval without evidence use.” | Critical if overclaimed | Phase 4C-C directly falsifies image-specific and spatial utilization for the current Reader | The manuscript must call it a failed/generic query compensator, not Method 2 |

## Reviewer B — Image Forensics

| Attack | Severity | Existing answer | Residual risk |
|---|---|---|---|
| “The masks are pseudo-masks or inaccurate forgery ground truth.” | High | They are deterministic unions of official SynthScars polygons for all references; no regeneration or relabeling | They annotate visible/explainable artifact regions, not every forged pixel; claims and losses must respect that scope |
| “The method exploits SynthScars generator/source artifacts and does not generalize.” | Critical | pHash split/test leakage was repaired; internal splits are fixed and content matched | Cross-generator localization is missing; fake training source remains SynthScars only |
| “Forensic cues are not actually used by the MLLM.” | Critical | Phase 4C-A proves representation learnability; Phase 4C-C honestly shows current downstream use is not image-specific | A second forensic method claim is unavailable until a causal interface test passes |

## Reviewer C — Vision / Segmentation

| Attack | Severity | Existing answer | Residual risk |
|---|---|---|---|
| “Localization quality is modest and the TF ceiling remains far higher.” | High | The paper can quantify G0/Phrase-Only/TF and frame the autonomous–oracle gap as the central problem | P1 remains far from oracle; this limits absolute segmentation claims |
| “SAM is doing the work; the proposed method does not improve spatial modeling.” | High | P1's contribution is language-to-grounding alignment, and tiny-set/trace audits separate capacity from deployment behavior | No new spatial decoder method is supported; do not claim one |
| “Direct supervision failures imply the spatial path is underdeveloped or poorly optimized.” | Medium | Prompt mismatch was found and corrected retrospectively; all matched trained checkpoints remained below step0; gradients/updates were verified | Seen-train fitting attribution is still unresolved; the negative result is recipe-bounded, not universal |

## Dataset and evaluation risk audit

| Risk | Audit status | Evidence | Required handling |
|---|---|---|---|
| Fake/Real balance | Controlled internally | Train/val/test are exactly balanced; category quotas match per side | Report source composition, not only balance |
| Synthetic fake source | High residual risk | Fake training population is SynthScars | Do not claim general forgery universality; add future cross-generator evaluation |
| Real-source mixture | Partial control | OpenImages, PASS, COCO, FFHQ, iNaturalist are versioned; RAISE is held out | Report source-wise false positives; note COCO pretraining familiarity and FFHQ/source-format shortcuts |
| Mask construction | Audited | Official polygons, per-image all-reference union, fixed rasterization | Never call pseudo-mask or pixel-perfect manipulation GT; distinguish from LEGION phrase-level units |
| Exact/near-duplicate leakage | Strongly controlled for frozen split | SHA256/pHash grouping; 18 official-test overlaps removed from internal pool; zero known post-fix overlap | Preserve frozen manifests and hashes |
| Category bias | Partial | Real/Fake internal category counts match | Per-image official1000 category labels are unavailable; category-level parity with LEGION remains unresolved |
| Generator/model bias | Missing | No generator-held-out fake training/evaluation protocol for the main method | Submission claim must be dataset-scoped unless new frozen benchmark is authorized |
| Codec/source shortcuts | Partial | Common loader; JPEG/noise robustness exists | Source and encoding remain correlated in raw data; perturbation robustness is not source generalization |
| Threshold and aggregation | Strongly controlled | Canonical logit threshold 0; direct batch=1 for formal classification boundary; per-image/global kept separate | No post-hoc threshold tuning or metric mixing |
| Selector/test isolation | Strong for recent phases | Phase 3/4 selectors use validation; internal test and official1000 sealed after historical final evaluations | Future final evaluation must be one-shot after method freeze |
| Official1000 semantics | Limited | Held-out Fake-only, zero known overlap, historical P1 results exist | It cannot measure FPR/balanced accuracy and cannot be used for development |
| Evaluation split fixedness | Strong | Frozen manifests and hashes; no loader resampling | Cite manifest version and preserve set equality checks |

## Generalization checklist

| Dimension | Status | Reason |
|---|---|---|
| Cross-generator generalization | `MISSING` | No completed generator-held-out localization evaluation for P1/current method |
| Semantic-category generalization | `PARTIAL` | Internal matching exists; authoritative official category mapping does not |
| Real false-positive behavior | `PARTIAL` | Balanced internal and historical RAISE evidence exist, but current final source-wise audit is incomplete |
| Unseen forgery style | `MISSING` | JPEG/noise changes are corruptions, not unseen generation mechanisms |
| Unseen source dataset | `PARTIAL` | Historical RAISE/LOKI classification exists; no current unified final evaluation |

## Highest paper risks in priority order

1. External/cross-generator generalization is the largest submission-level evidence gap.
2. A second method claim based on the current Reader would be directly contradicted by Phase 4C-C.
3. Mask semantics and metric/category mismatch can trigger serious reviewer objections if described loosely.
4. P1 novelty must be framed around learned autonomous language-to-grounding alignment, supported by the matched control, rather than around adding a phrase field alone.
