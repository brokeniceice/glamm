# Phase 4D-0 — Evidence Graph

## Central scientific graph

```text
Autonomous G0 is far below authoritative trajectories
  ├─ P1 phrase-aligned SFT improves matched autonomous G0
  ├─ Phrase Repair / Phrase-Only recover a large fraction, but not all
  └─ TF retains a large residual advantage
          ↓
Mixed bottleneck: autonomous query construction + downstream spatial path
          ↓
Generated replay / reward tuning / direct spatial / joint / AOGD routes fail
          ↓
Ask whether task-aligned forensic visual evidence can be learned
  ├─ standalone FEPN: global yes, dense no vs CLIP
  └─ CLIP-anchored residual adapter: dense yes under matched probe
          ↓
Ask whether P1 actually uses the learned evidence
  ├─ Reader improves weak G0 slightly
  ├─ forensic Reader does not exceed CLIP Reader
  ├─ matched ≈ wrong-image
  └─ matched ≈ spatial shuffle; Reader has no positional embedding
          ↓
Current interface is a generic visual-conditioned query compensator,
not demonstrated image-specific forensic evidence retrieval
          ↓
Missing causal bridge: correct image + correct spatial organization
must matter before a second method or held-out claim is justified
```

## Question A — Is autonomous language grounding a bottleneck?

### Evidence

- Phase 2D official1000: mean `TF−G0` FG IoU `+0.221020`; 119 severe cases had G0/TF `0.067323/0.813124`.
- Phase 3A P1 official1000: G0 / Phrase-Only / TF-Full `0.229544 / 0.332896 / 0.437609`.
- Phase 3C.0 validation: G0 / Phrase-Only / TF-Full `0.148233 / 0.247362 / 0.342928`; Phrase Repair on the paired eligible population gains `+0.108989`, CI `[+0.095485,+0.122814]`.
- Phase 3C.0 retains 476 cases with G0, Phrase-Only, and TF all at or below 0.30.

### Mechanistic interpretation

| Contrast | Supported interpretation | Unsupported overclaim |
|---|---|---|
| TF vs G0 | The autonomous trajectory/query is a major bottleneck; the same frozen downstream can do substantially better under authoritative context | TF is deployable; all error is language error |
| Phrase-Only vs G0 | Target phrase availability materially improves the predictor trajectory | A pure phrase-semantic causal effect; position, length, and hidden trajectory also change |
| Phrase Repair vs G0 | Errors in generated target-region content/query construction are practically recoverable | Single-factor phrase selection causality |
| TF vs Phrase-Only | Explanation/context trajectory and/or downstream spatial representation contributes additional capability | Long reasoning is intrinsically beneficial or the decoder is proven sufficient |
| Persistent oracle failures | A downstream visual/spatial limitation remains | SAM or mask decoder is the unique bottleneck |

### Answer

`AUTONOMOUS_LANGUAGE_GROUNDING_BOTTLENECK: SUPPORTED`.

The safest description is that P1 and the oracle interventions identify a mixed bottleneck: autonomous target semantics and causal predictor representation are important, while persistent oracle failures show that language repair alone is insufficient.

## Question B — What did P1 learn?

### Strongest controlled evidence

The matched Phase 3A.1 comparison changes the Fake target protocol while matching seed, optimizer, LR, batch, budget, data, and selector. P1 exceeds the new C0 by `+0.049758` official1000 G0 FG IoU, CI `[+0.034534,+0.064341]`, with `569/89/342` wins/ties/losses. The matched TF interaction is `+0.249720`, CI `[+0.228733,+0.271278]`: phrase context hurts the matched old-target C0 but helps P1 strongly.

### Answer

P1 learned a phrase-conditioned mapping from autonomous forensic language trajectories into the causal grounding representation. It is more precise to call this **autonomous language-to-grounding alignment** than generic localization improvement. The unavailable byte-identical original P1 step-0 snapshot prevents a strict claim about every hidden training-path variable, but the matched control is strong enough for a core method claim.

`METHOD_CONTRIBUTION_1: STRONG`.

## Question C — Can forensic-sensitive dense representation be learned?

### Matched protocol

- Source: frozen P1 CLIP ViT-L/14-336 layer `-2`, CLS removed, `1024×24×24`.
- Population: identical 8,836 train Fake and 1,106 validation Fake.
- Geometry: identical 336 resize/crop, mask transform and inverse mapping.
- Target: official polygon-derived all-reference union.
- Metric: original-space per-image FG IoU at logit threshold 0.

### Evidence

| Representation arm | Validation FG IoU |
|---|---:|
| Raw CLIP linear | 0.143842 |
| CLIP projection | 0.138264 |
| CLIP forensic adapter | 0.171424 |

Adapter−projection is `+0.033160`, CI `[+0.027825,+0.038722]`; adapter−raw is `+0.027581`, CI `[+0.022586,+0.032726]`. The output is non-collapsed and the three residual local blocks are the matched architectural difference.

### Answer

`FORENSIC_REPRESENTATION_LEARNABLE: TRUE` under the frozen matched validation probe. This proves task-aligned dense information is more decodable after the adapter. It does not prove held-out generalization, image-specific downstream use, or that the information improves P1.

## Question D — Does the current Reader use the forensic evidence?

### Phase 4C-B observation

CLIP and forensic Readers improve validation G0 over P1 by `+0.006100` and `+0.005624`, but forensic−CLIP is `-0.000476`, CI `[-0.001034,+0.000095]`. Strong Phrase-Only and TF queries are harmed rather than improved.

### Phase 4C-C causal input interventions

- Matched−cross-image G0: CLIP `-0.000026`, CI `[-0.000123,+0.000066]`; forensic `-0.000032`, CI `[-0.000163,+0.000109]`.
- Matched−spatial-shuffle G0 is numerically zero for both Readers. Because there is no positional embedding, token permutation invariance is an architectural property.
- Matched−zero G0: CLIP `+0.005061`, CI `[+0.002344,+0.007769]`; forensic `+0.004843`, CI `[+0.002330,+0.007328]`.
- Reader gain vs frozen `TF−G0` gap has positive Spearman correlation: CLIP `0.2806`, forensic `0.2650`.

### Answer

```text
IMAGE_SPECIFIC_EVIDENCE_UTILIZATION: FALSE
SPATIAL_EVIDENCE_UTILIZATION: FALSE
READER_AS_QUERY_COMPENSATOR: PARTIAL
FORENSIC_SPECIFIC_DOWNSTREAM_UTILIZATION: INCONCLUSIVE
PROCEED_TO_PHASE_4C_D: NO
```

Matched greater than zero rejects a pure query-only MLP description, but matched approximately equal to wrong-image and spatially shuffled evidence rejects real evidence retrieval. The most defensible mechanism is **generic visual-conditioned query compensation**: the nonzero feature distribution regularizes or adjusts weak autonomous queries, without demonstrated dependence on the correct image or spatial layout.

## Bottleneck decomposition

| Layer | Status | Evidence |
|---|---|---|
| Image acquisition / preprocessing | UNKNOWN as a scientific bottleneck | Geometry and preprocessing are audited, but source/codec bias remains and was not causally removed |
| Visual semantic representation | LIKELY BOTTLENECK | Raw CLIP probe is limited; learned adapter makes evidence more decodable |
| Forensic-sensitive representation | NOT A PROVEN ABSENCE; LEARNABILITY SUPPORTED | Phase 4C-A adapter improves matched dense probing significantly |
| Language reasoning / explanation | SUPPORTED BOTTLENECK | G0–oracle gaps and Phrase Repair gains; lexical/semantic reward limitations remain |
| Grounding phrase / `[SEG]` query | SUPPORTED BOTTLENECK | P1 matched gain, TF interaction, query-quality/Reader-gain correlation |
| Language–visual evidence interaction | SUPPORTED BOTTLENECK FOR CURRENT INTERFACE | 4C-C matched≈cross and matched≈shuffle |
| Spatial grounding representation | LIKELY BOTTLENECK | Persistent oracle failures; existing bounded direct optimization and current Reader fail; attribution to one module remains unresolved |
| SAM decoding | UNKNOWN, not uniquely implicated | Tiny-set overfit and TF swaps show capacity; 3D.2 does not establish global sufficiency or unique failure |
| Binary mask threshold / postprocess | NOT A SUPPORTED PRIMARY BOTTLENECK | Fixed threshold and matched evaluator audits; no evidence that tuning it is the scientific solution |

## Generalization evidence graph

| Question | Status | Existing evidence | Missing evidence |
|---|---|---|---|
| Semantic-category generalization | PARTIAL | Internal splits are category matched; official1000 contains four aggregate categories | Authoritative official1000 per-image category mapping is unavailable; P1 category-stratified matched effects are not established |
| Real false-positive control | PARTIAL | Balanced internal classification and historical RAISE evaluation exist; P1/P3 internal robustness was measured | Final frozen current-method RAISE/source-wise FPR and confidence intervals are missing |
| Unseen forgery style | MISSING for the current method | official1000 is held-out SynthScars; perturbation robustness is not generator shift | Cross-generator fake benchmark with localization-compatible evidence target |
| Unseen source dataset | PARTIAL | Historical RAISE/LOKI classification evidence exists | Current P1/any future method source-held-out validation, especially localization where labels permit it |
| Cross-generator generalization | MISSING | No completed generator-held-out localization experiment supports the main method | Pre-registered generator/source-held-out benchmark |

## Next scientific question

The next method experiment should not ask whether a deeper Reader can gain IoU. It should ask whether preserving 2D position makes **correct-image, correct-layout forensic evidence causally necessary**. That single question distinguishes:

- `H1`: forensic information exists, but the current permutation-invariant interface cannot access it spatially;
- `H2`: the adapter's probe gain cannot be transferred into language-conditioned segmentation by this route.
