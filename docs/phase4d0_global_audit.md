# Phase 4D-0 — Global Evidence Audit and Paper Route Consolidation

## Executive decision

The project currently supports **one core method plus strong systematic analysis**, not a two-method paper. P1 / Phrase-Aligned Forensic SFT is the only deployable localization method with a matched, statistically supported effect. Phase 4C-A establishes that a stronger forensic-sensitive dense representation is learnable, but Phase 4C-C shows that the present Reader does not use the correct image or spatial organization. Therefore the Reader cannot be promoted to a second method contribution.

The most important missing scientific evidence is a causal bridge from the learned forensic representation to image-specific, spatially organized grounding. If method R&D continues, the next action should be the design-only Phase 4D-1 position-aware minimal discriminative test; full training is not justified.

## What the complete experiment record establishes

### 1. Autonomous language-to-grounding alignment is a real, actionable bottleneck

- official1000 historical P1: G0 / Phrase-Only / TF-Full `0.229544 / 0.332896 / 0.437609`.
- validation P1: G0 / Phrase-Only / TF-Full `0.148233 / 0.247362 / 0.342928`.
- validation Phrase Repair paired gain: `+0.108989`, CI `[+0.095485,+0.122814]`.
- matched P1−C0 official1000 G0 gain: `+0.049758`, CI `[+0.034534,+0.064341]`, wins/ties/losses `569/89/342`.

These results support a mixed bottleneck: autonomous target phrase/query construction is important, but persistent oracle failures and the residual TF ceiling show that language is not the entire problem.

### 2. P1 is the first and currently only strong method contribution

P1's unique target change explicitly associates authoritative target-region language with the `[SEG]` predictor while holding the mask, architecture, objective family, G0 prompt, evaluator, selector, and threshold fixed. The matched C0 and TF interaction make the best-supported interpretation: P1 learns phrase-conditioned autonomous language-to-grounding alignment.

The unavailable byte-identical original P1 step-0 snapshot prevents absolute training-path causality, but does not erase the matched controlled effect. P1 should remain the frozen paper method and canonical checkpoint.

### 3. Stage-II post-training search did not yield a second method

Generated replay, reward formulation, direct spatial-path optimization, joint language-mask continuation, projected AOGD, and raw-hidden AOGD preflight all failed their respective validation or directional gates. These failures are recipe-bounded. They do not prove that all post-training or direct supervision is impossible, but they do justify stopping incremental Stage-II search.

### 4. Forensic representation exists; downstream forensic use is unproven

Phase 4C-A adapter−raw CLIP validation FG IoU is `+0.027581`, CI `[+0.022586,+0.032726]`, under a matched population, target, geometry, threshold, and evaluator. Thus `FORENSIC_REPRESENTATION_LEARNABLE=TRUE`.

However Phase 4C-B forensic Reader does not exceed CLIP Reader, and Phase 4C-C gives matched approximately equal to cross-image and spatial shuffle. Matched greater than zero evidence only shows dependence on a nonzero visual feature distribution. The current mechanism is generic visual-conditioned query compensation, not demonstrated forensic evidence retrieval.

## Paper storyline

```text
Problem
  Autonomous forensic explanations do not reliably produce effective
  localization queries despite a much stronger oracle-conditioned path.

Finding 1
  The autonomous target phrase / causal grounding representation is a major,
  but not exclusive, bottleneck.

Method 1
  Phrase-Aligned Forensic SFT (P1) explicitly aligns target-region language
  with the grounding trajectory and improves matched deployable G0.

Finding 2
  A CLIP-anchored adapter can learn more decodable forensic-sensitive dense
  representation.

Finding 3
  A naive permutation-invariant Reader does not use that representation by
  correct image or spatial layout; its small gain is query compensation.

Open method question
  Can a position-aware interface make correct-image, correct-layout forensic
  evidence causally necessary for grounding?
```

Paper form: `ONE_METHOD + STRONG_ANALYSIS`.

`SECOND_METHOD_NOT_JUSTIFIED`

## Route feasibility and scores

Scores are 1–5, where 5 is most favorable. For cost, risk, and target-dependence rows, 5 means lower cost/risk/dependence. The target is not a pseudo-mask; the legacy row label is interpreted as dependence on incomplete visible-artifact supervision.

| Criterion | Route 1: position-aware | Route 2: forensic residual | Route 3: direct spatial supervision |
|---|---:|---:|---:|
| Directly addresses proven bottleneck | 5 | 3 | 3 |
| Mechanistic clarity | 5 | 4 | 4 |
| Novelty potential | 4 | 4 | 3 |
| Training cost, 5=low | 4 | 4 | 3 |
| Implementation risk, 5=low | 3 | 3 | 3 |
| Evaluation clarity | 5 | 4 | 4 |
| Dependence on weak pseudo-mask, 5=low | 4 | 4 | 2 |
| Reviewer defensibility | 4 | 3 | 3 |
| Expected information gain | 5 | 4 | 3 |
| **Total / 45** | **39** | **33** | **28** |

### Route 1 — Position-aware forensic grounding

This route directly repairs the structural reason that spatial shuffle cannot matter: the current Reader has no positional embedding. The minimum change is fixed 2D position coding before the existing single-layer query-conditioned interaction. Correct-image derangement and spatial-content shuffle with positions held fixed provide direct falsification. It best separates the competing hypotheses and is therefore recommended, but only as a tiny preflight.

### Route 2 — Forensic residual interface

The subtraction is meaningful only as `R_forensic = F_forensic − F0` inside the adapter arm, because both are `256×24×24` and the local blocks are residual. It is not meaningful to subtract raw `1024×24×24` CLIP directly from adapted `256×24×24` features. Even for the valid residual, scale/normalization and the possibility that useful semantics are removed must be audited. Residual-only reading does not itself fix permutation invariance, so it is secondary to Route 1.

### Route 3 — Direct spatial forensic supervision

This is compatible with the visible/explainable artifact-union target, but existing bounded direct spatial and joint supervision routes produced no validation gain. A new version would change more variables and depend more heavily on incomplete artifact-region coverage. It has lower information gain than first testing the proven interface defect.

## Dataset and submission boundary

- Internal data are balanced and category matched; manifests and pHash grouping are frozen.
- The official1000 overlap issue was repaired; the set is Fake-only and cannot establish FPR.
- The mask is official polygon-derived visible/explainable evidence, not pseudo or exhaustive forgery GT.
- Current method cross-generator and unseen forgery-style generalization remain missing.
- Historical internal-test/official1000 facts may be reported with their original protocol, but no sealed split was opened in Phase 4D-0.
- A final one-shot held-out evaluation is a separate submission-stage decision after the method/interface is frozen; it must not be used for route selection.

## Recommended Phase 4D-1 boundary

The companion design in `docs/phase4d1_minimal_experiment_proposal.md` uses fixed 2D positions, frozen P1/adapter/SAM, 512 optimizer steps per matched CLIP/forensic arm, and correct-image/cross-image/spatial-shuffle/zero interventions. Its primary endpoint is the paired matched−cross-image FG IoU effect. Raw matched IoU gain alone cannot pass the gate.

No experiment was executed in Phase 4D-0. Any Phase 4D-1 implementation, training, cache construction, or evaluation requires a new explicit user instruction.

## Final route decision

```text
PAPER_STATUS: ONE_METHOD_PLUS_ANALYSIS

CONTRIBUTION_1_STATUS: STRONG

CONTRIBUTION_2_STATUS: MEDIUM

MAIN_PROVEN_BOTTLENECK: Autonomous target-phrase/SEG-query construction is a major bottleneck, and the current Reader interface fails to use correct-image spatial evidence.

CURRENT_FORensic_REPRESENTATION_STATUS: LEARNABLE on matched validation probes; held-out generalization and downstream causal utility are unproven.

CURRENT_EVIDENCE_INTERFACE_STATUS: Generic visual-conditioned query compensation; image-specific utilization FALSE, spatial utilization FALSE, forensic-specific utilization INCONCLUSIVE.

MOST_IMPORTANT_MISSING_EVIDENCE: A causal demonstration that correct-image, correctly arranged forensic features improve autonomous grounding through a position-aware interface.

RECOMMENDED_NEXT_ROUTE: ROUTE_1

NEXT_EXPERIMENT_SHOULD_BE: Phase 4D-1 minimal fixed-2D-position Reader test with matched/cross-image/spatial-shuffle/zero controls; design only until separately authorized.

FULL_TRAINING_ALLOWED: NO
```
