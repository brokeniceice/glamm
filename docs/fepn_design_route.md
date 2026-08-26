# Current Status

Phase 4A is complete under `GATE_GLOBAL_ONLY_EVIDENCE_LEARNED`. Phase 4B-G is authorized as a bounded global-interface study; dense integration remains prohibited.

# Current Working Hypothesis

Task-aligned global forensic evidence may improve autonomous language reasoning before dense evidence is separately redesigned.

# Current Authorized Phase

Phase 4B-G — Global Forensic Evidence Injection into original P1.

# Next Decision Gate

Validation-only comparison of P1, PROJ-ONLY and PROJ-LORA under canonical Detection/G0/language gates.

# Research Route

The route follows evidence-first, integration-later:

1. Phase 4A: standalone global+dense evidence learnability.
2. Phase 4B: controlled global-to-language and dense-to-grounding interface study, only after separate authorization.
3. Phase 4C: freeze an effective interface and test the minimum necessary joint adaptation.
4. Phase 4D: freeze architecture, interfaces, trainable set, loss, selector, thresholds and evaluator before held-out evaluation.

Long-term FEPN output is conceptually split into global forensic evidence `E_global` and dense forensic evidence `F_dense`. Global evidence must eventually influence autonomous generation rather than only a classifier. Dense evidence must eventually interact with a language-conditioned grounding query rather than act as an unconditional mask prior. These are working hypotheses, not frozen Phase 4B architectures.

# Design Constraints

- Evidence first, integration later.
- Fixed residual filtering is an input transformation, not a frozen forensic expert.
- Phase 4A trains no P1, LLM, CLIP, SAM, SEG predictor, NPR, SRM or FOCAL parameter.
- Fake samples alone receive dense supervision; Real samples are not assigned zero-mask ground truth.
- Primary selection is validation Fake mean foreground IoU, with foreground F1 tie-break and a global-classification safety gate.
- No validation threshold tuning; dense logit threshold is fixed at zero.
- Historical expert results are background only unless population, target, geometry, evaluator and aggregation match.
- A failed Phase 4A does not authorize direct integration or an architecture zoo; it triggers an explicit representation-design decision.

# Decision History / Change Log

## Version 0.1 — 2026-08-24 / route initialization

- Previous assumption: Stage-II language/spatial/joint post-training or autonomous-oracle representation alignment might close the autonomous-oracle grounding gap.
- New evidence: generated replay, reward optimization, spatial-only and joint post-training, projected-256 AOGD, and raw-4096 AOGD preflight did not yield a supported route. Phase 3G stopped because its mandatory one-step mean alignment direction was invalid.
- Route change: FEPN becomes the current main route, beginning with standalone evidence learnability rather than immediate P1 integration.
- Reason: the next experiment should test whether explicit task-aligned forensic visual evidence is learnable before testing whether the MLLM can consume it.
- Status: post-hoc route transition based on completed Stage-II evidence; Phase 4A protocol is frozen before observing Phase 4A results.

# Current Status Footer

CURRENT MAIN ROUTE: FEPN

CURRENT AUTHORIZED PHASE: Phase 4B-G — global interface only

P1 STATUS: Frozen best-supported training baseline

POST-TRAINING STATUS: Closed after Phase 3G

HELD-OUT TEST: Sealed


## Version 0.2 — 2026-08-24 / Phase 4A outcome

- Previous assumption: FEPN-v0 RGB+Residual evidence may exceed matched frozen CLIP while retaining global authenticity information.
- New evidence: selected epoch 6; FEPN−CLIP IoU delta -0.075590, CI [-0.085346, -0.065996]; global Accuracy 0.930832, ROC-AUC 0.981889.
- Route change: terminal gate `GATE_GLOBAL_ONLY_EVIDENCE_LEARNED`; Phase 4B remains separately authorized only.
- Reason: application of the frozen Phase 4A dense and global gates.
- Status: confirmatory with respect to the frozen Phase 4A protocol; any later interface design remains a new phase.

## Version 0.3 — 2026-08-24 / Phase 4B-G route reset and authorization

- Previous assumption: Phase 4B might jointly study global-to-language and dense-to-grounding integration.
- New evidence: Phase 4A learned strong global classification evidence but its dense head was significantly below the matched CLIP spatial baseline.
- Route change: authorize only the global-first interface `P4B-GLOBAL-FEPN`; postpone the original broader integration design and prohibit dense FEPN integration.
- Frozen method: epoch-6 FEPN is evaluated and frozen; its 128D pre-classifier pooled representation is projected by one fixed 2-layer MLP into four continuous 4096D tokens placed after the 576 original visual embeddings and before text.
- Controlled arms: original P1, projector-only, and projector+P1-LoRA. Language CE is the only training objective; both Real and Fake receive four evidence tokens.
- Scope: canonical internal validation only. Internal test and official1000 remain sealed.
- Status: protocol frozen before formal training or checkpoint validation results.


## Version 0.4 — Phase 4B-G outcome

- Final gate: `GATE_GLOBAL_FEPN_NOT_USEFUL_TO_P1`.
- Selected PROJ-ONLY step: 1000; selected PROJ-LORA step: 500.
- PROJ-LORA vs P1 G0 mean FG IoU delta: -0.027954, 95% CI [-0.039136, -0.016378].
- Matched SFT trigger: False; evidence-specific attribution supported: False.
- Internal test and official1000 remained sealed. The Phase 4B-G hard stop is active; the next route requires new explicit authorization.


## Version 0.5 — Phase 4C-A outcome

- Phase 4B-G closed under `GATE_GLOBAL_FEPN_NOT_USEFUL_TO_P1`: global tokens maintained/improved classification, G0 significantly degraded, and LoRA did not recover the interface. The global-token injection route is closed.
- Route hypothesis changed to specializing rather than replacing the existing CLIP spatial representation.
- Phase 4C-A was authorized as the bounded matched raw/projection/three-block-adapter learnability study; its frozen working hypothesis was that CLIP should be specialized rather than replaced.
- Phase 4C-A final gate: `GATE_CLIP_ANCHORED_FORENSIC_ADAPTER_LEARNABLE`.
- Evidence Reader authorized: True.
- Internal test and official1000 remained sealed.

## Version 0.6 — Phase 4C-B authorization

- Phase 4C-A outcome: `GATE_CLIP_ANCHORED_FORENSIC_ADAPTER_LEARNABLE`. Adapter exceeded CLIP-PROJ and RAW CLIP with paired-bootstrap lower bounds above zero; residual-block attribution was supported.
- Phase 4C-B is AUTHORIZED directly by the Phase 4C-A positive result.
- Current working hypothesis: a language-derived grounding query may improve autonomous localization by actively retrieving task-aligned spatial forensic evidence.
- Only the matched single-layer Evidence Reader and zero-initialized residual beta are trainable; P1, CLIP, Adapter and SAM remain frozen.

## Version 0.7 — Phase 4C-B outcome

- Final gate: `GATE_SPATIAL_REACCESS_EFFECTIVE_FORENSIC_GAIN_NOT_SEPARABLE`.
- Fresh validation G0 Reader attribution is reported in `docs/phase4c_b_evidence_reader.md`.
- Phase 4C-A Adapter downstream utilized: False.
- Recommended next step: do not start joint adaptation; revisit interface after report.
- Hard stop active; no joint adaptation or held-out evaluation was started.
