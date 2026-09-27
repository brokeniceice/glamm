# Phase 6G.18 — Utility Context Architecture Audit

Status: **COMPLETE STOP — AUDIT ONLY**. No training, checkpoint mutation, test, Official1000, or OOD access was performed.

## 1. Executive conclusion

The current Utility is not a generic correction-quality assessor. It is a **source-reliability and cross-context compatibility estimator**: it constructs a language/SAM context from `(S64, q_seg, z_L)`, constructs a forensic-source context from `(F24, z_F24, geometry, support)`, compares the two, and predicts a spatial gate `U_F`. It never observes the actual Rectifier proposal `C` or resulting state `S_rect` during context construction.

The code supports three central conclusions.

1. `z_L` and `z_F24` are raw dense mask logits, not authenticity-classification logits or calibrated confidence. They are best described as **proposal maps accompanying their source representations**. Frozen evidential heads subsequently convert `(representation, raw mask logit)` into non-negative two-class evidence and Dirichlet opinions.
2. `C = S_rect - S64` is the exact signed proposed update that the Utility ultimately gates. It already encodes the Rectifier's interaction between SAM state, forensic evidence, geometry and support. `S_rect` additionally repeats the full absolute `S64` state. Since the language path already explicitly encodes `S64`, `C` is the cleaner primary variable for a forensic-side context redesign.
3. The recommended interaction point is **after deterministic forensic geometry alignment to 64x64 and before CMX**, not at the 24x24 source grid and not after the Utility head. At that point `E64` and `C` share SAM geometry, while source identity can still be preserved before cross-context interaction.

Primary recommendation: a **structured dual-source forensic encoder using `C`**, with separate lightweight encoders for aligned evidence and correction followed by an explicit merge into one `ForensicContext64`. The best control is the same architecture with `S_rect`. A cheaper joint-projection version is the first complexity control. FiLM-style modulation is scientifically secondary because it defines `C` as conditioning rather than as co-equal forensic-side content.

This differs categorically from Phase6G.17. G17 retained the complete old context and added a late residual to its output logit. The recommended design rebuilds `Fctx64 = Phi_F(F24, z_F24, C)` before CMX, so every downstream compatibility feature can depend on the actual proposal.

## 2. Current Utility exact architecture

The deployed graph materialized by `scripts/phase4g1q_conditional_utility.py::utility_forward` is:

```text
Language source opinion                 Forensic source opinion
S64,q_seg,z_L                           F24,z_F24
 -> frozen LanguageEvidentialHead        -> frozen ForensicEvidentialHead
 -> evidence [B,2,64,64]                 -> evidence [B,2,24,24]
 -> Dirichlet masses/posterior            -> Dirichlet masses/posterior
                                            -> deterministic geometry map to 64x64

Trainable language context              Trainable forensic context
S64,q_seg,z_L                           aligned F64, aligned z_F64, support64
 -> LanguageContext64 [B,64,64,64]       -> ForensicContext64 [B,64,64,64]
                  \                       /
                   CMXJointRectification
                  channel and spatial bidirectional rectification
                              |
                     LocalCrossExchange
                  bidirectional local 7x7 attention
                              |
 [Lr,Fr,Lr*Fr,|Lr-Fr|,cos,p_L,p_F,source conflict]
                              |
              Conv3x3 -> GN -> GELU -> Conv1x1
                              |
                 utility_logit -> sigmoid -> U_F
                              |
                     S_adapt = S64 + U_F*C
```

Parameter ownership in the current CSCU Utility is: `LanguageContext64=116,352`, `ForensicContext64=63,488`, CMX rectification `6,682`, local cross exchange `33,536`, and Utility head `151,745`. The two evidential source heads are frozen.

### What each operation actually does

- **Projection/normalization:** source-specific `1x1` projections reduce 256 channels to 64; `q_seg` is projected from 256 to 64 and broadcast spatially; raw mask logits are projected from one channel to 16. GroupNorm and GELU are representation construction, not calibration.
- **Spatial mapping:** `_map_forensic()` calls `resample_clip_to_original_normalized`. It is deterministic coordinate resampling from CLIP crop geometry to the requested grid and constructs hard support. For masses it fills unsupported cells with vacuous mass; for ordinary features/logits it fills them with zero. It has no learned feature transformation.
- **CMX-style channel rectification:** global average/max summaries of both streams generate separate sigmoid channel weights. Each stream receives weighted features from the other through an additive residual.
- **CMX-style spatial rectification:** channel mean/max maps of both streams generate two spatial sigmoid maps; each stream again receives a weighted residual from the other.
- **Local cross exchange:** four-head, 7x7 local cross-attention lets language positions read supported forensic neighbors and forensic positions read language neighbors. It preserves two identities rather than immediately collapsing them.
- **Comparison:** the head receives both streams, product, absolute difference, local cosine, two source posteriors, and evidential conflict. It therefore mixes reliability estimation with representation compatibility.
- **Output:** `sigmoid(utility_logit)*support` is a dense scalar intervention strength. It gates the Rectifier correction; it does not generate that correction.

## 3. Tensor provenance audit

| Tensor | Exact producer and inputs | Shape | Frozen/detached and numerical meaning |
|---|---|---:|---|
| `S64` | Frozen SAM image encoder feature, cached/reloaded through `Phase4FStore`; current C1-specific code obtains the same spatial tensor via `phase4f_spatial_batch` | `[B,256,64,64]` | Frozen BF16 source; detached inside both context and evidential heads. Dense absolute SAM image state; no sigmoid/softmax. |
| `q_seg` | The C1 G0 generation path's actual `[SEG]` hidden state, projected by the existing `text_hidden_fcs`/SEG path to SAM prompt dimension and cached with the C1 checkpoint hash | `[B,256]` | Frozen BF16 global task/query vector. No sigmoid/softmax. It is phrase/task-conditioned and is also consumed by frozen SAM's prompt decoder. |
| `z_L` | `FrozenP1SAMPath(q_seg, S64)` gives low-resolution mask logits; `sam_lowres_to_original_normalized(..., output_hw=(256,256))` maps them into canonical original-normalized coordinates | `[B,1,256,256]` | Frozen preliminary C1/SAM mask **raw logit**, dense; not an authenticity logit, probability, or standalone learned confidence. It is directly segmentation-related and deterministically depends on `S64`, `q_seg`, and frozen SAM decoder. |
| `F24` | Current source: frozen CLIP block11+17 patch features -> frozen selected cross-layer attention fusion -> frozen selected 1024-to-256 projection | `[B,256,24,24]` | Frozen BF16 dense forensic representation; no sigmoid/softmax. It contains localized forensic appearance but not SAM state. |
| `z_F24` | Frozen `Conv2d(256,1,1)` training head loaded by `phase6g13_utility_train.load_head`, applied as `head(F24.float())` | `[B,1,24,24]` | Frozen dense **raw mask logit**. It is fully predictable from `F24` by the known linear head, so it is not independent information. It carries an explicitly mask-oriented projection of `F24`. |
| `C` | Frozen collapsed Phase6G.15-A2 Rectifier/Translator: `S_rect - S64`; equivalently the support- and gamma-governed injected correction before Utility scaling | `[B,256,64,64]` | Frozen FP32 view in G16/G17 and detached before Utility. Dense signed update, no sigmoid. It depends on `S64`, `F24`, geometry, learned cross-attention/translator, gamma, and hard support. |
| `S_rect` | Rectifier output `image_embeddings`, before Utility gating | `[B,256,64,64]` | Frozen proposed resulting SAM state, `S64+C`; dense, no sigmoid. Contains the entire absolute base state plus the signed proposal. |
| `support/M_valid` | Rectifier support is determined by SAM/evidence coordinates and evidence validity; Utility support comes from CLIP-to-original geometry mapping. In the current center-crop setting both identify where forensic evidence is defined, but they originate in separate alignment implementations and must be asserted equal/aligned in a new interface | `[B,1,64,64]` for Utility; Rectifier internally `[B,4096]` | Boolean deterministic geometry, frozen. It is not learned reliability. |
| `LanguageContext64` | `Conv1x1(S64)->64`, `Linear(q_seg)->64` broadcast, `Conv1x1(z_L)->16` after bilinear resize, concat 144 channels, `Conv3x3->64`, GN/GELU | `[B,64,64,64]` | Trainable Utility representation; all three source tensors are detached. Dense semantic/task/proposal context. |
| `ForensicContext64` | deterministically aligned `F24/z_F24`, then `Conv1x1(F64)->64`, `Conv1x1(zF64)->16`, concat support, `Conv3x3->64`, GN/GELU, multiply support | `[B,64,64,64]` | Trainable Utility representation over frozen sources. Dense source-evidence/proposal context restricted to valid crop support. |
| `utility_hidden` | First three layers of `ConditionalUtilityHead`: comparison `[B,263,64,64] -> Conv3x3(263,64)->GN->GELU` | `[B,64,64,64]` | Trainable dense pre-logit. It already mixes language, forensic, posterior and conflict information. |
| `utility_logit` | Final `Conv1x1(64,1)` | `[B,1,64,64]` | Raw dense gate logit; sigmoid is applied only afterward to obtain `U_F`. |

### Source-head supervision and redundancy

The frozen `LanguageEvidentialHead` consumes `(S64,q_seg,z_L)` and the frozen `ForensicEvidentialHead` consumes `(F24,z_F24)`. Each produces two non-negative class-evidence channels through `softplus`; `evidence_to_dirichlet` turns these into masses, ignorance and posterior. Their fitting objective was TMC expected cross-entropy plus an annealed incorrect-evidence KL on dense binary masks. Thus their posteriors are mask/evidence-source predictions, not the C1 H2 authenticity classifier.

`z_F24 = linear_head(F24)` exactly, so `(F24,z_F24)` is not two independent sensors. It is best interpreted as **representation plus its frozen dense mask proposal**. Likewise `z_L` is deterministically decoded from `(S64,q_seg)` by frozen SAM, making `(S64,q_seg,z_L)` **semantic image state plus task query plus its preliminary mask proposal**, not “semantic state plus calibrated reliability.” The evidential heads add learned uncertainty/reliability materialization later.

## 4. LanguageContext responsibility

`LanguageContext64` represents the current SAM visual state under the actual C1 segmentation request, augmented with the mask state that frozen SAM would produce before forensic correction. `q_seg` is global but broadcast to every spatial cell; `S64` and resized `z_L` retain spatial variation. Therefore it is more accurately named **task-conditioned base-state context** than language-only context.

It implicitly contains redundant views by design: `z_L` is decoded from `S64,q_seg`, but the decoder is nonlinear and frozen, so exposing its output supplies a task-aligned summary that a shallow context encoder need not reconstruct.

## 5. ForensicContext responsibility

`ForensicContext64` represents aligned CLIP-derived forensic features, their frozen dense mask proposal, and geometric availability. It does not contain the Rectifier's interpretation of those features against `S64`; it only contains the source evidence before correction generation.

The overall current Utility therefore performs three mixed duties:

1. source reliability estimation through frozen Dirichlet posteriors/conflict;
2. language-forensic cross-context compatibility through CMX and local exchange;
3. intervention-strength prediction through the final comparison head.

It cannot directly perform proposal-quality assessment because neither `C` nor `S_rect` enters those computations.

## 6. C vs S_rect information analysis

### `C`

`C` is principally the proposed change. It preserves signed channel and spatial direction, magnitude, hard support, Rectifier gamma, and the learned interaction between `S64` and forensic tokens. It does **not** preserve the absolute SAM state by itself. It overlaps with `F24/z_F24` because both contribute to its generation, but is not a deterministic shallow restatement of them: it additionally includes base-state queries, cross-attention geometry, translation and support.

Semantically `(F24,z_F24,C)` means: **raw forensic source, source's mask-oriented proposal, and the concrete SAM-space update derived from that source in this image/task state**.

### `S_rect`

`S_rect` is the proposed resulting state. It contains all of `S64` plus `C`, so it is sufficient for absolute post-proposal content but does not explicitly separate what changed. In a forensic path it duplicates `S64`, which already dominates the language path.

This repetition can be meaningful only if the architecture explicitly compares `S64` with `S_rect` as “current versus proposed state.” If `S_rect` is merely concatenated into a forensic encoder, the model must rediscover subtraction while coping with a large shared component. That is an avoidable confound.

Semantically `(F24,z_F24,S_rect)` means: **raw forensic source, its mask proposal, and the proposed absolute SAM state**. It is viable as a controlled alternative, but less responsibility-coherent than `C`.

The identity `S_rect=S64+C` does not make the two inputs equivalent to a finite shallow network with separate normalization and regularization. Their conditioning, shortcut opportunities and inductive biases differ.

## 7. Literature mechanism review

| Paper | Task/domain | Inputs and fusion location/operator | Alignment/source identity/reliability | Transferable lesson | What does not transfer |
|---|---|---|---|---|---|
| [CMX](https://arxiv.org/abs/2203.04838) | RGB-X semantic segmentation | Separate modality features are rectified bidirectionally by channel/spatial weighting, then exchanged/fused | Features are stage-aligned; source identity is preserved through rectification; uncertainty motivates selective interaction | Encode heterogeneous sources separately, align them, rectify before mixing | Sensor modalities are independently observed; `C` is a derived proposal, not a second sensor |
| [FiLM](https://arxiv.org/abs/1709.07871) | Language-conditioned visual reasoning | Conditioning input generates feature-wise affine modulation of a visual representation | Conditioning identity is not retained as a co-equal stream | Modulation is appropriate when one variable changes how another should be interpreted | It does not establish that an actual correction proposal should be demoted to conditioning |
| [TokenFusion](https://arxiv.org/abs/2204.08721) | Multimodal transformer vision | Dynamically replaces uninformative tokens using projected aligned inter-modal tokens | Explicit positional alignment and selective use; modality streams largely preserved | Extra information need not enter unconditionally; alignment must be explicit | Token replacement and transformer stacks are unnecessary and prohibited here |
| [RDFNet](https://openaccess.thecvf.com/content_iccv_2017/html/Park_RDFNet_RGB-D_Multi-Level_ICCV_2017_paper.html) | RGB-D segmentation | Modality-specific encoders plus residual fusion/refinement blocks | Separate streams precede residual complementary fusion | Residual fusion can preserve a strong base while learning complementary content | Multi-level encoders/decoder would add capacity and supervision confounds |
| [RFBNet](https://arxiv.org/abs/1907.00135) | RGB-D segmentation | Interaction stream progressively aggregates modality-specific features; gated residual blocks return complementary information | Source and interaction streams coexist | An explicit interaction representation is preferable to raw concatenation when sources have different roles | Deep bottom-up multi-stage interaction is excessive for this audit |
| [CAFuser](https://arxiv.org/abs/2410.10791) | Robust multimodal driving perception | Modality adapters align inputs; condition token guides fusion | Alignment precedes condition-aware selection | Separate alignment from reliability/condition control | Environmental condition is global/exogenous; `C` is dense and endogenous |
| [Review Residuals](https://arxiv.org/abs/2606.31859) | Transformer residual gating | Gate is conditioned on current state and proposed update before addition | Explicit state/update roles | A gate deciding `S + U*C` should in principle observe the proposed `C` | Evidence is on Transformer residual streams at much larger scale; it is conceptual precedent, not direct validation for frozen SAM localization |

## 8. Transferable design lessons

1. **Align first, then interact.** Heterogeneous representations should enter a common spatial frame before learned fusion. Here that means forensic evidence is mapped to 64x64 before meeting `C/S_rect`, which already lives in SAM 64x64 geometry.
2. **Preserve roles before merging.** Evidence and proposed update should receive separate projections/normalization because their distributions and semantics differ.
3. **Reliability is not geometry.** `support` is deterministic availability, whereas source posterior/conflict and `U_F` are learned reliability quantities.
4. **Conditioning is a semantic claim.** FiLM is not merely a cheaper concat: it asserts that `X` controls interpretation of evidence rather than contributing co-equal content.
5. **The decision should see the proposal before cross-context compatibility is computed.** Late logit repair, as in G17, cannot change CMX, local exchange, product/difference/cosine, or evidential-context comparisons.

## 9. Candidate A — Joint projection after alignment

```text
F24,z_F24 --geometry--> F64,zF64,support
F64 -> Conv1x1 256->64
zF64 -> Conv1x1 1->16
X64 (C or Srect, 256 channels)
concat [64 + 16 + 1 + 256] = 337 channels
Conv3x3 337->64 + GN + GELU
* support
-> Fctx64 -> unchanged CMX/exchange/comparison/head
```

This replaces, rather than patches, the old `ForensicContext64`. A literal 337-to-64 3x3 merge is about 194k parameters (versus 63.5k currently); using a mandatory `1x1 256->64` X projection before a 145-to-64 merge lowers it to about 100k and avoids X dominating by channel count. Normalization must occur per source before concatenation.

- **C version:** natural; direct joint representation of evidence and signed proposal.
- **Srect version:** feasible but risks copying `S64` through a high-capacity shortcut.
- **Main question:** is joint context construction sufficient, without preserving two internal forensic substreams?
- **Confound:** increased merge capacity; parameter-matched controls are required.

## 10. Candidate B — Structured dual-source forensic encoder

```text
Evidence branch:
  aligned F64,zF64,support -> existing-style Phi_E -> E64 [B,64,64,64]

Proposal branch:
  X [B,256,64,64] -> Conv1x1 256->64 -> GN -> GELU -> X64

Merge:
  concat(E64,X64) -> Conv3x3 128->64 -> GN -> GELU -> *support
  -> Fctx64 -> unchanged CMX/exchange/comparison/head
```

Estimated new X encoder plus merge is about 90.5k parameters before accounting for the reused evidence encoder. This is the clearest design: alignment occurs before fusion, roles remain inspectable, and `Fctx64` is genuinely defined from all forensic-side inputs. It is not G17 because `X` influences CMX, local exchange and every comparison feature.

- **C version:** most coherent recommended primary arm.
- **Srect version:** useful matched scientific control; consider exposing `[S64,Srect]` only in a separate explicit state-comparison candidate, otherwise duplication is implicit.
- **Main question:** does preserving evidence/proposal identities before merging enable proposal-aware forensic context?
- **Confound:** extra branch capacity; compare with a zero-X arm of identical architecture and a shuffled-X diagnostic after selection.

## 11. Candidate C — Conditioning/modulation context

```text
E64 = Phi_E(aligned F64,zF64,support)
X -> Conv1x1 256->128 -> split gamma_X,beta_X
Fctx64 = GN(E64) * (1 + bounded(gamma_X)) + beta_X
Fctx64 *= support
-> unchanged CMX/exchange/comparison/head
```

The X-to-modulation projection is about 33.2k parameters. It is compact and preserves evidence as a base representation.

- **C version:** plausible if `C` is interpreted as “how this proposal should alter the interpretation of forensic evidence,” but this reverses the more direct responsibility: evidence generated `C`, and Utility must judge `C`.
- **Srect version:** less natural because an absolute resulting state is not an obvious conditioning code; it also allows the repeated base state to modulate evidence everywhere.
- **Main question:** is proposal information better used as an interpretation condition than as forensic content?
- **Confound:** modulation operator and bounding choice add an inductive-bias difference beyond information availability.

Candidate C is retained as a secondary mechanism audit, not the first experiment.

## 12. C-version vs Srect-version

| Criterion | `C` | `S_rect` |
|---|---|---|
| Semantic role | proposed signed update | proposed absolute resulting state |
| Includes base `S64` | only indirectly through Rectifier computation | exactly and additively |
| Redundancy with LanguageContext | moderate/structured | high/direct |
| Direction and magnitude of change | explicit | implicit; network must compare/subtract |
| Natural for Candidate A/B | yes | control, not preferred |
| Natural for modulation | possible but secondary | weak |
| Shortcut risk | update norm/direction shortcut | absolute-state and duplicated-S64 shortcut |

The recommended main input is `C`. `S_rect` should be a matched alternative, not concatenated alongside `C`, because `(S64,C,Srect)` is algebraically redundant.

## 13. Redundancy / information audit

The following conclusions require no new training:

- `z_F24` has perfect architectural predictability from `F24`: it is a frozen 1x1 linear head. It is redundant in Shannon-information terms but useful as an explicit task-oriented coordinate of that representation.
- `z_L` is deterministically generated from `S64,q_seg` by frozen SAM. It similarly adds no independent information but exposes the frozen decoder's nonlinear proposal.
- `S_rect` is algebraically determined by `(S64,C)`. Putting it into the forensic path duplicates the exact base state already encoded by LanguageContext.
- `C` is computed from `(S64,F24,geometry)` by a frozen nonlinear Rectifier/Translator. It is derived, but it is a task-relevant sufficient artifact of that interaction that the shallow Utility otherwise would have to emulate.

Existing Phase6G.16 diagnostics further show that correction-aware Utility is not merely reading correction norm: correlation between correction norm and gate was modest and negative (`Pearson=-0.253`, `Spearman=-0.254` spatially; per-image `-0.118/-0.126`). Norm-preserving spatial shuffle caused a large mean absolute gate change (`0.24294`, 95% CI `[0.23877,0.24712]`), with support and norms preserved. This supports spatial/directional information in `C`, but does not prove that incorporating it improves final localization.

No new CCA/SVCCA or linear probe was run: doing so would require a newly specified sample population and probe selector and would exceed a purely static audit. A future preflight may compute fixed, non-selected descriptive correlations on internal TRAIN only.

## 14. Recommended architectures

### R1 — Structured dual-source `C` forensic context (primary)

- **Dataflow:** aligned `(F64,zF64,support)->E64`; `C->C64`; concat and 3x3 merge -> `Fctx64`; unchanged CMX/exchange/comparison/head.
- **Shapes:** `E64,C64,Fctx64=[B,64,64,64]`.
- **New modules:** `C` 1x1 projection/GN/GELU and 128-to-64 merge.
- **Reused:** frozen evidential heads, mapper, LanguageContext, CMX, exchange, comparison definition and gate semantics.
- **Removed:** old one-shot ForensicContext merge.
- **Estimate:** roughly 90.5k new proposal/merge parameters plus reused evidence encoder; total should be capacity-matched explicitly.
- **Why here:** `C` and evidence are spatially aligned, source identity is preserved, and actual proposal is known before compatibility computation.
- **Precedent:** CMX separate encoding/alignment before rectification; Review Residuals current-state/update organization.
- **Difference from G17:** changes context construction and all downstream interactions, not output logit refinement.
- **Ablations:** real-C versus zero-C identical architecture; C versus spatially shuffled C; C versus Srect matched architecture.

### R2 — Capacity-controlled joint projection with `C` (simplicity control)

- **Dataflow:** independently normalized projected `F64`, `zF64`, `C`, and support are concatenated once and mapped to 64 channels before CMX.
- **New modules:** C projection and one joint merge; no second internal interaction block.
- **Estimate:** approximately 100k with C compressed to 64 channels; exact count frozen before training.
- **Why:** tests whether explicit structured dual encoding is necessary or whether aligned joint construction suffices.
- **Main confound:** less interpretability and stronger early mixing.
- **Ablations:** identical zero-C and real-C arms; no late delta head.

### R3 — `C`-conditioned modulation (secondary only)

- **Dataflow:** old-style evidence context first, then C generates spatial/channel affine modulation before CMX.
- **Estimate:** about 33.2k for a 256-to-128 modulation projection.
- **Why retained:** tests the alternative hypothesis that C should condition evidence interpretation.
- **Why not primary:** it treats the actual proposal as metadata rather than content, and operator choice introduces an additional assumption.

For each architecture, `S_rect` is a matched information-role control. It should not silently replace C in the primary design.

## 15. Controlled experiment proposal

The next authorized phase should first freeze one architecture definition and run an initialization/invariance preflight, then train only the Utility context stack under the existing internal TRAIN/internal-validation contract.

Recommended minimal matrix:

```text
A0: current source-aware Utility (exact historical reuse)
A1: R1 architecture, X=0, identical capacity and initialization to A2
A2: R1 architecture, X=C
A3: R1 architecture, X=S_rect
```

Controls and rules:

- frozen C1, CLIP/fusion/projection, Rectifier/Translator, evidential heads and SAM;
- same train/validation population, order, loss (`seg+relative+ranking`), optimizer, budget and selector;
- cross/shuffle negatives must pass through the frozen Rectifier so X matches the perturbed forensic source;
- no late residual head, auxiliary objective, threshold tuning or architecture-specific LR;
- A1 versus A2 is the causal test for correction information; A2 versus A3 tests update versus resulting-state organization; A0 is historical architectural reference, not the strict capacity control;
- report parameter counts, initialization hashes, gate distributions, paired IoU/F1 statistics and post-selection zero/channel/spatial/flip diagnostics;
- internal validation only until the architecture decision is frozen.

Only if A2 is stable should R2 be run as a simplicity/structure control. R3 should be tested only if A2 establishes that C is useful but R1 does not explain whether it is content or conditioning.

## 16. Unsupported designs / rejected ideas

- **G17-style `old_logit + delta(h_U,C)`:** rejected for this question because C arrives after context interaction.
- **A second gate after Utility:** duplicates responsibility without redefining forensic context.
- **Raw 24x24 fusion with C:** rejected because C is already SAM-aligned 64x64; downsampling C discards proposal detail and entangles geometry with representation learning.
- **Keeping X independent until LocalCrossExchange:** too late; CMX rectification would still be proposal-blind.
- **Keeping X separate inside CMX as a third modality:** scientifically possible but changes CMX structure and attribution more than necessary; not first-line minimal design.
- **Concatenating both C and Srect:** algebraically redundant with S64 and creates shortcut/capacity ambiguity.
- **Transformer/MoE/new decoder/new objective:** outside the complexity and responsibility contract.

## 17. Final recommendation

### Q1 — What does `LanguageContext64` represent?

It is a dense **task-conditioned base SAM state**: absolute SAM image features, the C1 `[SEG]` task query, and frozen SAM's preliminary mask-logit proposal. It is neither purely language nor a calibrated reliability representation.

### Q2 — What does `ForensicContext64` represent?

It is a dense **aligned forensic-source context**: CLIP-derived forensic representation, its frozen linear dense-mask proposal, and geometric support. It contains source evidence but not the actual Rectifier proposal or resulting SAM state.

### Q3 — How should C or Srect enter?

`C` should become a native, separately encoded component of `ForensicContext64` **after geometry alignment and before CMX**. It should not remain separate until LocalCrossExchange and should not be only late conditioning. `S_rect` should be evaluated through the same interface as a matched alternative, but is not preferred because it duplicates `S64` and hides the signed update.

```text
PRIMARY_RECOMMENDATION = STRUCTURED_DUAL_SOURCE_FORENSIC_CONTEXT_WITH_C
PRIMARY_INTERACTION_POINT = AFTER_64x64_ALIGNMENT_BEFORE_CMX
SRECT_ROLE = MATCHED_INFORMATION_ORGANIZATION_CONTROL
G17_LATE_RESIDUAL_EQUIVALENT = NO
PHASE6G18 = COMPLETE STOP
```
