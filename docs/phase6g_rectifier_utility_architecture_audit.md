# R1 Rectifier + Utility Architecture Audit

**Status:** COMPLETE — READ-ONLY ARCHITECTURE AUDIT  
**Date:** 2026-09-19  
**Firewall:** no training, no evaluation rerun, no checkpoint/model modification, no new module, no internal test, no Official1000, no OOD.

## 1. Executive conclusion

The stronger `block11+17` signal does enter the Rectifier. It is retained by K/V projection and is amplified at the raw cross-attention response (`R2`). The first non-stable conversion occurs after `R2`, while converting an easily decoded forensic response into a correction that a frozen SAM decoder can use. This is not evidence that geometry-aware cross-attention failed, and it is not evidence that the 256-channel interface is too narrow.

The exact selected Phase6G.13 graph also rules out a simple explanation that the main translator is merely an old block22 translator reused on a new source. Its main translator is Phase6G.10 A0, trained from matched initialization on `block11+17` evidence and selected at epoch 8. A source-interface mismatch remains a useful general concern, but it is **not the leading explanation for this selected model**.

The most defensible diagnosis is mixed:

1. **Translation/objective mismatch:** raw forensic decodability and frozen-SAM-compatible correction are different objectives. Phase6G.10 directly demonstrates their rankings can reverse.
2. **Optimization/parameterization mismatch:** the two-stage main translator is useful as a composition, but its functional split is not independently established. A zero-initialized side branch improves a frozen main, whereas jointly freeing main and side does not add stable gain.
3. **Utilization mismatch:** Utility produces a stable downstream gain, but its current `relative` and `ranking` targets train source-relative forensic usefulness and mismatch sensitivity—not side-specific correction benefit. Its gate therefore should not be interpreted as selecting when the side path helps.

Consequently, R1's hierarchy should be preserved: evidence-to-SAM interaction, spatial support, residual correction, and a downstream Utility layer all have empirical or strong mechanistic support. The exact double-projection decomposition, main/side factorization, and Utility objective semantics remain open.

**Recommended ONE next scientific question:**

> With block11+17 evidence and the same frozen C1/SAM, does retraining a *single matched SAM-compatible translator* from the same initialization outperform both (a) the present double-projection main and (b) frozen-main plus side, under the identical final localization objective?

This question separates translator parameterization from side-path compensation without adding an architecture zoo.

## 2. Historical facts that must be preserved

- R1 is useful overall: this audit does not revisit whether `C1 + new R1` should exist.
- `block11+17` is stronger dense forensic evidence than block22. Phase6G.8/8A/8B also show that projection-only `1024→256` is sufficient and the former three-local-block adapter is redundant for this source.
- Cross-attention is not the primary failure: Phase6G.9 shows the block11+17 advantage grows at raw response `R2`.
- Phase6G.10 proves that higher `Delta` linear decodability does not imply a better SAM-compatible correction.
- Phase6G.11 supports a zero-init residual side on a frozen main; Phase6G.12 does not support extra gain from jointly freeing main+side.
- Phase6G.13 supports Utility on the combined correction, but does not support a side-specific selector interpretation.

## 3. Exact current computation graph

### 3.1 Selected artifact provenance

| Component | Selected artifact | SHA256 | Selection/provenance |
|---|---|---|---|
| block11+17 fusion | `outputs/phase6g2_multilevel_attention/phase6g2a/selected.pt` | `85fe439d...6139ed4` | epoch 4; trained fusion+probe, later frozen |
| projection-only interface | `outputs/phase6g8b_multilevel_evidence_interface_audit/interfaces/projection_256/selected_projection.pt` | `9d7733aa...065302` | epoch 11; projection and dense head |
| main Rectifier | `outputs/phase6g10_post_attention_projection_bypass/arms/A0/selected_checkpoint.pt` | `8d4caa91...3a4fc` | block11+17 matched training; epoch 8, update 8840 |
| zero-init side | `outputs/phase6g11_zero_init_correction_side_path/training/selected_side.pt` | `60fcdd2f...fad` | epoch 4; main frozen during training |
| Utility | `outputs/phase6g13_utility_on_main_side_rectifier/selected_utility.pt` | `442a187e...017f4` | epoch 10; main+side frozen during training |

All are frozen in the audited selected inference graph. “Trainable” below means historical training ownership, not audit-time state.

Initialization audit: the fusion attention/projections and Phase6G.8B `Conv1x1` projection/head use their module-default random initialization under the recorded phase seed, followed by their own selected dense-probe training. Phase6G.10 reconstructs the Rectifier from matched initialization rather than loading the old block22 selected Rectifier: `Linear` layers use PyTorch `nn.Linear` defaults, LayerNorm starts at weight 1/bias 0, and `gamma_main` starts at .01. The side `Conv1x1` weight and bias are explicitly all zero. Phase6G.13 calls the historical `a2` Utility constructor: source evidential heads and temperatures are loaded and frozen from Phase4G-1C, while the 371,803-parameter Utility branch is seed-3407 random initialization before Phase6G.13 training. The frozen C1/SAM state is inherited unchanged from its selected checkpoint lineage.

### 3.2 Graph

```text
frozen CLIP image encoder
  ├─ block11 patch tokens [B,576,1024]
  └─ block17 patch tokens [B,576,1024]
       ↓ selected per-position cross-layer MHA (historically trainable; now frozen)
       Q=block17; K,V={block11,block17}; gamma_fusion=0.0120294
       ↓ [B,576,1024] → [B,1024,24,24]
selected projection-only Conv1x1 1024→256 (262,400 params; frozen)
       ├─ F24 [B,256,24,24]
       └─ selected dense Conv1x1 head → z_F24 [B,1,24,24]

C1/frozen SAM image encoder → S_base [B,256,64,64]
C1 [SEG] state             → q_seg [B,256]
C1 language mask logit     → z_L [B,1,256,256]

Rectifier main:
  S tokens [B,4096,256] --semantic LayerNorm + coordinate PE--> Q projection
  F tokens [B, 576,256] --forensic LayerNorm + coordinate PE--> K projection
  raw F values [B,576,256] -------------------------------> V projection
  geometry-biased 8-head cross-attention (sigma=.25)
      ↓ R2, input to attention out_proj [B,4096,256]
  out_proj 256→256
      ↓ R3 [B,4096,256]
  rectification.projection 256→256
  × scalar gamma_main=0.0609020
  × shared semantic_support [B,4096,1]
      ↓ Delta_main [B,256,64,64]

Rectifier side:
  the same R2 [B,256,64,64]
  → zero-initialized Conv1x1 256→256
  × the same semantic_support
      ↓ Delta_side [B,256,64,64]

Delta_total = Delta_main + Delta_side
S_rect = S_base + Delta_total

Utility inputs (all source tensors detached):
  S64 [B,256,64,64], q_seg [B,256], z_L [B,1,256,256]
  F24 [B,256,24,24], z_F24 [B,1,24,24], clip geometry/support
      ↓ context projection + bidirectional CMX-style rectification
      ↓ bidirectional local 7x7 cross exchange
      ↓ comparison features and ConditionalUtilityHead
  U_F64 [B,1,64,64] → geometry aligned gate [B,1,64,64]

S_adapt = S_base + U_F * Delta_total
         = S_base + U_F * (S_rect - S_base)
      ↓ frozen SAM prompt encoder + mask decoder, conditioned by q_seg
final mask [B,1,256,256]
```

### 3.3 Node-level contract

| Node | Shape | Parameters | Historical training/supervision | Normalization / geometry / scale | Resolution/channel change | Active now? |
|---|---:|---:|---|---|---|---|
| Cross-layer fusion | `[B,576,1024]` | 4,198,401 | Phase6G.2A dense-mask probe objective | standard MHA; learned scalar residual gamma | two layer tokens → one 1024-D token, no spatial change | yes, frozen |
| Projection-only | `[B,1024,24,24]→[B,256,24,24]` | 262,400 | Phase6G.8B dense supervision | no norm/residual/gate | channel only | yes, frozen |
| `z_F24` head | `[B,256,24,24]→[B,1,24,24]` | 257 | same selected projection checkpoint | none | channel only | yes, frozen |
| `S_base` | `[B,256,64,64]` | upstream frozen C1/SAM | C1/P1 lineage | SAM space | none here | yes |
| Rectifier norms | tokenwise 256 | 1,024 | Phase6G.10 final mask loss | two LayerNorms | none | yes |
| Q/K/V | tokenwise 256 | 3×65,792 | Phase6G.10 final mask loss | 8 heads; fixed coordinate bias added before Q/K; raw F is V | no spatial/channel change | yes |
| attention `out_proj` | 256→256 | 65,792 | Phase6G.10 final mask loss | no norm; first learned post-attention translation | channel basis translation | yes |
| `rectification.projection` | 256→256 | 65,792 | Phase6G.10 final mask loss | no norm; before scalar gamma | channel basis translation | yes |
| `gamma_main` | scalar | 1 | init .01; Phase6G.10 final mask loss | selected .0609020 | amplitude only | yes |
| semantic support | `[B,4096]` bool | 0 | deterministic coordinate extent | hard spatial validity | no learned change | yes |
| side projection | Conv1x1 256→256 | 65,792 | zero init; Phase6G.11 final mask loss with main frozen | no norm/gamma; shared hard support | channel translation only | yes, frozen |
| Utility frozen source heads | L/F evidential maps | 178,820 | Phase4G-1C lineage | detached; calibrated temperatures | mapped to 64/256 grids | yes, frozen |
| Utility trainable branch | produces `U_F64` | 371,803 | Phase6G.13 `seg+relative+ranking` | GN/LN, sigmoid, support | inputs to width64; preserves 64×64 | yes, frozen |
| SAM decoder | mask logits `[B,1,256,256]` | frozen C1/P1 lineage | not updated here | BF16 interface as historical path | 64→256 | yes |

Rectifier selected state has 329,985 learned values total. Utility checkpoint contains 550,625 values: 178,820 in frozen source heads plus temperatures and 371,803 historically trainable Utility parameters. The latter split is: language context 116,352; forensic context 63,488; CMX rectification 6,682; local exchange 33,536; utility head 151,745.

## 4. Rectifier component audit

| Component | Q1 original problem | Q2 still needed for block11+17? | Q3 evidence | Q4 overlap | Q5 removal loses | Q6 status |
|---|---|---|---|---|---|---|
| `forensic_norm` | stabilize/standardize foreign evidence before keys | yes: projection-only evidence is not SAM-distributed | **indirect:** R0→K/V retains advantage; no norm ablation | partial with learned K/V | feature alignment and optimization stability | training stabilizer; unresolved necessity |
| `semantic_norm` | normalize frozen SAM query tokens | yes | **indirect** only | partial with Q projection | stable query geometry/basis | training stabilizer |
| `coordinate_sincos` | give both token sets absolute original-coordinate identity | yes, because 24×24 evidence and 64×64 SAM grids differ | **indirect:** geometry path works; no PE-only ablation | overlaps fixed distance bias semantically, not mathematically | absolute position representation beyond distance | plausible necessity; not directly isolated |
| `q_proj` | learn SAM-space query subspace | yes | **indirect:** cross-attn useful; no q-only ablation | norm/PE prepare, do not replace it | query direction learning | architectural necessity within MHA |
| `k_proj` | learn evidence matching subspace | yes | **direct stage evidence:** A1 advantage survives at K | no direct duplicate | semantic-forensic matching direction | supported architectural mechanism |
| `v_proj` | learn transmitted evidence subspace | yes | **direct stage evidence:** advantage survives at V | potential overlap with downstream translators | value basis translation | useful but exact split unresolved |
| geometry-aware attention | align 4096 SAM queries to 576 evidence tokens | yes | **direct:** advantage amplified at R2 | coordinate PE and distance bias both encode geometry | spatially aligned evidence retrieval | KEEP |
| `out_proj` | recombine heads and translate concatenated MHA response | yes in standard MHA; whether this exact layer is independently required is unclear | **direct mixed:** A1 removal is not stably better in task objective; A2 bypass worsens task | strong overlap with next 256→256 projection | head mixing/direction translation | QUESTION, not remove |
| `rectification.projection` | map MHA output to residual correction basis | yes | **direct mixed:** full bypass improves Delta probe but harms task objective | strong overlap with out_proj | SAM-compatible residual direction | KEEP function, QUESTION factorization |
| `gamma_main` | start residual small and calibrate branch amplitude | yes as optimization device; expressiveness can be absorbed into projection | **direct:** grew .01→.060902 in selected model; original Phase4F trajectory also grew | amplitude overlaps projection norms and Utility | stable residual introduction/global amplitude | training stabilizer + branch calibration |
| semantic support | disallow correction outside evidence coverage | yes for crop geometry/invalid cells | **mechanistic direct:** deterministic and shared; full-FOV work shows coverage matters | spatial gate overlaps Utility only in “where,” but support is validity, not usefulness | hard spatial validity and safe endpoints | KEEP |
| side projection | add correction directions after fixed main | empirically useful, but not yet proven inherently necessary | **direct:** +.004302, CI positive; joint freedom adds no stable gain | duplicates linear translation capacity of main | extra residual capacity/optimization path | QUESTION |

`coordinate_sincos` and distance bias are a possible representational overlap, but they encode different things: the PE lets learned projections use absolute location, whereas the Gaussian term imposes a fixed relative-locality prior on attention logits. There is no ablation supporting removal of either.

## 5. Main translator audit

### 5.1 Why two consecutive 256→256 projections?

In code, `GeometryAwareCrossAttention.out_proj` is standard multi-head output mixing. It recombines concatenated head responses. `CrossAttentiveSemanticRectification.projection` is a second residual translator whose output is multiplied by `gamma_main` and support before entering SAM space.

This is a coherent conceptual split—**head recombination** followed by **residual-basis translation**—but the implementation does not enforce distinct functions: both are unconstrained affine 256→256 maps with no nonlinearity or normalization between them. Ignoring biases, they collapse to one linear map. Their functional separation is therefore historical/conventional rather than identifiable from architecture alone.

Phase6G.10 supplies the decisive qualification:

| Arm | Change | Rectifier objective | R2 probe | Delta probe |
|---|---|---:|---:|---:|
| A0 | both projections | 0.181244 | 0.147642 | 0.118960 |
| A1 | remove `out_proj` | 0.179407 | 0.137850 | 0.118242 |
| A2 | bypass both | 0.173795 | 0.150160 | 0.128590 |

A2 makes the residual more linearly mask-decodable but makes frozen-SAM localization stably worse. Thus the double map is not justified by preserving maximal forensic information; it is justified by learning a correction compatible with the downstream frozen representation. Conversely, A1 does not establish that both layers are necessary: its task difference from A0 crosses zero. The correct label is **POTENTIAL_ROLE_OVERLAP / unresolved factorization**, not “harmful low rank.”

The A0 composition's effective Shannon rank (42.935) and high condition number (≈4.56×10^5) show strong directional shaping. They do not prove pathological information loss because the task objective prefers A0 over full bypass.

### 5.2 `gamma_main`

- Form: one learnable scalar, not channel-wise or spatial.
- Initialization: 0.01 (non-zero is enforced by code).
- Selected Phase6G.10 A0: 0.0609020144.
- Historical Phase4F forensic arm: increased from .01 through ≈.02978 at epoch 1 to ≈.05003 at epoch 10. Phase6G.10 independently learned its selected value.
- Role: amplitude calibration and optimization stabilization; it does not learn correction direction.
- Expressiveness: for a freely trainable preceding projection, a non-zero scalar is algebraically absorbable into its weights. Its value is therefore primarily parameterization, training dynamics, auditability, and branch-level calibration.
- Overlap: global amplitude overlaps projection norms; spatial/sample amplitude overlaps Utility only conceptually. Utility cannot fully replace the small-init optimization role because it is trained later and is spatially varying.

Conclusion: retain for now as a stabilizer/calibration control, but do not treat it as an additional representational mechanism.

### 5.3 Source mismatch hypothesis

The selected current main is **not** a block22-trained translator reused unchanged. `phase6g10_train_arm.py` seeds and constructs matched Rectifiers, feeds selected block11+17 fusion plus projection-only evidence, and trains the Rectifier with final mask loss. A0 was selected at epoch 8 after 8,840 updates. Phase6G.11 then freezes this A0 while learning the side; Phase6G.13 freezes both.

Therefore:

- “Main learned only the old block22 correction subspace” is contradicted by selected-checkpoint provenance.
- A subtler optimization mismatch remains possible: A0 retains the *same architecture and recipe* inherited from the block22 era, which may impose an optimization bias unsuitable for richer R2. But this is a recipe/parameterization hypothesis, not a checkpoint-source mismatch.
- Phase6G.12 weakens a simple capacity explanation: freeing main+side jointly does not stably beat frozen-main+side.

## 6. Side-path audit

### 6.1 Why can one linear side improve?

With the main frozen, zero initialization makes the initial function exactly A0. Final mask loss then learns only the residual correction not already supplied by the fixed main. This is an optimization advantage: it prevents co-adaptation and preserves a strong reference function while exposing a direct gradient path from frozen SAM compatibility back to R2. This resembles zero-initialized control/residual branches used to introduce new conditioning without disrupting a locked pretrained function ([ControlNet](https://arxiv.org/abs/2302.05543), [ReZero](https://arxiv.org/abs/2003.04887)).

It is reasonable to call it “residual correction after the fixed main,” but not an orthogonal or semantically unique correction unless constrained or independently identified.

### 6.2 What does the gain mean?

| Interpretation | Current support | Current counter-evidence / limitation |
|---|---|---|
| H1: main misses new useful directions | moderate: stable +.004302; cosine(main,side)≈.406; 53.68% side input outside main top-8 | these statistics do not establish causal orthogonality; joint training adds no stable gain |
| H2: side compensates old block22 source mismatch | weak/contradicted for selected graph | main A0 was retrained on block11+17 from matched initialization |
| H3: side is mainly amplitude/linear refinement | plausible: one affine Conv1x1 is sufficient; side/main norm≈1.76 | cosine .406 and top-8 statistic suggest it is not merely a scalar rescale, but remain descriptive |

The large side/main norm is not proof that the side “dominates”: main and side can cancel, complement, or occupy differently scaled feature bases. Likewise, cosine .406 is evidence against exact collinearity, not evidence of a clean independent subspace.

### 6.3 Should there be `gamma_side`?

`gamma_side * W_side` is functionally equivalent to a rescaled `W_side` when both are trainable and unconstrained. A scalar side gamma would add no expressiveness. Its possible value would be:

- optimization: separate branch amplitude from direction and preserve a controlled start;
- interpretability: expose a comparable global branch-strength statistic;
- calibration: regularize/limit a branch whose selected norm is large.

But the current side already has an exact zero-function initialization, which serves the main stabilization purpose more strongly than a small scalar on random weights. Adding `gamma_side` would also overlap with `W_side` scale and downstream `U_F`. Therefore lack of gamma is not an architectural asymmetry that must be “fixed.” It is at most a parameterization hypothesis, not a missing capability.

## 7. Utility architecture audit

### 7.1 Inputs and responsibilities

| Input | Shape before Utility mapping | Source/status | Information | Audit classification |
|---|---:|---|---|---|
| `S64` | `[B,256,64,64]` | frozen C1/SAM | semantic/image state to be corrected | necessary semantic context |
| `q_seg` | `[B,256]` | frozen C1 actual SEG query | requested localization/query context | likely necessary, but partly redundant with `S64,z_L`; no isolation |
| `z_L` | `[B,1,256,256]` | frozen language-side localization logit | language localization confidence/content | useful confidence/context; overlaps `S64,q_seg` |
| `F24` | `[B,256,24,24]` | frozen block11+17 projection | dense forensic feature | necessary forensic context |
| `z_F24` | `[B,1,24,24]` | frozen dense head | forensic foreground score/confidence proxy | likely useful but redundant with `F24` in expressiveness |
| clip geometry/support | metadata / masks | deterministic | coordinate mapping and validity | necessary spatial validity |
| frozen evidential source posteriors | L/F masses and probabilities | Phase4G-1C heads | calibrated source-specific belief/conflict | historically motivated; necessity in present correction-gate use unresolved |

The architecture maps language and forensic sources to width 64, then uses:

1. bidirectional CMX-style channel and spatial rectification;
2. one bidirectional 7×7 local cross-exchange;
3. comparison features `[Lr, Fr, Lr*Fr, |Lr-Fr|, cosine, p_L, p_F, conflict]`;
4. a 3×3 head and sigmoid to obtain `U_F`.

The minimal information a useful correction gate needs is: (i) forensic confidence, (ii) semantic/query context, (iii) the proposed correction state or a sufficient proxy, and (iv) spatial validity. Current Utility has (i), (ii), and (iv), but notably **does not observe `Delta_main`, `Delta_side`, or `Delta_total` directly**. It predicts usefulness from the sources that produced the correction. This explains why it can improve utilization overall while failing to identify where the side specifically helps.

This is a central semantic mismatch, not proof that Utility should be removed. CMX supports cross-modal rectification as a general mechanism, but it does not establish that source comparison alone is a calibrated correction-benefit gate ([CMX paper](https://arxiv.org/abs/2203.04838), [official code](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)).

### 7.2 Architecture classification

- **Necessary:** coordinate mapping/support, forensic context, semantic/query context, bounded gate endpoint semantics.
- **Empirically useful as a block:** full Utility; Phase6G.13 gives stable +.006935.
- **Potentially redundant:** simultaneous raw features, dense logits, evidential probabilities, and conflict may encode overlapping confidence; `q_seg`, `S64`, and `z_L` also overlap.
- **Historical legacy/unresolved:** frozen ECoLaF source heads and CMX/local exchange were inherited from earlier Utility development. Current Phase6G.13 proves the combined 371,803-parameter branch works, not that each submodule or input is necessary.
- **Missing for the claimed side-selection interpretation:** direct correction/side state. The gate cannot be expected to identify side benefit from a target it never receives and an objective that never labels it.

## 8. Utility objective audit

Phase6G.13 uses equal-weight:

```text
L = L_seg + L_relative + L_ranking
L_ranking = 0.5 L_cross + 0.5 L_shuffle
```

| Term | Exact operational definition | Gradient path in Phase6G.13 | What it optimizes | Overlap / evidence |
|---|---|---|---|---|
| `seg` | `2*BCEWithLogits + 0.5*soft-Dice` on frozen SAM output from `S_base + U_F*Delta_total` | Utility only; main, side, SAM, sources detached/frozen | end-task SAM-compatible gate values | direct task supervision; necessary contribution not isolated in G13 |
| `relative` | soft target `sigmoid((BCE(p_L,y)-BCE(p_F,y))/tau)`; image-balanced BCE-with-logits on Utility logit | Utility only; `p_L,p_F,target` target construction detached | whether forensic source posterior is locally better than language posterior | trains source-relative usefulness, not realized correction benefit; overlaps comparison features |
| `cross ranking` | hinge `relu(.1-(U_match-U_cross))`, cross-image forensic source | Utility only | prefer matched forensic-language relation over wrong-image evidence | mismatch sensitivity; no proof it selects helpful correction |
| `shuffle ranking` | same hinge against spatially shuffled forensic evidence | Utility only | prefer correct spatial arrangement | spatial correspondence; overlaps geometry/local exchange priors |

The observed near-zero correlation between gate and side benefit is expected under these definitions. `relative` asks “is forensic posterior better than language posterior?”; ranking asks “is this evidence correctly paired/aligned?”; neither asks “does adding `Delta_side` improve the final mask at this location?” The segmentation term can still learn combined-correction utilization, explaining the stable Phase6G.13 gain.

No Phase6G.13 ablation isolates the necessity of `relative` or either ranking half. Historical mismatch-control behavior is indirect evidence for ranking, not proof of present end-task necessity. Therefore the current objective should be described as **task loss plus source-reliability/matching regularization**, not a side-benefit objective.

## 9. Rectifier–Utility responsibility boundary

| Mechanism | Direction | Amplitude | Spatial validity | Sample/spatial usefulness | Evidence extraction |
|---|---|---|---|---|---|
| main translator | primary correction direction | implicit via weight norms | no | no | no |
| `gamma_main` | no | global branch amplitude | no | no | no |
| support | no | hard 0/1 only | **primary** | no; validity is not usefulness | no |
| side | residual correction direction; possibly linear refinement | implicit via weight norms | shared hard support | no | no |
| Utility | no new correction direction | **spatial/sample amplitude** in `[0,1]` | respects support | **primary current mechanism**, but trained mainly from source proxies + task loss | compares already extracted evidence; does not extract CLIP evidence |

Operational answers:

- **往哪改:** main translator plus side projection.
- **改多强:** projection norms and `gamma_main` set global/channel amplitude; Utility sets spatial/sample amplitude.
- **哪里允许改:** deterministic semantic support.
- **值不值得改:** Utility, but only for combined correction; not demonstrably side-specific.

Marked overlaps:

- `out_proj` vs `rectification.projection`: **POTENTIAL_ROLE_OVERLAP** in linear direction translation.
- projection norms vs `gamma_main`: **POTENTIAL_ROLE_OVERLAP** in global amplitude, with different optimization semantics.
- side vs main: **POTENTIAL_ROLE_OVERLAP** in unconstrained linear correction capacity.
- `gamma_main` vs Utility: partial amplitude overlap, but gamma is global/stabilizing while Utility is spatial/conditional.
- support vs Utility: superficial spatial overlap; responsibilities should remain distinct—validity versus usefulness.

## 10. Phase6G.8–G.13 gain-flow reconstruction

Measurements below are deliberately separated by protocol.

### 10.1 Fresh matched linear-probe evidence

| Stage | block22 | block11+17 | Delta | 95% CI / stability | Interpretation |
|---|---:|---:|---:|---|---|
| G9 R0 evidence | .164403 | .179338 | +.014935 | [.007983,.021897] | stable source advantage |
| G9 R1 K | .170793 | .184366 | +.013573 | [.006727,.020527] | retained |
| G9 R1 V | .165111 | .178491 | +.013380 | [.006471,.020415] | retained |
| G9 R2 raw attention | .134598 | .169290 | +.034692 | [.024972,.044205] | amplified |
| G9 R3 after out-proj | .151838 | .166079 | +.014240 | [.004956,.023355] | attenuated but stable |
| G9 R4 residual pre-support | .131250 | .139374 | +.008124 | [-.001303,.017654] | first loss of stable advantage |
| G9 R5 rectified state | .149737 | .148921 | -.000815 | [-.009239,.007594] | no stable advantage |
| G9 R6 residual/Delta | .126298 | .138800 | +.012502 | [.003561,.021621] | probe recovers, showing stage labels are not a monotonic causal chain |

Phase6G.8 used different frozen final models and found `F_forensic` +.014796, `Delta_F` +.003107 (CI crosses zero), and `U_F*Delta_F` -.010157 (stable negative). It is consistent with utilization loss but must not be numerically merged with G9.

### 10.2 Rectifier task objective, not probes

| Experiment | Arm | Internal-val mean IoU | Paired conclusion |
|---|---|---:|---|
| G10 | A0 double projection | .181244 | reference |
| G10 | A1 remove out-proj | .179407 | -0.001837; CI crosses zero |
| G10 | A2 bypass both | .173795 | -0.007448; CI fully negative |
| G11 | frozen A0 + zero-init side | .185546 | +.004302; CI [.000917,.007823] |
| G12 | jointly train main+side | .185791 | +.000246 vs frozen-main+side; CI crosses zero |

The first strong mechanistic break is between raw attention and residual translation, but G10 shows the translator's apparent information compression is partly task-compatible shaping. The remaining issue is thus not “recover all R2 decodability”; it is how to learn the right frozen-SAM-compatible correction reliably.

### 10.3 Utility/final segmentation

| Experiment | Arm | Mean IoU | Delta / conclusion |
|---|---|---:|---|
| G13 | frozen main+side, no Utility | .185546 | reference |
| G13 | same correction + trained Utility | .192480 | +.006935, CI [.004509,.009336], 561/245/300 |

Utility is a genuine utilization mechanism. However, gate-side-benefit correlations (Pearson .00675, Spearman .02027) and nearly equal help/harm gate means (.57146/.56578) reject the narrower claim that it learned side-specific routing.

### 10.4 Bottleneck taxonomy

| Candidate | Finding |
|---|---|
| representation bottleneck | not primary before R2; block11+17 is stronger and attention amplifies it |
| translation bottleneck | supported in the sense of difficult SAM-compatible conversion; exact double-projection blame not established |
| optimization mismatch | supported by frozen-main zero-init side succeeding while joint freedom does not |
| scale mismatch | possible, but not isolated; gamma and Utility both help at different levels |
| objective mismatch | strongly supported: probe decodability and task objective reverse; Utility auxiliary targets do not label correction benefit |
| utilization mismatch | supported: Utility adds stable gain, yet is not side-selective |

## 11. External literature survey

### 11.1 Feature rectification and cross-modal fusion

CMX performs bidirectional feature rectification and feature fusion for RGB-X segmentation. It supports retaining cross-modal comparison/rectification as a useful principle, but its modalities and decoder are trained for the segmentation task; it does not prove that an unconstrained correction is compatible with a frozen SAM space ([paper](https://arxiv.org/abs/2203.04838), [code](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)). TransForensics and OMG-Fuser likewise show dense cross-source interaction can help forgery localization, but they are end-to-end localization systems rather than frozen-decoder residual correction interfaces ([TransForensics](https://openaccess.thecvf.com/content/ICCV2021/html/Hao_TransForensics_Image_Forgery_Localization_With_Dense_Self-Attention_ICCV_2021_paper.html), [OMG-Fuser](https://openaccess.thecvf.com/content/CVPR2024W/WMF/papers/Karageorgiou_Fusion_Transformer_with_Object_Mask_Guidance_for_Image_Forgery_Analysis_CVPRW_2024_paper.pdf)).

### 11.2 Frozen foundation-model adaptation

AutoSAM trains a new image encoder against a frozen SAM prompt encoder/decoder, demonstrating that compatibility with a fixed decoder must be learned by the upstream interface rather than assumed from spatial decodability ([paper](https://arxiv.org/abs/2306.06370)). Domain-Rectifying Adapter explicitly maps target-domain features toward the representation expected by a fixed source segmenter, closely matching the “SAM-compatible translation” interpretation ([paper](https://openaccess.thecvf.com/content/CVPR2024/html/Su_Domain-Rectifying_Adapter_for_Cross-Domain_Few-Shot_Segmentation_CVPR_2024_paper.html)). SAM-Adapter adds task-specific prompts to SAM components, while MASA transforms frozen foundation features into task-suitable multi-scale features; both support interface adaptation but not this exact dense correction/gate decomposition ([SAM-Adapter](https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/Chen_SAM-Adapter_Adapting_Segment_Anything_in_Underperformed_Scenes_ICCVW_2023_paper.pdf), [MASA](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Matching_Anything_by_Segmenting_Anything_CVPR_2024_paper.pdf)). “Adapters Strike Back” further shows adapter placement and residual scaling affect frozen-model adaptation, supporting careful parameterization rather than adding depth by default ([paper](https://openaccess.thecvf.com/content/CVPR2024/html/Steitz_Adapters_Strike_Back_CVPR_2024_paper.html)).

### 11.3 Forensic-specific adaptation

Forensics Adapter learns forensic-oriented interactions on CLIP features, supporting specialized forensic feature adaptation but targeting face forgery detection rather than frozen-SAM localization compatibility ([paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Cui_Forensics_Adapter_Adapting_CLIP_for_Generalizable_Face_Forgery_Detection_CVPR_2025_paper.pdf), [code](https://github.com/OUC-VAS/ForensicsAdapter)). ForensicsSAM uses multiple forgery experts and conditional adversary experts; its routing is expert/task routing, not a scalar spatial correction gate ([paper/code](https://github.com/siriusPRX/ForensicsSAM)). TruFor predicts anomaly and confidence maps and uses both downstream, supporting the distinction between evidence and confidence, but its confidence is jointly designed for its own detector rather than a frozen decoder residual ([paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.pdf)).

### 11.4 Residual, scale, and routed adaptation

Residual adapters isolate task/domain-specific changes while preserving a shared backbone ([Residual Adapters](https://proceedings.neurips.cc/paper/2017/hash/e7b24b112a44fdd9ee93bdf998c6ca0e-Abstract.html)). ReZero and LayerScale introduce small/zero residual scaling for optimization stability rather than extra representational expressiveness ([ReZero](https://arxiv.org/abs/2003.04887), [LayerScale/CaiT](https://openaccess.thecvf.com/content/ICCV2021/papers/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.pdf)). ControlNet's zero convolutions demonstrate an exact non-interference start for a learned conditioning branch, closely analogous to the optimization rationale of the R1 side path ([paper](https://arxiv.org/abs/2302.05543), [code](https://github.com/lllyasviel/ControlNet)). None makes `gamma_side` necessary when the branch projection itself is zero-initialized.

## 12. Literature comparison matrix

| Paper | Year | Task | Frozen backbone? | Auxiliary evidence | Interaction point | Correction/fusion | Scale/gate | Supervision | Similarity to R1 | Not directly transferable |
|---|---:|---|---|---|---|---|---|---|---|---|
| [CMX](https://arxiv.org/abs/2203.04838) | 2022 | RGB-X segmentation | no, task trained | X modality | multi-stage features | bidirectional rectification + fusion | channel/spatial gates | segmentation | closest Utility lineage | not a frozen SAM correction interface |
| [TransForensics](https://openaccess.thecvf.com/content/ICCV2021/html/Hao_TransForensics_Image_Forgery_Localization_With_Dense_Self-Attention_ICCV_2021_paper.html) | 2021 | forgery localization | no | multi-scale RGB features | dense encoder/decoder | dense self-attention | attention weights | mask + boundary | dense forensic interaction | end-to-end decoder, no frozen semantic space |
| [TruFor](https://openaccess.thecvf.com/content/CVPR2023/papers/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.pdf) | 2023 | forgery localization/detection | task trained | RGB/noise-sensitive traces | anomaly/confidence outputs | learned fusion and confidence | confidence map | mask/detection/confidence | separates evidence and usefulness confidence | confidence target and decoder differ |
| [OMG-Fuser](https://openaccess.thecvf.com/content/CVPR2024W/WMF/papers/Karageorgiou_Fusion_Transformer_with_Object_Mask_Guidance_for_Image_Forgery_Analysis_CVPRW_2024_paper.pdf) | 2024 | forgery localization/detection | task trained | multiple forensic signals + object masks | token fusion | transformer fusion | attention | loc + detection | separate evidence streams fused spatially | no frozen SAM-compatible residual |
| [Forensics Adapter](https://openaccess.thecvf.com/content/CVPR2025/papers/Cui_Forensics_Adapter_Adapting_CLIP_for_Generalizable_Face_Forgery_Detection_CVPR_2025_paper.pdf) | 2025 | face forgery detection | CLIP largely preserved/adapted | blending artifacts | CLIP-layer interaction | forensic adapter | learned interaction | classification | forensic specialization of CLIP | classification, not dense frozen-decoder correction |
| [ForensicsSAM](https://github.com/siriusPRX/ForensicsSAM) | 2025 | forgery localization | SAM-derived/task adapted | forgery/adversary experts | expert feature path | multi-expert fusion | conditional expert activation | localization | conditional forensic use | expert routing is not spatial correction gating |
| [AutoSAM](https://arxiv.org/abs/2306.06370) | 2023 | medical segmentation | SAM prompt/decoder frozen | new image encoder | SAM decoder input | learned compatible encoder | no analogous Utility | segmentation | directly demonstrates frozen-decoder compatibility learning | replaces encoder; not residual evidence branch |
| [Domain-Rectifying Adapter](https://openaccess.thecvf.com/content/CVPR2024/html/Su_Domain-Rectifying_Adapter_for_Cross-Domain_Few-Shot_Segmentation_CVPR_2024_paper.html) | 2024 | cross-domain segmentation | fixed source model | target-domain features | before source segmenter | domain rectification | adapter residual | source/target episodic seg | closest “translate into expected feature space” rationale | domain adaptation setting and targets differ |
| [SAM-Adapter](https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/Chen_SAM-Adapter_Adapting_Segment_Anything_in_Underperformed_Scenes_ICCVW_2023_paper.pdf) | 2023 | underperformed-scene segmentation | much of SAM retained | task-specific prompts/features | SAM blocks | adapter/prompt injection | learned residual influence | segmentation | adapts SAM with limited trainable path | injection points and decoder training contract differ |
| [MASA](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Matching_Anything_by_Segmenting_Anything_CVPR_2024_paper.pdf) | 2024 | matching | foundation features frozen | object masks/features | frozen-feature adapter | task-suitable multiscale adapter | no analogous gate | matching/seg-derived | frozen features need task-compatible translation | different output task and no R1 Utility |
| [Adapters Strike Back](https://openaccess.thecvf.com/content/CVPR2024/html/Steitz_Adapters_Strike_Back_CVPR_2024_paper.html) | 2024 | visual adaptation | yes | task data | after frozen blocks | residual adapters | learnable scaling variants | downstream task | supports adapter placement/scale audit | classification-oriented results do not validate dense correction |
| [Residual Adapters](https://proceedings.neurips.cc/paper/2017/hash/e7b24b112a44fdd9ee93bdf998c6ca0e-Abstract.html) | 2017 | multi-domain vision | shared backbone | domain task | residual branch | small residual modules | implicit branch scale | task loss | side-path principle | no forensic/spatial/frozen-SAM interface |
| [ReZero](https://arxiv.org/abs/2003.04887) | 2020 | deep-network optimization | N/A | residual branch | residual merge | zero-init scalar residual | scalar alpha | task loss | explains gamma as stabilization | does not motivate Utility or side semantics |
| [LayerScale/CaiT](https://openaccess.thecvf.com/content/ICCV2021/papers/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.pdf) | 2021 | image classification | no | residual transforms | transformer residual | channel-wise small-init scaling | LayerScale | classification | scaling separates stable optimization from direction | channel-wise training result not evidence for R1 gate |
| [ControlNet](https://arxiv.org/abs/2302.05543) | 2023 | controlled generation | locked diffusion backbone | spatial condition | copied-block residual injections | zero convolution side path | zero-init projection | diffusion loss | exact non-interference side start | generative U-Net, not localization or Utility routing |

## 13. What is actually supported

1. Block11+17 evidence and projection-only 256-D interface are adequate and stronger than block22.
2. Geometry-aware cross-attention successfully consumes that evidence and amplifies its probe advantage at R2.
3. Some learned post-attention translation is required for frozen-SAM task performance; direct R2 residual bypass is harmful despite better Delta probe decodability.
4. A zero-init side on a frozen matched main gives a small stable task gain.
5. Utility on frozen main+side gives a larger stable task gain.
6. Hard geometric support and conditional usefulness have distinct roles and should not be conflated.
7. External work broadly supports learned compatibility interfaces, residual non-interference starts, and separating evidence from confidence; it does not uniquely validate the present double translator or Utility inputs/losses.

## 14. What remains speculative

- That two consecutive 256→256 affine maps are both independently necessary.
- That the side learns a semantically independent correction subspace.
- That a `gamma_side` would improve expressiveness or results.
- That Utility's frozen evidential heads, CMX block, local exchange, and all comparison features are each necessary.
- That `relative` and ranking losses are required once final segmentation loss is present.
- That current Utility estimates realized correction benefit; its targets instead encode source-relative quality and matching.
- That low effective rank is harmful; present task evidence argues against this simple reading.
- That the selected main suffers old block22 checkpoint mismatch; provenance argues the opposite.

## 15. KEEP / QUESTION / REDESIGN

### KEEP

- **Geometry-aware evidence-to-SAM interaction:** direct internal evidence shows the signal reaches and strengthens at R2.
- **Learned SAM-compatible translation function:** bypass evidence proves raw decodability is insufficient.
- **Residual correction into frozen SAM space:** preserves the validated R1 principle.
- **Hard semantic support:** clear deterministic validity responsibility.
- **Utility layer as a hierarchy level:** Phase6G.13 stable paired gain.
- **Zero/non-interfering initialization principle for auxiliary correction:** supported internally and by residual adaptation literature.

### QUESTION

- Exact `out_proj → rectification.projection` factorization.
- Main+side decomposition versus one matched translator.
- `gamma_main` as a permanent architectural element versus training parameterization.
- Absence/presence of `gamma_side`; likely optimization/calibration only.
- Utility input set, especially duplicated raw/logit/evidential signals.
- Whether CMX rectification and local exchange are both needed inside Utility.
- Relative/ranking auxiliary objectives under the present final task loss.

### REDESIGN CANDIDATE

- **Utility objective semantics:** if the scientific claim is correction-benefit routing, current targets do not express that claim.
- **Historical double-linear decomposition:** not deletion-ready, but a controlled single-translator comparison is justified.
- **Interpreting Utility as side selector:** remove this interpretation now; the data reject it.

No current evidence justifies removing the entire Rectifier, Utility, support, or cross-attention.

## 16. Maximum three candidate hypotheses

### H-A — One matched translator is sufficient

- **Motivation:** two affine 256→256 maps have overlapping function, while direct bypass is too weakly constrained for SAM compatibility.
- **Internal evidence:** A1 removal is not stably worse/better; A2 proves translation itself is needed; side gives only a small gain.
- **Literature basis:** frozen-model adapters and domain rectification emphasize a learned compatibility interface, not necessarily stacked linear maps.
- **Minimal change:** replace the main translator factorization with one matched trainable SAM-compatible map; no new evidence, gate, or decoder.
- **Scientific question:** is factorization useful, or is a single task-trained interface enough?
- **Complexity:** fewer parameters and clearer responsibility.
- **Failure meaning:** the two-stage head-mixing/residual-basis factorization supplies real optimization or task value.

### H-B — Side gain is optimization compensation, not missing architecture

- **Motivation:** frozen-main+side improves, but jointly freeing both does not.
- **Internal evidence:** G11 positive; G12 no stable increment; source-mismatch explanation is weakened by matched A0 provenance.
- **Literature basis:** zero-init residual branches often protect a pretrained function during adaptation (ControlNet/ReZero), making optimization—not expressiveness—the leading explanation.
- **Minimal change:** compare identical total translator capacity under frozen-reference residual learning versus unified training; do not duplicate a full translator.
- **Scientific question:** does non-interference optimization cause the side gain?
- **Complexity:** changes training parameterization, not evidence or decoder.
- **Failure meaning:** side may encode genuinely complementary directions that unified optimization does not recover.

### H-C — Utility is useful, but its objective is source utility rather than correction utility

- **Motivation:** stable final gain coexists with near-zero side-benefit correlation.
- **Internal evidence:** G13 +.006935; actual targets use `p_L,p_F` and matched/corrupted evidence, not `Delta` benefit.
- **Literature basis:** TruFor separates anomaly from confidence; routed/expert systems train gates for their actual routing target. CMX supports interaction but not the present target semantics.
- **Minimal change:** first ablate/clarify objective semantics under the unchanged Utility architecture; no new gate/module.
- **Scientific question:** which part of the gain comes from final segmentation supervision versus source-relative/ranking regularization?
- **Complexity:** no architectural growth; potentially fewer objectives.
- **Failure meaning:** auxiliary source-reliability supervision is genuinely needed to regularize end-task Utility.

## 17. Recommended next scientific question

**Only one question is recommended:**

> With block11+17 evidence and frozen C1/SAM, is one freshly trained, matched SAM-compatible translator sufficient to equal or beat both the current double-projection main and frozen-main+side under the same final localization objective?

Why this comes first: it directly resolves the earliest uncertain conversion boundary, avoids confusing probe quality with task compatibility, tests whether the side is compensating parameterization/optimization, and does not require changing Utility simultaneously. The audit ends here; no experiment has been started.
