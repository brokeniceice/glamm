# R1 Responsibility Decomposition + Correction-Aware Utility Deep Literature Review

Date: 2026-09-19  
Scope: static code/history analysis and literature review only  
Forbidden and not performed: training, new experiments, implementation, checkpoint modification, internal test, Official1000, OOD  
Status labels used throughout: **INTERNALLY_PROVEN**, **LITERATURE_SUPPORTED**, **MECHANISTIC_HYPOTHESIS**, **SPECULATION**.

## 1. Executive summary

1. **INTERNALLY_PROVEN — R1 is useful and is not being reconsidered as a whole.** The established comparison `C1 + new R1 > C1` is the premise. The current question is why a much stronger block11+17 forensic representation is only partially converted into final localization gain.
2. **INTERNALLY_PROVEN — Evidence and Interaction are not the present first-order suspects.** Block11+17 improves dense forensic evidence; projection-only is sufficient; the advantage survives K/V and grows at raw cross-attention `R2`.
3. **INTERNALLY_PROVEN — a learned SAM-compatibility translation is necessary.** Direct `R2 -> correction` improved linear mask decodability but harmed frozen-SAM localization. G14 further showed that two consecutive affine maps are unnecessary: the matched single affine translator was slightly but stably better.
4. **INTERNALLY_PROVEN — Utility has task value.** Phase6G.13 improved mean FG IoU from 0.185546 to 0.192480. The near-zero relationship between side-specific benefit and gate value means that result supports utilization of the combined correction, not side selection.
5. **Static-code-confirmed — current Utility is source-aware, not correction-aware.** It receives semantic/language and forensic contexts, posterior/conflict information, geometry support, and query context. It does not receive the actual correction `C`, `Delta_F`, or `S_rect=S64+C`.
6. **LITERATURE_SUPPORTED — frozen consumers often need a learned compatibility interface**, trained either through the frozen downstream objective or with explicit alignment. Domain-Rectifying Adapter, AutoSAM/SAM-Adapter, AVFormer, X-Adapter, AdaptFormer and ControlNet support this broad principle. They do not identify one universally optimal translator family.
7. **LITERATURE_SUPPORTED but not directly equivalent — dynamic residual scaling is common**, including token-, domain-, channel-, or sample-conditioned adapters. Most reviewed gates decide from the input/state before or while producing the update; they do not review the realized proposed update.
8. **Closest direct precedent is weak evidence.** Review Residuals explicitly uses `gate(current state, proposed update)`, but it is a **PREPRINT / NOT PEER REVIEWED**, studies from-scratch language Transformers, and does not establish spatial forensic correction into a frozen segmentation model.
9. **MECHANISTIC_HYPOTHESIS — the most economical missing variable is `C`.** Given identical source contexts, different translators or correction directions can propose helpful, irrelevant, or destructive updates. A source-only Utility cannot distinguish those cases. Seeing `C` exposes direction, magnitude, state compatibility and local conflict; magnitude alone is not usefulness.
10. The recommended next scientific question is therefore only: **with Translator, validity support, frozen SAM, objective and capacity controlled, does adding the actual proposed correction to Utility inputs yield stable benefit beyond the current source-aware Utility?** This report does not start that experiment.

## 2. Internal facts that literature must not override

| Fact | Status | Consequence |
|---|---|---|
| `C1 + new R1` improves localization over `C1` | INTERNALLY_PROVEN | Do not delete R1 because another paper uses a different interface. |
| block11+17 is stronger dense forensic evidence than block22 | INTERNALLY_PROVEN | Evidence acquisition is not the immediate bottleneck. |
| projection-only 1024→256 essentially preserves the new evidence | INTERNALLY_PROVEN | Do not add adapter depth merely from precedent. |
| block11+17 advantage survives K/V and is amplified at `R2` | INTERNALLY_PROVEN | Do not mechanically replace geometry-aware cross-attention. |
| direct `R2` correction is more probe-decodable but worse for frozen SAM | INTERNALLY_PROVEN | Mask decodability is not SAM compatibility; keep a learned Translator. |
| matched single affine beats factorized double affine: +0.001933, CI [+0.000383,+0.003445] | INTERNALLY_PROVEN | Double affine is historical redundancy, not a required function class. This does **not** prove affine sufficiency. |
| frozen main + zero-init side improves 0.181244→0.185546 | INTERNALLY_PROVEN | It shows residual optimization capacity after a frozen main; it does not prove two correction subspaces. |
| Utility improves 0.185546→0.192480, CI [+0.004509,+0.009336] | INTERNALLY_PROVEN | Utility must be retained and responsibility-aligned, not removed. |
| gate vs side-benefit correlation is approximately zero | INTERNALLY_PROVEN | Do not claim Utility learns when the side branch helps. |

No incomplete Phase6G.15 result is used as evidence here.

## 3. Responsibility decomposition

The clean abstraction is

```text
F_forensic = Evidence(image)
R2         = Interaction(S64, F_forensic)
C          = Translator(R2)
S_adapt    = S64 + M_valid * U * C
U          = Utility(current state, proposed correction, optional context)
```

### 3.1 Evidence

- Responsibility: expose *what forensic clue exists and where*.
- Current implementation: frozen CLIP block11+17, selected cross-layer fusion, minimal projection.
- Status: **KEEP; INTERNALLY_PROVEN**. Strong dense evidence does not itself guarantee downstream compatibility.

### 3.2 Interaction

- Responsibility: relate forensic tokens to the corresponding SAM semantic positions.
- Current implementation: normalized geometry, validity mask and geometry-aware cross-attention.
- Status: **KEEP; INTERNALLY_PROVEN**. The new evidence advantage survives and grows at `R2`.

### 3.3 Translator

- Responsibility: produce a candidate correction in frozen SAM's feature space: `C=T(R2)`.
- It should answer *how to modify the embedding if forensic evidence is used*, not *whether this correction should be accepted*.
- Status: **KEEP, simplify where matched evidence supports it**. A learned interface is internally necessary; double affine is not.

### 3.4 Validity

- Responsibility: deterministic geometric eligibility only: evidence coverage exists or does not.
- Current support is a hard constraint, not learned confidence. `M_valid=0` must force zero forensic correction.
- Status: **KEEP separate from Utility**. Reliability cannot invent coverage.

### 3.5 Utility

- Responsibility: decide how much of this *specific proposed correction* should be accepted.
- Current implementation instead estimates utility from source and context states.
- Status: **KEEP; task value is INTERNALLY_PROVEN.** Correction awareness is a hypothesis about responsibility alignment, not a reason to discard the hierarchy.

## 4. Round-1 literature findings

Four broad findings survive closer reading:

1. **Compatibility translation is real but task-specific.** [Domain-Rectifying Adapter](https://openaccess.thecvf.com/content/CVPR2024/html/Su_Domain-Rectifying_Adapter_for_Cross-Domain_Few-Shot_Segmentation_CVPR_2024_paper.html) explicitly maps shifted features toward a space usable by an established source segmenter and adds cyclic alignment. [AVFormer](https://openaccess.thecvf.com/content/CVPR2023/html/Seo_AVFormer_Injecting_Vision_Into_Frozen_Speech_Models_for_Zero-Shot_AV-ASR_CVPR_2023_paper.html) uses a learned projection and adapters to make visual tokens useful to a frozen speech system. [X-Adapter](https://openaccess.thecvf.com/content/CVPR2024/html/Ran_X-Adapter_Adding_Universal_Compatibility_of_Plugins_for_Upgraded_Diffusion_Model_CVPR_2024_paper.html) remaps old-plugin features into an upgraded frozen diffusion model. These support the *problem*, not one exact Translator.
2. **Near-identity initialization protects pretrained behavior.** Residual Adapters, AdaptFormer, ReZero, LayerScale and ControlNet use static scales, small/zero initialization or zero convolutions. This supports treating `gamma` as an optimization/stability device rather than a distinct semantic function.
3. **Most dynamic adapter gates are input-aware rather than update-aware.** [DAPT](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Dynamic_Adapter_Meets_Prompt_Tuning_Parameter-Efficient_Transfer_Learning_for_Point_CVPR_2024_paper.html) generates per-token scales from token significance. [SEMA](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Self-Expansion_of_Pre-trained_Models_with_Mixture_of_Adapters_for_Continual_CVPR_2025_paper.html) detects representation distribution shift and routes/expands adapters. [DPW](https://openaccess.thecvf.com/content/CVPR2026/html/Jang_Enhancing_Continual_Learning_of_Vision-Language_Models_via_Dynamic_Prefix_Weighting_CVPR_2026_paper.html) derives token-level prefix and adapter weighting from input-token relevance. None is a direct test of a realized spatial correction against a frozen SAM state.
4. **Forensic confidence is usually prediction/evidence reliability, not correction utility.** [TruFor](https://openaccess.thecvf.com/content/CVPR2023/html/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.html) separates anomaly localization and reliability maps and uses confidence for image-level integrity decisions. That is a strong conceptual precedent for separating evidence and reliability, but its reliability map does not review a candidate feature correction before a frozen decoder.

## 5. Translator literature deep review

### 5.1 What is supported

- **LITERATURE_SUPPORTED:** a frozen downstream system can supply the task gradient that shapes a small interface. AutoSAM trains an image-to-prompt interface while SAM components are frozen; SAM-Adapter injects task knowledge without full SAM fine-tuning; AVFormer trains projection/adapters around a frozen ASR model.
- **LITERATURE_SUPPORTED:** explicit feature alignment can help when a reference feature space is definable. Domain-Rectifying Adapter uses perturbation plus reverse/cyclic rectification supervision.
- **LITERATURE_SUPPORTED:** stable residual initialization is often more important than nominal depth. [AdaptFormer](https://papers.nips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html), [ControlNet](https://arxiv.org/abs/2302.05543), [ReZero](https://arxiv.org/abs/2003.04887), and [LayerScale/CaiT](https://openaccess.thecvf.com/content/ICCV2021/html/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.html) protect the pretrained path with scaled/zero-initialized residual injection.
- **LITERATURE_SUPPORTED:** there is no general theorem that a bottleneck MLP or nonlinearity is required for compatibility. [Adapters Strike Back](https://openaccess.thecvf.com/content/CVPR2024/html/Steitz_Adapters_Strike_Back_CVPR_2024_paper.html) shows structure, placement and scaling matter across vision tasks, but does not make one function family universal.

### 5.2 What is not supported

- The reviewed literature does not prove `Linear→GELU→Linear` is better than the matched affine Translator for this spatial correction.
- It does not provide a feature-space target for “SAM-compatible correction” that is known to be superior to the actual frozen-SAM segmentation objective.
- It does not justify making the Translator also estimate reliability. Keeping proposal generation and proposal acceptance distinct is cleaner and falsifiable.
- G14 establishes only `single affine > factorized affine` under its matched comparison. **Affine sufficiency remains unknown.** Any nonlinear Translator remains a hypothesis until matched evidence exists.

### 5.3 Gamma

`gamma*C` can be absorbed into Translator weights and adds no function class. The literature supports gamma/small-init/zero-init as optimization safeguards. Therefore:

- retaining gamma is legitimate for stable initialization;
- interpreting gamma as a separate learned responsibility is not justified;
- deleting it without a matched initialization audit is also not justified.

## 6. Dynamic adapter / residual gating literature

The family divides into four mechanisms:

1. **Static residual control:** Residual Adapters, ReZero, LayerScale, ControlNet. These protect the base path but do not make sample-dependent utility decisions.
2. **State/token-conditioned scaling:** Highway Networks, DAPT, DPW. Gates depend on the current activation/token relevance. They may control an adapter output, but normally do not inspect that output as a proposal.
3. **Domain/distribution-conditioned routing:** SEMA and DCRM-ViT. Routers infer domain or shift and select/synthesize residual capacity. This is source/context-aware routing, not post-proposal acceptance.
4. **Update-conditioned review:** Review Residuals explicitly computes

   ```text
   h_new = h + r(h,u) * u
   r = sigmoid(W [RMSNorm(h), RMSNorm(u)])
   ```

   It is the closest semantic match to `S64 + U(S64,C)*C`. However, it is a 2026 **PREPRINT / NOT PEER REVIEWED**, concerns every-layer updates in from-scratch language Transformers, reports scale-dependent benefit, and does not establish spatial utility, external forensic evidence, frozen-model compatibility, or realized-benefit supervision.

The practical conclusion is deliberately narrow: **update awareness is a credible mechanism to test, not a literature-settled best practice.**

## 7. Correction-aware / update-aware gating literature

### 7.1 Input mismatch versus objective mismatch

These are separate hypotheses.

**Input mismatch:** Current Utility cannot distinguish two corrections produced from identical source contexts. Adding `C` provides:

- correction direction relative to `S64`;
- magnitude and spatial concentration;
- local compatibility/conflict with semantic features;
- whether the Translator proposes over-correction or near-no-op;
- differences between translators that source confidence alone cannot reveal.

This is a **MECHANISTIC_HYPOTHESIS**. `||C||` alone is not utility: a large correction may be necessary or destructive, and a small correction may be precisely aligned.

**Objective mismatch:** Current relative target asks whether forensic posterior is locally better than language posterior; ranking asks matched evidence to outrank crossed/shuffled evidence. Neither directly labels the actual downstream gain of injecting `C`. A counterfactual target could conceptually compare frozen-SAM output with versus without correction. But this target is nonlocal, decoder-dependent, potentially noisy, and could reward transient threshold effects rather than compatible representation. The surveyed direct precedent is weak: Review Residuals uses the end task loss, not an oracle “realized update gain”; TruFor supervises confidence/reliability but not feature-update benefit.

**Recommendation from evidence:** isolate the input question first. Changing both Utility input and target would destroy attribution and is not justified by current literature.

### 7.2 Additive versus convex gating

Review Residuals reports that additive identity-preserving `h+r*u` is more depth-stable than a Highway-style convex mix. This is relevant but not decisive: R1 has one spatial injection into a frozen decoder rather than dozens of recurrent residual layers. The current additive form with a hard validity mask remains the correct controlled baseline.

## 8. Forensic reliability literature

- **TruFor — MECHANISTICALLY_RELEVANT:** explicitly separates anomaly and reliability, and its global decision consumes both. Reliability pertains to localization error propensity, not acceptance of a frozen-model feature correction.
- **OMG-Fuser — MECHANISTICALLY_RELEVANT:** fuses an arbitrary set of forensic signals with object-mask guidance and task supervision. It demonstrates multi-signal integration, not harmful-update verification.
- **Forensics Adapter — BACKGROUND_ONLY for Utility:** adapts CLIP to blending artifacts and shows task-specific forensic interfaces can improve generalization; no correction-aware gate into a frozen decoder.
- **ForensicsSAM — MECHANISTICALLY_RELEVANT but preprint:** uses forgery/adversary experts and conditional adversary handling. Its routing concern is adversary/expert identity, not realized correction benefit.
- **TransForensics — BACKGROUND_ONLY:** dense self-attention supports spatial forensic localization but has no separate reliability/utility role.
- **DiffForensics, CFL-Net and JPEG domain adaptation — BACKGROUND_ONLY:** improve forensic representations/robustness rather than dynamically reject a proposed correction.
- **Uncertainty/reliability analysis work — MECHANISTICALLY_RELEVANT:** pixel-level epistemic/aleatoric uncertainty can flag unreliable predictions, but uncertainty of a prediction is not the causal benefit of a feature update.

No reviewed forensic method directly instantiates all of: external forensic evidence → proposal in frozen SAM feature space → spatial gate that sees both current state and proposal → frozen decoder task outcome. Absence from this review is **not proof of novelty**.

## 9. Current-state vs proposed-update vs adapted-state taxonomy

| Interface | Information available | Strength | Missing / risk | Assessment |
|---|---|---|---|---|
| Type I: `G(S,F,confidence,q)` | base/source state and reliability | Proven current task value; can reject weak forensic sources before examining a correction | Cannot distinguish different `C` from the same sources; conflates source quality with proposal compatibility | Current R1; retain as baseline |
| Type II: `G(S,C,q)` | base state and actual proposal | Minimum direct information for direction, magnitude, conflict and compatibility | May lose useful uncertainty/source provenance; can shortcut on amplitude | Best minimal scientific test; **MECHANISTIC_HYPOTHESIS** |
| Type III: `G(S,C,F,confidence,q)` | proposal plus all provenance | Maximum available context | redundancy, shortcuts, parameter/capacity confound, harder attribution | Not the first test |
| Before/after: `G(S,S+C,q)` | explicit candidate state comparison | `C` is exactly recoverable as `(S+C)-S`; may ease optimization | no new information relative to `(S,C)` and doubles highly correlated state inputs | Alternative parameterization, not extra evidence |
| `G(S,C,S+C,...)` | all three | computational convenience only | mathematically redundant; can obscure what gate uses | Do not use initially |

Thus `S+C` may be useful as an inductive bias but does not increase identifiability when both `S` and `C` are present. The minimum sufficient correction-review interface is `(S,C)` plus query and externally enforced `M_valid`. Whether source posterior/confidence contributes additional predictive value should be a later, controlled question.

## 10. 20+ paper comparison matrix

Abbreviations: `S` current state; `U` proposed update; `A` adapted state; `sp` spatial; `tok` token; `ch` channel; `sam` sample. “No” under a gate field means the paper does not implement the target gate, not that the model never uses that tensor elsewhere.

| Paper | Venue/Year | Peer reviewed? | Task | Frozen model? | Proposed update? | Gate sees S / U / A? | Gate level | Gate target | Translator type | Supervision | R1 relevance | Non-transferable aspect | Label |
|---|---|---:|---|---:|---:|---|---|---|---|---|---|---|---|
| [Review Residuals](https://arxiv.org/abs/2606.31859) | arXiv 2026 | **No; preprint** | language modeling | No, from scratch | Yes | Yes / **Yes** / No | token×channel | accept residual update | linear gate over normalized S,U | task loss | closest `G(S,C)` precedent | language, every layer, scale-emergent, no frozen decoder/spatial evidence | **DIRECTLY_RELEVANT** |
| [Highway Networks](https://arxiv.org/abs/1505.00387) | arXiv 2015 | Preprint lineage | deep networks | No | Yes | Yes / No / No | element/ch | carry-transform balance | transformed branch | task loss | historical state-conditioned gating | convex replacement, not post-update review | MECHANISTICALLY_RELEVANT |
| [Residual Adapters](https://proceedings.neurips.cc/paper/2017/hash/e7b24b112a44fdd9ee93bdf998c6ca0e-Abstract.html) | NeurIPS 2017 | Yes | multi-domain vision | Mostly | Yes | No / No / No | static | domain adaptation | bottleneck residual adapter | task loss | residual compatibility interface | no dynamic utility | MECHANISTICALLY_RELEVANT |
| [ReZero](https://arxiv.org/abs/2003.04887) | UAI 2020 | Yes | deep optimization | No | Yes | No / No / No | scalar | stable residual injection | arbitrary branch + zero scalar | task loss | near-zero initial correction | static, not reliability | MECHANISTICALLY_RELEVANT |
| [CaiT / LayerScale](https://openaccess.thecvf.com/content/ICCV2021/html/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.html) | ICCV 2021 | Yes | image classification | No | Yes | No / No / No | channel | optimize deep residuals | diagonal residual scale | task loss | small-init stabilization | static; not frozen interface | MECHANISTICALLY_RELEVANT |
| [AdaptFormer](https://papers.nips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html) | NeurIPS 2022 | Yes | visual transfer | Yes | Yes | No / No / No | static | task adaptation | parallel bottleneck MLP + scale | task loss | frozen-ViT adapter design | no proposal-aware gate or spatial SAM correction | MECHANISTICALLY_RELEVANT |
| [ControlNet](https://arxiv.org/abs/2302.05543) | ICCV 2023 | Yes | controlled diffusion | locked base copy | Yes | context / No / No | spatial/features | conditioning injection | copied blocks + zero conv | diffusion loss | zero-init noninterference | generative diffusion, no utility review | MECHANISTICALLY_RELEVANT |
| [DAPT](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Dynamic_Adapter_Meets_Prompt_Tuning_Parameter-Efficient_Transfer_Learning_for_Point_CVPR_2024_paper.html) | CVPR 2024 | Yes | point-cloud transfer | Yes | Yes | Yes / No / No | token | token significance | bottleneck dynamic adapter | task loss | spatial/token adaptive contribution | gate precedes/does not inspect adapter output | MECHANISTICALLY_RELEVANT |
| [Adapters Strike Back](https://openaccess.thecvf.com/content/CVPR2024/html/Steitz_Adapters_Strike_Back_CVPR_2024_paper.html) | CVPR 2024 | Yes | visual transfer | Yes | Yes | usually No / No / No | static/ch | adaptation quality | studied bottleneck/placement/scaling | task loss | Translator architecture principles | no correction-aware acceptance | MECHANISTICALLY_RELEVANT |
| [SEMA](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Self-Expansion_of_Pre-trained_Models_with_Mixture_of_Adapters_for_Continual_CVPR_2025_paper.html) | CVPR 2025 | Yes | continual learning | Yes | Yes | descriptors / No / No | layer/sample | distribution shift, reuse/expand | mixture of adapters | continual task objectives | separates representation shift and module routing | pre-routing; model growth; no spatial correction | MECHANISTICALLY_RELEVANT |
| [Dynamic Prefix Weighting](https://openaccess.thecvf.com/content/CVPR2026/html/Jang_Enhancing_Continual_Learning_of_Vision-Language_Models_via_Dynamic_Prefix_Weighting_CVPR_2026_paper.html) | CVPR 2026 | Yes | VLM continual learning | Yes | Yes | token / No / No | token | task relevance; adapter only when needed | prefix + adapter residual weighting | CL task loss | fine-grained adapter weighting | adapter weight derives from prefix score, not update review | MECHANISTICALLY_RELEVANT |
| [Keep It Frozen / DCRM-ViT](https://openaccess.thecvf.com/content/CVPR2026/html/Khan_Keep_It_Frozen_Domain-Routed_Conditional_Residual_Modulation_for_Multi-Domain_Vision_CVPR_2026_paper.html) | CVPR 2026 | Yes | multi-domain classification/segmentation | Yes | Yes | input/domain / No / No | sample/domain | domain identity/shift | synthesized low-rank residual modulation | bi-level task objective | frozen model and conditional residual | router precedes residual, not correction benefit | MECHANISTICALLY_RELEVANT |
| [Domain-Rectifying Adapter](https://openaccess.thecvf.com/content/CVPR2024/html/Su_Domain-Rectifying_Adapter_for_Cross-Domain_Few-Shot_Segmentation_CVPR_2024_paper.html) | CVPR 2024 | Yes | cross-domain few-shot segmentation | established source segmenter | Yes | No gate | n/a | domain-space compatibility | feature rectification adapter | segmentation + reverse/cyclic alignment | explicit compatible-space translation | domain style correction, not external proposal utility | **DIRECTLY_RELEVANT** to Translator |
| [AutoSAM](https://arxiv.org/abs/2306.06370) | arXiv 2023 | **No; preprint** | medical segmentation | SAM encoder/decoder frozen | prompt proposal | No explicit gate | n/a | prompt compatibility | image-to-prompt encoder | BCE+Dice through frozen SAM | frozen decoder constrains interface | prompt generation, not feature correction | MECHANISTICALLY_RELEVANT |
| [SAM-Adapter](https://openaccess.thecvf.com/content/ICCV2023W/VCL/html/Chen_SAM-Adapter_Adapting_Segment_Anything_in_Underperformed_Scenes_ICCVW_2023_paper.html) | ICCVW 2023 | Yes (workshop) | camouflage/shadow segmentation | largely SAM adaptation setting | Yes | No / No / No | feature | task knowledge injection | adapters/prompts | segmentation loss | SAM-specific interface precedent | modifies SAM pathways; no utility separation | MECHANISTICALLY_RELEVANT |
| [AVFormer](https://openaccess.thecvf.com/content/CVPR2023/html/Seo_AVFormer_Injecting_Vision_Into_Frozen_Speech_Models_for_Zero-Shot_AV-ASR_CVPR_2023_paper.html) | CVPR 2023 | Yes | audio-visual ASR | speech and vision models frozen | Yes | No explicit gate | token | cross-modal compatibility | linear projection + bottleneck adapters | ASR loss, staged curriculum | external evidence into frozen consumer | speech, not spatial; adapters also handle domain shift | MECHANISTICALLY_RELEVANT |
| [X-Adapter](https://openaccess.thecvf.com/content/CVPR2024/html/Ran_X-Adapter_Adding_Universal_Compatibility_of_Plugins_for_Upgraded_Diffusion_Model_CVPR_2024_paper.html) | CVPR 2024 | Yes | diffusion plugin compatibility | old copy/upgraded model frozen | guidance | No explicit utility gate | multi-level | version compatibility | trainable mapping layers | image-text diffusion objective | explicit bridge between incompatible feature spaces | diffusion denoisers; complex multiscale bridge | MECHANISTICALLY_RELEVANT |
| [Medical SAM Adapter / Med-SA](https://doi.org/10.1016/j.media.2025.103547) | Medical Image Analysis 2025 | Yes | medical segmentation | SAM largely frozen | adapter residuals | No / No / No | feature | domain/task adaptation | SAM adapters | segmentation losses | frozen SAM adaptation | domain adapter, no proposal acceptance | BACKGROUND_ONLY |
| [TruFor](https://openaccess.thecvf.com/content/CVPR2023/html/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.html) | CVPR 2023 | Yes | forgery localization/detection | No | prediction map | reliability uses fused features/prediction context; not C | spatial + image | localization reliability/error propensity | RGB+Noiseprint fusion, confidence head | localization/confidence/detection objectives | separates forensic evidence, prediction and reliability | confidence is not frozen-feature correction utility | **DIRECTLY_RELEVANT** to responsibility separation |
| [OMG-Fuser](https://openaccess.thecvf.com/content/CVPR2024W/WMF/html/Karageorgiou_Fusion_Transformer_with_Object_Mask_Guidance_for_Image_Forgery_Analysis_CVPRW_2024_paper.html) | CVPRW 2024 | Yes (workshop) | forgery localization/detection | forensic inputs may be external | fused features | semantic/object guided, not C review | token/spatial | evidence fusion | per-signal streams + fusion Transformer | task loss | adaptive multi-forensic integration | end-to-end fusion, no harmful-update gate | MECHANISTICALLY_RELEVANT |
| [Forensics Adapter](https://openaccess.thecvf.com/content/CVPR2025/html/Cui_Forensics_Adapter_Adapting_CLIP_for_Generalizable_Face_Forgery_Detection_CVPR_2025_paper.html) | CVPR 2025 | Yes | face forgery detection | CLIP adapted lightly | adapter features | No correction utility | token/feature | blending-trace adaptation | forensic CLIP adapter | detection loss | forensic evidence adaptation | classification; no spatial frozen-decoder correction | BACKGROUND_ONLY |
| [ForensicsSAM](https://github.com/siriusPRX/ForensicsSAM) | arXiv/code 2025 | **No; preprint** | forgery localization | SAM-based | expert outputs | adversary detector/router, not C | sample/expert | adversary type/activation | forgery and adversary experts | task/adversarial objectives | conditional forensic expert activation | routing by attack condition, not realized update benefit | MECHANISTICALLY_RELEVANT |
| [TransForensics](https://openaccess.thecvf.com/content/ICCV2021/html/Hao_TransForensics_Image_Forgery_Localization_With_Dense_Self-Attention_ICCV_2021_paper.html) | ICCV 2021 | Yes | forgery localization | No | No distinct proposal | No gate | n/a | dense localization | dense self-attention | mask supervision | spatial forensic context | no reliability/compatibility decomposition | BACKGROUND_ONLY |
| [DiffForensics](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_DiffForensics_Leveraging_Diffusion_Prior_to_Image_Forgery_Detection_and_Localization_CVPR_2024_paper.html) | CVPR 2024 | Yes | forgery detection/localization | diffusion prior leveraged | No distinct C | No correction gate | n/a | robust forensic features | diffusion-prior feature learner | forensic task losses | forensic robustness | no frozen SAM correction review | BACKGROUND_ONLY |
| [CFL-Net](https://openaccess.thecvf.com/content/WACV2023/html/Niloy_CFL-Net_Image_Forgery_Localization_Using_Contrastive_Learning_WACV_2023_paper.html) | WACV 2023 | Yes | forgery localization | No | No distinct C | No gate | n/a | contrastive forensic separation | localization network | contrastive + mask losses | forensic representation learning | no reliability or adapter interface | BACKGROUND_ONLY |
| [Weakly-Supervised Deepfake Localization](https://openaccess.thecvf.com/content/WACV2024/html/Tantaru_Weakly-Supervised_Deepfake_Localization_in_Diffusion-Generated_Images_WACV_2024_paper.html) | WACV 2024 | Yes | diffusion-image localization | pretrained features | local-score proposal | attention-based, not C review | spatial | localization evidence | weakly supervised localization | image labels | local forensic evidence | no pixel-GT reliability or frozen-SAM correction | BACKGROUND_ONLY |

Counts: 12 gating/residual/dynamic-adapter papers; 7 frozen/compatibility-translator papers; 7 forensic reliability/adaptive-fusion papers. Categories overlap where scientifically appropriate.

## 11. Current R1 mapped against literature

| Responsibility | Current implementation | What literature commonly does | Gap / redundancy | Evidence-based disposition |
|---|---|---|---|---|
| Evidence | block11+17 fusion, projection-only | multi-level/multi-signal representation learning | no current first-order gap; old deep local adapter was redundant for new evidence | KEEP |
| Interaction | geometry-aware cross-attention with support | attention, object guidance, aligned token fusion | no demonstrated conversion failure at R2 | KEEP |
| Translator | affine projection(s), gamma, historically main+side | bottleneck adapters, remapping layers, explicit alignment, zero/small-init residuals | double affine redundant; nonlinear sufficiency unknown; side may be optimization compensation | KEEP learned translation; prefer simplest empirically supported form |
| Validity | deterministic geometry support | masks/availability constraints often implicit | conflation risk if called confidence | KEEP as hard external constraint |
| Utility | source-aware semantic/forensic contexts and evidential comparison; relative/ranking/task losses | confidence heads, state/token/domain routers; rarely true update review | does not see `C`/`S+C`; auxiliary targets do not directly encode realized correction gain | KEEP current baseline; test responsibility alignment separately |

Static implementation evidence:

- `model/csculf.py:41-68`: language context consumes `S64,q_seg,z_L`; forensic context consumes mapped `F,z_F,support`.
- `model/csculf.py:71-166`: source contexts are jointly rectified/exchanged and compared via product, absolute difference, cosine, posteriors and conflict.
- `scripts/phase4g1q_conditional_utility.py:134-166`: `utility_forward` receives `S64,q_seg,z_L,F24,z_F24,clip_geometries`; no correction tensor or corrected state enters the graph.
- `scripts/phase4g1q_conditional_utility.py:175-181`: the relative target is derived from forensic-vs-language pixel BCE difference, not actual downstream improvement from injecting `C`.
- `scripts/phase6g13_utility_train.py:146-163`: frozen Rectifier output is computed separately, Utility output gates it, and final segmentation loss plus relative/ranking losses train Utility. Thus the end task can shape Utility, but Utility cannot condition its prediction on the correction it gates.
- `model/tf_fdg.py:105-171`: geometry-aware attention produces the residual proposal, support masks it, and a small gamma initializes injection conservatively.

## 12. What current R1 already gets right

1. It separates forensic evidence acquisition from semantic interaction.
2. It aligns evidence geometrically and enforces hard support.
3. It translates interaction output before changing frozen SAM features.
4. It injects additively, preserving an identity path.
5. It has a spatial Utility rather than a single global confidence.
6. Utility combines semantic, query, forensic, posterior and conflict information rather than using correction magnitude as a proxy.
7. Final frozen-SAM segmentation loss participates in training, so compatibility is not judged only by a dense probe.
8. Internal ablations establish real positive value for both R1 and Utility; literature cannot override those facts.

## 13. What current R1 may be missing

### 13.1 Most defensible gap: the gated object is hidden from the gate

Current Utility knows whether sources look reliable but not what the Translator actually proposes. This creates a conditional-identifiability gap: for fixed `(S,F,q,confidence)`, any changes in Translator parameters or local `C` geometry are invisible to Utility. The proposal could align with, oppose, or rotate the semantic feature while receiving the same gate.

This is a **MECHANISTIC_HYPOTHESIS**, not yet an empirical fact.

### 13.2 Possible but unproven gap: objective alignment

Relative forensic-vs-language reliability and matched-vs-corrupted ranking are proxies. They may not predict frozen-SAM benefit. But an explicit counterfactual-gain target may itself be unstable and decoder-specific. Literature does not currently justify replacing the existing objectives. The final task loss already provides an implicit correction-outcome signal.

### 13.3 Translator capacity remains unresolved

Internal evidence rejects factorized double affine and direct raw-R2 correction; it does not decide affine versus minimal nonlinear translation. Ongoing/incomplete experiments are not interpreted here. Literature offers candidate principles, not a matched answer.

## 14. What literature does NOT justify changing

- Do not remove R1, Utility, geometry support, or learned translation.
- Do not replace geometry-aware cross-attention merely because newer fusion mechanisms exist.
- Do not restore a deep forensic adapter; internal evidence says projection-only preserves the selected evidence.
- Do not interpret a generic sigmoid, router, confidence map, or MoE gate as proof of correction-aware Utility.
- Do not adopt Type III because it has more inputs.
- Do not change Utility inputs and supervision simultaneously.
- Do not add an explicit alignment loss without a defensible SAM-compatible feature target.
- Do not infer that nonlinear Translator is better from adapter papers alone.
- Do not turn `M_valid` into learned confidence or allow Utility to override absent coverage.
- Do not treat `gamma` as semantic novelty; it is best viewed as residual optimization/stability.

## 15. Novelty-risk analysis

### 15.1 Search conclusion

The exact phrase “correction-aware Utility” is not a novelty claim. Multiple naming families were checked: residual/update gating, adapter-output gating, dynamic residual modulation, proposal verification, feature-correction confidence, routing, forensic reliability and frozen-model compatibility.

### 15.2 Closest overlap

- **Review Residuals:** very high equation-level overlap with `state + gate(state,update)*update`. Novelty risk is **high** if the contribution is described only as “the gate sees the update.” Distinguishing factors are spatial forensic correction, external evidence, hard geometric validity, a frozen SAM consumer, and potentially supervision/diagnostics around correction compatibility. Those differences require evidence; renaming is insufficient.
- **TruFor:** moderate responsibility-level overlap: prediction and reliability are separate. It does not gate a feature correction into a frozen model.
- **DAPT/DPW/DCRM/SEMA:** moderate dynamic-adaptation overlap, but their decision variable is token relevance/domain/shift rather than a realized correction.
- **Domain-Rectifying Adapter/X-Adapter/AutoSAM:** moderate Translator overlap, but not proposal acceptance.

### 15.3 Safe novelty language now

Only the following is defensible: *the reviewed literature motivates studying proposal-aware acceptance at a spatial frozen-SAM forensic interface; it does not establish that this precise problem/mechanism is novel or superior.* A publication-level novelty claim requires a broader systematic search and, more importantly, a mechanism-plus-evidence contribution beyond the generic Review Residual equation.

## 16. Maximum three design hypotheses

### H1 — Minimal SAM-compatible Translator rather than deeper generic adapter

- **Scientific problem:** preserve forensic information while mapping it into the frozen SAM-compatible correction space.
- **Current internal evidence:** raw `R2` is not compatible; a learned translator is necessary; single affine beats factorized double affine; affine sufficiency is unknown.
- **External support:** frozen-consumer interfaces and explicit rectification in Domain-Rectifying Adapter, AutoSAM, AVFormer, X-Adapter, AdaptFormer.
- **What R1 lacks:** not a confirmed module, but a settled minimum sufficient function family.
- **Minimal conceptual change:** compare one carefully initialized minimal nonlinear residual translator to the matched selected affine translator, with Utility fixed.
- **Genuinely new aspect:** forensic-interaction-to-frozen-SAM correction compatibility, not generic adapter depth.
- **Prior overlap:** strong with adapter/remapping literature.
- **Main risk:** extra capacity improves training metrics but worsens frozen-SAM compatibility or confounds optimization.
- **Falsifier:** no stable paired localization gain over the matched affine translator, or improved probe decodability with degraded frozen-SAM masks.
- **Status:** **MECHANISTIC_HYPOTHESIS**; not the recommended immediate question because Translator optimization is/has been investigated separately.

### H2 — Correction-aware Utility is the minimum responsibility-aligned extension

- **Scientific problem:** the current gate controls `C` without observing `C`.
- **Current internal evidence:** Utility adds stable value, but gate values do not track side-specific benefit; stronger evidence is only partially converted downstream.
- **External support:** Review Residuals directly conditions on state and update; dynamic adapters establish fine-grained conditional scaling; TruFor supports separating prediction from reliability.
- **What R1 lacks:** proposal direction, magnitude, compatibility and local conflict at the acceptance decision.
- **Minimal conceptual change:** retain current Translator, support and additive injection; add the actual `C` to a matched-capacity Utility review of `S64`.
- **Genuinely new aspect:** spatial forensic correction review at a frozen SAM interface, if supported beyond generic update gating.
- **Prior overlap:** high equation-level overlap with Review Residuals; moderate conceptual overlap with TruFor and dynamic adapters.
- **Main risk:** the gate learns amplitude shortcuts, duplicates the Translator, or gains nothing because task loss already makes current source context sufficient.
- **Falsifier:** no stable gain over matched source-aware Utility, or unchanged predictions under correction permutation/zeroing after accounting for capacity.
- **Status:** **MECHANISTIC_HYPOTHESIS; highest-priority controlled question.**

### H3 — Hybrid source reliability plus correction compatibility only if each is incrementally identifiable

- **Scientific problem:** source reliability may provide information not recoverable from `S,C`, while correction awareness may repair the current blind spot.
- **Current internal evidence:** existing source-aware Utility is useful; therefore discarding source reliability is not automatically safe.
- **External support:** TruFor and multimodal reliability work support source/prediction confidence; Review Residuals supports update review.
- **What R1 lacks:** evidence that both information sets add complementary benefit.
- **Minimal conceptual change:** only after Type II is established, test source context as an incremental input at matched capacity.
- **Genuinely new aspect:** explicit separation of hard validity, source reliability and correction compatibility in spatial forensics.
- **Prior overlap:** combines two mature ideas; novelty risk is substantial without a distinct objective/analysis.
- **Main risk:** redundancy and shortcut learning make Type III larger but not scientifically cleaner.
- **Falsifier:** source context adds no stable gain beyond `(S,C,q)`, or benefit disappears under matched-capacity controls.
- **Status:** **SPECULATION / deferred**, not the next step.

## 17. ONE recommended next scientific question

> **Keeping the selected Translator, hard geometry validity, frozen SAM, final segmentation objective, data, optimization budget and Utility capacity controlled, does a spatial Utility that directly observes `(S64, C, q_seg)` produce a stable paired localization gain over the current source-aware Utility—and does its behavior demonstrably depend on the identity of `C` rather than only its norm?**

This question isolates input responsibility before changing supervision, Translator or architecture depth. It is the only recommended next question. No experiment is launched by this report.

---

**STOP.** Literature and static responsibility audit complete; no training, implementation, checkpoint mutation, test, Official1000 or OOD access was performed.
