# Phase 3G — 4096D Raw Hidden Autonomous–Oracle Grounding Distillation

## Outcome

Phase 3G terminated at the mandatory one-step directional preflight. Its terminal gate is:

> `GATE_RAW4096_GRADIENT_DIRECTION_INVALID`

Formal 4500-step training was not started. Consequently, there is no Phase 3G trained checkpoint, selector result, validation comparison, population representation comparison, or matched-SFT control.

## Controlled hypothesis and representation indexing

This was the final post-hoc representation-target test motivated by Phase 3F sample-level correlations. The only intended core change from Phase 3F was replacing the projected 256D cosine target with a raw 4096D cosine target before frozen `text_hidden_fcs`.

The raw representation was not the hidden state of the `[SEG]` token itself. Both autonomous and authoritative trajectories used the shared causal extraction helper and selected the expanded hidden position immediately preceding the final valid `[SEG]`, i.e. the state that predicts `[SEG]`.

P1, the Phase 3B canonical batch-1 rollout cache, the Phase 3F schedule, canonical prompt, teacher trajectory, LoRA-only trainable boundary, optimizer, and nominal exposure budget were otherwise frozen. Phase 3F's historical gate remains unchanged.

## Mandatory 32-Fake preflight

The oracle-direction and gap checks passed:

- mean autonomous G0 foreground IoU: `0.198618`;
- mean authoritative TF-full foreground IoU: `0.436491`;
- mean raw-4096 cosine gap: `0.440486`;
- mean raw-4096 relative L2: `0.941983`;
- mean projected-256 cosine gap, diagnostic only: `0.170656`.

The raw-only representation loss reached LoRA with gradient norm `0.920358`. No frozen parameter tensor received a gradient. Thus the loss graph was connected to the authorized trainable parameters.

The preregistered four-sample one-step directional audit failed. Mean raw-4096 cosine gap changed from `0.6678253710` to `0.6686340272`, a worsening of `+0.0008086562`. Two samples worsened, one improved, and one was nearly unchanged; the required mini-batch mean inequality `gap_after < gap_before` was false.

The temporary optimizer step changed LoRA only. `text_hidden_fcs`, mask decoder, and all other frozen groups were unchanged, after which the original P1 LoRA state was restored exactly.

## Stopping decision

The instruction explicitly required `STOP` if the one-step direction audit failed. Repeating the preflight with a favorable seed, changing mini-batch membership, lowering the learning rate, tuning loss weights, or launching training despite the gate would convert the preflight into post-hoc tuning. None was done.

All downstream Phase 3G activities are therefore `NOT_RUN_BY_PREFLIGHT_GATE`: 4500-step training, checkpoint selection, canonical G0/Phrase-Only/TF-full validation evaluation, paired bootstrap, population representation analysis, qualitative selection, and matched SFT. Internal test and official1000 remained sealed.

## Answers to the final protocol questions

1. Phase 3F failed because neither its G0 improvement nor its population 256D/4096D alignment changes were reliable; that historical result is not rewritten.
2. Phase 3G had post-hoc motivation from the stronger sample-level 4096D alignment-change/localization-change association in Phase 3F.
3. The intended unique target change was projected 256D cosine alignment to pre-projection raw 4096D causal-predictor cosine alignment.
4. The 4096D loss genuinely entered LoRA; its preflight gradient norm was positive and frozen gradients were absent.
5. The temporary one-step update did not move the mini-batch mean toward oracle, so the mandatory direction gate failed.
6. Population 4096D and 256D changes were not measured because no trained checkpoint exists.
7. No checkpoint was selected; P1 remains the frozen reference rather than a Phase 3G selection.
8. G0, Phrase-Only, TF-full, classification, structure, and delta-gap/delta-IoU comparisons were not run.
9. Matched SFT was not triggered because a significant positive G0 result could not exist without training.
10. No gain can be attributed to raw-4096 distillation.
11. Stage-II representation-distillation search is formally stopped.
12. FEPN is the next main route, but it requires a separate explicit authorization and preregistration.

The supported interpretation is narrower than a model-capacity claim: under the frozen-rollout, LoRA-only configuration and the preregistered one-step optimizer audit, direct raw-4096 cosine alignment did not demonstrate a valid local optimization direction. This does not establish that all raw-hidden objectives or all representation learning are intrinsically impossible.
