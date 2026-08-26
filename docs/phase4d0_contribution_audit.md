# Phase 4D-0 — Contribution Audit

## Contribution 1

### Claim

Phrase-Aligned Forensic SFT aligns autonomous forensic language trajectories with localization targets and yields a statistically supported deployable grounding improvement.

### Supporting Evidence

- The method change is explicit and narrow: add authoritative `Target regions:` text before `[SEG]`; architecture, union target, losses, G0 prompt, generation, selector, and threshold remain fixed.
- Matched C0 vs P1 on official1000: G0 FG IoU `0.179786→0.229544`, delta `+0.049758`, 95% CI `[+0.034534,+0.064341]`, wins/ties/losses `569/89/342`, Wilcoxon `p=5.36e-15`.
- Matched TF context interaction: `+0.249720`, CI `[+0.228733,+0.271278]`, supporting phrase-conditioned adaptation rather than generic extra context benefit.
- Phase 3C.0 Phrase Repair gains `+0.108989`, CI `[+0.095485,+0.122814]`, and P1 retains a large TF ceiling, linking the method to an identified language/query bottleneck.
- Classification is non-regressive and improved in the matched comparison.

### Novel Method or Analysis?

`METHOD`

### Strength

`STRONG`

### Main Reviewer Attack

The phrase target comes from annotations and may be viewed as richer supervision rather than a sufficiently novel method; the exact original P1 step-0 state is unavailable, and the main localization result is on one synthetic forgery benchmark.

### Missing Experiment

No additional same-init C0 experiment is needed; it already exists. The important missing evidence is frozen, pre-registered external/generalization evaluation and, if authoritative labels permit, category/source stratification. This is a paper-validation need, not a reason to retrain P1.

### Stage-I judgment

The statement “Stage-I supervision aligns autonomous language grounding with forgery-localization targets” is supported when written as a controlled empirical claim, not a strict decomposition of every training-path cause. `METHOD_CONTRIBUTION_1: STRONG`.

## Contribution 2

### Claim

A CLIP-anchored residual adapter learns a more forensic-sensitive dense representation under a target-, geometry-, and evaluator-matched probe protocol.

### Supporting Evidence

- Raw CLIP / projection / adapter validation FG IoU: `0.143842 / 0.138264 / 0.171424`.
- Adapter−projection: `+0.033160`, CI `[+0.027825,+0.038722]`.
- Adapter−raw: `+0.027581`, CI `[+0.022586,+0.032726]`.
- Same 8,836 train Fake and 1,106 validation Fake, same official-annotation-derived union target, 336 geometry, original-space inversion, zero threshold, and per-image aggregation.
- `F_forensic` is non-collapsed; residual-block attribution is supported.

### Novel Method or Analysis?

`ANALYSIS`

### Strength

`MEDIUM`

### Main Reviewer Attack

The result is validation-probe learnability, not an end-to-end method, and the downstream Reader did not show forensic-specific advantage. A reviewer can reasonably call it an auxiliary representation study rather than a second contribution.

### Missing Experiment

A causal transfer test in which correct-image and correct spatial arrangement of the forensic representation are necessary for grounding. Held-out generalization is also missing, but should follow only after the interface mechanism is supported.

## Contribution 3

### Claim

Controlled evidence interventions identify an interface bottleneck: the current Reader's small G0 gain is generic visual-conditioned query compensation rather than image-specific spatial forensic retrieval.

### Supporting Evidence

- Phase 4C-B Reader gains over P1 are significant, but forensic does not exceed CLIP: delta `-0.000476`, CI `[-0.001034,+0.000095]`.
- Phase 4C-C matched−cross-image is approximately zero for both Readers.
- Matched−spatial-shuffle is exactly/numerically zero; the Reader has no positional embedding and is permutation-invariant.
- Matched−zero is positive for both Readers: CLIP `+0.005061`, forensic `+0.004843`, with CIs excluding zero.
- Reader gain correlates with the frozen TF−G0 gap (`ρ=.2806/.2650`) and stronger oracle queries are less helped or harmed.

### Novel Method or Analysis?

`MECHANISTIC_FINDING`

### Strength

`STRONG`

### Main Reviewer Attack

This is a negative mechanistic result on one frozen Reader architecture and validation population; it does not prove that forensic evidence is intrinsically unusable.

### Missing Experiment

No experiment is needed to support the narrow negative conclusion about the current Reader. A minimal position-aware causal test is needed only to turn the finding into a new method direction.

## Candidate Phase 4C claim decisions

| Candidate | Decision | Reason |
|---|---|---|
| B1: We learn a forensic-sensitive dense representation | `PAPER_ANALYSIS_CONTRIBUTION` | Strong matched probe evidence, but no held-out or downstream causal utilization |
| B2: We introduce a Reader that allows LLM grounding queries to access forensic evidence | `NOT_SUPPORTED` | Wrong-image and spatial shuffle do not matter; forensic Reader does not beat CLIP Reader |
| B3: We identify an interface bottleneck between forensic representation and autonomous grounding | `PAPER_ANALYSIS_CONTRIBUTION` | Supported by the matched/cross/shuffle/zero intervention matrix and query-gap correlation |

## Paper contribution structure

The defensible structure is:

```text
one strong method
  +
systematic mechanistic analysis of the autonomous–oracle gap,
representation learnability, and the failed evidence interface
```

`SECOND_METHOD_NOT_JUSTIFIED`

Phase 4C-A alone should not be promoted into a second end-to-end method, and the current Reader must not be described as forensic evidence retrieval.
