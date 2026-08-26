# Phase 4D-1B — Interface Signal Viability Audit

## 1. Executive Summary

- Step-0 beta gradient is nonzero, but Reader-body gradient at beta=0 is exactly zero.
- Nonzero beta restores Reader gradients; tiny/zero gate suppression is **TRUE**.
- The raw Reader residual exists, but Phase 4D-1's learned gate reduces the effective residual to only a few parts per million of the original query norm.
- BF16 survival and frozen SAM response are scale-dependent. Signal viability does not by itself establish image-specific utilization or localization gain.
- Primary failure mode: **ZERO_OR_TINY_GATE_GRADIENT_SUPPRESSION**.
- A corrected Phase 4D-1R design is justified as a controlled follow-up, but it was not executed.

## 2. Gradient Flow

For POS-FORENSIC at beta=0, the beta gradient is `2.481e-4`, while the Reader-body gradient is `0`. At beta `0.01 / 0.1 / 1.0`, the Reader-body gradient norm becomes `8.984e-5 / 1.736e-3 / 1.805e-2`. The POS-CLIP arm shows the same zero-at-beta-0 and recovery-at-nonzero-beta structure. Audit A performed no optimizer step.

This establishes the gate causality mechanism: beta itself can initially update, but a zero gate blocks the loss gradient from reaching the Reader body. The exact magnitude is scale-dependent and is not interpreted as an optimization hyperparameter comparison.

## 3. Signal Magnitude

For POS-FORENSIC step 512 on the frozen 256-sample population:

- mean query norm: `5276.108774`
- mean raw Reader residual norm: `7.723689`
- mean effective residual norm: `0.018301`
- mean raw residual/query ratio: `1.601e-3`
- mean effective residual/query ratio: `3.794e-6`

The Reader therefore produces a nonzero raw residual, but the learned Phase 4D-1 beta (`-0.0023695`) attenuates it to a very small effective scale. For matched-vs-shuffle image-specific differences, the attenuation is more severe: the mean FP32 query difference is `4.814e-4`, and only `3.125%` of samples retain any changed BF16 query element.

## 4. BF16 Survival

The alpha grid was fixed before inspection and was not selected by IoU. For POS-FORENSIC, survival relative to the baseline query is:

| alpha | BF16 sample survival | matched-vs-shuffle survival |
| ---: | ---: | ---: |
| 0.001 | 12.11% | 1.17% |
| 0.003 | 28.12% | 2.34% |
| 0.01 | 54.69% | 7.03% |
| 0.03 | 87.89% | 21.88% |
| 0.1 | 100.00% | 54.30% |
| 0.3 | 100.00% | 84.77% |
| 1.0 | 100.00% | 99.22% |

The pre-registered D1 rule selected beta `0.03` because it is the smallest grid value with at least 80% POS-FORENSIC survival versus baseline q. This was a numerical-viability rule, not a matched-vs-shuffle, mask, validation-IoU, or performance selector.

![BF16 survival curve](/home/yz/groundingLMM_official/outputs/phase4d1b_interface_signal_audit/figures/phase4d1b/bf16_survival_curve.png)

## 5. SAM Logit Sensitivity

Frozen SAM continuous logits respond increasingly as the residual scale grows. At alpha `0.03`, POS-FORENSIC matched-vs-baseline changes survive, while matched-vs-shuffle effects remain much smaller. Binary changes appear later than continuous-logit changes and were not used to choose alpha.

The deterministic generic-direction control also changes SAM logits after BF16, showing that the prompt path is not globally insensitive to query perturbations. This only establishes interface sensitivity; it does not establish useful forensic direction or improved localization.

![SAM logit sensitivity](/home/yz/groundingLMM_official/outputs/phase4d1b_interface_signal_audit/figures/phase4d1b/sam_logit_sensitivity.png)

## 6. Reader vs Generic Perturbation

Matched-norm generic-direction controls and Reader-direction results are both recorded in `phase4d1b_sam_logit_sensitivity.csv`. The present audit supports a gate/scale defect, but does not establish that the Reader direction is uniquely effective or uniquely ineffective. Direction quality remains unresolved without a controlled localization rerun.

## 7. Optional 64-Step Diagnostic

The optional diagnostic was executed only after Audit A established exact zero Reader-body gradient at beta=0 and recovery at nonzero beta. D0 and D1 used the same first 256 Phase 4D-1 training samples, order, architecture, loss, optimizer, and schedule; only beta initialization differed.

| diagnostic | final beta | step-64 Reader grad norm | matched-vs-shuffle BF16 survival | mean SAM logit abs. diff. | binary masks changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 original gate | `-5.236e-7` | `2.672e-9` | 0.00% | `0` | 0.00% |
| D1 beta=0.03 | `0.0299983` | `1.050e-4` | 18.75% | `1.083e-4` | 0.39% |

D1 clearly restores Reader optimization and transmits a nonzero image-specific signal through BF16 into continuous SAM logits. The binary-mask effect remains sparse (1/256), so this diagnostic supports correcting the gate but does not predict an IoU improvement.

## 8. Final Decision

```text
READER_GRADIENT_VIABLE: PARTIAL
TINY_BETA_SUPPRESSES_OPTIMIZATION: TRUE
BF16_SIGNAL_SURVIVAL: SCALE_DEPENDENT
SAM_QUERY_SENSITIVITY: SCALE_DEPENDENT
PRIMARY_FAILURE_MODE: ZERO_OR_TINY_GATE_GRADIENT_SUPPRESSION
CORRECTED_MINIMAL_RERUN_JUSTIFIED: YES
PROCEED_TO_PHASE_4D_2: NO
```

Phase 4D-1 did not adequately train its Reader body because the zero/tiny gate suppressed both optimization and downstream signal magnitude. This audit does not show that positional encoding works, does not establish localization gain, and does not show that forensic information is transferable or impossible to transfer. Internal test and official1000 remained sealed.
