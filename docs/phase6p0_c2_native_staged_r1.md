# Phase6P0 — C2-native staged R1

Status: **COMPLETE STOP AFTER OFFICIAL1000**.

## Protocol

Replicated the Phase6E.3 route on frozen C2: C2-native Rectifier training, then Utility training with the selected Rectifier frozen, then joint training from both selected checkpoints. C2, C2-native SAM, CLIP forensic adapter and evidential source heads remained frozen. No P1-trained Rectifier or Utility state was loaded.

Canonical TRAIN: 8,836 images, 8,682 with exactly one C2 [SEG] eligible for optimization. Canonical DEV: 1,106 images, 1,084 valid; invalid samples count as zero. Each stage used ten epochs. Only DEV Mean FG IoU selected checkpoints, with earlier epoch winning ties. The Rectifier stage also considered epoch 0.

Preflight included full per-image C2-G0 DEV parity, C2-native SAM state, frozen forensic adapter parity, C2 TRAIN geometry gamma audit, trainable parameter counts, gradient routing and frozen-state hashes.

## Internal DEV results

| Model or stage | Selected epoch | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|
| C2-G0 | — | 0.155826 | 0.219360 | 0.149181 | 0.259631 |
| C2-native R1 rectifier | 8 | 0.175306 | 0.248826 | 0.197611 | 0.330009 |
| C2-native R1 utility | 8 | 0.181434 | 0.255671 | 0.190761 | 0.320402 |
| C2-native R1 joint | 8 | 0.194156 | 0.273981 | 0.223905 | 0.365887 |

## Selected joint R1 versus C2-G0

Paired Mean FG IoU difference: 0.038329; 2,000-draw bootstrap 95% CI [0.028556, 0.048753]; W/T/L 556/200/350; Wilcoxon p = 4.55e-16.
Global FG IoU difference: 0.074724; paired bootstrap 95% CI [0.055523, 0.093208].
The checkpoint was selected on this same DEV population; intervals describe the selected result and are not independent confirmation.

## Historical context and boundary

Phase6E.3 C1-native staged R1 selected epoch 8 with DEV Mean FG IoU 0.204904. C1 and C2 differ in backbone and SEG-valid populations, so this is contextual and not a matched causal comparison.

Complete stage histories, selected checkpoint hashes and paired F1/global statistics are in the output directory. Official1000 was accessed only after the internal DEV selector was frozen. C2-native R1 did not use internal test or localization OOD.

## Selected joint R1 on Official1000

| Model | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|
| C2-G0 | 0.241224 | 0.334328 | 0.233257 | 0.378278 |
| C2-native staged R1 | 0.300914 | 0.417736 | 0.326449 | 0.492215 |

Paired Mean FG IoU difference R1−C2-G0: 0.059690, 95% bootstrap CI [0.048098, 0.071275], W/T/L 628/38/334, Wilcoxon p=4.65e-25. Global FG IoU difference: 0.093192, paired bootstrap CI [0.071592, 0.114406].

C2 generation tokens, IDs, GT foreground counts and live C2 forward mask confusion counts matched the frozen C2-G0 Official1000 record per image. The standalone BF16 SAM runtime differed from live C2 on 964 images; its unadapted counts and paired result are recorded separately. Official1000 was not used to retrain or reselect the joint checkpoint. Historical access to this test set limits independence of this comparison.
