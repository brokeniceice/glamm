# Phase 6B.1 — Frozen CLIP-CLS Classifier External Confirmation

## Conclusion

Frozen C1 external inference is complete with no training, calibration, threshold sweep, ensemble, filtering, or resplit. The final interpretation is generated below from the frozen results.

- external confirmation: **SUPPORTED**
- formal LLM `[CLS]` → CLIP CLS replacement: **GO**, candidate C1-L; integration requires a separate stage/authorization.

## Main results

### Accuracy

| Dataset | R1 | C1-S | C1-L | LEGION-retrained |
|---|---:|---:|---:|---:|
| internal | 0.9837 | 0.9884±0.0007 | 0.9879±0.0009 | 0.9864 |
| aigi_holmes | 0.8312 | 0.8693±0.0052 | 0.8741±0.0049 | 0.8881 |
| genimage | 0.6420 | 0.6994±0.0103 | 0.7059±0.0111 | 0.7292 |
| loki | 0.5426 | 0.5575±0.0091 | 0.5634±0.0076 | 0.5837 |
| raise998 | N/A | N/A | N/A | N/A |

### ROC-AUC

| Dataset | R1 | C1-S | C1-L | LEGION-retrained |
|---|---:|---:|---:|---:|
| internal | 0.9984 | 0.9992±0.0000 | 0.9992±0.0000 | 0.9993 |
| aigi_holmes | 0.9565 | 0.9723±0.0007 | 0.9728±0.0005 | 0.9761 |
| genimage | 0.8713 | 0.9254±0.0017 | 0.9267±0.0025 | 0.9315 |
| loki | 0.6542 | 0.6724±0.0055 | 0.6732±0.0064 | 0.6812 |
| raise998 | N/A | N/A | N/A | N/A |

### Fake recall

| Dataset | R1 | C1-S | C1-L | LEGION-retrained |
|---|---:|---:|---:|---:|
| internal | 0.9792 | 0.9879±0.0014 | 0.9894±0.0014 | 0.9918 |
| aigi_holmes | 0.6670 | 0.7437±0.0111 | 0.7536±0.0105 | 0.7839 |
| genimage | 0.2993 | 0.4155±0.0220 | 0.4288±0.0230 | 0.4769 |
| loki | 0.3455 | 0.3867±0.0176 | 0.4004±0.0149 | 0.4457 |
| raise998 | N/A | N/A | N/A | N/A |

### TNR

| Dataset | R1 | C1-S | C1-L | LEGION-retrained |
|---|---:|---:|---:|---:|
| internal | 0.9882 | 0.9888±0.0021 | 0.9864±0.0016 | 0.9810 |
| aigi_holmes | 0.9954 | 0.9949±0.0007 | 0.9945±0.0006 | 0.9923 |
| genimage | 0.9847 | 0.9834±0.0014 | 0.9830±0.0009 | 0.9814 |
| loki | 0.8311 | 0.8074±0.0050 | 0.8019±0.0032 | 0.7856 |
| raise998 | 0.9940 | 0.9937±0.0023 | 0.9913±0.0015 | 0.9890 |

### FPR

| Dataset | R1 | C1-S | C1-L | LEGION-retrained |
|---|---:|---:|---:|---:|
| internal | 0.0118 | 0.0112±0.0021 | 0.0136±0.0016 | 0.0190 |
| aigi_holmes | 0.0046 | 0.0051±0.0007 | 0.0055±0.0006 | 0.0077 |
| genimage | 0.0153 | 0.0166±0.0014 | 0.0170±0.0009 | 0.0186 |
| loki | 0.1689 | 0.1926±0.0050 | 0.1981±0.0032 | 0.2144 |
| raise998 | 0.0060 | 0.0063±0.0023 | 0.0087±0.0015 | 0.0110 |

### F1

| Dataset | R1 | C1-S | C1-L | LEGION-retrained |
|---|---:|---:|---:|---:|
| internal | 0.9836 | 0.9884±0.0007 | 0.9879±0.0009 | 0.9865 |
| aigi_holmes | 0.7980 | 0.8505±0.0069 | 0.8568±0.0065 | 0.8751 |
| genimage | 0.4553 | 0.5800±0.0212 | 0.5929±0.0223 | 0.6378 |
| loki | 0.4730 | 0.5093±0.0164 | 0.5213±0.0137 | 0.5598 |
| raise998 | N/A | N/A | N/A | N/A |

## C1-L minus R1 and LEGION gap closure

| Dataset | ΔAcc | ΔAUC | ΔRecall | ΔTNR | ΔFPR | Gap closure Acc/AUC/Recall |
|---|---:|---:|---:|---:|---:|---:|
| internal | +0.0042 | +0.0009 | +0.0103 | -0.0018 | +0.0018 | 1.167 / 0.955 / 0.714 |
| aigi_holmes | +0.0429 | +0.0163 | +0.0866 | -0.0009 | +0.0009 | 0.654 / 0.808 / 0.638 |
| genimage | +0.0639 | +0.0553 | +0.1295 | -0.0017 | +0.0017 | 0.586 / 0.873 / 0.579 |
| loki | +0.0207 | +0.0190 | +0.0549 | -0.0293 | +0.0293 | 0.297 / 0.452 / 0.379 |

Gap closure is descriptive only and is emitted only when LEGION-retrained improves over R1 for that metric.

## Full mixed-dataset metrics

| Dataset/arm | Balanced Acc | Precision | Brier | ECE-15 |
|---|---:|---:|---:|---:|
| internal / R1 | 0.9837 | 0.9881 | 0.0144 | 0.0134 |
| internal / C1-S | 0.9884±0.0007 | 0.9888±0.0021 | 0.0099±0.0001 | 0.0066±0.0005 |
| internal / C1-L | 0.9879±0.0009 | 0.9865±0.0015 | 0.0096±0.0001 | 0.0054±0.0013 |
| internal / LEGION-retrained | 0.9864 | 0.9812 | 0.0106 | 0.0097 |
| aigi_holmes / R1 | 0.8312 | 0.9931 | 0.1537 | 0.1646 |
| aigi_holmes / C1-S | 0.8693±0.0052 | 0.9932±0.0008 | 0.1088±0.0041 | 0.1304±0.0069 |
| aigi_holmes / C1-L | 0.8741±0.0049 | 0.9928±0.0007 | 0.1048±0.0039 | 0.1251±0.0063 |
| aigi_holmes / LEGION-retrained | 0.8881 | 0.9903 | 0.0985 | 0.1067 |
| genimage / R1 | 0.6420 | 0.9513 | 0.3328 | 0.3444 |
| genimage / C1-S | 0.6994±0.0103 | 0.9615±0.0011 | 0.2549±0.0083 | 0.2798±0.0105 |
| genimage / C1-L | 0.7059±0.0111 | 0.9618±0.0005 | 0.2489±0.0091 | 0.2736±0.0110 |
| genimage / LEGION-retrained | 0.7292 | 0.9625 | 0.2391 | 0.2544 |
| loki / R1 | 0.5883 | 0.7496 | 0.4272 | 0.4237 |
| loki / C1-S | 0.5971±0.0072 | 0.7460±0.0061 | 0.3880±0.0044 | 0.3784±0.0049 |
| loki / C1-L | 0.6011±0.0059 | 0.7472±0.0041 | 0.3811±0.0034 | 0.3711±0.0034 |
| loki / LEGION-retrained | 0.6156 | 0.7526 | 0.3878 | 0.3769 |

RAISE998 is Real-only: only TNR/FPR/TN/FP are interpreted as primary; its Accuracy, recall, F1, ROC-AUC, and balanced accuracy are not used for claims.

## Representation versus capacity

- internal: C1-L−C1-S ΔAcc=-0.0005, ΔAUC=+0.0001
- aigi_holmes: C1-L−C1-S ΔAcc=+0.0048, ΔAUC=+0.0005
- genimage: C1-L−C1-S ΔAcc=+0.0064, ΔAUC=+0.0012
- loki: C1-L−C1-S ΔAcc=+0.0059, ΔAUC=+0.0008
- raise998: C1-L−C1-S ΔAcc=-0.0023, ΔAUC=N/A

## RAISE998 Real-only primary counts

| Arm | TNR | FPR | TN | FP |
|---|---:|---:|---:|---:|
| R1 | 0.9940 | 0.0060 | 992 | 6 |
| C1-S | 0.9937±0.0023 | 0.0063±0.0023 | 991.6667±2.3094 | 6.3333±2.3094 |
| C1-L | 0.9913±0.0015 | 0.0087±0.0015 | 989.3333±1.5275 | 8.6667±1.5275 |
| LEGION-retrained | 0.9890 | 0.0110 | 987 | 11 |

## Statistical analysis

Representative seed 3407 was frozen from internal validation before external inference.

| Dataset | Comparison | McNemar p | ΔAcc bootstrap 95% CI | ΔRecall 95% CI | ΔTNR 95% CI |
|---|---|---:|---:|---:|---:|
| internal | C1-L vs R1 | 0.337 | [-0.0023, +0.0086] | [+0.0009, +0.0172] | [-0.0100, +0.0045] |
| internal | C1-L vs LEGION-retrained | 1 | [-0.0036, +0.0045] | [-0.0082, +0.0000] | [-0.0018, +0.0109] |
| aigi_holmes | C1-L vs R1 | 0 | [+0.0354, +0.0390] | [+0.0711, +0.0781] | [-0.0009, +0.0005] |
| aigi_holmes | C1-L vs LEGION-retrained | 9.63e-288 | [-0.0208, -0.0186] | [-0.0445, -0.0402] | [+0.0023, +0.0035] |
| genimage | C1-L vs R1 | 0 | [+0.0489, +0.0533] | [+0.0987, +0.1070] | [-0.0019, +0.0005] |
| genimage | C1-L vs LEGION-retrained | 0 | [-0.0375, -0.0346] | [-0.0775, -0.0720] | [+0.0017, +0.0034] |
| loki | C1-L vs R1 | 0.113 | [-0.0023, +0.0266] | [+0.0190, +0.0577] | [-0.0467, -0.0044] |
| loki | C1-L vs LEGION-retrained | 5.21e-09 | [-0.0388, -0.0194] | [-0.0759, -0.0486] | [+0.0078, +0.0322] |

Full discordant counts and 10,000-repeat paired bootstrap metadata are in `statistics/paired.json`. ROC-AUC remains a point estimate.

## GenImage per generator

All ADM, BigGAN, GLIDE, Midjourney, SD-v1.4, SD-v1.5, VQDM, and Wukong rows are in `tables/genimage_per_generator.csv`; thresholds remain 0.5.

## Leakage and claim boundary

AIGI-Holmes contains previously audited overlap with internal train. The result is retained under the frozen ACTIVE override and is not described as passing a leakage-free OOD audit.

LEGION-public classification is N/A because the public intermediate checkpoint has no trained prediction head.

## Final answers

1. **C1-L solves a material part of the current OOD weakness: SUPPORTED.** It improves mean Accuracy, ROC-AUC, and Fake recall over R1 on AIGI-Holmes, GenImage, and LOKI, while retaining Internal2208 accuracy. Representative-seed LOKI Accuracy CI still crosses zero, so that dataset-level gain is less certain than AIGI/GenImage.
2. **C1-L approaches but does not generally reach LEGION-retrained.** Accuracy gap closure is 65.4% on AIGI-Holmes, 58.6% on GenImage, and 29.7% on LOKI; it remains below LEGION on all three external mixed datasets, while slightly exceeding it on Internal2208.
3. **C1-S is already strong; the large head adds modest OOD operating-point value.** C1-L adds 0.48–0.64 percentage-point Accuracy on the three external mixed sets, but only 0.0005–0.0012 AUC. This supports CLIP representation as the primary factor, with a secondary capacity effect.
4. **Real specificity is modestly impaired.** C1-L mean FPR rises versus R1 by 0.0009 on AIGI-Holmes, 0.0017 on GenImage, 0.0293 on LOKI, and 0.0027 on RAISE998. RAISE FPR 0.0087 remains below LEGION-retrained 0.0110; LOKI shows a real sensitivity-specificity tradeoff.
5. **Replacement recommendation: GO for C1-L**, with the specificity tradeoff retained explicitly. This is an architecture decision, not a threshold-tuning decision.
6. **Actual integration is required next**, but only under separate authorization: route frozen CLIP CLS through the frozen C1-L head inside R1 inference while leaving localization and all current checkpoints unchanged.

**Phase 6B.1 STOP.**
