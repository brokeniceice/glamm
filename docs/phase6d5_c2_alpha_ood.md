# Phase6D.5 — C2/RINE alpha OOD

Status: **COMPLETE**. Reused the frozen C2 classification OOD probabilities and Phase6D.5 RINE raw margins on identical sample IDs. The C2 raw margin is reconstructed with `logit(p)`; GPU 1 replayed all saturated probabilities and representative original batch-8 groups to check the inversion.

The formal alpha is the historical validation-selected **0.3**. The 0.0–1.0 grid is exploratory; no OOD parameter selection was done. C2 and RINE each use their own frozen internal-TRAIN mean/std; a score > 0 predicts Fake.

C2 checkpoint SHA256 `4a67e6a87c453d554fa5bd6cf93329ae54c1c2853f0925dc7a63397eef27e8ce`; RINE checkpoint SHA256 `5286b05c82416e3a11d067b1f449b566c39360ffe7133499d09aecd0fcb3b562`.

## aigi_holmes (n=99999)

GPU 1 audit: 224 images, including all 18 saturated probabilities; maximum non-saturated `logit(p)` reconstruction difference 0.125762. At alpha=0.3, 0 decisions straddle zero within the saved float32 probability half-ULP interval.

| alpha | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.876909 | 0.983475 | 0.755555 | 0.998260 | 0.001740 | 0.859907 |
| 0.1 | 0.876239 | 0.986653 | 0.754175 | 0.998300 | 0.001700 | 0.859030 |
| 0.2 | 0.875359 | 0.988983 | 0.752395 | 0.998320 | 0.001680 | 0.857882 |
| 0.3 | 0.874429 | 0.990538 | 0.750475 | 0.998380 | 0.001620 | 0.856660 |
| 0.4 | 0.873059 | 0.991471 | 0.747715 | 0.998400 | 0.001600 | 0.854866 |
| 0.5 | 0.871379 | 0.991964 | 0.744315 | 0.998440 | 0.001560 | 0.852655 |
| 0.6 | 0.869189 | 0.992175 | 0.739875 | 0.998500 | 0.001500 | 0.849759 |
| 0.7 | 0.865969 | 0.992200 | 0.733315 | 0.998620 | 0.001380 | 0.845468 |
| 0.8 | 0.861359 | 0.992096 | 0.723994 | 0.998720 | 0.001280 | 0.839280 |
| 0.9 | 0.854839 | 0.991891 | 0.710754 | 0.998920 | 0.001080 | 0.830401 |
| 1.0 | 0.845458 | 0.991603 | 0.691774 | 0.999140 | 0.000860 | 0.817393 |

## genimage (n=100000)

GPU 1 audit: 80 images, including all 0 saturated probabilities; maximum non-saturated `logit(p)` reconstruction difference 0.00632009. At alpha=0.3, 0 decisions straddle zero within the saved float32 probability half-ULP interval.

| alpha | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.768740 | 0.942834 | 0.549040 | 0.988440 | 0.011560 | 0.703627 |
| 0.1 | 0.767470 | 0.950890 | 0.546260 | 0.988680 | 0.011320 | 0.701421 |
| 0.2 | 0.765770 | 0.957229 | 0.542640 | 0.988900 | 0.011100 | 0.698495 |
| 0.3 | 0.763850 | 0.961716 | 0.538660 | 0.989040 | 0.010960 | 0.695216 |
| 0.4 | 0.761610 | 0.964509 | 0.533880 | 0.989340 | 0.010660 | 0.691313 |
| 0.5 | 0.759170 | 0.965978 | 0.528720 | 0.989620 | 0.010380 | 0.687051 |
| 0.6 | 0.755390 | 0.966516 | 0.520780 | 0.990000 | 0.010000 | 0.680411 |
| 0.7 | 0.749910 | 0.966423 | 0.509100 | 0.990720 | 0.009280 | 0.670583 |
| 0.8 | 0.742010 | 0.965892 | 0.492380 | 0.991640 | 0.008360 | 0.656183 |
| 0.9 | 0.730510 | 0.965035 | 0.468780 | 0.992240 | 0.007760 | 0.634971 |
| 1.0 | 0.716520 | 0.963918 | 0.439840 | 0.993200 | 0.006800 | 0.608085 |

## loki (n=2217)

GPU 1 audit: 77 images, including all 1 saturated probabilities; maximum non-saturated `logit(p)` reconstruction difference 0.0180114. At alpha=0.3, 0 decisions straddle zero within the saved float32 probability half-ULP interval.

| alpha | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.676590 | 0.756000 | 0.560364 | 0.846667 | 0.153333 | 0.673051 |
| 0.1 | 0.674335 | 0.758443 | 0.555809 | 0.847778 | 0.152222 | 0.669716 |
| 0.2 | 0.672982 | 0.761206 | 0.552771 | 0.848889 | 0.151111 | 0.667584 |
| 0.3 | 0.672982 | 0.764151 | 0.551253 | 0.851111 | 0.148889 | 0.666973 |
| 0.4 | 0.672530 | 0.767090 | 0.549734 | 0.852222 | 0.147778 | 0.666053 |
| 0.5 | 0.671628 | 0.769830 | 0.546697 | 0.854444 | 0.145556 | 0.664207 |
| 0.6 | 0.667569 | 0.772033 | 0.538345 | 0.856667 | 0.143333 | 0.658005 |
| 0.7 | 0.658548 | 0.774122 | 0.520881 | 0.860000 | 0.140000 | 0.644434 |
| 0.8 | 0.648173 | 0.775718 | 0.500380 | 0.864444 | 0.135556 | 0.628217 |
| 0.9 | 0.642760 | 0.776804 | 0.485194 | 0.873333 | 0.126667 | 0.617391 |
| 1.0 | 0.628778 | 0.777734 | 0.457099 | 0.880000 | 0.120000 | 0.593981 |

## raise998 (n=998)

GPU 1 audit: 51 images, including all 0 saturated probabilities; maximum non-saturated `logit(p)` reconstruction difference 1.40203e-07. At alpha=0.3, 0 decisions straddle zero within the saved float32 probability half-ULP interval.

| alpha | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.998998 | - | 0.000000 | 0.998998 | 0.001002 | 0.000000 |
| 0.1 | 0.998998 | - | 0.000000 | 0.998998 | 0.001002 | 0.000000 |
| 0.2 | 0.998998 | - | 0.000000 | 0.998998 | 0.001002 | 0.000000 |
| 0.3 | 0.998998 | - | 0.000000 | 0.998998 | 0.001002 | 0.000000 |
| 0.4 | 0.998998 | - | 0.000000 | 0.998998 | 0.001002 | 0.000000 |
| 0.5 | 1.000000 | - | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| 0.6 | 1.000000 | - | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| 0.7 | 1.000000 | - | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| 0.8 | 1.000000 | - | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| 0.9 | 1.000000 | - | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| 1.0 | 1.000000 | - | 0.000000 | 1.000000 | 0.000000 | 0.000000 |

## Mixed OOD macro (AIGI-Holmes, GenImage, LOKI)

| alpha | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.774080 | 0.894103 | 0.621653 | 0.944456 | 0.055544 | 0.745528 |
| 0.1 | 0.772681 | 0.898662 | 0.618748 | 0.944919 | 0.055081 | 0.743389 |
| 0.2 | 0.771370 | 0.902473 | 0.615935 | 0.945370 | 0.054630 | 0.741320 |
| 0.3 | 0.770420 | 0.905469 | 0.613463 | 0.946177 | 0.053823 | 0.739616 |
| 0.4 | 0.769066 | 0.907690 | 0.610443 | 0.946654 | 0.053346 | 0.737411 |
| 0.5 | 0.767392 | 0.909257 | 0.606577 | 0.947501 | 0.052499 | 0.734638 |
| 0.6 | 0.764049 | 0.910241 | 0.599667 | 0.948389 | 0.051611 | 0.729392 |
| 0.7 | 0.758142 | 0.910915 | 0.587765 | 0.949780 | 0.050220 | 0.720162 |
| 0.8 | 0.750514 | 0.911235 | 0.572251 | 0.951601 | 0.048399 | 0.707894 |
| 0.9 | 0.742703 | 0.911244 | 0.554909 | 0.954831 | 0.045169 | 0.694254 |
| 1.0 | 0.730252 | 0.911085 | 0.529571 | 0.957447 | 0.042553 | 0.673153 |

RAISE998 contains only Real images, so ROC-AUC and Fake recall are undefined.
GenImage per-generator results and paired per-image scores are retained in the result artifacts.
Precision limit: unsaturated C2 margins are reconstructed from saved float32 probabilities, so ROC-AUC values are numerical approximations to a full raw-logit replay. All 19 saturated cases use exact GPU 1 margins. At formal alpha=0.3, no decision threshold is ambiguous under the saved probability half-ULP intervals.

## Formal alpha=0.3 comparison

| Dataset | C2-center Acc / AUC / F1 | C2+RINE alpha=0.3 Acc / AUC / F1 | Historical C1+RINE alpha=0.3 Acc / AUC / F1 |
|---|---|---|---|
| aigi_holmes | 0.876909 / 0.983475 / 0.859907 | 0.874429 / 0.990538 / 0.856660 | 0.903369 / 0.992995 / 0.893475 |
| genimage | 0.768740 / 0.942834 / 0.703627 | 0.763850 / 0.961716 / 0.695216 | 0.802950 / 0.967524 / 0.759880 |
| loki | 0.676590 / 0.756000 / 0.673051 | 0.672982 / 0.764151 / 0.666973 | 0.713126 / 0.767009 / 0.726334 |
| raise998 | 0.998998 / - / 0.000000 | 0.998998 / - / 0.000000 | 0.998998 / - / 0.000000 |

At the frozen alpha=0.3, C2+RINE increases ROC-AUC over C2-center on AIGI-Holmes, GenImage, and LOKI, while fixed-threshold accuracy and F1 decrease on all three. Historical C1+RINE alpha=0.3 has higher accuracy, ROC-AUC, and F1 on those same populations. The alpha sweep remains exploratory.
