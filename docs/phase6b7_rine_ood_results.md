# Phase 6B.7 — RINE-on-C1 OOD Confirmation

## 结论

冻结的 `RINE-on-C1` 已在全部四个预注册数据集上完成评测。总体结论是：**RINE package 明显提高排序能力和跨域整体表现，但并非在每个数据集的固定 0.5 阈值指标上都超过 C1-Exact。**

- 三个 mixed OOD 数据集的 macro ROC-AUC 从 `0.862917` 提高至 `0.911070`（`+0.048152`）；三个数据集均提高。
- mixed OOD macro Accuracy 从 `0.733647` 提高至 `0.779021`（`+0.045375`），Fake recall 从 `0.568829` 提高至 `0.634097`，TNR 也从 `0.919772` 提高至 `0.940719`。
- GenImage 和 LOKI 的固定阈值指标显著改善；AIGI-Holmes 的 AUC/TNR/FPR 改善，但 Accuracy、Fake recall 和 F1 小幅下降。
- GenImage 八个 generator 的 AUC 全部提高，但 Accuracy/Fake recall 只在五个提高；`Midjourney`、`VQDM`、`Wukong` 的固定阈值 recall 下降。
- 纯真实 RAISE998 上 FPR 从 `0.011022` 降至 `0.001002`，没有出现 specificity 牺牲。

因此，Phase 6B.7 支持把 `RINE-on-C1` 视为比 C1-Exact/LEGION-retrained **整体更强的 OOD classifier candidate**，尤其适合强调跨域 AUC、GenImage/LOKI 识别和真实图低误报；但不能声称它在所有 OOD 数据集、所有固定阈值指标上严格支配 C1。

## 1. 冻结协议与完成性

| 项目 | 冻结值 |
|---|---|
| selected epoch | `1` |
| checkpoint | `outputs/phase6b6_rine_training/selected_checkpoint.pt` |
| checkpoint SHA256 | `5286b05c82416e3a11d067b1f449b566c39360ffe7133499d09aecd0fcb3b562` |
| threshold | `0.5` |
| positive class | Fake |
| protocol SHA256 | `1d1ad31f510457d96bdd298b082260e9c673227828f12fafd054d5b17c58bd11` |
| datasets | AIGI-Holmes, GenImage, LOKI classification, RAISE998 |
| inference runtime | `1311.13 s`（约 `21.85 min`） |
| final status | `COMPLETE` |

本阶段没有 training、threshold tuning、calibration、epoch/checkpoint reselection、`q/D'/xi` 修改、OOD 选层、fusion 或 AIDE。`C1-Exact / LEGION-retrained` 复用已冻结且同样本对齐的 prediction；Phase 6B.1b 已确认两者推理等价。

## 2. Dataset-level results

### AIGI-Holmes official TestSet

| Model | N | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1-Exact / LEGION-retrained | 99,999 | **0.888109** | 0.976110 | **0.783896** | 0.992320 | 0.007680 | **0.875091** |
| RINE-on-C1 | 99,999 | 0.884399 | **0.991609** | 0.771135 | **0.997660** | **0.002340** | 0.869632 |
| RINE - C1 | — | -0.003710 | +0.015499 | -0.012760 | +0.005340 | -0.005340 | -0.005459 |

RINE 的排序能力和 specificity 明显提高，但在冻结阈值 0.5 下更加保守，少识别了 638 张 Fake，导致 Accuracy/F1 小幅下降。

### GenImage test

| Model | N | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1-Exact / LEGION-retrained | 100,000 | 0.729160 | 0.931478 | 0.476880 | 0.981440 | 0.018560 | 0.637779 |
| **RINE-on-C1** | 100,000 | **0.773820** | **0.963910** | **0.560920** | **0.986720** | **0.013280** | **0.712641** |
| **RINE - C1** | — | **+0.044660** | **+0.032432** | **+0.084040** | **+0.005280** | **-0.005280** | **+0.074862** |

GenImage 上 RINE 同时提高 Fake recall 和 TNR，不是以增加真实图误报换取 recall。

### LOKI classification

| Model | N | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1-Exact / LEGION-retrained | 2,217 | 0.583672 | 0.681164 | 0.445710 | 0.785556 | 0.214444 | 0.559847 |
| **RINE-on-C1** | 2,217 | **0.678845** | **0.777690** | **0.570235** | **0.837778** | **0.162222** | **0.678410** |
| **RINE - C1** | — | **+0.095174** | **+0.096526** | **+0.124525** | **+0.052222** | **-0.052222** | **+0.118563** |

LOKI 是本阶段最强的相对改善：recall 和 specificity 同时提升。

### RAISE998（Real-only）

| Model | N | TNR | FPR | False positives |
|---|---:|---:|---:|---:|
| C1-Exact / LEGION-retrained | 998 | 0.988978 | 0.011022 | 11 |
| **RINE-on-C1** | 998 | **0.998998** | **0.001002** | **1** |

RAISE998 不含 Fake，因此不解释 ROC-AUC、Fake recall 或 F1。RINE 的 FPR 为 `0.1002%`，相较 C1 少 10 个 false positives，当前是可接受且更优的真实图鲁棒性。

## 3. Mixed OOD macro summary

该 macro 只平均 AIGI-Holmes、GenImage、LOKI 三个同时含 Real/Fake 的数据集，不包含 Real-only RAISE998。

| Model | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|
| C1-Exact / LEGION-retrained | 0.733647 | 0.862917 | 0.568829 | 0.919772 | 0.080228 | 0.690906 |
| **RINE-on-C1** | **0.779021** | **0.911070** | **0.634097** | **0.940719** | **0.059281** | **0.753561** |
| **RINE - C1** | **+0.045375** | **+0.048152** | **+0.065268** | **+0.020947** | **-0.020947** | **+0.062655** |

RINE 赢得 `2/3` 个 mixed dataset 的 Accuracy、`3/3` 的 ROC-AUC。

## 4. GenImage per-generator

| Generator | Model | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ADM | C1 | 0.622000 | 0.875066 | 0.264500 | 0.979500 | 0.020500 | 0.411673 |
| ADM | **RINE** | **0.643667** | **0.926686** | **0.302333** | **0.985000** | **0.015000** | **0.459008** |
| BigGAN | C1 | 0.755000 | 0.962562 | 0.526000 | 0.984000 | 0.016000 | 0.682231 |
| BigGAN | **RINE** | **0.890333** | **0.992310** | **0.791667** | **0.989000** | **0.011000** | **0.878328** |
| GLIDE | C1 | 0.760167 | 0.952912 | 0.537833 | 0.982500 | 0.017500 | 0.691599 |
| GLIDE | **RINE** | **0.887500** | **0.989863** | **0.787667** | **0.987333** | **0.012667** | **0.875023** |
| Midjourney | **C1** | **0.806750** | 0.950696 | **0.635333** | 0.978167 | 0.021833 | **0.766771** |
| Midjourney | RINE | 0.786167 | **0.964630** | 0.586167 | **0.986167** | **0.013833** | 0.732708 |
| SD v1.4 | C1 | 0.787667 | 0.962765 | 0.592167 | 0.983167 | 0.016833 | 0.736068 |
| SD v1.4 | **RINE** | **0.841167** | **0.985078** | **0.694833** | **0.987500** | **0.012500** | **0.813940** |
| SD v1.5 | C1 | 0.792125 | 0.962061 | 0.602375 | 0.981875 | 0.018125 | 0.743443 |
| SD v1.5 | **RINE** | **0.851813** | **0.985990** | **0.715125** | **0.988500** | **0.011500** | **0.828350** |
| VQDM | **C1** | **0.578083** | 0.836111 | **0.175000** | 0.981167 | 0.018833 | **0.293173** |
| VQDM | RINE | 0.565667 | **0.895332** | 0.146833 | **0.984500** | **0.015500** | 0.252653 |
| Wukong | **C1** | **0.710500** | 0.938608 | **0.440000** | 0.981000 | 0.019000 | **0.603153** |
| Wukong | RINE | 0.698250 | **0.960630** | 0.411333 | **0.985167** | **0.014833** | 0.576838 |

汇总：RINE 在 `8/8` generator 上提高 AUC、`8/8` 不降低 TNR，但只在 `5/8` 提高 Accuracy 和 Fake recall。因此 multi-layer 增益在排序能力上普遍成立；固定阈值决策增益主要集中于 ADM、BigGAN、GLIDE、SD v1.4 和 SD v1.5，而不是所有 generator。

## 5. Same-sample paired comparison

| Dataset | Both correct | RINE-only correct | C1-only correct | Both wrong | Disagreement | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| AIGI-Holmes | 85,130 | 3,309 | 3,680 | 7,880 | 6,989 | `9.56e-6` |
| GenImage | 68,984 | 8,398 | 3,932 | 18,686 | 12,330 | `< machine-representable range` |
| LOKI | 1,218 | 287 | 76 | 636 | 363 | `5.79e-30` |
| RAISE998 | 987 | 10 | 0 | 1 | 10 | `0.001953` |

McNemar 的方向与 Accuracy 一致：GenImage、LOKI、RAISE998 显著偏向 RINE；AIGI-Holmes 显著但方向小幅偏向 C1。GenImage JSON 中 exact p 因浮点下溢记录为 `0.0`，应解释为极小，而不是数学意义上的严格零。

## 6. 对预注册问题的回答

1. **RINE 是否在 mixed OOD 上稳定超过 C1-Exact？** 以 ROC-AUC 而言是，`3/3` 数据集提高；以固定阈值 Accuracy/F1 而言不是，AIGI-Holmes 小幅下降。macro 层面明确超过。
2. **AUC 提升是否跨数据集成立？** 是。AIGI `+0.015499`、GenImage `+0.032432`、LOKI `+0.096526`，并覆盖 GenImage `8/8` generators。
3. **Fake recall 是否提高而没有明显牺牲 TNR？** macro 上 recall 和 TNR 同时提高；GenImage、LOKI 同时提高。AIGI recall 下降 `0.012760`，但 TNR 提高 `0.005340`。没有任何主数据集出现 TNR 牺牲。
4. **GenImage 是否普遍改善？** AUC/TNR 普遍改善，但固定阈值 Accuracy/recall 只改善 `5/8`；不是所有 generator 的全面支配。
5. **RAISE998 FPR 是否可接受？** 是，`0.001002`（1/998），并优于 C1 的 `0.011022`（11/998）。

## 7. 最终定位

当前证据支持：

```text
RINE-on-C1 = preferred overall OOD classifier candidate
```

理由是它在不触碰 localization 的前提下，获得跨三个 mixed OOD 数据集一致的 AUC 提升、mixed macro 全指标改善，并显著降低 RAISE998 FPR。保留限制：AIGI-Holmes 与三个 GenImage generator 在 threshold=0.5 下仍由 C1 获得更高 Accuracy/recall/F1；本阶段禁止调阈值，因此不对这一差异做事后修正。

## 8. Artifacts

- machine-readable results: `outputs/phase6b7_rine_ood/results.json`
- frozen protocol: `outputs/phase6b7_rine_ood/protocol.json`
- completion marker: `outputs/phase6b7_rine_ood/worker_complete.json`
- detached log: `outputs/phase6b7_rine_ood/detached_pipeline.log`
- predictions: `outputs/phase6b7_rine_ood/predictions/<dataset>/predictions.pt`

Phase 6B.7 已完成并 **STOP**。未自动进入训练、调参、校准、融合或后续 OOD 实验。
