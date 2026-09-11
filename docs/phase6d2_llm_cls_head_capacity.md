# Phase 6D.2 — Frozen LLM [CLS] Head Capacity Audit

LLM、LoRA、vision、localization 与 RINE 全冻结；仅训练 H1/H2/H3 classification head。DEV-OOD 未参与 LR、epoch、width 或 checkpoint 选择。

## Frozen contract

- hidden cache: Phase 6B.2 train `17672` / validation `2212`，storage FP16，所有 arms 完全共享。
- optimizer: AdamW；LR candidates `[0.0001, 0.0003, 0.001]`；epoch budget `10`；batch `256`；seeds `[3407, 3408, 3409]`。
- selector: 每个 arm/seed 按 validation ROC-AUC、Accuracy、较早 epoch、较小 LR 依次选择。
- DEV-OOD DIAGNOSTIC: `2560`，8 generators 各 160 Real + 160 Fake；manifest `/data/yz/groundingLMM_official/outputs/phase6d2_cls_head_capacity/dev_ood_genimage_2560.jsonl`，SHA256 `56c30931a4f1c67985e59a5b2f45b2fc5229cbae4eb96a256a36592554580348`。

### Selected LR/epoch per seed

| Arm | Seed | LR | Epoch | Validation Accuracy | Validation ROC-AUC |
|---|---:|---:|---:|---:|---:|
| H1 | 3407 | 0.0003 | 5 | 0.985081 | 0.999091 |
| H1 | 3408 | 0.0003 | 10 | 0.985986 | 0.999070 |
| H1 | 3409 | 0.0003 | 5 | 0.984629 | 0.999069 |
| H2 | 3407 | 0.0003 | 5 | 0.984629 | 0.999084 |
| H2 | 3408 | 0.0003 | 7 | 0.983273 | 0.999093 |
| H2 | 3409 | 0.0001 | 5 | 0.984629 | 0.999080 |
| H3 | 3407 | 0.0001 | 10 | 0.984629 | 0.999124 |
| H3 | 3408 | 0.0001 | 7 | 0.984177 | 0.999083 |
| H3 | 3409 | 0.0001 | 5 | 0.985533 | 0.999093 |

## validation

| Arm | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 0.985533 | 0.998649 | 0.978300 | 0.992767 | 0.007233 | 0.985428 |
| H1 | 0.985232 | 0.999077 | 0.981314 | 0.989150 | 0.010850 | 0.985174 |
| H2 | 0.984177 | 0.999086 | 0.983424 | 0.984931 | 0.015069 | 0.984167 |
| H3 | 0.984780 | 0.999100 | 0.982520 | 0.987040 | 0.012960 | 0.984747 |

## internal_test

| Arm | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 0.982790 | 0.998403 | 0.979167 | 0.986413 | 0.013587 | 0.982727 |
| H1 | 0.983394 | 0.998634 | 0.983998 | 0.982790 | 0.017210 | 0.983404 |
| H2 | 0.983243 | 0.998617 | 0.987017 | 0.979469 | 0.020531 | 0.983306 |
| H3 | 0.982941 | 0.998633 | 0.984903 | 0.980978 | 0.019022 | 0.982974 |

## dev_ood

| Arm | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 0.644141 | 0.875483 | 0.305469 | 0.982812 | 0.017188 | 0.461902 |
| H1 | 0.670573 | 0.887791 | 0.362240 | 0.978906 | 0.021094 | 0.523709 |
| H2 | 0.684766 | 0.889076 | 0.395573 | 0.973958 | 0.026042 | 0.555694 |
| H3 | 0.677083 | 0.888767 | 0.378646 | 0.975521 | 0.024479 | 0.538345 |

## RINE complementarity

Validation RINE-error rescues: `H0=7`, `H1=6`, `H2=6`, `H3=6`.

### validation

| Head | Both correct | Head only | RINE only | Both wrong | Head-only Real/Fake | RINE-only Real/Fake | Disagreement | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 | 2173 | 7 | 26 | 6 | 7/0 | 4/22 | 0.014919 | 0.951126 | 0.904116 |
| H1 | 2173 | 6 | 26 | 7 | 6/0 | 7/19 | 0.014467 | 0.950539 | 0.900452 |
| H2 | 2173 | 6 | 26 | 7 | 6/0 | 10/16 | 0.014467 | 0.949532 | 0.898999 |
| H3 | 2171 | 6 | 28 | 7 | 6/0 | 9/19 | 0.015371 | 0.949928 | 0.898839 |

### internal_test

| Head | Both correct | Head only | RINE only | Both wrong | Head-only Real/Fake | RINE-only Real/Fake | Disagreement | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 | 2163 | 7 | 30 | 8 | 7/0 | 8/22 | 0.016757 | 0.951292 | 0.907833 |
| H1 | 2164 | 7 | 29 | 8 | 7/0 | 12/17 | 0.016304 | 0.949704 | 0.905964 |
| H2 | 2166 | 7 | 27 | 8 | 7/0 | 14/13 | 0.015399 | 0.949373 | 0.905123 |
| H3 | 2165 | 8 | 28 | 7 | 8/0 | 13/15 | 0.016304 | 0.949959 | 0.905238 |

### dev_ood

| Head | Both correct | Head only | RINE only | Both wrong | Head-only Real/Fake | RINE-only Real/Fake | Disagreement | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 | 1617 | 32 | 367 | 544 | 9/23 | 17/350 | 0.155859 | 0.825885 | 0.830933 |
| H1 | 1668 | 47 | 316 | 529 | 6/41 | 20/296 | 0.141797 | 0.843253 | 0.844595 |
| H2 | 1695 | 57 | 289 | 519 | 6/51 | 29/260 | 0.135156 | 0.844076 | 0.845154 |
| H3 | 1680 | 51 | 304 | 525 | 6/45 | 23/281 | 0.138672 | 0.842849 | 0.845368 |

## Gates

`LLM_CLS_HEAD_BOTTLENECK = INCONCLUSIVE`
`BEST_INTERNAL_HEAD = H3`
`DEV_OOD_GENERALIZATION_GAIN = YES`
`BEST_RINE_COMPLEMENT_HEAD = H0`
`ALLOW_RINE_LLM_FUSION_NEXT = NO`

阶段完成并 STOP。
