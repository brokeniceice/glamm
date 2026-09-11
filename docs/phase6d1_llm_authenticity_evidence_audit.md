# Phase 6D.1 — LLM Authenticity Evidence Audit

本审计仅使用 internal validation，A/B/C 全部冻结；未训练、融合、拟合权重、调阈值或校准。

## Evidence contract

- A: RINE-on-C1 原始 binary logit。
- B: 固定 `[CLS]` hidden 经过现有 classification head 后的 `z_Fake-z_Real`。
- C: canonical prompt 末尾固定 `[CLS]` 位置的 LM-head true next-token logits，取 `z_[FAKE]-z_[REAL]`。
- `[REAL]` 与 `[FAKE]` 均经运行时检查为单一 special token。输入不含 GT verdict，未 teacher-force authenticity token。
- 三者统一以 score > 0 判 Fake；保存原始 logits，未 calibration。

B 使用 Phase 6B.2 已冻结、用于此前 B+RINE 比较的原始 logits。本次同前向重放 B 未数值复现历史 cache（详见 `b_cache_reproduction_audit.json`），因此没有用重放值替换历史 B；C 仍来自本次经位置审计的冻结 next-token forward。

## Metrics

| Evidence | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|
| A_RINE | 0.994123 | 0.999845 | 0.998192 | 0.990054 | 0.009946 | 0.994147 |
| B_LLM_CLS | 0.985533 | 0.998649 | 0.978300 | 0.992767 | 0.007233 | 0.985428 |
| C_GEN_TOKEN | 0.986438 | 0.999155 | 0.985533 | 0.987342 | 0.012658 | 0.986425 |

## A_vs_B

| Scope | N | Both correct | First only | Second only | Both wrong | Disagreement |
|---|---:|---:|---:|---:|---:|---:|
| all | 2212 | 2173 | 26 | 7 | 6 | 0.014919 |
| Real | 1106 | 1091 | 4 | 7 | 4 | 0.009946 |
| Fake | 1106 | 1082 | 22 | 0 | 2 | 0.019892 |

Score correlation: Pearson `0.951126`, Spearman `0.904116`.

## A_vs_C

| Scope | N | Both correct | First only | Second only | Both wrong | Disagreement |
|---|---:|---:|---:|---:|---:|---:|
| all | 2212 | 2177 | 22 | 5 | 8 | 0.012206 |
| Real | 1106 | 1087 | 8 | 5 | 6 | 0.011754 |
| Fake | 1106 | 1090 | 14 | 0 | 2 | 0.012658 |

Score correlation: Pearson `0.955375`, Spearman `0.908060`.

## B_vs_C

| Scope | N | Both correct | First only | Second only | Both wrong | Disagreement |
|---|---:|---:|---:|---:|---:|---:|
| all | 2212 | 2173 | 7 | 9 | 23 | 0.007233 |
| Real | 1106 | 1092 | 6 | 0 | 8 | 0.005425 |
| Fake | 1106 | 1081 | 1 | 9 | 15 | 0.009042 |

Score correlation: Pearson `0.990429`, Spearman `0.982599`.

## RINE error rescue

RINE errors: `13`; B rescues `7` (`0.538462`); C rescues `5` (`0.384615`); C-B = `-2`.

## Gates

`GEN_TOKEN_DISTINCT_FROM_CLS = YES`
`GEN_TOKEN_COMPLEMENTS_RINE = YES`
`GEN_TOKEN_BETTER_COMPLEMENT_THAN_CLS = NO`
`ALLOW_COLLABORATIVE_DECODING_EXPERIMENT = NO`

阶段完成并 STOP；没有运行 collaborative decoding。
