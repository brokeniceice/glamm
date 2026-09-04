# Frozen expert relationship 与 oracle headroom

## 固定 population

只使用 `TRAIN-AUDIT` 的 1303 个 valid canonical-G0 Fake；其 IDs SHA256 为 `aae5f13f1b236f6002d47ccdd54fbb6e0c9fd6d50d9831d7636484197e4b10cd`。同 fold 的 22 个 invalid-G0保持 formal failure，排除出 expert-correctness 条件概率。所有 pixel统计位于 original-normalized 256 grid的 joint CLIP support，固定 logit threshold 0，未调阈值。

## Frozen source关系

| Scope | P(F correct｜L wrong) | P(L correct｜F wrong) | joint error | disagreement | error overlap |
|---|---:|---:|---:|---:|---:|
| all supported pixels | 0.474040 | 0.337988 | 0.029304 | 0.041373 | 0.414622 |
| foreground supported pixels | 0.201342 | 0.163317 | 0.572470 | 0.256063 | 0.690944 |
| image，FG IoU≥0.5 | 0.067485 | 0.063380 | 0.816577 | 0.114351 | 0.877164 |

boundary disagreement为 `0.224297`；CLIP support平均覆盖 `0.945061`。

预注册分类取两个 foreground rescue conditional的较小值：≥.25 STRONG、≥.10 MODERATE、≥.01 WEAK，否则 ABSENT。因此：

`EXPERT_COMPLEMENTARITY = MODERATE`，diagnostic only。

## Frozen oracle

| Reference | mean FG IoU |
|---|---:|
| P1-only | 0.178355 |
| Forensic-only | 0.203794 |
| static equal-probability fusion | 0.194874 |
| image oracle | 0.255476 |
| pixel oracle | 0.375432 |
| conflict-region oracle | 0.375432 |

pixel oracle相对最佳 frozen single source的 headroom为 `+0.171639`。按 ≥.10 HIGH、≥.03 MODERATE、否则 LOW：

`FROZEN_EXPERT_ORACLE_HEADROOM = HIGH`，diagnostic only。

二分类时，只要两个 source disagreement，其中必有一个 pixel label正确，因此本定义下 pixel oracle与 conflict-region oracle相同。这一 oracle只描述 frozen source prediction互补，不是可使用的模型、不是 PCERF upper bound，也不授权训练。

机器可读结果见 `outputs/phase4g05/expert_relationship.json` 与 `outputs/phase4g05/frozen_oracle_headroom.json`。
