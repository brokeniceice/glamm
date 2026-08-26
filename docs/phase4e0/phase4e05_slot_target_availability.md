# Phase 4E-0.5 — Slot Target Availability

## Annotation schema 结论

SynthScars grouped manifest 在 union rasterization 前保留每个 authoritative reference：

```text
ref_id
annotation_id
phrase
explanation
polygons
```

train Fake 8,836 张，共 19,173 refs；所有 ref 都有唯一 `ref_id`、`annotation_id`、非空 phrase 和 polygon。当前 release 中每个 ref 恰有一个 polygon，但协议仍把 `polygons` 视为一个 ref 的 component list。无需新 annotation、LLM/SAM relabel 或 pseudo label 即可逐 ref rasterize mask。

```text
PER_REGION_POLYGON_AVAILABLE: YES
REGION_ID_AVAILABLE: YES
REGION_PHRASE_MAPPING_AVAILABLE: YES
CAN_RECONSTRUCT_SLOT_LEVEL_MASKS_WITHOUT_NEW_ANNOTATION: YES
```

## Region 数量与 K=4 capacity

| Split | Fake | Mean refs/image | Max | M≤4 | M>4 |
|---|---:|---:|---:|---:|---:|
| train | 8,836 | 2.1699 | 18 | 8,118 | 718（8.126%） |
| validation | 1,106 | 2.1438 | 15 | 1,025 | 81（7.324%） |

因此不能声称 K=4 永远具有“一 slot 一 region”的身份。

## 冻结 mixed slot supervision

### M≤K

对每个 authoritative ref 单独 rasterize `G_m`，每个 slot 输出 continuous logit `Z_k`。assignment cost 固定为：

```text
C(k,m) = 1.0 * DiceCost(sigmoid(Z_k), G_m)
       + 1.0 * BCEWithLogitsCost(Z_k, G_m)
```

cost tensor stop-gradient，Hungarian 后 matched slot 使用 BCE+Dice；未匹配 slot target 为 authoritative no-region/empty。另保留 full union loss。

### M>K overflow

不按 annotation order、面积或几何邻近强行合并 ref，因为这会创造不存在的 slot identity。本图退化为 **union-only coordinated query formulation**：所有 ref 仍进入 authoritative union，slot 不做伪一对一 supervision，并强制记录 collapse diagnostics。该 overflow 规则在训练前固定，不由结果改变 K。

## Union operator

禁止 learned weighted average。对 slot logit `z_k`：

```text
p_k = sigmoid(z_k)
log_not_union = Σ_k log1p(-clamp(p_k, eps, 1-eps))
p_union = -expm1(log_not_union) = 1 - Π_k(1-p_k)
z_union = logit(clamp(p_union, eps, 1-eps))
L_union = BCEWithLogits(z_union, G_union) + Dice(p_union, G_union)
```

`eps=1e-6`；全程 continuous，不 threshold。

## Slot-collapse diagnostics

每个 formal checkpoint 记录：

- query diversity：off-diagonal `cos(Q_k,Q_j)`；
- attention diversity：每 layer/path 的 pairwise JS divergence 与 cosine；
- mask overlap：pairwise soft IoU；
- slot mass：`mean(sigmoid(Z_k))`；
- slot confidence：`mean(2*|sigmoid(Z_k)-0.5|)`；
- active slot：`mass>=0.01 AND confidence>=0.10`；
- union contribution：`mean(p_union - p_union_without_k)`。

单图 `collapsed` 定义为：`active_slot_count≤1`，或同时满足 mean off-diagonal mask soft-IoU `≥0.90` 且至少三个 slot 的 union contribution `<0.001`。人群判定：collapsed fraction `≥0.80 → TRUE`；`[0.40,0.80) → PARTIAL`；`<0.40 → FALSE`。

```text
SLOT_COLLAPSE: training-time result, not evaluated in Phase 4E-0.5
```

若未来为 TRUE/PARTIAL，K=4 run 不能用于否定 multi-query hypothesis。机器可读 schema 与 histogram 见 `outputs/phase4e05_tf_fdg_hardening/slot_target_availability.json`。

