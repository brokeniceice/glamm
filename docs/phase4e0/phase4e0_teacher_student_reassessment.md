# Phase 4E-0 — Teacher–Student 路线重审

## 过去真正否定了什么

Phase 3F 否定的是 projected 256D G0/TF vector 的 cosine AOGD recipe；Phase 3G preflight 否定的是 raw 4096D pointwise cosine 的局部 gradient direction。两者都没有训练 teacher/student query set，没有 spatial attention、decoder feature 或 mask-logit distillation，也没有 query assignment。因此不能外推为 TF teacher 无用。

```text
256D cosine: TESTED / FAILED
4096D cosine: NOT ADEQUATELY TESTED
structured KD: UNTESTED
query KD: UNTESTED
attention KD: UNTESTED
decoder-aware KD: UNTESTED
mask-logit KD: UNTESTED
```

## TF teacher 应教什么

TF 的价值不是提供一个“正确坐标的 4096D 向量”，而是提供在 authoritative phrase/explanation 条件下产生的 **grounding behavior**：query slots 之间的关系、query 对图像位置的 attention、decoder 中间 feature 与最终 mask logit。推荐 teacher signal 为：

`BEST_TEACHER_SIGNAL: MULTI_LEVEL`

由四层组成：

1. **Query relation**：对 teacher/student query 做 Hungarian matching，再匹配归一化 pairwise cosine/Gram relation；不直接匹配 raw 4096 coordinate。
2. **Spatial attention**：匹配每个已配对 query 在 forensic/SAM lattice 上的 normalized spatial distribution，采用 Channel-Wise KD 式 temperature KL。
3. **Decoder feature**：只匹配两个预注册 decoder level，经 learned 1×1 adapter 与 per-channel normalization 后用 HCL/L1；避免无界多层 loss。
4. **Mask logit**：teacher/student 在相同 inverse geometry 前的 continuous logits 做 temperature BCE/KL，并保留 authoritative GT mask loss。

## 不采用或降级的信号

| Loss | 决策 | 原因 |
|---|---|---|
| `L_query_raw` | 不采用 | Phase 3G 局部方向无效，teacher/student coordinate system 无同一性保证 |
| `L_relation` | 采用 | DETRDistill/relational KD 先匹配 query identity 再传结构 |
| `L_attention` | 采用 | 直接传 spatial behavior，补足 Phase 3F 缺失信号 |
| `L_feature` | 采用但仅两层 | Knowledge Review/Structured KD 支持；控制 loss 堆叠 |
| `L_logit` | 采用 | teacher 与 student 输出空间天然对齐，最直接 behavior target |
| `L_language` | 保留 canonical CE | 防止 Stage-II 破坏 P1 explanation/detection；不是新 teacher loss |

## Teacher 与 student 信息边界

```text
TF teacher：authoritative phrase/trajectory → teacher queries/attention/features/logits
G0 student：canonical user prompt + autonomous generation → student queries/attention/features/logits
GT union mask：只用于 training loss，不输入 student inference
```

teacher branch 训练期 stop-gradient；student 不得接收 teacher phrase、teacher token 或 GT mask。部署完全移除 teacher。

```text
TF teacher → teacher grounding queries → teacher spatial interaction → teacher logits
                         │ assignment / relation / attention / feature / response
G0 student → student grounding queries → student spatial interaction → student logits
```

## 4096D 路线判断

`4096D_G0_TF_ROUTE: WORTH_REOPENING`，但含义严格限定为：4096D causal state 用作 query generator 的输入，并通过下游 structured behavior KD 学习；不重启 raw-hidden cosine tuning。

## 证伪

在相同 full training 下，若 `-teacher KD` 与 full 的 deployable G0 差异 CI 跨 0，且 query/attention/logit alignment 改善不与 IoU 改善相关，则 TF teacher 不是 Stage-II 有效贡献，应从最终方法移除；不能通过换权重无限续试。
