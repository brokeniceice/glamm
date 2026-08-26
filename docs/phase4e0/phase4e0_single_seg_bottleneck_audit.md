# Phase 4E-0 — Single-`[SEG]` Bottleneck 审计

## 审计问题

“single `[SEG]`”在本项目中准确指：每个目标 mask 最终只由一个 `[SEG]` 前一 causal state，经 `4096→256` MLP 后成为一个 SAM sparse prompt。它不指数据中永远只有一个 `[SEG]` occurrence；GLaMM 可为多个 target 产生多个独立 mask，但单个 union target 没有 coordinated multi-token state。

## 信息压缩链

```text
整段 G0 trajectory [L×4096]
  → 只保留 [SEG] 前一 state [1×4096]
  → 非线性投影 [1×256]
  → SAM sparse prompt [1×256]
  → 与 frozen semantic image embedding 交互
```

该路径发生三次结构损失：trajectory 内 token relation 消失；4096D state 被压成 256D；forensic spatial lattice 不存在于 prompt path。Phase 3F/3G 仅改变单点 representation alignment，Phase 4C/4D Reader 仅把 dense map 再汇总回同一个 q，因此都没有真正移除这一结构约束。

## 文献对照

- LISA/GLaMM 证明 single special-token prompt 可工作，但不证明它对 forensic union mask 足够。
- PixelLM 的 segmentation codebook 用多个 token 表达不同 visual scale/target context，并在 pixel decoder 内联合消费。
- PSALM 的 object/mask query set 与 multi-scale pixel decoder 保留 slot 与 spatial state。
- OMG-LLaVA/VisionLLM v2 允许 task decoder 接收更丰富的 perception/task state，而不是单点 prompt。

## Mandatory 九问

1. **4096D hidden 承担什么信息？** 它是生成 `[SEG]` 的 causal predictor state，混合当前图像 token、user instruction、此前 explanation/answer token 与 grounding intent。
2. **是否含 target semantics？** 是；Phase 3A/3C 的 phrase intervention 证明 target wording 会改变该路径及 mask，但不能分离具体 subspace。
3. **是否含 explanation context？** 是；该 state 位于 explanation 之后，TF-full 相对 phrase-only 的差异说明 trajectory context 继续影响 grounding。
4. **是否含 grounding intent？** 是；它直接预测 `[SEG]`，是当前实现唯一送往 segmentation path 的 language state。
5. **spatial forensic information 是否被迫压入其中？** language path 若要表达 forensic location，只能将图像相关线索压入该 state；独立 4C-A lattice 并未到达 canonical decoder。
6. **`text_hidden_fcs` 后维度？** `4096→4096→256`，最终 256D。
7. **SAM 实际 prompt dimension？** 每个 mask 一个 sparse text prompt，shape `[1,256]`（batch/target 维之外）。
8. **一枚 prompt 是否足以表达 artifact union？** 现有证据不能证明“不足”，但多 region、多尺度、低层 forensic artifact 共存时存在明显容量与角色混叠风险，因此判为 LIKELY bottleneck。
9. **为何成熟方法使用更丰富 schema？** PixelLM 用 codebook 表达多 scale/target context；PSALM 用 object/mask query 与 pixel decoder；OMG-LLaVA 保留 object/pixel/perception-prior token。这些设计让 dense prediction state 不必全部通过一个普通 language vector。

## 三种竞争解释

| 假设 | 现有证据 | 当前判断 | 将来 matched test |
|---|---|---|---|
| H1：single-query 容量不足 | TF gap、direct path failure、Reader failure、成熟 multi-query 先例 | LIKELY | full `K=4` vs `K=1` |
| H2：主要是 G0 semantic content 错 | Phrase Repair 与 Phrase-only gain | SUPPORTED，但不排斥 H1 | 固定 decoder 比较 G0/Phrase/TF |
| H3：SAM image feature 缺 forensic evidence | 4C-A positive + Reader null | LIKELY | full forensic path vs `-forensic` 与 shuffled control |

## 结论边界

`SINGLE_SEG_BOTTLENECK: LIKELY`，不是 `SUPPORTED`。它是值得一次正式 matched architecture ablation 的高优先级解释，但不能称为唯一或首要已证实原因。若 full method 中 `K=1` 与 `K=4` 在相同参数预算和训练协议下等效，则该解释被证伪，Stage-II gain 应归因于 dense forensic fusion 或 teacher KD。

```text
SINGLE_SEG_IS_LIKELY_BOTTLENECK: YES
```
