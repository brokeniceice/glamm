# Phase 4E-0 — 最终架构冻结决定

## 决策摘要

推荐 Stage-II 为 **TF-FDG：Teacher-Guided Forensic Dual-Path Grounding**。它不是在 P1 上再加一个 Reader，而是把单一 SAM sparse-prompt 接口替换为 coordinated query-set dense decoder：P1 G0 trajectory 提供 deployable language condition，4C-A adapter 提供保持二维结构的 forensic evidence，SAM 提供 semantic dense feature；训练期 TF branch 只提供 structured grounding behavior teacher。

## 训练信息流

符号：`F`=frozen，`T`=trainable，`SG`=stop-gradient。`K=4`。

```text
image [B,3,H,W]
 ├─ P1 CLIP/LLaVA (F) ── canonical G0 generation
 │    └─ [SEG]前一 causal state h_G0 [B,4096]
 │          └─ QueryGenerator_G0 (T, gradient←所有 student loss)
 │                └─ Q_G0 [B,K,256]
 │
 ├─ same P1, authoritative TF trajectory (F, training only)
 │    └─ h_TF [B,4096]
 │          └─ frozen teacher QueryGenerator (SG)
 │                └─ Q_TF [B,K,256]
 │
 ├─ SAM image encoder (F) ── S_sem [B,256,64,64]
 └─ CLIP grid (F) [B,1024,24,24]
      └─ selected 4C-A adapter (F in main) ── F_for [B,256,24,24]

S_sem ── semantic pyramid adapter (T) ─┐
F_for ── forensic pyramid adapter (T) ├─ DualPathDecoder (T, 4 layers)
Q_G0 ──────────────────────────────────┘    ├─ spatial attention A_G0
                                             ├─ decoder feature D_G0
                                             └─ student mask logits Z_G0 [B,1,h,w]

同一冻结 teacher snapshot：Q_TF + 相同 dense inputs → A_TF,D_TF,Z_TF (SG)

GT union mask [B,1,H,W] → L_mask；
matched(Q_G0,Q_TF) → L_relation；A → L_attention；D → L_feature；Z → L_logit。
```

Teacher 先以 TF condition 和 authoritative union mask 完成固定 warm-up，随后冻结 snapshot；student 从 teacher decoder 初始化但使用独立 G0 QueryGenerator。这样 teacher behavior 有 mask supervision 锚点，不是同一步自蒸馏的移动目标。

## 推理信息流

```text
image
 ├─ frozen P1 canonical G0 → h_G0 [B,4096] → QueryGenerator → Q [B,4,256]
 ├─ frozen SAM encoder → S_sem [B,256,64,64]
 └─ frozen CLIP + 4C-A adapter → F_for [B,256,24,24]
          Q + S_sem + F_for → DualPathDecoder → mask logits → canonical postprocess
```

推理不需要 authoritative phrase、TF explanation、GT mask、teacher branch 或 oracle evidence。Detection/G0/TF user prompt 均继续显式固定为 canonical；正式 threshold-boundary 指标使用 direct batch=1。

## Decoder 冻结规格

- QueryGenerator：`LayerNorm(4096) → Linear(4096,1024) → GELU → Linear(1024,K×256)`，第一 slot 加入旧 P1 256D projection 作为 semantic anchor，其他 slot 为 learned residual。
- 两个 spatial path 各建两层 pyramid：`24×24` 与 `48×48` forensic；SAM 为 `32×32` 与 `64×64` semantic。每层都显式记录 inverse geometry。
- DualPathDecoder：4 层、256D、8 heads；每层依次 query self-attention、semantic cross-attention、forensic deformable/cross-attention、FFN；不增加 MoE 或第三 encoder。
- mask head：共享 high-resolution fused mask feature与 K 个 mask embedding 内积，K 个 slot logits 以 learned normalized nonnegative weights 聚合为 per-image all-ref union logit。
- teacher/student 使用同构结构；teacher snapshot 冻结，student 可训练。

## Loss 决策

| Loss | 决策 | teacher/student signal、维度与 normalization | 文献先例 / 既往边界 |
|---|---|---|---|
| `L_mask` | 使用 | student logit 与 GT union mask；BCE + Dice，按 image 平均 | LISA/GLaMM/PixelLM；唯一 authoritative pixel target |
| `L_query` raw | 不使用 | 不匹配 raw 4096D/256D coordinate | Phase 3F/3G negative boundary |
| `L_relation` | 使用 | Hungarian assignment 后，`[B,K,K]` normalized Gram 的 Smooth-L1 | DETRDistill、relational KD |
| `L_attention` | 使用 | 每层/slot spatial attention 经 temperature softmax，KL | Channel-Wise KD；传 spatial distribution |
| `L_logit` | 使用 | aligned continuous mask logits，temperature BCE/KL | Structured KD；输出 behavior 对齐 |
| `L_feature` | 使用两层 | 1×1 adapter 后 channel-normalized HCL/L1 | Knowledge Review；禁止无限多层 |
| `L_language` | 不进入 Stage-II optimizer | P1 language path 冻结；只做 CE/Detection non-regression audit | 避免改变 Stage-I 与 prompt contract |

## 每个组件的证据链

### Coordinated query codebook

```text
COMPONENT: K=4 grounding query codebook
SOURCE PAPER: PixelLM; PSALM
SOURCE MECHANISM: multiple segmentation/object queries jointly condition dense mask decoding
OUR OBSERVATION: current per-mask path retains only one 256D sparse prompt
WHY IT ADDRESSES OUR OBSERVATION: preserves several semantic/spatial roles and supports query-relation KD
WHAT WOULD FALSIFY IT: matched K=1 ablation is non-inferior
```

### Dual-path forensic decoder

```text
COMPONENT: semantic/forensic dual-path dense fusion
SOURCE PAPER: PSALM; VisionLLM v2; Propose and Rectify
SOURCE MECHANISM: task decoder consumes spatial specialist features; forensic cues rectify segmentation features
OUR OBSERVATION: 4C-A representation is learnable but one-query Reader ignores image/layout
WHY IT ADDRESSES OUR OBSERVATION: forensic lattice survives until mask logits
WHAT WOULD FALSIFY IT: matched is not better than cross-image/shuffle, or -forensic is non-inferior
```

### Multi-level TF teacher

```text
COMPONENT: assigned query + attention + feature + logit KD
SOURCE PAPER: DETRDistill; Structured KD; Channel-Wise KD; Knowledge Review
SOURCE MECHANISM: transfer structured dense behavior rather than a raw vector
OUR OBSERVATION: stable TF advantage but 256D/raw pointwise cosine fails
WHY IT ADDRESSES OUR OBSERVATION: uses the teacher signal at decoder-relevant levels
WHAT WOULD FALSIFY IT: -KD is non-inferior under the frozen full recipe
```

## 最终机器可读结论

```text
PRIMARY_CONFIRMED_PROBLEM:
P1 的 autonomous language trajectory 与可学习 dense forensic evidence 尚无能保留 query relation 和 spatial structure 到 mask prediction 的有效接口。

SINGLE_SEG_BOTTLENECK:
LIKELY

4096D_G0_TF_ROUTE:
WORTH_REOPENING

BEST_TEACHER_SIGNAL:
MULTI_LEVEL

BEST_FORENSIC_INTERFACE:
HYBRID

RECOMMENDED_STAGE_II_ARCHITECTURE:
TF-FDG — Teacher-Guided Forensic Dual-Path Grounding

KEY_LITERATURE_PRECEDENTS:
LISA/GLaMM; PixelLM; PSALM; OMG-LLaVA; VisionLLM v2; Propose and Rectify; DETRDistill; Structured KD; Channel-Wise KD; Knowledge Review

WHAT_IS_OURS:
依据项目内 G0–TF gap、4C-A dense representation positive result 与 Reader causal failure，首次把 deployable autonomous forensic trajectory、spatial forensic lattice 和 training-only structured TF teacher 组合为可证伪的 dual-path query decoder。

WHY_PREVIOUS_MINIMAL_METHODS_FAILED_TO_TEST_THIS:
它们始终保留或重新制造 single-query compression，只做 projected/raw pointwise alignment，或没有让 forensic lattice 到达 dense decoder；因此没有同时检验 multi-query capacity、spatial evidence use 与 teacher decoder behavior。

FULL_TRAINING_JUSTIFIED:
YES
```

`YES` 只授权形成 Phase 4E-1 proposal，不等于授权执行。Phase 4E-0 到此停止。

