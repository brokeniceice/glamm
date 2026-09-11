# Phase 6A — Classification / Multi-SEG Architecture Audit

## 审计边界与总论

本阶段仅审计源码、冻结 internal TRAIN 标注、tensor contract 与 checkpoint key。没有训练、微调、生成 checkpoint、运行模型推理，也没有访问 Internal test、Official1000 或任何 external/final benchmark。

核心结论：R1 的 localization improvement **没有进入当前 classification path**；但 R1 内存在可在 verdict 前取得的 image-only representation，可用于后续严格控制实验。另一方面，P1/R1 的 unified forensic 数据路径确实把多 reference 压成了 single-SEG，而底层 GLaMM 核心仍原生支持 multi-SEG。SynthScars TRAIN 的有序 reference 与 polygon 仍可恢复，因此进入 P1-multi/R1-multi 设计与预检具有可行性。

## Classification

### 1. 当前 R1 localization improvement 是否已经进入 classification？

**没有。** P1/R1 正式分类读取固定 assistant-prefix `[CLS]` 位置的 LLM 最终 hidden `[B,4096]`，经 `Linear(4096,2)` 输出 Real/Fake。R1 selected checkpoint 只含 `arm`、`utility_state`、`rectifier_state` 等 Phase4 状态，不含 classification head、LLM、LoRA 或 SEG projection 权重；R1 分类是 P1 的 EXACT REUSE。

完整证据、shape 与梯度路径见 `docs/phase6a_classification_trace.md`。

### 2. 哪些 R1 feature 是 pre-verdict、classification-safe？

| Feature | Shape | 结论 | 原因 |
|---|---|---|---|
| CLIP patch grid | `[B,1024,24,24]` | YES | image-only，verdict 前可得 |
| forensic `F24` | `[B,256,24,24]` | YES | image-only forensic arm |
| `z_F24` | `[B,1,24,24]` | YES | 仅依赖 `F24` |
| SAM `S64` | `[B,256,64,64]` | YES | image-only，但计算成本较高 |
| R1 rectified `S64` / residual | `[B,256,64,64]` | YES | image-only rectifier，可在 verdict 前计算 |
| isolated `Fctx64` | `[B,256,64,64]` | YES | image-only；必须与 language utility 隔离 |
| `q_seg`, `z_L`, joint utility `U_F`, adapted `S64`, final mask | variable | NO | 依赖 phrase/[SEG] 或初始 mask，位于 verdict/generation 之后 |

逐字段因果审计见 `outputs/phase6a_architecture_audit/r1_feature_accessibility.csv`。这里的 YES 只说明不存在标签/生成循环，不等于已证明含有分类增益。

### 3. 是否存在无需改 backbone 即可复用的 forensic representation？

**存在。** 首选是 `GAP(F24)`：复用冻结的 Phase4 forensic arm，仅增加小型 MLP，不改 CLIP、LLM、SAM 或 R1 checkpoint。第二候选是 pooled R1 rectifier residual；它也合法且现有 R1 权重可直接加载，但需要额外 SAM image-encoder 成本。

不能复用完整 CSCU/utility/final-mask 分支作为 primary classifier 输入，因为该分支需要生成或 teacher-forced phrase/[SEG]，会形成“先产生 Fake grounding，再判断 Fake”的因果回路。

### 4. LEGION Stage 2 为什么可能更 OOD-generalizable？

官方 LEGION Stage 2 使用冻结 CLIP ViT-L/14@336 的 penultimate-layer CLS `[B,1024]`，经 `1024→2048→2` MLP；训练只更新 2,103,298 参数的 prediction head，不经过 LLM、prompt grammar、explanation、SEG 或 SAM。相比 P1 的 prompt-conditioned LLM `[CLS]` hidden，这一设计把分类与生成语法隔离，并保留 pretrained global image summary，因此**可能**减少域外生成表征耦合。

这只是源码导出的机制假说，不是 benchmark 结论；feature、head capacity 和训练 recipe 尚未通过 matched control 解耦。

### 5. C0/C1/C2/C3 下一步最值得先跑哪两个？

建议未来优先跑 **C1 + C2**：

- C1：冻结 CLIP CLS → LEGION-style MLP，直接检验 feature/head 选择；新增 2,103,298 参数。
- C2：冻结 `GAP(F24)` → `256→512→2` MLP，检验已有 image-only forensic representation；新增 132,610 参数。

C0 是现有参照。C3（current CLS 与 F24 fusion，新增 1,114,882 参数）应在 C2 独立显示有效后再做，否则无法区分 representation gain 与额外容量。完整方案见 `docs/phase6a_classification_attribution.md`。

## Multi-SEG

### 6. 当前 P1/R1 是否确实退化成 single-SEG？

**是，但退化发生在 unified forensic 数据与 Phase4 wrapper/cache，不在 GLaMM 核心。** 当前 adapter 将同图多条 authoritative phrases 去重后用分号拼接为一段文本，将全部 reference masks 先 union 为 `[1,H,W]`，assistant target 只放一个 `[SEG]`。随后 q-seg、R1 cache、rectifier/SAM 路径自然都只看到一个 prompt/mask。

单点假设清单及风险见 `outputs/phase6a_architecture_audit/single_seg_assumptions.json`。

### 7. GLaMM / LEGION native multi-SEG 如何工作？

两者的原生语义都是：按 caption span 排序，为每个 phrase 构造 `<p>phrase_i</p>[SEG]`，保留有序 `[K,H,W]` GT；模型抽取所有 SEG 前一位置的 hidden，经共享 projection 得到 `[K,256]`，SAM 对同一 image embedding 解码 K 个 mask。现有 image-level evaluator 对 mask logits 取 max/二值 OR，并保持阈值 `>0`。

差异是当前 hardened GLaMM 对预测/GT 数量不一致会拒绝该样本，而官方 LEGION 会把 GT 截到预测数量；两者原 mask loss 都按 batch 内总 mask 数归一化。拟议 MultiSEG-R1 必须使用“先每图内 phrase mean，再 batch image mean”，避免多 phrase 图片权重更大。完整源码调用链见 `docs/phase6a_glamm_multiseg_trace.md`。

### 8. SynthScars 能否恢复 phrase ↔ mask supervision？

**YES，带两个显式门禁条件，且不需重新人工标注。** 冻结 TRAIN 的 8,836 张 Fake 图选择 8,971 条原始 annotation，共有 19,173 个有序 phrase refs 与 19,173 个 polygon regions。一对一关系在当前 union 前仍存在；未发现空 phrase、缺 polygon 或两 phrase 共用完全相同 decoded region。

K 分布：1 条 4,181（46.6057%），2 条 2,229（24.8467%），3 条 1,224（13.6440%），4 条 639（7.1230%），5+ 条 698（7.7806%）。存在 1 个 decoded empty mask，必须预先冻结 deterministic invalid/zero/exclusion policy；另有一条 18-reference 长 target 在官方 token limit 下只保留 12 个 SEG，必须成对截断 phrase-mask 并记录 indices，不能静默错位。

机器统计见 `outputs/phase6a_architecture_audit/synthscars_phrase_mask_audit.json`。

### 9. R1 rectifier 支持 multi-SEG 需要改什么？

R1 权重本身按 batch row 共享，不需要复制 K 套模块。推荐把各图 K 个 SEG flatten 为 `T=ΣK_b`：

```text
q_seg [T,256] + image_index [T]
image-only S64/F24 [B,...]（每图只算一次）
→ 按 image_index repeat/gather
→ shared CSCU / rectifier / SAM over T
→ offsets 分组还原为每图 [K_b,H,W]
```

需要修改 unified dataset/target builder、完整 pair 的长序列截断、variable-K cache 与 offsets、Phase4 wrapper 的 flatten/group、每图归一化 mask loss、generation parser 和 phrase-level artifact 保存。核心 SEG extraction、共享 projection、SAM 多 prompt decoder 与 rectifier参数形状无需改；现有 R1 weights 可完整初始化这些未变模块。

### 10. multi-SEG 属于什么改动范围？

**MODERATE。** 它不是 backbone 或新算法改造，GLaMM/SAM 核心已经支持 K masks，R1 权重也完全共享；但它不只是局部 shape patch，因为必须端到端保证 phrase、SEG、mask、cache 和生成输出的次序/数量一致，并有意改变 mask loss 的样本权重语义。逐组件级别见 `docs/phase6a_multiseg_change_scope.md`。

### 11. 是否推荐进入 P1-multi / R1-multi 2×2？

**GO，进入下一阶段的实现与 TRAIN-only preflight；本报告本身不授权训练。** 四臂定义为 L0=P1-single、L1=R1-single、L2=P1-multi、L3=R1-multi，后续报告：

```text
R1 single effect   = L1 - L0
MultiSEG on P1     = L2 - L0
MultiSEG on R1     = L3 - L1
interaction        = (L3 - L2) - (L1 - L0)
```

K=1 backward parity、K={1,2,variable} synthetic tests、严格 state-dict load、phrase-mask count/order assertions 与 TRAIN-only 单 batch forward/backward 必须成为未来实施门禁。

## Final recommendation

| Direction | Decision | Boundary |
|---|---|---|
| classification feature reuse | **GO** | 仅限 pre-verdict image-only `F24/z_F24` 或 rectifier residual |
| LEGION-style classifier control | **GO** | matched TRAIN/validation protocol；冻结 backbone |
| R1-feature fusion | **CONDITIONAL GO** | C2 独立证明有信号后才做 C3；完整 utility/mask feedback 为 NO-GO |
| P1/R1 multi-SEG | **GO** | 先实现与 TRAIN-only preflight；不得直接进入 final benchmark |

## Generalization firewall

Classification OOD 仍是独立待解决问题；P1→R1 localization 在不同 GT 语义下表现不一致也不应在本阶段被解释为单一泛化机制。本审计没有加入 domain loss、新数据、augmentation search 或 benchmark-specific tuning，也没有根据已有 test 结果选择超参数。

**Phase 6A 到此 STOP。**
