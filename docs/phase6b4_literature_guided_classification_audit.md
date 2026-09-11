# Phase 6B.4 — Literature-Guided Classification Architecture Audit

## 结论先行

本阶段完成文献与官方代码审计，**没有训练、没有 external benchmark inference、没有修改模型或既有 checkpoint**。

下一步最值得优先做的 controlled experiment 是：

1. **RINE-style multi-layer CLIP（首选）**：保留当前 C1-Exact 使用的冻结 CLIP ViT-L/14@336，但把单一 `hidden_states[-2][:,0]` 替换为官方 RINE 的“所有 Transformer block CLS → 投影 → TIE → 加权求和 → 投影 → classifier”分类侧路。它直接测试当前 LEGION/C1-Exact 缺失的中浅层信息，并且不触碰 LLM、`[SEG]`、SAM 或 localization。
2. **AIDE PFE low-level forensic complement（第二优先）**：若 multi-layer CLIP 仍受同一 CLIP 表征上限约束，再严格复用 AIDE 的 DCT 选取高/低频 patch、固定 SRM 滤波和低层 expert，并通过 AIDE 的 feature-concat + MLP 范式与冻结 C1-Exact semantic branch 结合。这是较大但来源明确的分类侧改动，同样可以保持 localization 完全不动。

不建议把普通 temperature scaling 当成分类改进实验；它在正温度和固定 `0.5` 阈值下不能改变任何二分类决定。也不建议把更多 CLIP/LLM score fusion 作为第一优先级：它只能重新组合已经存在的两路信息，而 RINE/AIDE 分别增加了当前 C1-Exact 没有使用的层级信息或低层取证信息。

> 审计口径：结构建议来自论文与公开官方代码；没有根据当前 OOD 数值选择结构。SNIFF 截至本次审计可核验到出版社正文摘要，但没有找到作者公开代码，且可访问材料未披露精确 block index，因此本文不会臆造其层号。

### 审计来源身份（2026-09-09 UTC）

| 项目 | 论文/出版社 | 官方代码 | 审计时 remote HEAD |
|---|---|---|---|
| UFD | CVPR 2023 CVF Open Access | `WisconsinAIVision/UniversalFakeDetect` | `030495aea3300a8b54c0ec37ec7fe1dd7e63c619` |
| RINE | ECCV 2024 ECVA | `mever-team/rine` | `9b7fd5857cc205d0412be6aeee0d7611b95bd620` |
| AIDE | ICLR 2025 proceedings | `shilinyan99/AIDE` | `6725b710d5c437ab2f59792908ce0377dfc907de` |
| SNIFF | ICT Express, online 2026-06-02 | 未找到作者公开 repo | N/A |

以上 SHA 由只读 `git ls-remote <official-repo> refs/heads/main` 取得；本阶段没有 clone、checkout 或修改这些外部项目。

## 1. 当前基线：C1-Exact / LEGION-retrained 到底用了什么

当前可复现的 C1-Exact 与 LEGION-retrained Stage 2 分类路径是：

```text
CLIP ViT-L/14@336
→ output_hidden_states=True
→ hidden_states[-2][:, 0]            # penultimate block CLS, 1024-D
→ Linear(1024, 2048)
→ ReLU
→ Linear(2048, 2)
```

本地代码证据：

- LEGION 在 [`external/LEGION_official/model/Legion.py`](../external/LEGION_official/model/Legion.py#L364-L378) 中明确取 `hidden_states[-2]` 的第 0 个 token，再送入 `prediction_head`。
- 官方 Stage-2 trainer 在 [`external/LEGION_official/scripts/cls/train.py`](../external/LEGION_official/scripts/cls/train.py#L211-L219) 中先冻结全模型，再只解冻 `prediction_head`。
- C1-Exact 已确认只有 4 个 prediction-head 参数张量、共 `2,103,298` 个可训练参数；分类 forward 不经过 explanation、`[SEG]` 或 SAM，见 [`docs/phase6b1a_exact_stage2_control.md`](phase6b1a_exact_stage2_control.md#L11-L24)。

因此，后续只新增/替换分类 sidecar、同时冻结定位路径，在工程上是清晰可行的。需要特别避免一个术语误区：**“CLIP CLS”不是唯一固定张量**；不同工作取的是不同 block、是否经过 CLIP 最终 LayerNorm/projector 也不同。

## 2. UniversalFakeDetect / UFD（CVPR 2023）

### 2.1 为什么 frozen CLIP feature 泛化较好

[UFD 论文](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html)的核心诊断是：针对某一生成器端到端训练的 detector 容易只学到该 fake domain 的特定低层伪迹，Real 则成为容纳“不像训练 fake”的样本的 sink class。UFD 因而不再为真假任务重训 encoder，而是在未被 real/fake 目标塑形的、internet-scale vision-language feature space 上做 nearest-neighbor 或 linear probe。论文还显示，单纯 ImageNet 规模的预训练表征不足以复现这一泛化优势，关键不只是“冻结”，也是大规模、跨语义域的 CLIP 预训练。

这支持当前“冻结 CLIP、只训练小分类头”的大方向，但不证明任意 CLIP 层或任意 head 都同样有效。

### 2.2 层、token 与 classifier 的精确形式

[UFD 官方代码](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/models/clip_models.py)调用 OpenAI CLIP 的 `model.encode_image(x)`，再接一个 `nn.Linear`。对 ViT-L/14，`encode_image` 返回的是经过视觉 Transformer 最终归一化和 CLIP projection 的**最终全局图像 embedding（768-D）**；它不是 Hugging Face `hidden_states[-2][:,0]` 的原始 1024-D penultimate CLS。冻结模式下，[trainer](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/main/networks/trainer.py)只保留线性层参数可训练，并用 BCE-with-logits 目标训练。

因此：

| 项目 | UFD | 当前 C1-Exact / LEGION-retrained |
|---|---|---|
| CLIP 接口 | OpenAI CLIP `encode_image` | HF/LEGION vision tower hidden states |
| 特征位置 | 最终 pooled/projected image embedding | `hidden_states[-2]` 第 0 token |
| 维度（ViT-L/14） | 768 | 1024 |
| head | 单线性 probe | 1024→2048→ReLU→2 |
| CLIP | frozen | frozen |

UFD 是当前方案的重要理论基础，但不是 C1-Exact 的逐算子同构实现。

## 3. RINE（ECCV 2024）

### 3.1 多个 Transformer block 的 CLS 如何融合

[RINE 论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09160.pdf)与[官方仓库](https://github.com/mever-team/rine)的路径为：

```text
frozen CLIP ViT 的每个 Transformer block CLS
→ stack K ∈ R^(B×N×D)
→ Q1: 每层共享方式映射到 D'（q-layer projection）
→ TIE: 对每个 layer × projected-channel 学一个 importance logit
→ 在 layer 维 softmax
→ importance-weighted sum across N blocks
→ Q2 projection
→ two-layer classification head
```

官方 [`src/models.py`](https://github.com/mever-team/rine/blob/main/src/models.py)通过 hook 每个视觉 Transformer block 的 `ln_2` 输出，抽取 class token 后堆叠；`alpha` 是按 layer 和 projected feature channel 学习的参数，softmax 在 layer 维进行。这里的 TIE **不是每层一个标量**，而是允许不同 projected channel 对不同 block 有不同权重。

### 3.2 TIE / importance weighting

TIE 的作用是避免预先固定某一层，也避免朴素平均假定每层同等重要。RINE 的消融显示：去掉 intermediate representations 的损失最大；去掉 TIE 也下降，但幅度较小。这支持“多层信息”是一级假设，“学习层重要性”是其上的二级假设，而不是把全部收益都归于 TIE。

### 3.3 contrastive loss 是否必要

RINE 的训练目标是：

```text
L = BCE + ξ · supervised contrastive loss
```

对 Q2 后的表示施加 supervised contrastive loss，目的是让同类 embedding 更密集。官方 [`src/utils.py`](https://github.com/mever-team/rine/blob/main/src/utils.py)实现了 BCE 与对比项。论文消融中，去掉 contrastive loss 后仍能正常训练且仍明显优于只用最终层的退化版本，但性能低于完整 RINE。

所以结论应分两层：

- **它不是多层 CLS fusion 在数学上成立的必要条件**，也不是证明“中间层有用”所必需；
- **若实验命名为 exact/full RINE-style reproduction，则应保留论文的 contrastive 项及官方 ξ，不应随意删除。**

本报告推荐的首个实验是完整的 RINE 聚合/损失路径，而不是自创的无对比损失变体。若之后要做机制消融，BCE-only 必须单列为 ablation，不能冒充 RINE full。

### 3.4 能否低侵入接入当前 R1

可以，且边界清楚：复用同一次冻结 CLIP forward 的多层 CLS，只替换/新增 classification branch；R1 的 LLM generation、`[SEG]` hidden projection、SAM image encoder、mask decoder和 prompt 均不需要改变。

但要明确这将是 **RINE architecture on current frozen CLIP**，不是零差异复现 RINE：官方 RINE 基于 OpenAI CLIP ViT-L/14 接口，而当前 C1-Exact 是 ViT-L/14@336 的 HF hidden-state 接口。为了与 C1-Exact 做 controlled comparison，应冻结当前 CLIP、输入尺寸、预处理、训练/验证 manifests 与 selector，只改变“单层 CLS → 多层 RINE aggregator”。

RINE 论文报告的可训练参数随 `q`、`D'` 和训练配置变化（约 `0.28M–10.52M`，常用配置约 `6.32M`），不能脱离其配置只给一个固定参数数。实施前应从官方给出的配置中预注册唯一设置，不能看 OOD 后选配置。

## 4. AIDE（ICLR 2025）

### 4.1 high-level semantic branch

[AIDE 正式 ICLR 页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b0303773962ea1b5394c3a83cc7dd066-Abstract-Conference.html)将高层分支称为 Semantic Feature Extraction（SFE）：使用冻结的 OpenCLIP ConvNeXt-XXL 对完整图像编码，得到高层语义/上下文 feature，经线性降维与全局聚合形成 256-D 表示。它与当前 C1-Exact 都利用大规模 vision-language pretraining，但 backbone、token/feature map 和预处理并不相同。

### 4.2 DCT / SRM high-low-frequency patch branch

[论文正文](https://arxiv.org/html/2406.19435)和[官方代码](https://github.com/shilinyan99/AIDE)中的 Pixel-level Feature Extraction（PFE）为：

1. 把图像划分为 `32×32` patches；
2. 用 DCT band-pass 统计给 patch 排序；
3. 取 2 个最高频分数和 2 个最低频分数 patch；
4. 对 patch 使用固定的 30 个 SRM high-pass filters 暴露噪声/残差模式；
5. 高、低两组分别通过 ResNet-50-style expert，组内/patch 间聚合为低层表示。

该路径同时保留“异常高频伪迹”和“异常平滑/低频区域”，并非只做一张全图 DCT，也不是把 SRM 响应直接当最终 logit。

### 4.3 multi-expert 如何融合

AIDE 的最终融合并不是动态 gating 或 attention：四个低层 patch embedding 聚合成 2048-D PFE feature，SFE 形成 256-D semantic feature，二者在 channel 维 concat 成 2304-D，进入 `Linear(2304,1024) → GELU → Linear(1024,2)`。论文使用 multi-expert 描述不同 feature extractor，但最终决策是确定性的 feature concatenation + MLP。

### 4.4 哪部分最可能补足当前 CLIP-only 上限

最可能提供正交增量的是 **PFE 的 DCT-selected high/low patch + fixed SRM + trainable low-level experts**，而不是再增加一套高层 SFE：当前 C1-Exact 已有冻结 CLIP semantic branch。AIDE 消融也显示高/低频 PFE 与 SFE 的组合优于任何单分支；这只能作为“互补性合理”的文献证据，不能保证迁移到本项目后必然超过 LEGION。

代价同样必须正视：官方 PFE 包含 ResNet-50-style experts，训练参数与显存/算力远大于 score fusion 或 SNIFF，且 DCT patch selection 和 SRM preprocessing 增加了数据路径复杂度。因此它适合作为第二个正交实验，不适合在 RINE 之前展开大量变体。

## 5. SNIFF（ICT Express 2026）

[SNIFF 出版社页面](https://www.sciencedirect.com/science/article/pii/S2405959526000901)公开的可核验结构是：

```text
frozen CLIP ViT
→ selected intermediate layers 的 class-token features
→ element-wise average
→ streamlined two-layer linear classifier
```

论文同时配套 SAFE preprocessing/data augmentation；完整系统报告 `0.66M` 个可训练参数。与 RINE 相比：

| 维度 | SNIFF | RINE |
|---|---|---|
| 多层融合 | element-wise average | Q1 + trainable TIE + weighted sum + Q2 |
| 层权重 | 无可训练权重 | layer×channel importance |
| head | 简化 two-layer linear classifier | projections + deeper classification path |
| 辅助损失 | 可访问摘要未声称需要 SupCon | BCE + supervised contrastive |
| 报告参数 | 0.66M | 依配置约 0.28M–10.52M |
| 实现复杂度 | 低 | 中等 |

审计限制：出版社摘要确认“selected multiple intermediate layers”，但本次可访问的正式材料没有给出精确 block indices；也没有定位到作者公开官方代码。因而现在不能把任意层集合称为“exact SNIFF”。SAFE 还包含数据增强，而本阶段禁止用泛化增强先行；这也使“完整 SNIFF”不适合作为第一个纯结构 controlled experiment。

SNIFF 的价值主要是证明“无需 TIE 的多层平均”有正式文献依据，并提示参数效率；在拿到可核验层号前，它适合作为 RINE 后续的轻量消融方向，而不是现在臆造配置并直接实现。

## 6. CLIP logits + LLM logits：fusion、stacking 与 calibration

先把两路二分类 logits 化为 logit margin：

```text
s_CLIP = z_CLIP,Fake − z_CLIP,Real
s_LLM  = z_LLM,Fake  − z_LLM,Real
```

### 6.1 哪些能改变决策边界

| 方法 | 形式 | 能否改变固定 0.5 决策 | 能否改变样本排序 / ROC-AUC | 本质 |
|---|---|---:|---:|---|
| Temperature scaling | `sigmoid(s/T), T>0` | **不能** | **不能** | 只调置信度尺度 |
| Platt scaling | `sigmoid(a·s+b), a>0` | 能，因 `b` 移动阈值 | 不能 | 单模型 calibration / threshold shift |
| Fixed score sum | `0.5s_C+0.5s_L` | 能（相对任一 base） | 能 | 无参数线性 fusion |
| Convex alpha fusion | `αs_C+(1−α)s_L` | 能 | 能 | 一参数、无截距的受限线性 fusion |
| Logistic stacking | `w_Cs_C+w_Ls_L+b` | 能 | 能 | 三参数线性 meta-classifier |
| Nonlinear stacking | `MLP(s_C,s_L)` | 能 | 能 | 更强但更易对两维输入过拟合 |

[Guo et al. 的 temperature scaling](https://proceedings.mlr.press/v70/guo17a)是 calibration 方法，不是特征融合。因为 `T>0`：

```text
sigmoid(s/T) ≥ 0.5  ⇔  s ≥ 0
```

所以它在本项目固定 `prob_fake ≥ 0.5` 时严格保持 prediction、Accuracy、Fake recall、TNR/FPR 与 F1；正比例缩放也保持排序，因此 ROC-AUC 不变。它可以改善 NLL/ECE/Brier 等 calibration 指标，但不能被宣称为提升 0.5 阈值分类。

Platt scaling 的截距可以改变固定阈值下的结果，但若 `a>0` 仍不改变排序；这相当于校准/移动 operating point，不是增加新的真假证据。

[Stacked generalization](https://doi.org/10.1016/S0893-6080(05)80023-1)允许学习两路 score 的独立权重与截距，因此比当前 F2-alpha 的“权重和为 1、无 bias”更一般，确实能旋转二维 score 空间中的分类边界。规范 stacking 应使用 out-of-fold base predictions 或独立的 internal calibration split 训练 meta-learner，避免用 base learner 的 in-sample score 造成乐观偏差；internal validation 仍应只承担冻结选择/报告角色。

它仍有根本限制：stacking 不产生新图像表示，只能利用 F0/F1 已经编码且在分数层可见的互补性。两路高度相关时，三参数 stacking 可能只重设阈值/尺度。非线性两分数 MLP 虽更灵活，但没有足够机制理由优先于新增多层或低层证据，当前不推荐。

### 6.2 与现有 F2-alpha 的关系

本地 [`scripts/phase6b2_fusion.py`](../scripts/phase6b2_fusion.py#L41-L50)冻结了：

```text
F0 = R1/P1 LLM [CLS] head
F1 = C1-Exact CLIP CLS head
F2-alpha = α·z_CLIP + (1−α)·z_LLM
```

其 alpha 只在 internal TRAIN 上拟合，不含 bias；所以它是受限 score fusion，不是完整 logistic stacking。若未来授权路线 C，唯一有清晰文献增量的最小控制是两路 margin 的线性 logistic stacking（`w_C,w_L,b`）并采用 OOF/独立 calibration protocol；temperature scaling、再做一个 alpha sweep、或 dataset-specific alpha 都不构成新的结构证据。

## 7. 方法总表

| 方法 | 使用特征 | 融合方式 | 新参数 | 是否冻结 CLIP | 与当前问题匹配度 | 实现成本 |
|---|---|---|---:|---|---|---|
| UFD linear probe | CLIP 最终 pooled/projected image embedding | 无；单特征线性 probe | 很低 | 是 | 中：支持 frozen-CLIP 范式，但特征位置与 C1 不同 | 低 |
| C1-Exact / LEGION Stage 2 | penultimate block CLS，1024-D | 无；1024→2048→2 | 2.103M | 是 | 当前基线 | 已完成 |
| RINE full | 所有 CLIP block CLS | Q1 + TIE layer/channel weights + sum + Q2 + head | 依配置约 0.28M–10.52M | 是 | **很高**：直接补单层 CLS 缺失信息 | 中 |
| RINE BCE-only ablation | 同 RINE | 同上，但无 SupCon | 同结构 | 是 | 中：可做机制消融，不应冒充 full RINE | 中 |
| SNIFF | selected intermediate CLS | element-wise average + two-layer head | 0.66M（论文） | 是 | 高，但精确层号/官方代码尚不可核验 | 低 |
| AIDE full | frozen OpenCLIP semantic map + DCT/SRM high/low patch experts | feature concat + 2304→1024→2 MLP | 高，含 ResNet-50-style PFE experts | semantic CLIP 是；PFE 可训练 | 高：增加正交低层线索，但改动大 | 高 |
| AIDE-PFE + C1 semantic | AIDE low-level PFE + 当前 C1 CLIP feature | AIDE-style feature concat + MLP | 高 | 是 | **很高**：直接瞄准 CLIP-only 上限 | 高 |
| F2-fixed | CLIP + LLM logits | 0.5/0.5 logit sum | 0 | 两个 base 均冻结 | 中：仅重组已有证据 | 极低 |
| F2-alpha | CLIP + LLM logits | convex weighted logit sum | 1 | 两个 base 均冻结 | 中：已有候选，表达力受限 | 极低 |
| Linear logistic stacking | 两路 logit margins | `w_Cs_C+w_Ls_L+b` | 3 | 两个 base 均冻结 | 中：能学边界，但不增加表征 | 低至中（需 OOF） |
| Temperature scaling | 单路或融合 logits | 正温度缩放 | 1 | 是 | 低：只能 calibration | 极低 |

## 8. 三条候选路线的明确比较

### A. Multi-layer CLIP：RINE / SNIFF style

- **新增信息**：同一冻结 CLIP 内，中浅层纹理/局部细节与高层语义的层级信息。
- **相对 C1-Exact**：C1 只读一个 penultimate CLS；A 路线直接改变这一表征瓶颈。
- **超过 LEGION 的机制依据**：LEGION-retrained Stage 2 与 C1-Exact 是同一个单层 CLS contract；RINE 引入它没有看到的 intermediate block representations。
- **风险**：仍受同一 CLIP backbone 与预处理约束；RINE 配置需要预注册，SNIFF 精确层号当前不可核验。
- **结论**：三条路线中优先级最高；先做来源最完整的 RINE full。

### B. CLIP + low-level forensic：AIDE style

- **新增信息**：频率异常、噪声残差、过度平滑/高频 artifact patch，不依赖 CLIP semantic embedding 必须显式保留这些信号。
- **相对 C1-Exact**：保留其强 semantic classifier，同时增加正交 PFE，而不是再训练另一个语义 head。
- **超过 LEGION 的机制依据**：LEGION Stage 2 没有 DCT、SRM 或 patch experts；AIDE 的论文消融支持 semantic + low-level 的互补性。
- **风险**：参数、显存、预处理与实现成本最高；将是“官方 AIDE PFE + 当前 C1 semantic”的可追溯 hybrid control，而非 exact full-AIDE reproduction，报告必须如此命名。
- **结论**：第二优先；适合在 multi-layer CLIP 后验证正交证据是否突破 CLIP-only 上限。

### C. CLIP + LLM score fusion：stacking / calibration style

- **新增信息**：不新增图像表示；只组合两路已存在的 scalar decision evidence。
- **相对 F2-alpha**：linear logistic stacking 多出独立两路权重和 bias，表达力严格更强；temperature scaling 则不能改变 0.5 决策。
- **可能超过 LEGION 的条件**：只有当 LLM score 在 CLIP 错误子集上存在稳定且可由两维线性边界利用的互补性时。
- **风险**：logit scale、in-sample meta-training 和 calibration/分类提升混淆；OOD 下权重稳定性没有结构保证。
- **结论**：可作为便宜的后续 control，但不列入最先做的 1–2 个实验。

## 9. 冻结推荐：仅两个 controlled experiments

### Experiment 1（优先）：RINE full aggregation on current frozen CLIP

```text
same RGB / same CLIP ViT-L/14@336 / same internal train-val
all Transformer-block CLS
→ official RINE Q1
→ official TIE + layer-wise weighted sum
→ official Q2 + classifier
→ official BCE + ξ·SupCon
```

- **为什么可能超过 LEGION**：LEGION/C1-Exact 只读 penultimate CLS，RINE 使用中浅层到深层的完整 hierarchy；这是论文与消融直接支持的缺失信息。
- **与 C1-Exact/F2-alpha 的关系**：以 C1-Exact 为 matched single-layer baseline；不复用或调节 F2-alpha，先判断 CLIP 表征本身能否提升。
- **改动大小**：中等；增加 hidden-state collection 与 RINE classification head，不动 backbone。
- **新训练参数**：有，仅 Q1/TIE/Q2/head；CLIP、LLM、SAM、R1 全冻结。唯一官方配置须在训练前根据 RINE 发布配置预注册，禁止 OOD 选型。
- **localization**：完全不动；分类 sidecar 不读取 explanation、`[SEG]` 或 mask。
- **命名边界**：应命名 `RINE-on-C1-CLIP` 或等价名称，而非“exact RINE”，因为分辨率/API 与原论文实现不同。

### Experiment 2（第二优先）：AIDE PFE complement to frozen C1-Exact

```text
frozen C1-Exact semantic feature
⊕ official AIDE DCT high/low patch selection
⊕ official fixed 30-filter SRM preprocessing
⊕ official AIDE low-level experts
→ AIDE-style channel concat + MLP → Real/Fake
```

- **为什么可能超过 LEGION**：引入 LEGION single-CLS 完全没有的低层频率/噪声证据，且 AIDE 正式消融支持其与 semantic branch 互补。
- **与 C1-Exact/F2-alpha 的关系**：C1-Exact 作为冻结 semantic expert；不同于 F2-alpha 的末端 score 混合，它在 feature level 加入新的视觉证据。
- **改动大小**：较大；需要官方 DCT/SRM preprocessing 和 PFE experts，但不允许改造成无出处的自定义轻量模块。
- **新训练参数**：有；训练 AIDE PFE 与 fusion MLP，C1-Exact/CLIP/LLM/SAM 全冻结。参数量与 exact implementation 应在 preflight 实测，不在本审计中猜测。
- **localization**：完全不动；作为独立 classification branch 接入。
- **命名边界**：这是 `AIDE-PFE + C1 semantic controlled hybrid`，不是 full AIDE；若要声称 full AIDE，必须连其 OpenCLIP ConvNeXt-XXL SFE 一起严格复现并作为独立 baseline。

## 10. STOP 条件与下一阶段入口

本阶段只冻结结构选择，不授权实现或训练。若进入下一阶段，应先单独完成：

1. 官方 RINE repo commit/配置/checkpoint provenance 固定；
2. current HF CLIP 与 OpenAI CLIP block-output 对应关系的单 batch shape audit；
3. 唯一 `q / D' / ξ` 配置预注册，以及 trainable-parameter manifest；
4. internal train/validation-only firewall；
5. 明确 C1-Exact matched control，禁止先看 external/OOD 再选层、选 loss 或选配置。

阶段状态：`COMPLETE_AND_STOPPED_BEFORE_IMPLEMENTATION_OR_TRAINING`。
