# Codex 下一阶段研发计划：GLaMM 统一可解释 AIGC 取证框架

## 0. 项目目标

当前项目是在毕业论文工作的基础上继续完善，目标不是简单复现 LEGION，而是形成一套可投稿会议/期刊的统一 AIGC 图像取证框架。

当前核心方向：

1. 基座由 LISA 切换到 GLaMM；
2. 使用 LEGION / SynthScars 的伪造图像、瑕疵解释和瑕疵区域标注；
3. 补充真实图像数据，来源包括 OpenImages、COCO、PASS、FFHQ、iNaturalist；
4. 真实图按照 SynthScars 内部 Human / Animal / Scene / Object 的类别比例进行采样；
5. PASS 无原始类别标签，已使用 GPT-5.6 Sol 自动分类；
6. 设计分类、解释、定位一体化训练；
7. 新增分类 token / 分类头，并考虑分类头输出与语言模型文本真假标签的一致性损失；
8. 保留 NPR 专家网络，并新增 SRM 分支；
9. 下一步需要解决 GLaMM 与 NPR/SRM 的融合方式；
10. 在完成结构确认和 baseline 后，才开始正式训练。

---

# 1. 当前最重要的原则

## 1.1 暂时不要直接开始完整训练

在以下问题确认之前，不进行正式长训练：

- `[CLS]` 当前到底是如何进入序列的；
- `[CLS]` 是训练目标中的生成 token，还是输入中的固定 token；
- 分类头到底读取哪个 hidden state；
- `[Real] / [Fake]` 在训练和推理时如何产生；
- Real 样本当前是否包含 `[SEG]`；
- Real 样本是否参与 segmentation loss；
- GLaMM 中 `[SEG]` hidden state 到 mask decoder 的完整数据流；
- NPR/SRM 当前输出的 tensor shape 和空间尺度；
- 真实/伪造数据的采样、预处理、增强是否完全一致。

Codex 第一阶段的任务是**代码审计和机制确认**，不是立即改结构。

---

# 2. 需要回答的核心研究问题

## RQ1：当前 `[CLS]` 到底是“生成 token”还是“固定 forensic query token”？

当前方案描述为：

```text
[CLS] [Real]/[Fake] explanation ... [SEG]
```

但目前尚未确认代码中 `[CLS]` 的实际生命周期。

Codex 必须从代码中追踪：

```text
tokenizer 注册
    ↓
dataset conversation 构造
    ↓
input_ids
    ↓
labels
    ↓
collate
    ↓
model.forward
    ↓
hidden_states
    ↓
classification head
    ↓
model.generate
```

重点确认：

- `[CLS]` 是否出现在 `input_ids`；
- `[CLS]` 对应 label 是否为 `-100`；
- `[CLS]` 是否需要语言模型预测；
- 推理阶段是否必须先生成 `[CLS]` 才能分类；
- 分类 head 是否读取生成前的 hidden state，还是 teacher forcing 下目标序列中的 hidden state。

### 目标

最终在代码审计后比较两个方案：

### 方案 A：Generated CLS

```text
Assistant target:
[CLS] [Fake] explanation [SEG]
```

语言模型需要预测 `[CLS]`。

### 方案 B：Fixed Forensic Query

```text
Input / assistant prefix:
[FORENSIC]
```

`[FORENSIC]` 固定存在于模型输入，不作为预测目标。

```text
h_forensic -> classification head
```

语言模型再预测：

```text
[Real] / [Fake] + explanation + [SEG]
```

**不要提前决定哪种方案，先依据当前代码结构给出改造成本、稳定性和训练/推理差异。**

---

# 3. 统一训练方案

项目拟研究：

> Authenticity-Conditioned Unified Forensic Learning

即分类、文本解释、瑕疵区域定位共享同一个 GLaMM 框架，但不同样本根据真实/伪造标签启用不同监督。

---

## 3.1 Fake 样本目标

建议 grammar：

```text
[CLS or FORENSIC]
[Fake]
<artifact explanation>
[SEG]
```

监督包括：

```text
L_cls
L_text
L_seg
L_consistency
```

如果后续使用 SynthScars 的瑕疵类型标签，则可增加：

```text
L_artifact
```

---

## 3.2 Real 样本目标

建议 grammar：

```text
[CLS or FORENSIC]
[Real]
No identifiable synthetic artifact evidence is detected.
```

Real 样本：

```text
L_cls        ✓
L_text       ✓
L_seg        ×
L_artifact   ×
L_consistency ✓
```

Real 样本默认不输出 `[SEG]`，但必须通过实验确认这种 conditional grammar 是否引入任务捷径。

---

# 4. 一致性损失

项目已经观察到类似 SIDA 的现象：

- classification head 输出准确率高于从生成文本中解析 `[Real]/[Fake]`；
- classification head 与语言模型文本真假标签存在不一致。

因此必须实现并评估一致性约束。

设：

```text
p_cls = classification head 的真假概率
p_lm  = LLM 在 [Real]/[Fake] 位置上的真假概率
```

可候选：

### 方案 1：KL

```text
L_cons = KL(p_cls || p_lm)
```

### 方案 2：Symmetric KL

```text
KL(p_cls || p_lm) + KL(p_lm || p_cls)
```

### 方案 3：JS Divergence

用于更稳定的双向一致性。

第一版优先实现单向或 symmetric KL，不要一开始引入复杂设计。

---

# 5. Real 不输出 `[SEG]` 的潜在 shortcut 问题

需要专门设计实验确认。

风险并不是“模型在输入中看到是否有 `[SEG]`”，因为 `[SEG]` 是目标输出；真正风险是：

- Fake explanation 总是更长；
- Fake 总是需要产生 `[SEG]`；
- Real explanation 总是固定短句；
- 模型可能利用输出模板差异形成 task shortcut。

## Codex 需要准备以下可切换 grammar

### Grammar A

```text
Real:
[Real] No synthetic artifact evidence is detected.

Fake:
[Fake] explanation [SEG]
```

### Grammar B

让 Real/Fake 具有更接近的文本结构：

```text
Real:
[Real]
Evidence assessment: no reliable synthetic artifact is detected.
Localization: no artifact region is available.

Fake:
[Fake]
Evidence assessment: ...
Localization: [SEG]
```

仍然只对 Fake 计算 segmentation loss。

通过配置项切换：

```yaml
target_grammar: conditional_short
# or
target_grammar: conditional_balanced
```

---

# 6. 定位评估必须拆分

对于 GT 为 Fake 的图像，如果模型分类成 Real，因此不输出 mask，不能只用一个 IoU 指标混在一起解释。

必须同时实现三类定位指标。

## 6.1 Oracle Localization

已知样本为 Fake，强制进入定位流程。

用途：

> 单独评价 mask decoder / grounding 能力。

## 6.2 Conditional Localization

仅在模型预测 Fake 且成功产生 mask 的 GT Fake 样本上计算：

- IoU
- Dice / pixel F1

用途：

> 评价模型在“已经识别为 Fake 后”的定位质量。

同时必须报告：

```text
mask_coverage_rate
=
产生有效 mask 的 GT Fake 样本数 / GT Fake 总数
```

避免只在少量容易样本上计算高 IoU。

## 6.3 Joint Detection-Localization

GT Fake：

- Prediction Real / 无 mask → IoU = 0
- Prediction Fake → 正常计算 mask IoU

用途：

> 评价端到端部署效果。

最终论文至少报告：

```text
Detection AUC/F1
Oracle IoU/F1
Conditional IoU/F1
Mask Coverage Rate
Joint IoU/F1
```

---

# 7. 数据集审计

当前真实图来源：

```text
OpenImages
COCO
PASS
FFHQ
iNaturalist
```

采样已经按照 SynthScars 的：

```text
Human
Animal
Scene
Object
```

类别比例进行平衡。

PASS 的类别由 GPT-5.6 Sol 自动分类。

## Codex 需要验证，而不是重新设计采样

输出完整统计：

```text
SynthScars Fake:
Human
Animal
Scene
Object

Real Total:
Human
Animal
Scene
Object

OpenImages:
Human / Animal / Scene / Object

COCO:
Human / Animal / Scene / Object

PASS:
Human / Animal / Scene / Object

FFHQ:
Human

iNaturalist:
Animal
```

检查：

- Real / Fake 总量；
- 四类别比例；
- 各真实数据源占比；
- train / val / test 是否存在重复；
- 文件名重复；
- pHash / perceptual duplicate（如果已有工具则复用）；
- preprocessing 是否完全一致。

---

# 8. Dataset Shortcut 检查

虽然已经做了 content balance，但仍然可能存在：

- JPEG quality；
- resolution；
- aspect ratio；
- crop；
- dataset watermark；
- sensor noise；
- metadata / preprocessing；
- 数据集来源风格。

需要生成 source statistics：

```text
width / height
aspect ratio
file format
JPEG quality（可获取时）
mean / std
compression artifacts
```

建议后续增加一个轻量 source classifier 实验：

```text
image -> dataset source
```

用于判断数据集来源是否非常容易被识别。

此实验不属于第一轮核心训练，可放在数据审计完成后。

---

# 9. LEGION-style 与 Unified Training 必须公平比较

这是论文最核心的实验之一。

## Baseline A：LEGION-style decoupled training

尽量在相同 GLaMM 初始化和相同数据条件下模拟：

### Stage A1

Synthetic-only：

```text
Explanation + Localization
```

### Stage A2

Real/Fake：

```text
Classification
```

分类任务与解释/定位不做统一联合优化。

---

## Baseline B：Unified Training

Real/Fake 同时进入：

```text
Classification
Text
Conditional Segmentation
Consistency
```

目标不是预设 Unified 一定更好，而是回答：

> 联合真实性判别、瑕疵 grounding 和自然语言解释，是否能提高跨任务一致性，同时保持或提升定位和泛化能力？

必须记录：

```text
classification
localization
explanation
classification-text consistency
OOD generalization
```

---

# 10. NPR + SRM 专家网络

当前状态：

- NPR 按官方实现；
- 新增 SRM 分支；
- 测试集和 LOKI 上已观察到一定性能提升；
- 暂未与 GLaMM 融合。

不要直接照搬毕业论文融合方式，先分阶段验证。

---

# 11. 专家融合 V1：只进入分类头

第一版优先做低风险 late fusion：

```text
GLaMM
   ↓
h_cls / h_forensic

NPR
   ↓
g_npr

SRM
   ↓
g_srm

[h_cls, g_npr, g_srm]
       ↓
Fusion
       ↓
Classification Head
```

候选 Fusion：

1. concat + MLP；
2. gated weighted sum；
3. small attention fusion。

第一轮至少实现：

```text
concat + MLP
gated fusion
```

目的：

> 先确认 NPR/SRM 在 GLaMM unified baseline 中是否仍能提升分类泛化。

此阶段不要让 NPR/SRM 影响 explanation 和 segmentation。

---

# 12. 专家融合 V2：Forensic Evidence Injection

只有 V1 证明专家有效后，再实现。

结构建议：

```text
GLaMM Semantic Tokens
          |
          | Q
          v
Gated Cross Attention
          ^
          | K/V
Forensic Tokens
    ↑
NPR + SRM
```

输出：

```text
V' = V_sem + alpha * CrossAttn(V_sem, E_forensic)
```

关键要求：

```text
alpha 初始为 0 或非常小
```

保证训练初期：

```text
V' ≈ V_sem
```

避免破坏 GLaMM 原有视觉表示。

需要支持配置：

```yaml
forensic_fusion:
  mode: none
  # late_cls
  # gated_cross_attn
```

---

# 13. 专家冻结策略

第一阶段：

```text
NPR frozen
SRM frozen
```

训练：

```text
classification head
fusion module
projector
LoRA / selected GLaMM trainable params
```

第二阶段若结果稳定：

仅解冻 NPR / SRM 后部 block，并使用较小学习率。

所有 trainable params 必须在启动日志中完整打印。

---

# 14. SynthScars artifact 类型标签

如果当前数据中保留了：

```text
Physics
Distortion
Structure
```

则先确保 dataset 能读取和返回。

第一阶段不一定马上加入主模型，但预留：

```text
artifact_type
```

字段。

后续可增加：

```text
Artifact Type Head
```

仅 Fake 样本计算：

```text
L_artifact
```

这个任务以后可以帮助分析不同专家对应不同瑕疵类型的作用。

---

# 15. Codex 任务执行阶段

---

## Phase 0：代码审计（现在立即做）

### 目标

**不修改模型结构。**

检查：

1. tokenizer special token 注册；
2. `[CLS]`；
3. `[Real]`；
4. `[Fake]`；
5. `[SEG]`；
6. dataset conversation；
7. labels mask；
8. collate；
9. GLaMM forward；
10. generate；
11. classification head；
12. segmentation loss；
13. text CE；
14. NPR；
15. SRM；
16. SynthScars / Real 采样。

### 输出文件

建议生成：

```text
docs/current_pipeline_audit.md
```

内容必须包括：

```text
当前 token flow
当前 loss flow
当前 Real target
当前 Fake target
当前 classification flow
当前 segmentation flow
NPR/SRM tensor shape
数据集统计
发现的问题
```

### 验收

必须能明确回答：

> 当前 `[CLS]` 是生成 token 还是固定输入 token？

在这个答案确认前，不进入 Phase 1。

---

## Phase 1：固化 Unified GLaMM Baseline

暂时：

```text
NO NPR
NO SRM
NO evidence injection
```

实现：

```text
GLaMM
+ classification token/head
+ Real/Fake mixed data
+ conditional segmentation
```

配置中必须能关闭：

```yaml
use_npr: false
use_srm: false
consistency_loss: false
```

先验证模型正常学习。

---

## Phase 2：Consistency Loss

增加：

```text
classification head probability
vs
LM [Real]/[Fake] probability
```

记录：

```text
cls_accuracy
text_label_accuracy
cls_text_agreement
```

必须额外输出四类：

```text
CLS correct / Text correct
CLS correct / Text wrong
CLS wrong / Text correct
CLS wrong / Text wrong
```

再比较：

```text
No Consistency
vs
Consistency
```

---

## Phase 3：定位评估重构

实现：

```text
oracle localization
conditional localization
joint localization
mask coverage rate
```

不能只保存一个 IoU。

---

## Phase 4：LEGION-style 对照

实现独立配置：

```yaml
training_strategy: legion_style
```

和：

```yaml
training_strategy: unified
```

确保：

- 相同初始化；
- 相同数据；
- 相同训练 budget（尽可能）；
- 相同 eval。

---

## Phase 5：NPR/SRM Late Fusion

实验：

```text
Unified GLaMM
Unified + NPR
Unified + SRM
Unified + NPR + SRM
```

先只影响 classification。

---

## Phase 6：Forensic Evidence Injection

如果 Phase 5 OOD 有稳定提升：

实现 gated residual cross-attention。

比较：

```text
Late Classification Fusion
vs
Evidence Injection
```

观察：

- Classification；
- Localization；
- Explanation；
- OOD；
- Consistency。

---

# 16. 必须准备的消融实验

最终代码结构需要支持下列配置。

```text
A0 GLaMM baseline
A1 GLaMM + Unified Training
A2 A1 + Consistency
A3 A2 + NPR
A4 A2 + SRM
A5 A2 + NPR + SRM Late Fusion
A6 A5 + Evidence Injection
```

训练策略：

```text
LEGION-style
vs
Unified
```

token 策略：

```text
Generated CLS
vs
Fixed Forensic Query
```

target grammar：

```text
conditional_short
vs
conditional_balanced
```

---

# 17. 实验集使用规则

## In-domain

SynthScars + matched Real split。

## OOD

LOKI 保持锁定测试集。

从现在开始原则上：

- 不使用 LOKI 训练；
- 不使用 LOKI 做 early stopping；
- 不依据 LOKI 单次结果频繁修改超参数。

如果此前已经多次根据 LOKI 结果改过模型，需要在实验记录中注明。

---

# 18. 统一输出格式

每次 evaluation 输出：

```json
{
  "sample_id": "...",
  "image_path": "...",
  "dataset": "...",
  "source": "...",
  "content_type": "human|animal|scene|object",
  "gt_label": "real|fake",
  "cls_prob_fake": 0.0,
  "cls_pred": "real|fake",
  "lm_prob_fake": 0.0,
  "lm_pred": "real|fake",
  "cls_lm_agree": true,
  "has_gt_mask": true,
  "has_pred_mask": true,
  "oracle_iou": null,
  "conditional_iou": null,
  "joint_iou": null,
  "explanation": "...",
  "artifact_type": null
}
```

---

# 19. 实验管理要求

每个实验保存：

```text
config.yaml
git_commit.txt
train.log
metrics.json
predictions.jsonl
checkpoint/
visualizations/
```

日志必须打印：

- trainable parameter count；
- frozen parameter count；
- special token ids；
- loss weights；
- dataset sizes；
- content distribution；
- Real/Fake distribution；
- NPR/SRM enabled state；
- fusion mode；
- token strategy；
- target grammar。

---

# 20. Codex 编码规则

1. 不删除已有可运行实现；
2. 新实验通过 config 开关实现；
3. 不将不同方案写死在一个 forward 中；
4. 每个新模块加入 shape test；
5. 修改 token 后必须测试 tokenizer save/load；
6. 修改 dataset grammar 后必须提供可视化样本；
7. 修改 loss 后必须打印每项 loss；
8. Real batch 必须确认 segmentation loss 为 0 / skipped；
9. Fake batch 必须确认 segmentation gradient 正常；
10. 所有新增实验保持 backward compatibility。

---

# 21. 当前第一条 Codex 指令

请直接执行以下任务，不要开始训练：

```text
请对当前 brokeniceice/glamm 项目进行完整的“token / dataset / loss / model flow”代码审计，本轮不要修改模型结构。

重点回答：

1. 当前 [CLS] token 是如何注册到 tokenizer 的？
2. dataset 中 [CLS] 是 input token 还是 assistant target token？
3. [CLS] 对应 labels 是否为 -100？
4. 训练时 [CLS] 是否需要通过语言模型 CE 预测？
5. 推理时 classification head 的 hidden state 从哪里获得？
6. 当前 [Real]/[Fake] 是如何生成和监督的？
7. [SEG] 在 Real 和 Fake 样本中分别如何出现？
8. Real 样本当前是否计算 segmentation loss？
9. GLaMM 如何从 [SEG] hidden state 得到 mask？
10. classification head 的输入 tensor shape 是什么？
11. NPR 和 SRM 的输入、输出 shape、checkpoint、冻结状态分别是什么？
12. 当前 SynthScars + OpenImages + COCO + PASS + FFHQ + iNaturalist 的采样逻辑是什么？
13. 请统计 Real/Fake 与 Human/Animal/Scene/Object 的实际数量和比例。
14. 请检查 Real/Fake 是否经过完全相同的 resize、crop、augmentation 和 normalization。
15. 不要改变任何训练逻辑。

输出：
- docs/current_pipeline_audit.md
- 当前 token flow 图（文本形式即可）
- 当前 loss flow
- 当前 dataset flow
- 关键代码文件与行号
- 潜在 bug / label leakage / shortcut 风险
- 对 Generated CLS 与 Fixed Forensic Query 两种方案的代码级改造成本比较

完成审计后停止，不要继续实现。
```

---

# 22. Phase 0 完成后的第二条指令

只有第一阶段审计完成且人工确认后再执行：

```text
基于 current_pipeline_audit.md，新增一个可配置的 classification token strategy。

要求支持：

1. generated_cls
2. fixed_forensic_query

不要删除当前实现。

对于 fixed_forensic_query：
- token 固定进入 assistant prefix / model input；
- 该 token 不参与语言模型 CE；
- 从该 token hidden state 读取 classification representation；
- 后续 [Real]/[Fake] 仍由语言模型预测；
- 保持 [SEG] 原有机制不变。

同时输出：
- tokenization 示例；
- input_ids / labels 示例；
- hidden state index；
- 单 batch forward test；
- generate test；
- 两种模式的差异说明。

本阶段暂不加入 NPR/SRM。
```

---

# 23. 建议开发顺序

严格按照以下顺序：

```text
代码审计
↓
确认 CLS 实现
↓
Unified GLaMM baseline
↓
Consistency Loss
↓
Localization evaluation
↓
LEGION-style vs Unified
↓
NPR/SRM late fusion
↓
Forensic Evidence Injection
↓
完整训练
↓
论文实验
```

不要提前跳到 NPR/SRM 与 GLaMM 的深层融合。

---

# 24. 最终论文希望形成的故事

## Contribution 1

**Authenticity-Conditioned Unified Forensic Learning**

统一真实性分类、artifact explanation 和 artifact grounding，同时对 Real/Fake 使用条件监督。

## Contribution 2

**Discriminative-Generative Authenticity Consistency**

对齐 classification head 与语言模型生成的真假判断，降低分类与解释矛盾。

## Contribution 3

**Forensic Expert Enhancement**

利用 NPR + SRM 提供通用视觉编码器缺失的低层 synthetic traces。

## Contribution 4（在实验有效时再保留）

**Gated Forensic Evidence Injection**

通过残差门控 cross-attention 将低层取证证据注入 GLaMM，而不是简单 concat。

---

# 25. 当前不要做的事情

- 不要现在开始完整训练；
- 不要直接重构整个 GLaMM；
- 不要默认当前 `[CLS]` 一定是生成 token；
- 不要同时改 CLS、loss、dataset、NPR fusion；
- 不要在同一次实验里加入多个未经单独验证的模块；
- 不要使用 LOKI 作为训练/调参集；
- 不要只报告一个混合了分类误差的 mask IoU；
- 不要假设 Unified 一定优于 LEGION-style，必须通过公平对照证明。

