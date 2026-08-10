# 统一分类、解释与定位的 AIGC 取证模型路线图

## 1. 项目目标

在官方 GLaMM 基础上构建一次前向即可完成以下任务的统一模型：

1. 图像级真实性分类；
2. 与分类结论一致的取证解释；
3. 与解释中的局部证据一致的 artifact 掩码。

核心目标不是简单叠加三个任务头，而是让三种输出共享同一个判定与证据链，同时提高未见生成器、未见数据集和常见图像扰动下的泛化能力。

当前阶段明确不接入 NPR。纯 NPR+SRM best 只作为后续可选辅助专家保留，不作为提示词、标签来源或当前主模型输入。

当前第一版训练范围进一步限定为：**所有 fake 样本只来自 SynthScars**。不使用 AIGI、LOKI、FakeBench、DeepfakeJudge 或其他数据集中的 fake 样本参与训练。其他 fake 数据只可能在后续作为完全未见域评估集使用，且需要用户再次确认。

## 2. 不可破坏的设计原则

- **单次推理**：一个统一入口同时返回分类、解释和掩码，不使用两个独立 checkpoint 或先后运行的任务程序。
- **单一判定源**：结构化 verdict token 与分类头必须来自同一 `[CLS]` 表征，并受显式一致性约束。
- **证据而非伪造像素**：SynthScars polygon 表示可见 artifact 证据区域，不表示图中其余区域是真实像素。
- **真图负监督**：真实图必须同时监督为 real、无可靠 AIGC artifact、空证据掩码。
- **当前 fake 均为局部证据样本**：SynthScars fake 必须同时具有 artifact 解释与非空 polygon mask；全局证据或无 mask fake 延后研究。
- **禁止专家泄漏**：NPR 结论不得转换成 hard prompt；无 NPR 时模型必须能够独立工作。
- **先建立无 NPR 基线**：只有统一主模型通过跨域验收后，才讨论 NPR 的软融合增益。

## 3. 统一输出协议草案

第一版只新增 verdict token：`[REAL]`、`[FAKE]`。保留已有 `[CLS]`、`[SEG]`；不引入 `[LOCAL]`、`[NONE]` 或其他 evidence-type token。

局部证据假图：

```text
[CLS] [FAKE] <artifact explanation> [SEG]
```

真图：

```text
[CLS] [REAL] No reliable synthetic artifact is found. [SEG]
```

`[SEG]` 在真图和假图中都存在。真图以全零 mask 监督；每个 SynthScars fake 在第一版只产生一个 `[SEG]`，其监督是该图片所有有效 refs/polygon mask 的布尔并集（union evidence mask）。第一版不处理全局证据或缺少 mask 标注的 fake。

第一版不设置独立 evidence type：fake 必须具有 artifact explanation 和非空 union evidence mask，real 必须使用固定负解释和全零 mask，这些属性可由 verdict 与 mask 监督直接表达。若未来引入 global evidence 或无 mask fake，再重新评估是否需要 evidence-type token，而不为尚不存在的训练状态预留 token。

推理时分类结论先确定 verdict token，后续解释和 `[SEG]` 在因果上依赖该 token。最终用户界面可以隐藏结构 token，但评估必须保留原始 token、分类概率和 mask logits，不能用后处理掩盖矛盾。

## 4. 统一样本数据契约

每个样本统一为：

```text
image
class_label                 # real/fake/unknown
explanation
union_evidence_mask         # 第一版模型监督；所有有效 ref masks 的布尔并集
ref_phrases                 # 保留用于追溯和后续 phrase-mask 对齐
ref_explanations
ref_masks
annotation_ids
has_class_label
has_explanation
has_mask_label
source
generator
```

损失根据 `has_*` 字段屏蔽缺失标签，但所有数据走同一个模型和推理协议。

数据角色：

- **SynthScars**：fake + explanation + 非空 union evidence mask；同时保留逐 ref 文本与 mask。
- **多来源真实图**：real + 固定负解释 + 全零 mask。
- 第一版不引入其他 fake、分类-only fake 或其他定位/解释数据。

SynthScars 官方包有 11,236 条训练标注和 11,182 个唯一文件名。合并 118 对同 stem 的 PNG/JPG 尺寸/编码变体后，共有 11,064 个唯一训练视觉身份；其中 171 个身份包含多条 annotation。数据阶段按视觉身份聚合有效 refs/polygon，以布尔 OR 生成单个 union evidence mask，避免无意重复采样；逐 ref 数据、原始尺寸、文件变体和 annotation id 必须完整保留以便追踪及阶段 7 使用。

## 5. 损失设计

基础任务损失：

```text
L_task = L_cls + L_verdict + L_lm + L_mask_bce + L_mask_dice
```

一致性损失：

```text
L_consistency =
    λ_ct * L_cls_token
  + λ_re * L_real_empty
  + λ_tm * L_text_mask
```

- `L_cls_token`：分类头概率与 `[REAL]/[FAKE]` token 概率的一致性约束。
- `L_real_empty`：真实图的 mask BCE/面积惩罚，防止真图硬找局部 artifact。
- `L_text_mask`：仅对 SynthScars fake 启用，约束解释所声明的证据与 `[SEG]` mask 同时存在并对齐。
- 当前所有 SynthScars fake 都必须有非空 union evidence mask；若数据审计发现异常空标注，应报告并排除，而不是改成全局 fake。

第一版先实现可验证的 `L_cls_token` 与 `L_real_empty`；语义级 `L_text_mask` 在基础链路稳定后加入，避免一次引入过多不可诊断变量。

## 6. 分阶段实施计划

### 阶段 0：冻结基线与验收定义

**目标**：建立后续所有实验的不可变参照。

工作项：

- 固定当前官方 GLaMM+[CLS] checkpoint、tokenizer 和 yacht mask 回归结果。
- 固定 SynthScars 原始文件哈希、拆分和统计。
- 固定无 NPR 原则；记录纯 NPR+SRM best，但不加载。
- 写明分类、mask、解释、一致性和泛化指标。

通过条件：基线能够重复推理；所有结果目录包含配置、commit、权重来源和数据统计。

### 阶段 1：SynthScars 与真实负样本数据层

**目标**：只解决数据正确性，不改模型。

工作项：

- 解析 SynthScars polygon，生成与原图坐标一致的逐 ref 二值 mask，并用布尔 OR 生成每图单个 union evidence mask。
- 审计越界点、空 polygon、多 polygon、多 refs、重复 annotation 和跨后缀图像变体。
- 生成统一样本契约和可视化抽检。
- 选择多来源、内容尽量匹配 SynthScars 的真实图，生成 `[REAL]`、固定负解释和全零 mask。
- 检查数据来源与标签的相关性，避免“SynthScars 来源=fake、单一来源=real”的数据集捷径。

交付物：dataset adapter、统计 JSON、至少 100 个可视化抽检样本、数据单元测试。

通过条件：所有标注引用有效；随机抽检 polygon 与解释对应；真实 mask 严格为空；无标签泄漏路径。

### 阶段 2：结构化统一推理协议

**目标**：先让一次运行稳定地产生三种输出，不训练长任务。

工作项：

- 增加 `[REAL]`、`[FAKE]` token 并初始化 embedding。
- 明确 `[CLS]` 分类位置、verdict token 位置和 `[SEG]` hidden state位置。
- 实现单一 `evaluate_unified`：返回分类概率、verdict、解释和单个 union evidence mask。
- 真图也运行 mask decoder，以便观察而不是隐藏错误。
- 加入语法约束：一个样本只能产生一个 verdict 和一个 `[SEG]`；`[REAL]` 后使用固定负解释并监督全零 mask，`[FAKE]` 后使用 artifact explanation 并监督非空 union evidence mask。

交付物：统一前向接口、结构化输出解析器、真实与 SynthScars 局部 fake 两类合成单元测试。

通过条件：分类和文字 verdict 在接口层不能分叉；现有 GLaMM mask 回归不被破坏。

### 阶段 3：基础损失与一致性损失

**目标**：建立可单独开关、可消融的一致性训练路径。

工作项：

- 实现 `L_cls`、`L_verdict` 和缺失标签屏蔽。
- 将真实图全零 mask 纳入 BCE/Dice/面积监督。
- 实现 `L_cls_token`；分别记录每个损失，不只记录总损失。
- 暂不加入 NPR 和复杂 text-mask 语义对齐。

交付物：损失单元测试、梯度流检查、每个损失权重的配置项。

通过条件：每种损失只影响预期参数；缺失标签不产生伪梯度；real/fake 极小样本可正确过拟合。

### 阶段 4：小样本闭环验证

**目标**：在正式训练前证明统一链路能够学习。

工作项：

- 使用少量 SynthScars + 真实图构造平衡小集。
- 进行 16/32/64 样本 overfit。
- 对每张样本同时检查 verdict、解释、mask 和 consistency 指标。
- 检查真图是否仍生成 artifact 文字或非空 mask。

通过条件：训练集接近完全拟合；真实图 mask 面积趋近 0；局部 fake mask 可拟合；分类与 verdict 一致率 100%。

### 阶段 5：无 NPR 的第一版联合基线

**目标**：训练真正统一的 GLaMM 基线，而不是 LEGION 式分离阶段。

训练数据只包含：SynthScars local fake 与多来源 real negatives。所有 batch 共享模型；不混入其他 fake 或 classification-only 数据。

建议先冻结 CLIP/SAM image encoder，仅训练必要的 LoRA/投影层、分类与 evidence 相关模块、text hidden projection 和 mask decoder；实际可训练参数表须在开训前审计。

保存与选择 best 时不能只看单一任务。验证评分应同时包含分类、mask、real 空掩码和一致性指标。

通过条件：

- SynthScars mask 指标显著高于未训练基线；
- 真实图非空 mask 比例和 artifact 幻觉率处于预设上限内；
- 分类与文字 verdict 一致率为 100%；
- 不依赖 NPR 即可完成统一推理。

### 阶段 6：SynthScars 范围内的泛化训练与未见域评估准备

**目标**：解决旧项目最主要的未见域退化问题。

工作项：

- fake 训练数据始终只使用 SynthScars；按 SynthScars 可用的生成器、内容类型和 artifact 类型设计分组划分。
- 设置 SynthScars 内部的 leave-one-generator/content/artifact-out，而不是只做随机划分。
- 其他数据集的 fake 若后续获准使用，只作为完全零样本的未见域测试，不回流训练或调参。
- 加入 JPEG、resize、blur、noise 等后处理鲁棒性评估；增强配置必须单独消融。
- 防止来源捷径：报告按 source/generator 的 real/fake 指标。
- 对分类、解释、mask 和一致性分别报告跨域下降量。

通过条件：相较阶段 5，held-out 数据平均指标提升，且已见域性能下降在可接受范围；不能只凭单个数据集提升进入下一阶段。

### 阶段 7：解释—掩码语义对齐

**目标**：从“格式一致”推进到“证据语义一致”。

工作项：

- 在保留第一版每图 union mask 结果用于整图证据评估的同时，将输出升级为每个语义 ref 短语绑定对应 `[SEG]` token 和独立 ref mask，而不是整段解释只对应一个 union mask。
- 引入 phrase-region 对齐或 masked visual feature 对比损失。
- 增加反事实检查：遮除预测证据区域后，相关解释置信度应下降。
- 单独评估 artifact category、短语 grounding 和 mask overlap。

通过条件：语义对齐指标和人工抽检均改善，且分类泛化不下降。

### 阶段 8：可选 NPR 软融合

**前置条件**：阶段 6、7 的无 NPR 模型通过验收。

约束：

- NPR 冻结；不生成 hard hint；不改变标签。
- 只以零初始化 residual/gate 方式注入共享 `[CLS]`/evidence 表征。
- 训练时使用 expert dropout，并加入 NPR 与主模型冲突样本。
- 必做三组对照：无 NPR、NPR feature、NPR hard hint（仅作为负面消融，不进入最终方案）。

通过条件：在多个 held-out 数据集上稳定增益，而非只改善域内分类；禁用 NPR 后模型仍保持可用。

## 7. 统一验收指标

### 分类

- ACC、AP、macro-F1、real accuracy、fake accuracy；
- 按数据集、来源、生成器分别统计；
- SynthScars generator/content/artifact-held-out 的性能下降；外部 fake 测试需另行批准。

### 定位

- fake local evidence：mIoU、F1/Dice；
- real：平均预测 mask 面积、非空 mask 比例、像素假阳性率；

### 解释

- verdict token accuracy；
- artifact category accuracy；
- 文本语义指标与固定人工抽检集。

### 一致性

- 分类头与 verdict token 一致率；
- `pred=real` 但文本声称 artifact 的矛盾率；
- `pred=real` 但 mask 非空的矛盾率；
- `[FAKE]` 且文本声称存在 artifact、但 union mask 为空的矛盾率；
- 分类、解释和定位同时正确的 end-to-end success rate。

## 8. 实验纪律与停止条件

- 每阶段只改变一类核心变量，并保留可运行的上一阶段 checkpoint。
- 先 smoke test，再小样本 overfit，最后正式训练。
- best checkpoint 选择公式在训练前固定。
- 任一阶段若基础 GLaMM mask 回归明显退化、真实图 artifact 幻觉上升或 held-out 泛化恶化，应停止并回退，不用后续模块掩盖问题。
- 在阶段 8 之前，任何代码路径都不得加载 NPR。

## 9. 当前下一步

阶段 1 数据层已经冻结完成，详见 `docs/unified_forensics_data_freeze_v1.md`。PASS 6,000 张已完成 GPT 内容分类；内部池在排除 18 个与 SynthScars 官方 test 重合的 train 视觉身份后，固定为 11,046 Real + 11,046 Fake，并按内容类别和 pHash 连通组完成 8:1:1 train/val/test 划分。SynthScars 官方 test 1,000 张和 RAISE clean held-out 998 张保持为独立测试资源。

下一阶段从阶段 2 的结构化统一推理协议开始；此时仍未启动正式训练，也未加载或改动 NPR 专家网络。除非建立新的显式数据版本，否则不得重新扫描、重采样或覆盖 v1 冻结 manifest。
