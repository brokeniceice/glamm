# Phase 5A-2A — LEGION Localization Evaluation Audit

## 0. 结论

**建议：Option A — LEGION 只应与 R1 G0 对比。**

更精确地说，公开 intermediate `legion_LE` 应按官方 `scripts/loc_exp/infer.py` 的默认协议运行：**仅输入图像和官方固定 user instruction，自由生成 explanation/`[SEG]`，再将全部 `[SEG]` 对应的 SAM mask 取并集**。这与 R1 G0 在“模型可见信息和推理形态”上同属 deployable image-only free generation，因此是三种 R1 条件中唯一可建立公平对应的条件。

官方仓库还存在一个可选的训练期 `--mask_validation` 分支。它将含有 GT caption、GT artifact phrases 和 GT `[SEG]` 位置的完整 assistant turn 直接作为 `input_ids` 做 causal forward，因此技术上是 teacher-forced；但它既不是公开 `infer.py` 的默认推理协议，也不是 R1 TF-PHRASE 的严格同构输入，更没有公开代码证明论文表格使用了该分支。它不能据此被包装成 Phase 5A-2 的“LEGION TF”等价组。

公开 commit **没有 Phrase-only 等价路径**，也没有发布一条从 free-generation 输出一直连接到论文 SynthScars / LOKI / RichHF-18K mIoU/F1 的可调用 evaluator。因此不能为了凑齐三组结果人为增加 LEGION Phrase/TF 条件。

状态标签：

- Option A: **SUPPORTED / RECOMMENDED**
- Option B: **NOT SUPPORTED**
- Option C: **NOT SUPPORTED**
- 论文表格的精确 localization evaluation protocol: **UNIDENTIFIABLE FROM PUBLIC COMMIT**

## 1. 审计范围与约束

- LEGION source：`external/LEGION_official`
- 审计 commit：`d21535dd45f6fea509337a83095966f0b86ac924`
- 审计后官方仓库：`main...origin/main`，无 tracked/untracked 改动，保持 clean
- 重点静态审计：`eval/`、`scripts/loc_exp/`、`dataset/`、`model/Legion.py`、`README.md`
- R1 静态审计：`eval/forensics_eval.py`、`scripts/phase3a_evaluate.py`、`scripts/p1_r1_reusable_matrix.py`、`dataset/forensics/unified.py`、现有 P1/R1 baseline 文档
- 本阶段未加载模型、未读取 benchmark 样本、未运行 R1/LEGION inference、未计算任何性能指标、未修改 prompt/threshold/model。

## 2. 官方 LEGION 实际公开了三种不同层级的证据

### 2.1 路径 L-FREE：公开默认 folder inference

这是 `scripts/loc_exp/infer.py` 暴露给使用者的 localization + explanation 推理路径。

```text
目录中的普通图像
→ 官方固定 user instruction
→ model.evaluate() 自回归 generate（max_tokens_new=512, num_beams=1）
→ 在生成 token 中寻找 [SEG]
→ 每个 [SEG] hidden state 经 text_hidden_fcs 形成 SAM text prompt
→ SAM prompt encoder + mask decoder
→ SAM postprocess_masks 恢复到原图尺寸
→ 每个 mask logit > 0
→ 所有 [SEG] mask 按像素 OR/union
→ mask.png
```

代码证据：

1. `scripts/loc_exp/infer.py:40-77` 构造只有 image + instruction 的空 assistant prompt，`bboxes=None`，然后调用 `model.evaluate(..., max_tokens_new=512)`。
2. 固定 instruction 位于 `scripts/loc_exp/infer.py:144-145`；训练数据的 `GCG_QUESTIONS` 在 `dataset/utils/utils.py:1-3` 也只有同一条 prompt，因此这里的 `random.choice` 实际没有 prompt 变体。
3. `model/Legion.py:289-312` 调用 `generate(..., num_beams=1)`，从生成序列的 `[SEG]` 位置提取 hidden state，并调用 SAM mask 路径。
4. `model/Legion.py:213-247` 显示每个 `[SEG]` embedding 进入 SAM prompt encoder、mask decoder，最后由 `postprocess_masks` 恢复至 `orig_size`。
5. `scripts/loc_exp/infer.py:80-97` 虽从生成文本解析 `<p>...</p>` phrase，但 phrase 只作为返回值；调用方没有把 phrase 再喂入模型或用它生成 mask。
6. `scripts/loc_exp/infer.py:160-180` 对 `pred_masks[0]` 使用 logit threshold `> 0`，再用 `torch.any(..., dim=0)` 合并全部 `[SEG]` mask 并保存 `mask.png`。
7. `scripts/loc_exp/infer.sh:1-4` 只是带 checkpoint/image/output 路径占位符的 launcher；它没有传 GT 或调用 metric evaluator。

因此 L-FREE 的答案如下：

| 问题 | 官方默认行为 |
|---|---|
| prompt | 固定的详细 artifact-analysis instruction，要求 explanation 中 interleave segmentation masks |
| 模型输入 | 图像 + 固定 user instruction |
| GT artifact phrase | 不提供 |
| GT explanation/text | 不提供 |
| teacher forcing | 否，自由生成 |
| 是否使用生成 phrase 再定位 | 否；phrase 仅从已生成文本中解析并返回，mask 直接来自生成 `[SEG]` hidden state |
| 多个 `[SEG]` | 每个产生一个 SAM mask；阈值化后按像素 OR/union |
| mask threshold | SAM mask logit `> 0` |
| GT mask / metric | 此脚本完全不加载 GT，也不计算 localization metric |

换言之，公开默认路径在 `mask.png` 处结束，不是一条完整的论文 evaluation pipeline。

### 2.2 路径 L-TFVAL：可选训练期 mask validation

`scripts/loc_exp/train.py` 另有 `--mask_validation` 分支：

```text
SynthScars/LEGION validation annotation
→ image + 固定 user instruction
→ GT caption 按 tokens_positive 插入 <p> GT phrase </p> [SEG]
→ 完整 user + GT assistant conversation 被 tokenized 为 input_ids
→ model_forward(inference=True)，不是 generate
→ 输入序列中每个 GT [SEG] hidden state → SAM mask
→ mask logit > 0
→ predicted mask 与对应 GT reference mask 逐个 zip
→ foreground/background intersection-and-union
→ 代码打印 giou / ciou
```

代码证据：

1. `dataset/gcg_datasets/GranDf_gcg_ds.py:101-119` 将 GT caption 中每个 `tokens_positive` span 改写为 `<p> phrase </p> [SEG]`，并把整个 tagged caption 作为 assistant answer。
2. `dataset/gcg_datasets/GranDf_gcg_ds.py:368-399` 从 SynthScars `refs[].sentence` 和 `refs[].segmentation` 构造 phrase、token span 和逐-reference GT mask。
3. `dataset/dataset.py:143-212` 无论 `inference` 标志为何，都把完整 conversation tokenized 为 `input_ids`；`inference=True` 只改变截断行为并传入模型。
4. `scripts/loc_exp/train.py:299-327` 仅把 `LegionGCGDataset(validation=True)` 接为 validation dataset，并仅在 `args.mask_validation` 时令 validation collator 使用 `inference=True`。
5. `model/Legion.py:135-176` 在 `inference=True` 时从现成 `input_ids` 中查找 `[SEG]`；`model/Legion.py:178-193` 调用普通 forward，不执行自回归 `generate`。
6. `scripts/loc_exp/train.py:585-639` 用 `prediction > 0`，按 `zip(gt_masks, predicted_masks)` 逐 reference 计算 intersection/union，并输出 `giou`、`ciou`；这里没有把多个 mask union 成一张 per-image mask，也没有计算论文表中的 F1。
7. 官方 `scripts/loc_exp/train.sh:3-22` 没有传 `--mask_validation`；默认 validation 走 loss 分支，而不是上述 mask metric 分支。

所以 L-TFVAL 的“inference”只是代码变量名。按信息流定义，它明确看到了完整 GT assistant target，属于 teacher forcing。它是训练/验证设施，不等于 L-FREE，也不能自动视为论文测试协议。

### 2.3 路径 L-PAPER：README 论文结果表

`README.md:133-139` 声称在 SynthScars、RichHF-18K、LOKI 上评估 localization；嵌入图片表格报告 mIoU 和 F1。但公开 commit 中：

- `scripts/loc_exp/infer.py` 只输出 explanation 和 union `mask.png`，不读 GT、不算指标；
- `scripts/loc_exp/train.py --mask_validation` 只接入 `LegionGCGDataset`，报告 `giou/ciou` 而非 README 表中的 mIoU/F1；
- `eval/utils.py:93-181` 虽定义 `compute_iou`、SynthScars polygon-union helper 和 LOKI bbox-union helper，但全仓库没有调用点；
- `eval/utils.py` 中的 `riou`、`riou_new`、`compute_map` 同样没有调用点；
- 没有 RichHF-18K localization GT loader/evaluator；
- 没有公开脚本说明论文表究竟用 free generation 还是 teacher-forced caption、如何处理无 `[SEG]`、如何从 per-reference mask 得到 per-image mask、F1 的像素/区域定义以及 dataset/image/category aggregation 顺序。

结论：公开代码不能把 L-FREE 或 L-TFVAL 无歧义地连接到 L-PAPER。**不能断言 SynthScars / LOKI / RichHF-18K 使用相同完整流程，也不能用 L-TFVAL 的 giou/ciou 代替论文 mIoU/F1。**

## 3. R1 三种现有条件的实际信息输入

三种条件都以图像为视觉输入；GT mask 只用于 evaluator 的 scoring target，不作为模型输入。区别在于 user prompt、assistant context 和是否生成。

### 3.1 G0

```text
模型输入：image
        + canonical user question
        + empty assistant turn
推理：greedy/free generation
GT authenticity：无
GT artifact phrase：无
GT explanation：无
GT mask：仅 evaluator 使用
```

- canonical question 是 `Determine whether this image is authentic and explain the forensic evidence.`（`dataset/forensics/unified.py:30-37`）。
- `scripts/phase3a_evaluate.py:273-294` 以 `provide_gt_fake=False`、`generation_mode="unified_fake_generate"` 调用 generation，并显式记录 `uses_gt_authenticity=False`、`uses_gt_explanation=False`。
- `eval/forensics_eval.py:486-610` 显示该模式使用空 assistant content，调用 `model.evaluate` 自由生成，再从生成 token 判定 `[SEG]` 和取出预测 mask。

### 3.2 Authoritative Phrase-only

```text
模型输入：image
        + legacy artifact-localization user question
        + GT-authenticity [FAKE]
        + "Target regions: <authoritative GT phrase> [SEG]"
推理：对完整已知 assistant content 做 causal forward，不生成 explanation
GT explanation：无
GT mask：仅 evaluator 使用
```

- `eval/forensics_eval.py:296-307` 从 frozen `normalized_training_phrase` 取得 authoritative phrase，并构造 `[FAKE] Target regions: <phrase> [SEG]`。
- `eval/forensics_eval.py:288-294,463-477` 显示 Phrase-only 走 `_causal_forward`，不是 generate；其默认 user prompt 是 legacy `FORENSICS_QUESTION`。
- 当前 reusable matrix 的 Phrase representation context 也明确使用 legacy `FORENSICS_QUESTION`（`scripts/p1_r1_reusable_matrix.py:189-206`）。这属于保留的 Phrase diagnostic contract，不应误称为 image-only 条件。
- `scripts/phase3a_evaluate.py:319-333` 标记它使用 GT authenticity 和 GT localization phrase，但不使用 GT explanation。

### 3.3 TF-PHRASE（当前 canonical；旧 legacy-prompt TF 已丢弃）

```text
模型输入：image
        + canonical user question
        + [FAKE]
        + 完整 authoritative GT explanation
        + "Target regions: <authoritative GT phrase> [SEG]"
推理：对完整已知 assistant content 做 causal forward
GT mask：仅 evaluator 使用
```

- `eval/forensics_eval.py:309-318` 构造完整 TF-PHRASE assistant target。
- `eval/forensics_eval.py:432-460` 用 canonical user prompt 调 `_causal_forward`，不执行自由生成。
- `scripts/phase3a_evaluate.py:305-318` 显式记录 `uses_gt_authenticity=True` 和 `uses_gt_explanation=True`。
- `scripts/p1_r1_reusable_matrix.py:204-206` 的当前 TF context 使用 `CANONICAL_UNIFIED_QUESTION`。
- `docs/p1_reusable_evaluation_baselines.md:198-207` 规定旧 legacy user-prompt TF 只保留为历史 mismatch，不是当前 baseline；G0 是 deployable primary，Phrase/TF 是 oracle diagnostics。

### 3.4 R1 mask 后处理语义

`scripts/phase3a_evaluate.py:92-110` 将一张图的多个预测 mask logits 取逐像素 `amax`，再以 logit `> 0` 二值化；GT 是该图全部 reference mask 的 union。它在二值语义上等价于“每个 mask `>0` 后 OR”，但 target/aggregation 必须在未来正式对比前另行冻结，不能直接拿历史数字与 LEGION README 表相减。

## 4. 条件映射表

| R1 condition | R1 向模型提供的信息 | LEGION 官方是否存在严格等价条件 | 是否公平可比 | 原因 |
|---|---|---|---|---|
| G0 | image + canonical user prompt；无 GT label/phrase/explanation；自由生成 | **存在同层级条件：L-FREE** | **是，但限于 condition-level parity** | 两者都是 image-only free generation，`[SEG]` 来自生成序列，mask threshold 均为 `>0`，多 mask 都做 union。两者官方/canonical prompt 文本不同；应各自保留默认 prompt，并在共同 target/evaluator 下比较，而不是声称逐 token 相同。 |
| Phrase | image + GT `[FAKE]` + authoritative GT phrase + `[SEG]`；causal forward；无 GT explanation | **不存在** | **否** | LEGION 没有只注入 GT phrase 的官方入口或 evaluator。L-FREE 不看 GT phrase；L-TFVAL 同时看完整 GT caption 和所有 phrase，信息量更大。人为新增 phrase prefix 会成为新协议。 |
| TF | image + GT `[FAKE]` +完整 GT explanation + authoritative GT phrase + `[SEG]`；canonical prompt；causal forward | **仅有近似 L-TFVAL，不是严格等价** | **否（Phase 5A-2）** | L-TFVAL 看完整 tagged GT caption、可能含多个 phrase/`[SEG]`，用 LEGION prompt 和 per-reference mask alignment；R1 TF 用自身 canonical prompt、单一 frozen assistant template 和 per-image union target。L-TFVAL 也不是公开 checkpoint 默认 inference，且未与论文 evaluator 建立连接。 |

“condition-level parity” 不表示两模型 prompt 必须逐字相同。对不同 MLLM 强行替换官方 prompt 本身会改变协议；更稳妥的正式设计是保留各自公开/canonical image-only prompt，同时统一样本、GT target、mask geometry、failure policy、threshold 的解释、per-image/global aggregation和统计方法，并把 prompt 差异作为模型协议差异公开记录。

## 5. 对 Phase 5A-2 的明确建议

采用 **Option A**：

1. LEGION 只运行官方 L-FREE：固定官方 prompt、自由生成、生成 `[SEG]`、SAM mask logit `>0`、多 mask union。
2. 只与 R1 G0 对应；Phrase-only 和 TF-PHRASE 在 LEGION 列标记 `N/A — no official equivalent`，不要补造结果。
3. Phase 5A-2 若要形成数值对比，必须先单独冻结一个共同 evaluator：相同 image set、相同 GT target construction、相同 resize/original-size policy、相同空/无 `[SEG]` failure policy、相同 per-image/global metric definitions。该动作不属于本审计，尚未执行。
4. 公开权重是 intermediate LE checkpoint，所以未来结果只能表述为“public intermediate checkpoint reference/diagnostic comparison”，不能冒充论文 final checkpoint 复现。
5. 不得把 README 表格数值、L-TFVAL 的 giou/ciou 与 R1 历史 FG IoU/F1 直接相减；它们的协议和 aggregation 未证明一致。

## 6. 停止点

Phase 5A-2A 静态审计完成。未进入 Phase 5A-2 推理或 evaluator 实现。
