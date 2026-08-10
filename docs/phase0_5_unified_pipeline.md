# Phase 0.5：统一取证训练流水线接入与 P0 修复

## 1. 范围与结论

本阶段已完成 frozen Real/Fake manifests 到 GLaMM 的统一数据接入、`fixed_cls_query` token flow、逐样本 segmentation 监督门控、mixed batch 修复和取证评估字段接入。

本阶段没有启动正式长训练，没有接入 NPR/SRM，也没有实现 consistency loss。

## 2. 新 token flow

训练和推理新增 `token_strategy=fixed_cls_query`。token 名称仍为 `[CLS]`，但语义改为固定分类查询：

1. tokenizer 将 `[CLS]`、`[REAL]`、`[FAKE]` 注册为 special tokens；`[REAL]` 和 `[FAKE]` 均强制检查为严格一个 token。
2. collate 在最后一个裸 `ASSISTANT:` prefix 后插入 `[CLS]`，因此 `[CLS]` 始终存在于 `input_ids`。
3. `[CLS]` 对应 `labels` 固定为 `-100`，不参与 causal LM CE。
4. classification head 读取最后层 `[CLS]` hidden state，并输出 Real/Fake 二分类 logits。
5. 同一个 `[CLS]` hidden 对应的 LM next-token logits 仅选取 `[REAL]`、`[FAKE]` 两个 token，再在这两个 logits 上做 softmax，得到 `lm_verdict_prob`。
6. generation prompt 自带 `[CLS]`；`fixed_cls_query` 不设置 forced decoder token，也不再生成第二个 `[CLS]`。

兼容入口仍保留旧的 `generated_cls` strategy，但统一取证训练入口默认使用 `fixed_cls_query`。

## 3. 固定 target grammar

Fake target：

```text
[CLS] [FAKE] <artifact explanation> [SEG]
```

示例：

```text
ASSISTANT: [CLS] [FAKE] A localized blending artifact is visible around the edited region. [SEG]
```

Real target：

```text
[CLS] [REAL] No identifiable synthetic artifact evidence is detected.
```

对应 labels 示例（token id 仅为当前本地 tokenizer 快照示例，不是跨 checkpoint 常量）：

```text
input token:  ...  [CLS]   [REAL]  No  identifiable ...
input id:          32007    32008   ...
label:              -100    32008   ...

input token:  ...  [CLS]   [FAKE]  explanation ... [SEG]
input id:          32007    32009   ...          32004
label:              -100    32009   ...          32004
```

因此 `[REAL]`/`[FAKE]`、explanation 和 Fake 的 `[SEG]` 进入 LM CE；`[CLS]` 不进入 LM CE；`[SEG]` 继续使用 GLaMM 原有 language-to-mask decoder 路径。

## 4. UnifiedForensicsDataset 与 preprocessing flow

`UnifiedForensicsDataset` 读取冻结的 `{train,val,test}_combined.jsonl`，不重新划分数据。每个样本输出 `cls_label`、`seg_valid`、来源元数据以及 GLaMM 所需图像张量。

Real 与 Fake 使用完全相同的两路图像预处理：

```text
原始 RGB 图像
├── global encoder: CLIPImageProcessor -> pixel_values
└── grounding encoder: ResizeLongestSide -> SAM normalize -> pad 到 1024×1024
```

监督语义：

- Fake：`cls_label=1`，`seg_valid=True`；从 frozen SynthScars evidence references 读取 polygon，重建 union evidence mask。缺失或空 evidence 被视为数据错误。
- Real：`cls_label=0`，`seg_valid=False`；`masks=None`，不再把 zero mask 当成有效 GT mask。

Real/Fake 的 global encoder 和 grounding encoder 输入路径、归一化及尺寸处理完全一致，域差异只保留在标签、文本 target 和有效 evidence mask 上。

## 5. mixed batch collate

collate 不再通过 `batch[0]` 决定整个 batch 的 mask 或 grounding 语义。`seg_valid`、`masks`、原始尺寸和 resize 状态均逐样本保留；可堆叠的图像张量只在全体样本均存在时统一 stack，部分缺失会显式报错。

对一个实际 Real + Fake mixed batch 的检查结果：

| 字段 | shape / 表示 |
|---|---|
| `global_enc_images` | `[2, 3, 336, 336]` |
| `grounding_enc_images` | `[2, 3, 1024, 1024]` |
| `input_ids` | `[2, 154]` |
| `labels` | `[2, 154]` |
| `cls_labels` | `[2]`，值为 `[0, 1]` |
| `seg_valid` | `[2]`，值为 `[False, True]` |
| `masks_list[0]` | `None`（Real） |
| `masks_list[1]` | `[1, H, W]`（Fake union evidence） |

实际样本 preprocessing smoke check：global tensor 为 `[3,336,336]`，grounding tensor 为 `[3,1024,1024]`；Fake mask 为 `[1,768,768]`，Real mask 为 `None`。

## 6. loss flow

```text
固定 [CLS] hidden
├── classification head -> cls_loss（有 cls_label 的样本）
└── LM head 的 [REAL]/[FAKE] 两个 next-token logits -> 仅用于 verdict evaluation

语言 token labels
└── causal LM CE；排除 [CLS]，保留 verdict/explanation/[SEG]

[SEG] hidden -> mask decoder -> pred masks
└── 逐样本检查 seg_valid
    ├── False / Real：跳过 BCE 与 Dice
    └── True / Fake：要求 GT 和 pred mask 数量严格一致，再计算 BCE + Dice
```

异常策略：

- Fake 缺少 GT mask：立即抛出错误。
- Fake 缺少必要 `[SEG]`/pred mask：增加 `seg_missing_pred_count`，发出明确 warning，并跳过该样本的 mask loss。
- Fake 的 pred/GT mask 数量不一致：增加 `seg_count_mismatch_count`，发出明确 warning；不允许 `min(pred, gt)` 静默截断。
- Real 意外产生 pred mask：不计算 BCE/Dice，并记录 `seg_unexpected_pred_count`。
- 模型保留累计 missing/mismatch counters，训练日志同步记录每 batch counter。

## 7. evaluation 输出

取证 validation 会写入：

```text
<log_dir>/forensics_predictions_epoch_<epoch>.jsonl
```

每条记录包含：

- `cls_pred`
- `cls_prob`（classification head 的 Fake 概率）
- `lm_verdict_pred`
- `lm_verdict_prob`（只对 `[CLS]` 位置的 `[REAL]`、`[FAKE]` 两个 LM logits 做 softmax 后的 Fake 概率）
- `cls_lm_agree`

还会保留 sample/image/source/content/GT 字段，便于后续按数据域和 content group 汇总。当前只记录一致性，不增加 consistency loss。

## 8. 修改文件列表

- `dataset/forensics/unified.py`：新增 frozen-manifest unified dataset wrapper。
- `dataset/forensics/__init__.py`：导出 unified dataset。
- `dataset/forensics/real_images.py`：Real mask 语义改为 `None` / invalid。
- `dataset/dataset.py`：fixed CLS 插入与 label masking；逐样本 mixed collate。
- `tools/utils.py`：新增 `[REAL]`、`[FAKE]` 常量。
- `model/GLaMM.py`：fixed CLS classification、LM verdict、`seg_valid` loss、异常 counters 和 generation flow。
- `train.py`：注册 special tokens、接入 unified train/val dataset、保存 tokenizer、记录取证评估 JSONL。
- `eval/forensics.py`：构造统一取证 prediction records。
- `app.py`：fixed CLS inference prompt 和 verdict 展示。
- `eval/gcg/infer.py`、`eval/region_captioning/infer.py`：fixed-query checkpoint 的 prompt 兼容。
- `tests/test_real_images_adapter.py`：Real invalid-mask 语义断言。
- `tests/test_unified_forensics_pipeline.py`：Phase 0.5 smoke tests。

## 9. smoke test 结果

执行环境：`glamm_official` conda environment。

```text
python -m unittest discover -v tests
Ran 34 tests in 0.266s
OK
```

Phase 0.5 要求的检查均已覆盖并通过：

- tokenizer special token test：通过；`[REAL]`/`[FAKE]` 各严格一个 token。
- dataset Real sample test：通过；固定 Real grammar、`cls_label=0`、`seg_valid=False`、`masks=None`。
- dataset Fake sample test：通过；固定 Fake grammar、`cls_label=1`、`seg_valid=True`、union evidence mask 非空。
- mixed batch collate test：通过；逐样本 mask/`seg_valid` 保留，`[CLS]` label 为 `-100`。
- single-batch forward test：通过；classification 和 LM verdict 输出 shape 正确，可反向传播。
- Real segmentation skip test：通过；BCE/Dice 均为零。
- Fake segmentation gradient test：通过；BCE/Dice 对 pred mask 产生梯度；数量不一致发 warning/counter 且不截断。
- generation test：通过；prompt 自带 `[CLS]`，未设置 forced `[CLS]`。
- tokenizer/model save-load test：通过；special token ids 和 `token_strategy` 保存/加载后保持一致。

附加检查：所有修改文件通过 `py_compile`，`git diff --check` 无 whitespace error；使用真实 frozen Real/Fake manifests 完成 dataset 和 mixed-collate preprocessing smoke check。未运行正式训练。
