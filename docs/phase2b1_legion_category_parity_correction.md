# Phase 2B.1：LEGION 类别级对齐修正

## 1. 修正原因

Phase 2B 正确识别了指标定义不一致的问题，但后续有一处比较混用了不同的类别范围。Phase 2A 的 official SynthScars test 结果是在全部 1000 张图像上计算的，而 LEGION Table 2 分别报告四种内容类别。本补充保留历史实验记录，同时修正这一推论；不进行训练、模型修改、checkpoint 重选、threshold sweep 或 proxy relabeling。

**修正后状态：`CATEGORY_PARITY_UNRESOLVED`（类别级对齐尚未解决）。**

## 2. Phase 2B 中的具体错误

Phase 2B 报告了 Phase 2A G0 的 global fg/bg mIoU `0.545930` 和 global foreground F1 `0.295168`，随后直接减去 LEGION 的 `0.5462/0.2990`，把差值描述为 −0.027/−0.383 个百分点，并称其为“最接近的可观测候选差距约为零”。

Phase 2A 这一行的范围是 `official test / overall / n=1000 / 全局像素聚合 / mask logit > 0`；LEGION 这一行的范围则是 `official test / Object / n=162 / 未公开的精确聚合与阈值`。类别、样本量、预测表示、聚合方式和阈值都不兼容，因此撤回该减法及其“模型差距接近零”的解释。

同一问题还影响了 internal-overall `0.544256` 与 LEGION Object `0.5462` 的比较、Phase 2C 继承的“接近”表述，以及历史 future-hypothesis 摘要。所有位置均记录在 `outputs/phase2b1_legion_category_correction/comparison_scope_audit.json`。

## 3. 仍然成立的结论

本次修正不推翻以下 Phase 2B 发现：

- Phase 2A 平均前景 IoU 约 0.1396，而 LEGION 报告的是前景/背景 mIoU；二者属于不同指标族，不能直接比较。原先看起来约 40 个百分点的差距被指标不一致严重放大。
- official SynthScars test 1000 与 Phase 2A train、validation、test 的 SHA256 重叠均为零，因此 official evaluation 不存在训练污染。
- internal TF→G0 的平均前景 IoU 差距约为 0.2045。
- `[SEG]` 触发率约为 99%，不是主要瓶颈。
- 单一 union-mask 监督与 phrase-level multi-mask 监督确实存在协议差异。

需要撤回的结论更窄：Phase 2A overall 与 LEGION Object 数值接近，不能证明真实模型差距接近零。

## 4. LEGION Table 2 的范围

[LEGION 官方仓库](https://github.com/opendatalab/LEGION)说明 SynthScars 包含四类内容，并链接到官方数据发布页。[ICCV 2025 论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Kang_LEGION_Learning_to_Ground_and_Explain_for_Synthetic_Image_Detection_ICCV_2025_paper.pdf)的 Table 2 分别给出四组 SynthScars 指标：

| 内容类别 | N | LEGION mIoU | LEGION F1 |
|---|---:|---:|---:|
| Object | 162 | 54.62 | 29.90 |
| Animal | 134 | 54.52 | 27.43 |
| Human | 587 | 60.82 | 39.44 |
| Scene | 117 | 53.67 | 24.51 |

样本数来自官方 supplement 的内容统计表。Table 2 没有报告 SynthScars 的 overall localization 指标。

## 5. Phase 2A official-1000 的范围

Phase 2A 使用冻结的 step-2500 checkpoint、官方发布的 1000 张 test split、union GT 和固定阈值 `mask_logit > 0`。已保存的总体结果如下：

| 模式 | N | Global fg/bg mIoU | Global FG F1 | 逐图 fg/bg mIoU | 逐图 FG F1 |
|---|---:|---:|---:|---:|---:|
| G0 | 1000 | 0.545930 | 0.295168 | 0.551843 | 0.268184 |
| G1 | 1000 | 0.545998 | 0.296476 | 0.552747 | 0.271495 |
| TF | 1000 | 0.709897 | 0.644910 | 0.675056 | 0.531387 |

这些数值全部是 **overall**，不是 Object 结果。

## 6. 原减法为什么无效

只有 dataset、split、category、metric family、target representation、aggregation 和 threshold 全部一致时，数值差才有明确含义。此处 `overall != Object`、`1000 != 162`，全局像素聚合也不等于未公开的 Table 2 聚合方式；Phase 2A 的单一 union mask 也尚未证明与 LEGION 的评测单元等价。数值处于相近量级，只能说明按 fg/bg 风格重算后进入了相似区间，不能识别受类别控制的模型差距。

`eval/comparison_scope.py` 现在提供 `ComparisonScope`、`assert_comparable_scope` 和 `scoped_delta`。overall-vs-Object、不同 split、不同 aggregation，以及将 proxy 冒充 official 的比较都会抛出 `ComparisonScopeError`。

## 7. 官方内容类别映射审计

审计范围包括 [SynthScars Hugging Face 官方仓库](https://huggingface.co/datasets/khr0516/SynthScars)、完整发布压缩包、annotations、目录结构、LEGION 官方仓库及其全部 27 个公开 commit、论文、supplement、公开评测代码和本地 official manifest。

- Hugging Face 的全部历史仅包含 `.gitattributes`、`README.md` 和 `SynthScars.zip`。
- 官方 test archive 采用扁平的 `test/images` 和 `test/annotations/test.json`，没有类别目录。
- 1000 条 test record 只有 `caption`、`img_file_name` 和 `refs` 字段。
- ref 只有 `bbox`、`explanation`、`segmentation` 和 `sentence` 字段。
- 因此 official manifest 中的 `content_category` 为 `null`。
- 论文和 supplement 只给出总体类别数量，没有 image-ID mapping。
- 官方仓库历史和评测代码中也没有找到 mapping、list 或逐图 content 字段。

因此审计结果为：`AUTHORITATIVE_PER_IMAGE_CONTENT_MAPPING_UNAVAILABLE`。已有的 CLIP/GPT 内容标签被明确排除，因为它们属于 proxy prediction；也禁止使用 filename、caption、人工看图或不相交的 internal-1104 标签进行推测。

## 8. 类别级重新评分

本阶段没有执行类别级重新评分。缺少 authoritative image-to-category mapping 时，把 1000 个已保存的 G0/G1/TF prediction 分配给四类，会人为制造比较目标。因此没有生成 `official1000_content_mapping.jsonl` 和 `phase2a_category_metrics.json`；相关 mapping integrity test 按要求使用明确原因 skip。

Internal fake test 的标签数量为 Human 581、Animal 152、Object 194、Scene 177，共 1104 张；它与 official-1000 的 SHA256 overlap 为零，不能作为替代。

## 9. 派生的计数加权诊断

使用论文给出的类别数量，并假设各类别指标可以按图像数线性相加，可得到：

- 派生 count-weighted mIoU：**58.13485%**
- 派生 count-weighted F1：**34.53837%**

该结果命名为 `LEGION_COUNT_WEIGHTED_CATEGORY_DIAGNOSTIC`，并保存 `derived=true`。它 **不是** 论文报告的 overall metric。如果 Table 2 使用非线性、按 class/pixel 或其他聚合方式，按数量加权不一定等于真正的 overall 结果。

## 10. 修正后的比较表

| 内容类别 | N | Phase 2A G0 mIoU 候选 | Phase 2A G0 F1 候选 | LEGION mIoU | LEGION F1 | 差值状态 |
|---|---:|---:|---:|---:|---:|---|
| Object | 162 | 不可得 | 不可得 | 54.62 | 29.90 | 无法计算 |
| Animal | 134 | 不可得 | 不可得 | 54.52 | 27.43 | 无法计算 |
| Human | 587 | 不可得 | 不可得 | 60.82 | 39.44 | 无法计算 |
| Scene | 117 | 不可得 | 不可得 | 53.67 | 24.51 | 无法计算 |
| Overall | 1000 | 54.5930 global / 55.1843 逐图 | 29.5168 global / 26.8184 逐图 | 未报告 | 未报告 | 无法计算 |

本阶段没有选择对 Phase 2A 更有利的候选列，也没有输出任何类别差值。

## 11. 修正后的科学结论

确定结论一：Phase 2A foreground IoU 与 LEGION fg/bg mIoU 的直接比较无效，原先约 40 个百分点的表观差距很大程度来自指标定义不一致。确定结论二：Phase 2A overall 的 fg/bg 风格结果与 LEGION 各类别结果处于相同数量级。尚未解决的问题是：Phase 2A 与 LEGION 在相同内容类别和相同精确聚合方式下究竟相差多少。

项目后续不能再表述为 Phase 2A 与 LEGION 完全一致、只低 0.027 个百分点，或模型差距约为零。目前有证据支持的说法是：指标对齐显著缩小了表观数值差距，但精确类别级对齐仍未解决。

## 12. 对 Phase 2C 的影响

本次修正不改变任何 Phase 2C 实验。Phase 2A 仍冻结在 step 2500；B1/B2/B3 checkpoint、validation-loss selector、internal test、external evaluation、McNemar 统计、calibration、invariance 和 NPR/SRM 结论全部保持不变。Phase 2C 独立证明了冻结 forensic residual fusion 能带来小幅分类增益，其训练和模型选择从未使用 LEGION localization 差值。

## 13. 未解决问题与最终回答

1. Phase 2A `0.545930/0.295168` 是 overall；LEGION `0.5462/0.2990` 是 Object。
2. 原减法无效，其 near-zero-gap 解释已经撤回。
3. Metric mismatch、zero split overlap、TF→G0 gap、高 `[SEG]` trigger rate 和监督单元差异仍然成立。
4. 没有找到 official-1000 的 authoritative per-image content label；官方 aggregate count 的确精确加和为 1000。
5. 在不制造标签的前提下，无法计算 Phase 2A 四类 G0 结果和类别候选差值。
6. Table 2 的精确聚合仍不可得，所以 exact metric parity 同样尚未解决。
7. 派生的 58.13485% mIoU 和 34.53837% F1 依赖线性加权假设，不是 LEGION official overall。
8. 本次修正既不改变 Phase 2A checkpoint，也不改变 Phase 2C 训练或模型选择。
9. Phase 2C forensic enhancement 的结论仍可独立成立。

全部产物位于 `outputs/phase2b1_legion_category_correction/`，测试位于 `tests/test_phase2b1_category_scope.py`。本阶段没有启动训练或推理。
