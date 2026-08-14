# New paired C0 与 P1：分类及 G0 评测汇总

## 1. 内部数据集的来源、构建与最终冻结版本

本文所称 **Internal / 内部数据集**，是 Phase 2A 起冻结使用的 unified forensics dataset；它不包含 official SynthScars test 1000，也不包含 LOKI、RAISE 或 AIGI-test。其构建过程如下。

| 步骤 | 处理 | 结果 |
|---|---|---|
| 1. 构建 Fake 候选集 | 读取 SynthScars 官方 annotation-centric JSON，按 image identity 合并同一图像的多条 annotation 与文件 variant；保留全部 reference phrase，并将该图的所有标注区域合成为一张 union evidence mask | 得到 11,064 个 SynthScars train Fake image identities；每个 Fake 样本具有分类标签、解释文本与有效定位监督 |
| 2. 汇集 Real 候选集 | 固定抽取 OpenImages 8,000、PASS 6,000、COCO 4,000、FFHQ 2,500、iNaturalist 1,000，共 21,500 张；按图像内容 SHA256 精确去重 | 删除 3 个精确重复项，剩 21,497 个 Real candidates |
| 3. 内容分类与配平 | 将图像统一归入 Human、Animal、Object、Scene 四类；PASS 6,000 张由 GPT 内容分类，其中 46 个 ambiguous 样本不参与匹配；以冻结的 SynthScars CLIP 内容预测计数作为 Fake 侧目标，Real 侧在每类内按稳定哈希顺序截取 | 选出 11,064 Real，与 11,064 Fake 在四个内容类别上逐类相等 |
| 4. 排除 official test 泄漏 | pHash 近重复审计发现 18 个内部 train Fake identities 与 official SynthScars test 近重复；保留 official test 不动，删除这 18 个 Fake，并同步删除 18 个同类别 Real 以维持平衡 | 最终内部池为 11,046 Real + 11,046 Fake = 22,092 |
| 5. 分组划分 | 在每个内容类别内按 8:1:1 划分 train / validation / test；同一 pHash connected component 整体进入同一 split，避免近重复图像跨 split | 得到冻结的 train、validation、test manifests |
| 6. 完整性复核 | 检查跨 split 的 sample ID、image name、Real 内容 SHA256 与 pHash connected component，并复核 official test 泄漏 | 上述跨 split 重复均为 0；修复后与 official SynthScars test 的近重复泄漏为 0 |

最终冻结内部池在 Real 与 Fake 两侧分别包含 Human 5,815、Animal 1,525、Object 1,939、Scene 1,767，类别数量逐类完全一致。各 split 内也保持 Real/Fake 及四类内容计数逐类相等。

| 最终内部 split | Real | Fake | 合计 | 阶段用途 |
|---|---:|---:|---:|---|
| train | 8,836 | 8,836 | 17,672 | 模型训练 |
| validation | 1,106 | 1,106 | 2,212 | 训练期间验证与冻结 selector 选模 |
| test | 1,104 | 1,104 | 2,208 | 冻结后的内部测试；不参与训练或选模 |

因此，本报告的 **Internal classification** 在全部 2,208 个 test 样本上评测；**Internal G0** 只在其中 1,104 个具有有效 union evidence mask 的 Fake 样本上评测。Real 样本用于真实性分类，但没有伪造区域定位 GT，不能纳入 G0 定位指标。

构建与冻结依据：`docs/current_pipeline_audit.md`、`outputs/data_audits/content_matched_real_v1/summary.json`、`outputs/data_audits/unified_forensics_split_v1/summary.json`，以及冻结的 `train/val/test_{real,fake}.jsonl` manifests。

## 2. 范围与模型身份

本文只汇总 **new paired C0** 与 **P1**，不把 historical Phase 2A/B0 当作 C0。已有结果直接复用；此前缺失的 new C0 internal G0、new C0 official1000 standalone classification，以及 C0/P1 在 LOKI、RAISE、AIGI-test 上的 classification-only 结果已补齐。所有补充实验均为冻结模型推理，没有训练、选模、阈值扫描或外部集校准。

| 简称 | 模型定义 | 冻结 checkpoint |
|---|---|---|
| C0 | new paired control；训练 target 不含 `Target regions:` phrase supervision | step 2500 / epoch 5；SHA256 `d220490d875107368c2385730da93c72b84d830762813d609f68704dd83aae32` |
| P1 | phrase-supervised model；训练 target 含 authoritative `Target regions:` | step 3500 / epoch 7；SHA256 `fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326` |

## 3. 数据范围

| 数据集 | 范围 | 用途与限制 |
|---|---:|---|
| Internal test | 2,208 = 1,104 Real + 1,104 Fake | classification 使用全部样本；G0 只对 1,104 个有有效 union mask 的 Fake 汇总 |
| Official1000 | 1,000 Fake | official SynthScars test；classification 是单类 Fake 测试，ROC-AUC 不可定义；G0 对全部 1,000 张汇总 |
| LOKI | 2,217 = 900 Real + 1,317 Fake | 复用 Phase 2C 已冻结的去重图像级外部集合 |
| RAISE | 998 Real | 复用 clean held-out manifest；原 1,000 张中 2 张不可解码，故实际 N=998；只适合报告 Real specificity/FPR |
| AIGI-test | 1,731 = 870 Real + 861 Fake | 使用 NPR 专家训练入口的冻结 `datasets/AIGI-Holmes-Dataset/dataset/test.jsonl`；沿用 NPR loader：`mask` 非空判 Fake，否则判 Real |

## 4. 英文缩写说明

| 缩写 | 含义 |
|---|---|
| C0 | Control 0；本报告专指 new paired control，不是 historical Phase 2A/B0 |
| P1 | Phrase-supervised model 1；接受 authoritative region phrase supervision 的模型 |
| CLS | Classification；模型的 fixed-`[CLS]` 专用分类头 |
| LM | Language Model；语言模型的 `[REAL]`/`[FAKE]` verdict 分支 |
| G0 | Grounding protocol 0；canonical unified prompt 下自由自回归生成，不提供 GT authenticity、GT explanation 或 authoritative phrase |
| GT | Ground Truth；真实标注 |
| FG / BG | Foreground / Background；前景 / 背景 |
| IoU | Intersection over Union；交并比 |
| mIoU | mean Intersection over Union；本文 `fg/bg mIoU` 是前景 IoU 与背景 IoU 的均值 |
| F1 | Precision 与 Recall 的调和平均；分类表中 Fake 为正类，定位表中为前景像素 F1 |
| ROC-AUC | Receiver Operating Characteristic – Area Under the Curve；ROC 曲线下面积，衡量排序能力，不依赖单一阈值 |
| FPR | False Positive Rate；假阳性率；在 RAISE 上表示真实图被误判为 Fake 的比例 |
| OOD | Out-of-Distribution；分布外数据 |
| N / N/A | Number of samples / Not Applicable；样本数 / 当前数据范围下不适用 |

## 5. Internal test classification

主分类结果来自 canonical standalone detection。Fake 为正类。

| 模型 | N | CLS Accuracy | CLS Precision | CLS Recall | CLS F1 | CLS ROC-AUC | LM Accuracy | LM F1 | LM ROC-AUC | CLS–LM agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 2,208 | 0.977355 | 0.984375 | 0.970109 | 0.977190 | 0.997506 | 0.975543 | 0.975297 | 0.997709 | 0.996377 |
| P1 | 2,208 | 0.983696 | 0.988117 | 0.979167 | 0.983621 | 0.998389 | 0.984149 | 0.984113 | 0.998407 | 0.996830 |
| P1 − C0 | — | +0.006341 | +0.003742 | +0.009058 | +0.006432 | +0.000883 | +0.008605 | +0.008815 | +0.000697 | +0.000453 |

机器可读来源：

- C0：`outputs/phase3a1_paired_control/evaluation/internal/new_c0/detection/metrics.json`
- P1：`outputs/phase3a_phrase_grounding/evaluation/internal/detection/metrics.json`

## 6. Internal test G0

以下均为 1,104 个 Fake 样本；mask logit threshold 固定为 0.0。

| 模型 | N | mean FG IoU | mean FG F1 | mean fg/bg mIoU | global FG IoU | global FG F1 | global fg/bg mIoU | `[SEG]` trigger rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 1,104 | 0.136540 | 0.193382 | 0.542746 | 0.128014 | 0.226973 | 0.539595 | 0.971014 |
| P1 | 1,104 | 0.166414 | 0.233295 | 0.553925 | 0.162135 | 0.279029 | 0.553051 | 0.983696 |
| P1 − C0 | — | +0.029874 | +0.039913 | +0.011179 | +0.034120 | +0.052056 | +0.013456 | +0.012681 |

机器可读来源：

- C0：`outputs/phase3a1_paired_control/evaluation/internal/new_c0/G0/metrics.json`
- P1：`outputs/phase3a_phrase_grounding/evaluation/internal/G0/metrics.json`

## 7. Official1000 classification

这是 standalone fixed-`[CLS]` detection，而不是从 G0 predictions 反推分类。由于 1,000 张全部为 Fake，Accuracy 等于 Fake recall；缺少 Real 负类，ROC-AUC 为 N/A。

| 模型 | N | CLS Accuracy / Fake recall | CLS F1 | LM Accuracy / Fake recall | LM F1 | CLS–LM agreement |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 1,000 | 0.969000 | 0.984256 | 0.970000 | 0.984772 | 0.993000 |
| P1 | 1,000 | 0.980000 | 0.989899 | 0.976000 | 0.987854 | 0.996000 |
| P1 − C0 | — | +0.011000 | +0.005643 | +0.006000 | +0.003083 | +0.003000 |

机器可读来源：

- C0：`outputs/phase3a1_paired_control/evaluation/g0/new_c0/detection/metrics.json`
- P1：`outputs/phase3a_phrase_grounding/evaluation/official1000/detection/metrics.json`

## 8. Official1000 G0

| 模型 | N | mean FG IoU | mean FG F1 | mean fg/bg mIoU | global FG IoU | global FG F1 | global fg/bg mIoU | `[SEG]` trigger rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 1,000 | 0.179786 | 0.251918 | 0.548158 | 0.175036 | 0.297925 | 0.546963 | 0.968000 |
| P1 | 1,000 | 0.229544 | 0.319323 | 0.571646 | 0.236949 | 0.383118 | 0.576302 | 0.968000 |
| P1 − C0 | — | +0.049758 | +0.067405 | +0.023487 | +0.061913 | +0.085194 | +0.029338 | +0.000000 |

机器可读来源：

- C0：`outputs/phase3a1_paired_control/evaluation/g0/new_c0/G0/metrics.json`
- P1：`outputs/phase3a_phrase_grounding/evaluation/official1000/G0/metrics.json`

## 9. External OOD classification：CLS head

所有外部结果使用相同 canonical prompt、fixed-`[CLS]` evaluator、BF16 inference、batch size 8 和固定 0.5 分类阈值。外部集未参与训练、checkpoint selection 或 threshold calibration。

### 9.1 LOKI

| 模型 | N | Accuracy | Precision | Fake recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 2,217 | 0.556608 | 0.757716 | 0.372817 | 0.499746 | 0.641226 |
| P1 | 2,217 | 0.543076 | 0.751656 | 0.344723 | 0.472670 | 0.653859 |
| P1 − C0 | — | −0.013532 | −0.006060 | −0.028094 | −0.027075 | +0.012634 |

P1 在固定 0.5 阈值下的 Accuracy/F1/recall 低于 C0，但 ROC-AUC 更高；这表示排序能力与固定阈值 operating point 给出不同方向，不能只报其中一项。

### 9.2 RAISE clean held-out

| 模型 | N | Real specificity | FPR | CLS–LM agreement |
|---|---:|---:|---:|---:|
| C0 | 998 | 0.988978 | 0.011022 | 0.995992 |
| P1 | 998 | 0.993988 | 0.006012 | 0.998998 |
| P1 − C0 | — | +0.005010 | −0.005010 | +0.003006 |

RAISE 全部为 Real，因而不能计算有意义的 Fake precision/recall/F1 或 ROC-AUC。P1 比 C0 少 5 个 false positives（C0=11，P1=6）。

### 9.3 AIGI-test

| 模型 | N | Accuracy | Precision | Fake recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 1,731 | 0.703062 | 0.716065 | 0.667828 | 0.691106 | 0.789395 |
| P1 | 1,731 | 0.733102 | 0.739496 | 0.715447 | 0.727273 | 0.818418 |
| P1 − C0 | — | +0.030040 | +0.023431 | +0.047619 | +0.036167 | +0.029023 |

P1 在 AIGI-test 的所有列示 CLS 指标上均高于 C0。

## 10. External OOD classification：LM verdict 诊断

| 数据集 | 模型 | LM Accuracy | LM F1 | LM ROC-AUC | CLS–LM agreement |
|---|---|---:|---:|---:|---:|
| LOKI | C0 | 0.550293 | 0.486876 | 0.648983 | 0.981958 |
| LOKI | P1 | 0.548038 | 0.484037 | 0.649589 | 0.981507 |
| RAISE | C0 | 0.988978 | N/A（全 Real） | N/A | 0.995992 |
| RAISE | P1 | 0.992986 | N/A（全 Real） | N/A | 0.998998 |
| AIGI-test | C0 | 0.705950 | 0.691702 | 0.789505 | 0.980936 |
| AIGI-test | P1 | 0.725014 | 0.720657 | 0.811107 | 0.983824 |

机器可读来源与逐样本记录：

- C0 summary：`outputs/phase3a1_paired_control/evaluation/external_classification/new_c0/summary.json`
- P1 summary：`outputs/phase3a1_paired_control/evaluation/external_classification/p1/summary.json`
- 各数据集逐样本：上述目录下的 `loki/`、`raise/`、`aigi_test/` 中 `predictions.jsonl`

## 11. 汇总结论与边界

1. Internal classification：P1 的 CLS Accuracy 比 C0 高 0.006341；Internal G0 mean FG IoU 高 0.029874。
2. Official1000：P1 的 standalone CLS Fake recall 高 0.011000，G0 mean FG IoU 高 0.049758。
3. External classification：P1 在 RAISE specificity 和 AIGI-test 的 Accuracy/F1/ROC-AUC 上更高；LOKI 则呈现 fixed-threshold Accuracy/F1 下降但 ROC-AUC 上升的混合结果，不能概括为全面 OOD 提升。
4. External 实验是 classification-only；不能据此推断 LOKI、RAISE 或 AIGI-test 上的 localization/G0 表现。
5. 以上是冻结 checkpoint 的观测比较。matched new C0 缩小了训练轨迹混淆，但原始 P1 step-0 snapshot 不可恢复的证据边界仍然存在。
