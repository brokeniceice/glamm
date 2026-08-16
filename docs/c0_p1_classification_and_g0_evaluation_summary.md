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
| LOKI classification | 2,217 = 900 Real + 1,317 Fake | 复用 Phase 2C 已冻结的去重图像级外部集合 |
| LOKI localization | 229 Fake | LEGION Table 2 使用的 fully-synthetic valid task scope；来自 LOKI `open_ended_vqa.json`，GT 是 regional bounding boxes 填充后取 union，不是像素级人工 mask |
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
| BBox | Bounding Box；边界框；LOKI localization GT 由 `[x,y,w,h]` 区域框填充为矩形 mask |
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

## 9. LOKI external localization：G0 自由生成与 G1 known-Fake

### 9.1 数据身份与 GT 口径

此前 classification 使用的 2,217 张图像与 Table 2 localization 的 229 张图像来自同一个官方 LOKI 数据集，但属于不同 task JSON 和不同评测范围，并非严格子集关系：两者有 94 张图重叠，另有 135 张 localization 图像不在 `true_or_false.json` 的 classification scope 中。本地完整媒体包已包含 229/229 张 localization 图片，无需重复下载；缺失的是官方 UTF-16 `open_ended_vqa.json`，现已补入原 LOKI 实体目录，SHA256 为 `578ce8e551e9815480896796ffce236acda7f0a55f96b887c6fefcbee87a9db8`。

LOKI 整体是一个包含 image、video、3D、text、audio 五种模态、26 个细分类别和约 18K 道题目的多模态 benchmark；这里的 18K 是跨模态、跨题型的**问题数**，不是图像数。其 image 模态公开了 7 个子类，共 2,217 张用于真假判断的唯一图片；每张图片对应两道语义相反的 True/False 问题，因此 `true_or_false.json` 有 4,434 行。分类集合构成为：

| 图像子类 | Real | Fake | 合计 |
|---|---:|---:|---:|
| Animal | 115 | 284 | 399 |
| Object | 58 | 196 | 254 |
| Person | 120 | 120 | 240 |
| Scene | 128 | 128 | 256 |
| Satellite | 107 | 214 | 321 |
| Medical | 200 | 200 | 400 |
| Document | 172 | 175 | 347 |
| **合计** | **900** | **1,317** | **2,217** |

LEGION Table 1 将 LOKI 的定位范围定义为 229 个“由常见生成器完全合成、写实风格且具有有效区域标注”的 valid samples，Table 2 再将其用于跨域 localization。这 229 张不是从上述 2,217 张分类集合中随机抽取的子集，而是 LOKI 单独发布的 `open_ended_vqa.json` task scope：全部为 Fake，每张图均有人类异常解释和 regional bounding boxes；合计 687 个区域框。官方两个 task JSON 只有 94 张图片路径相交。

LOKI 没有发布像素级人工 artifact masks。严格沿用 LEGION 公布的 `generate_loki_mask()`：将每个 `problems.regional[].region=[x,y,w,h]` 填充为矩形，再对同一图片的矩形取 union。本轮从 229 张图的 687 个区域框生成 229 张非空 bbox-union masks，保存在原路径 `datasets/LOKI/legion_localization/masks/`。因此下列结果应称为 **LOKI bbox-derived localization**，不能描述为像素级 mask localization。

### 9.2 与 LEGION Table 2 对齐的指标口径（推理协议未对齐）

Table 2 的 `mIoU` 使用全数据集累计像素后计算前景/背景 IoU 均值，`F1` 为全局前景像素 F1；表中统一使用百分数。

| 模型 / 协议 | N | global fg/bg mIoU (%) | global FG F1 (%) | 相对 LEGION mIoU | 相对 LEGION F1 |
|---|---:|---:|---:|---:|---:|
| New C0 / 本项目 G0 | 229 | 43.02 | 2.72 | −5.64 | −13.99 |
| P1 / 本项目 G0 | 229 | 43.22 | 3.09 | −5.44 | −13.62 |
| LEGION / Table 2 定位协议 | 229 | 48.66 | 16.71 | — | — |

本轮使用相同的公开 229-image scope、bbox rasterization 语义和最接近 Table 2 文字定义的 global-pixel aggregation，但**推理协议没有对齐**：LEGION 公布的 localization inference prompt 直接要求模型分析 artifacts 并输出 interleaved segmentation masks，不包含真实性判断；本项目 G0 则使用 unified authenticity-and-explanation prompt，让 LM 自由生成 `[REAL]` 或 `[FAKE]`。因此上表只能用于显示同一 GT/指标下的数值位置，不能称为 Table 2 的直接模型性能复现，也不能把差值解释为纯 localization 差距。

### 9.3 本项目完整 G0 指标

| 模型 | N | mean FG IoU | mean FG F1 | mean fg/bg mIoU | global FG IoU | global FG F1 | global fg/bg mIoU | `[SEG]` trigger rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 229 | 0.026546 | 0.044674 | 0.422165 | 0.013794 | 0.027212 | 0.430236 | 0.366812 |
| P1 | 229 | 0.034444 | 0.057179 | 0.429338 | 0.015674 | 0.030865 | 0.432244 | 0.375546 |
| P1 − C0 | — | +0.007898 | +0.012505 | +0.007173 | +0.001881 | +0.003653 | +0.002008 | +0.008734 |

逐图 paired mean FG IoU 差为 `+0.007898`，bootstrap 95% CI=`[-0.000445,+0.016429]`，win/tie/loss=`51/147/31`，Wilcoxon p=`0.015516`。大量 ties 主要来自两模型在同一批样本上都没有产生有效定位 mask。

这里必须区分“评测集合已知为 Fake”和“把 Fake 条件提供给模型”。本项目冻结的 G0 定义是：只在具有定位 GT 的 Fake 样本上统计，忽略独立 CLS head 的 classification gate，但仍使用 unified prompt 让 LM 自由生成真实性 verdict；它不向模型提供 GT Fake prefix。模型生成 `[REAL]` 时会按训练语法直接结束，生成 `[FAKE]` 时才继续解释并产生 `[SEG]`。逐样本生成路由如下：

| 模型 | GT Fake 总数 | LM 生成 `[REAL]` | LM 生成 `[FAKE]` | 触发 `[SEG]` | 总体 trigger | 条件 trigger：`P([SEG] \| [FAKE])` | `[FAKE]` 但无 `[SEG]` |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 229 | 144 | 85 | 84 | 36.68% | 98.82% | 1（重复生成至 400-token 上限） |
| P1 | 229 | 142 | 87 | 86 | 37.55% | 98.85% | 1（重复生成至 400-token 上限） |

因此，这个低 trigger 是本项目 G0 的“自由 LM verdict + localization”耦合结果：229 张 GT 全为 Fake，但 C0/P1 分别有 144/142 张被 LM 判成 `[REAL]`，所以没有 mask。条件于已经生成 `[FAKE]`，两者的 `[SEG]` 触发成功率都接近 99%；真正的“已判 Fake 但未触发 `[SEG]`”各只有 1 张。它不能用来断言 segmentation branch 自身只有约 37% 的触发能力。

项目历史上的 G1 使用相同 unified prompt 并加入结构性 GT `[FAKE]` continuation prefix，才对应“模型已知该图为 Fake、随后自由生成解释和 `[SEG]`”的内部协议；TF 则进一步提供 GT explanation/phrase。常见的 fake-only localization 以及 LEGION Table 2 更接近这种已知伪造条件下的定位问题。若要补充与当前 C0/P1 架构最公平的 known-Fake localization，应在同一 229-image scope 上报告 G1；若要声称严格复现 LEGION inference protocol，还需另行审计并对齐其 artifact-localization prompt，不能直接把现有 G0 改名代替。

### 9.4 G1 known-Fake 补充实验

G1 与上述 G0 使用完全相同的 229 张图片、bbox-union GT、checkpoint、预处理、BF16、greedy decoding、400 max-new-tokens 和 `mask logit > 0` threshold；唯一协议变化是给模型加入结构性 GT `[FAKE]` continuation prefix。结果未覆盖 G0，分别保存在各模型的 `G1/` 目录。

#### 9.4.1 与 Table 2 对齐的 global-pixel 指标

| 模型 / 协议 | N | global fg/bg mIoU (%) | global FG F1 (%) | `[SEG]` trigger | 相对 LEGION mIoU | 相对 LEGION F1 |
|---|---:|---:|---:|---:|---:|---:|
| New C0 / G1 | 229 | 42.90 | 9.54 | 96.51% | −5.76 | −7.17 |
| P1 / G1 | 229 | 42.54 | 14.39 | 96.07% | −6.12 | −2.32 |
| LEGION / Table 2 定位协议 | 229 | 48.66 | 16.71 | N/A | — | — |

G1 显著恢复了定位触发和前景指标，但两个方向不同：相对各自 G0，C0/P1 的 global FG IoU 分别从 0.013794/0.015674 升至 0.050079/0.077554，global FG F1 从 2.72%/3.09% 升至 9.54%/14.39%；与此同时 global BG IoU 从 0.846678/0.848813 降至 0.807960/0.773340。前景改善与背景下降在 fg/bg 均值中互相抵消，所以 global fg/bg mIoU 没有随 F1 上升，分别由 43.02%/43.22% 变为 42.90%/42.54%。这不是指标计算矛盾，而是 G1 预测更多前景时同时增加了 false-positive foreground pixels。

#### 9.4.2 本项目完整 G1 指标

| 模型 | N | mean FG IoU | mean FG F1 | mean fg/bg mIoU | global FG IoU | global FG F1 | global fg/bg mIoU | `[SEG]` trigger rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 229 | 0.054904 | 0.090162 | 0.422623 | 0.050079 | 0.095381 | 0.429019 | 0.965066 |
| P1 | 229 | 0.076893 | 0.126401 | 0.425346 | 0.077554 | 0.143945 | 0.425447 | 0.960699 |
| P1 − C0 | — | +0.021989 | +0.036239 | +0.002723 | +0.027476 | +0.048564 | −0.003572 | −0.004367 |

C0 的 8 个未触发样本全部为重复生成至 400-token 上限；P1 的 9 个未触发样本包括 6 个重复生成和 3 个非法 special-token trajectory。G1 消除了大多数 G0 `[REAL]` 路由失败，但不能消除长文本重复或生成中途重新出现 `[REAL]` 等语言解码失败。

P1−C0 的逐图 paired mean FG IoU 差为 `+0.021989`，bootstrap 95% CI=`[+0.004666,+0.039517]`，win/tie/loss=`127/35/67`，Wilcoxon p=`0.000253`；mean FG F1 差为 `+0.036239`，95% CI=`[+0.011988,+0.060807]`。mean fg/bg mIoU 差仅 `+0.002723`，95% CI=`[-0.009393,+0.014582]`，不能据此确认 P1 在两类平均指标上优于 C0。

G1 相对 G0 的逐图 mean FG IoU 改善为：C0 `+0.028359`，95% CI=`[+0.018293,+0.040784]`，win/tie/loss=`108/95/26`；P1 `+0.042450`，95% CI=`[+0.028426,+0.057245]`，win/tie/loss=`120/66/43`。这确认 LOKI G0 的低值有很大一部分来自自由真实性 verdict 未进入定位路径；但 G1 仍不是 LEGION prompt 的逐字节复现，不能把剩余差距归因于单一模型因素。

G1 机器可读来源：

- C0：`outputs/phase3a1_paired_control/evaluation/external_g0/loki/new_c0/G1/`
- P1：`outputs/phase3a1_paired_control/evaluation/external_g0/loki/p1/G1/`
- G0/G1 配对统计：`outputs/phase3a1_paired_control/evaluation/external_g0/loki/g1_comparison.json`
- 229 条逐样本四轨迹对照：`outputs/phase3a1_paired_control/evaluation/external_g0/loki/g1_per_sample_comparison.jsonl`

G0 机器可读来源：

- 数据审计：`datasets/LOKI/legion_localization/summary.json` 与 `manifest.jsonl`
- C0：`outputs/phase3a1_paired_control/evaluation/external_g0/loki/new_c0/G0/metrics.json`
- P1：`outputs/phase3a1_paired_control/evaluation/external_g0/loki/p1/G0/metrics.json`
- 配对比较：`outputs/phase3a1_paired_control/evaluation/external_g0/loki/comparison.json`

## 10. External OOD classification：CLS head

所有外部结果使用相同 canonical prompt、fixed-`[CLS]` evaluator、BF16 inference、batch size 8 和固定 0.5 分类阈值。外部集未参与训练、checkpoint selection 或 threshold calibration。

### 10.1 LOKI

| 模型 | N | Accuracy | Precision | Fake recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 2,217 | 0.556608 | 0.757716 | 0.372817 | 0.499746 | 0.641226 |
| P1 | 2,217 | 0.543076 | 0.751656 | 0.344723 | 0.472670 | 0.653859 |
| P1 − C0 | — | −0.013532 | −0.006060 | −0.028094 | −0.027075 | +0.012634 |

P1 在固定 0.5 阈值下的 Accuracy/F1/recall 低于 C0，但 ROC-AUC 更高；这表示排序能力与固定阈值 operating point 给出不同方向，不能只报其中一项。

### 10.2 RAISE clean held-out

| 模型 | N | Real specificity | FPR | CLS–LM agreement |
|---|---:|---:|---:|---:|
| C0 | 998 | 0.988978 | 0.011022 | 0.995992 |
| P1 | 998 | 0.993988 | 0.006012 | 0.998998 |
| P1 − C0 | — | +0.005010 | −0.005010 | +0.003006 |

RAISE 全部为 Real，因而不能计算有意义的 Fake precision/recall/F1 或 ROC-AUC。P1 比 C0 少 5 个 false positives（C0=11，P1=6）。

### 10.3 AIGI-test

| 模型 | N | Accuracy | Precision | Fake recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 1,731 | 0.703062 | 0.716065 | 0.667828 | 0.691106 | 0.789395 |
| P1 | 1,731 | 0.733102 | 0.739496 | 0.715447 | 0.727273 | 0.818418 |
| P1 − C0 | — | +0.030040 | +0.023431 | +0.047619 | +0.036167 | +0.029023 |

P1 在 AIGI-test 的所有列示 CLS 指标上均高于 C0。

## 11. External OOD classification：LM verdict 诊断

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

## 12. 汇总结论与边界

1. Internal classification：P1 的 CLS Accuracy 比 C0 高 0.006341；Internal G0 mean FG IoU 高 0.029874。
2. Official1000：P1 的 standalone CLS Fake recall 高 0.011000，G0 mean FG IoU 高 0.049758。
3. External classification：P1 在 RAISE specificity 和 AIGI-test 的 Accuracy/F1/ROC-AUC 上更高；LOKI 则呈现 fixed-threshold Accuracy/F1 下降但 ROC-AUC 上升的混合结果，不能概括为全面 OOD 提升。
4. LOKI G0 diagnostic：P1 的 global fg/bg mIoU/F1 为 43.22%/3.09%，C0 为 43.02%/2.72%；低于 38% 的总体 `[SEG]` trigger 主要由 LM 把大量 GT Fake 判为 `[REAL]` 导致。
5. LOKI G1 known-Fake：C0/P1 trigger 恢复至 96.51%/96.07%，global FG F1 升至 9.54%/14.39%；P1 的逐图 mean FG IoU 比 C0 高 0.021989，95% CI 不跨 0。但 global fg/bg mIoU 为 42.90%/42.54%，因新增前景预测同时降低 background IoU，不能只凭 trigger 或 FG F1 推断两类平均指标同步改善。
6. LOKI G0/G1 都使用 bbox-derived union masks；G1 比 G0 更接近 known-Fake localization，但仍与 LEGION 的 direct artifact-localization prompt 不同。RAISE 与 AIGI-test 仍只有 classification-only 结果。
7. 以上是冻结 checkpoint 的观测比较。matched new C0 缩小了训练轨迹混淆，但原始 P1 step-0 snapshot 不可恢复的证据边界仍然存在。
