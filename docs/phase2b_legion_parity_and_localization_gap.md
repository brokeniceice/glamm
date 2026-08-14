# Phase 2B：LEGION 对齐审计与定位差距分解

## 勘误：类别范围修正

**修正（Phase 2B.1）：** 此前将 Phase 2A official SynthScars-1000 的 `0.545930 / 0.295168` 与 LEGION Table 2 的 `0.5462 / 0.2990` 直接比较，混用了不兼容的类别范围。Phase 2A 数值是 **全部 1000 张图像的总体结果**，LEGION 数值则是 **Object 子集（n=162）**。二者不能相减，因此撤回此前 −0.027/−0.383 个百分点的差值以及“最接近的可观测候选差距约为零”的解释。下文保留原始数字和历史讨论作为实验记录，但其结论已由 [Phase 2B.1 类别对齐修正文档](phase2b1_legion_category_parity_correction.md)取代。

仍然成立的结论包括：仅前景 IoU `0.1396` 不能与前景/背景 mIoU 比较；official-1000 与 Phase 2A train/val/test 的 SHA256 重叠均为零；TF→G0 的平均前景 IoU 差距约为 `0.2045`；`[SEG]` 触发率约为 99%；单一 union mask 与 phrase-level multi-`[SEG]` 仍是实际存在的协议差异。Phase 2A 与 LEGION 的精确类别级对齐仍未解决。

## 结论

Phase 2B 严格保持 Phase 2A 冻结，没有训练、fine-tune、改 grammar、改 `[SEG]`、改 loss/LR/checkpoint selector、扫 test threshold，也没有加入 NPR/SRM/consistency loss。

最重要的结论是：Phase 2A 的 `Mean IoU=0.1396` 与 LEGION Table 2 的 `mIoU=0.5462` 不是同一指标，不能相减。Phase 2A 原值是逐图 foreground IoU；论文文字定义是 foreground/background 两类 IoU 的 mean。对同一批 Phase 2A binary predictions、同一 union GT、同一 `logit>0` threshold 重算后，internal test 的 per-image fg/bg mIoU 为：TF **0.653576**、G0 **0.544256**、G1 **0.542725**。表面上约 40.7 points 的 G0 gap 因口径对齐缩为约 0.19 points，但后者仍不是严格 paper-final model gap，因为 Table 2 的 aggregation/F1 代码没有公开，而且任务 supervision unit 不同。

官方 SynthScars test 1000 与 Phase 2A train/val/test 的 SHA256 交集均为 **0**，所以 official-1000 evaluation 不受训练污染。在该 split 上，Phase 2A G0 的 global fg/bg mIoU/F1 为 **0.545930/0.295168**，G1 为 **0.545998/0.296476**；LEGION Table 2 报告 **0.5462/0.2990**。最接近论文文字定义的 global 候选下，G0 只差 **0.027/0.383 percentage points**。但因 Table 2 exact aggregation/F1 code 与 paper-final checkpoint 均不可得，这只能证明“没有观测到显著的大 gap”，不能宣称 exact model parity。

真正仍然清晰存在的是 Phase 2A 自身的 TF→autoregressive gap：internal TF foreground IoU 0.3441，而 G0 为 0.1396。`[SEG]` trigger 约 99%，不是主因；surface explanation token-edit similarity 与 TF−G0 gap 无相关（Spearman ρ=-0.0106, p=0.724）。大 mask、较少/较集中的连通域使 TF 明显受益，但 G0 没有同步受益，说明生成 explanation 所形成的 `[SEG]` hidden representation 泛化是独立瓶颈。与此同时，多实例/多连通域会显著压低 TF，支持 single union query 的容量/监督表示也是第二个瓶颈，但不能据此宣称因果。

## 1. 来源与可复现边界

- 官方代码：[opendatalab/LEGION](https://github.com/opendatalab/LEGION)，本地只读审计 commit `d21535dd45f6fea509337a83095966f0b86ac924`。
- 论文：[ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Kang_LEGION_Learning_to_Ground_and_Explain_for_Synthetic_Image_Detection_ICCV_2025_paper.html) 与 [supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Kang_LEGION_Learning_to_ICCV_2025_supplemental.pdf)。
- 优先逐行审计 `README.md`、`scripts/loc_exp/train.sh|train.py|infer.py`、`LegionGCGDataset`、`tools/utils.py` 与 `eval/*`。
- README 说明 paper final weights 已遗失，只提供 [localization/explanation intermediate](https://huggingface.co/khr0516/legion_LE)。本次下载在 Hugging Face metadata request 阶段 connection timeout，未获得本地 snapshot，因此没有伪称 paper-final reproduction；失败日志与格式信息位于 `optional_legion_checkpoint/`。

## 2. 指标定义（核心表 1）

| Metric | Phase 2A implementation | LEGION paper definition | LEGION public repo implementation | Exact parity? | Notes |
|---|---|---|---|---|---|
| Mean IoU | per-image foreground IoU mean | Table 2 称 fg/bg mIoU | `giou` 是 phrase-mask pair foreground IoU mean | No | 三者不是同一量 |
| Global IoU | 全图汇总 TP/(TP+FP+FN) | 未说明 aggregation | `ciou` 是全部 phrase-mask pair 汇总 foreground IoU | Only repo↔formula | class index 1 |
| mIoU | 新增 per-image/global fg/bg mean | foreground/background regions 的 mean | repo validation 没有 fg/bg mean | No | paper per-image/global aggregation 未公开 |
| Pixel F1 | foreground，per-image mean 与 global 均报告 | “overall F1” | loc validation 未实现 F1 | No | paper 的 class/aggregation 仍 unresolved |
| gIoU | 不使用该含混名 | 未定义 | `trackers["gIoU"].avg[1]`，pair-weighted fg IoU | Yes for public code | 不是 bg/fg mean |
| cIoU | 不使用该含混名 | 未定义 | `(intersection.sum/union.sum)[1]` | Yes for public code | global pair fg IoU |

`intersectionAndUnionGPU(K=2)` 确实返回 class 0/1 两个 histogram，但公开 validation 最终只取 `[1]`。此外它用 `zip(gt_masks, predicted_masks)`，pred 数不足时会静默截断。`infer.py` 会把多个生成 mask 做 OR 以保存输出，但这不能证明 Table 2 使用 union-mask evaluation。

论文只足以把 Table 2 mIoU 归入候选 **C（foreground/background 两类 IoU 的 mean）**；无法从公开材料确定它是 per-image 还是 global aggregation。F1 的 class 与 aggregation 也无法确定。因此必须保留结论：

> public repository does not provide sufficient code to exactly reconstruct Table 2 aggregation

固定 threshold 两边均为 `mask_logit > 0`（等价 sigmoid>0.5）。本阶段没有任何 threshold sweep。

## 3. 数据集与划分（核心表 2）

| Split | 逻辑图像/冻结样本 | annotation rows | refs | 与 official test 1000 overlap |
|---|---:|---:|---:|---:|
| Phase 2A fake train | 8,836 | 8,971 | 19,173 | 0 |
| Phase 2A fake val | 1,106 | 1,128 | 2,371 | 0 |
| Phase 2A fake test | 1,104 | 1,119 | 2,354 | 0 |
| LEGION official train | 11,064 grouped logical images | 11,236 | released JSON 23,935 | 0 by definition |
| LEGION official test | 1,000 | 1,000 | released JSON 2,674 | 1,000 self |

严格 SHA256 overlap matrix：official train→our train/val/test/absent = **8836/1106/1104/18**；official test→our train/val/test/absent = **0/0/0/1000**。mapping 同时保留 relative path、basename、dimensions、bytes 与 SHA256，不是只按文件名匹配。

因此：

- `official_test_in_our_train=0`，official-1000 结果可标为 `ELIGIBLE_SPLIT_PARITY`，不是 `CONTAMINATED_DIAGNOSTIC`；
- current 1104 Fake test 与 official 1000 交集为 0；
- 对 official-train 中进入冻结 manifests 的 11,046 个同图样本，artifact text、refs/polygons、union mask 全部逐像素完全相同，最大 area-ratio 差为 0；
- release JSON 没有 Physics/Distortion/Structure 或逐图 Human/Animal/Object/Scene 字段，所以 artifact-type 分桶与 official-1000 逐 content 指标不能可信重建，均明确标记 unresolved。Supplement 只给 aggregate official test Human/Animal/Object/Scene = 587/134/162/117；current 1104 为 581/152/194/177（其标签来自冻结 content-label pipeline）。

补充材料声称 train/test artifact instances 为 23,913/2,653，但直接读取当前 release JSON 得到 23,935/2,674；本报告保留两组来源，不替作者猜测。

## 4. 预测目标、监督方式与预处理

LEGION 是 phrase-level multiple grounding queries：caption 中每个匹配 phrase 变为 `<p> phrase </p> [SEG]`，每个 `[SEG]` 对一个独立 mask。Phase 2A 是 image-level single union query：`[CLS] [FAKE] <entire explanation> [SEG]`，唯一 `[SEG]` 对 OR(all refs) 的 union mask。这不是“同一个 segmentation-head task”。

Official train/test 每图 refs 分布如下：

| Split | 1 | 2 | 3 | 4+ | mean | median | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| released train grouped | 5,102 | 2,752 | 1,532 | 1,678 | 2.163 | 2 | 4 | 18 |
| official test | 303 | 262 | 179 | 256 | 2.674 | 2 | 5 | 22 |

即 official test 有 69.7% 图像含多个 refs，而 Phase 2A 无论 refs 数量始终压成一个 union query。

Preprocessing 差异：两者 global encoder 都是 `CLIPImageProcessor(openai/clip-vit-large-patch14-336)`，SAM normalization/pad-to-1024 也同源；但 LEGION train 的实际 parser args 是 `ResizeLongestSide(512)`，Phase 2A train/eval 是 `ResizeLongestSide(1024)`。LEGION `infer.py` 默认又回到 1024。两边 model path 都用 resize/original-size list 将 mask 恢复到原图尺寸。

Prompt 也不同：LEGION train/infer 明确枚举 physical/structural/distortion artifacts 并要求 interleaved segmentation masks；Phase 2A 固定为 `Determine whether this image is authentic and explain the forensic evidence.`，并联合 Real/Fake verdict、explanation 与 single union mask。

## 5. 模型与训练配置（核心表 3）

| 项目 | Phase 2A | LEGION loc_exp public/paper |
|---|---|---|
| 数据 | 1:1 Real+Fake | Fake SynthScars |
| target grammar | `[CLS] [FAKE] explanation [SEG]` / Real grammar | caption 中每 phrase 后一 `[SEG]` |
| #SEG | Fake 固定 1 | 每个可匹配 ref 1 个 |
| mask target | all-ref union | independent phrase mask |
| classification jointly trained | Yes | loc_exp No |
| explanation jointly trained | Yes | Yes |
| loss | text + cls + 2.0 BCE + 0.5 Dice | CE 1.0 + BCE 0.4 + Dice 0.2 |
| LR | 3e-4 | 1e-4 |
| init | `MBZUAI/GLaMM-FullScope` | `MBZUAI/GLaMM-GranD-Pretrained` + SAM |
| grounding train resolution | 1024 | 512 |
| mask-supervised exposure | exact 50,000 Fake union-mask draws | exact paper exposure unresolved |
| unique mask images seen | exact 8,836/8,836 | random-with-replacement，exact unique 未记录 |
| checkpoint selector | min val total loss，step2500 | released train.sh 未开 mask_validation，走 val loss；paper selector 未说明 |

初始化不相同：Phase 2A FullScope 是独立 Hugging Face repo/revision，README 说明它经过多任务 mixed fine-tuning；LEGION 指向 GranD-Pretrained。FullScope config 中残留 `_name_or_path=GLaMM-GranD-Pretrained` 只能证明祖先 provenance，不能证明最终权重 byte-identical。完整 repo revisions、config/index/tokenizer 与 Phase 2A checkpoint SHA256 在 `source_audit/initialization_parity.json`。

Phase 2A canonical metrics 精确给出 100,000 total / 50,000 Fake mask exposures；按 seed=3407 与实际 2GPU→1GPU sampler history 重放，8,836 张 Fake 全部至少出现一次，mean 5.659、min 1、max 10。

LEGION release 存在不能忽略的 recipe 矛盾：paper 写 8×A100、batch 2/device；train.sh 写 batch 16/device 且没固定 world size；current `train.py` 还在 `epoch>=1` 时 `break`，所以 `--epochs 3` 实际只跑 epoch 0/703 steps。`HybridSegDataset` 忽略 sampler index 并调用 child[0]，随后 Legion dataset uniform random-with-replacement。若仅把 paper global batch 16 与 script 703×3 拼接，名义 draws 为 33,744，但它不是可由 released entry point 精确复现的 exposure，故不把它伪装成实测值。

## 6. 同口径统一指标组（核心表 4）

Internal frozen Fake test，n=1104，全部来自同一 binary predictions/union GT/threshold：

| Mode | Mean FG IoU | Global FG IoU | Mean BG IoU | Global BG IoU | Mean fg/bg mIoU | Global fg/bg mIoU | Mean FG F1 | Global FG F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF | .344065 | .388522 | .963087 | .964768 | .653576 | .676645 | .454585 | .559619 |
| G0 | .139568 | .133436 | .948944 | .951586 | .544256 | .542511 | .200415 | .235454 |
| G1 | .137260 | .132652 | .948189 | .950985 | .542725 | .541818 | .196853 | .234232 |

Human/Animal/Object/Scene 的完整八指标 suite 保存在 `metric_parity/phase2a_best_internal.json`。按 paper 文字最接近的是 `mIoU_fg_bg_per_image` 或 `mIoU_fg_bg_global`，但 exact aggregation 未知，所以两列都报告，不挑更有利的一列。

Official SynthScars test，n=1000，与 Phase 2A train 零重叠；Phase 2A checkpoint 与统一 `mask_logit>0` threshold：

| Source / Mode | Mean FG IoU | Global FG IoU | Mean fg/bg mIoU | Global fg/bg mIoU | Mean FG F1 | Global FG F1 |
|---|---:|---:|---:|---:|---:|---:|
| LEGION Table 2 | unresolved | unresolved | `.5462`（aggregation unresolved） | `.5462`（若按 global 候选） | `.2990`（aggregation unresolved） | `.2990`（若按 global-FG 候选） |
| Phase 2A TF | .408905 | .475917 | .675056 | .709897 | .531387 | .644910 |
| Phase 2A G0 | .187885 | .173136 | .551843 | .545930 | .268184 | .295168 |
| Phase 2A G1 | .189926 | .174037 | .552747 | .545998 | .271495 | .296476 |

LEGION 行中的两次候选展示不代表 exact implementation 已确认；它只用于显示论文文字允许的候选口径。由于 released paper-final weights 已遗失、intermediate checkpoint 下载失败，本表没有 LEGION 模型的本地重跑行。

## 7. 差距分解

### 掩码面积

| GT area | n | TF IoU | G0 IoU | G1 IoU | TF F1 | G0 F1 | G1 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1% | 419 | .2779 | .0968 | .0929 | .3747 | .1435 | .1389 |
| 1–2% | 168 | .3586 | .1815 | .1766 | .4794 | .2494 | .2418 |
| 2–5% | 272 | .3526 | .1667 | .1628 | .4739 | .2408 | .2348 |
| 5–10% | 126 | .4123 | .1592 | .1694 | .5329 | .2299 | .2419 |
| 10–20% | 69 | .4126 | .1494 | .1563 | .5261 | .2152 | .2203 |
| 20%+ | 50 | .5366 | .1463 | .1305 | .6391 | .1988 | .1798 |

Tiny masks 是困难因素：0–1% 占 37.95%，TF 只有 .278。但它不是 G0 低值的充分解释，因为 20%+ 时 G0 仍只有 .146，而 TF 达 .537。

### 伪影标注与连通域

| refs | n | TF IoU | G0 IoU | G1 IoU |
|---|---:|---:|---:|---:|
| 1 | 528 | .3697 | .1166 | .1143 |
| 2 | 258 | .3375 | .1616 | .1575 |
| 3 | 145 | .3294 | .1628 | .1556 |
| 4+ | 173 | .2880 | .1572 | .1619 |

| GT components | n | TF IoU | G0 IoU | G1 IoU |
|---|---:|---:|---:|---:|
| 1 | 550 | .3785 | .1202 | .1163 |
| 2 | 250 | .3201 | .1594 | .1593 |
| 3 | 152 | .3192 | .1575 | .1533 |
| 4+ | 152 | .2840 | .1592 | .1608 |

TF 随 refs/components 增多明显下降；G0/G1 没有相同单调关系，说明 area/content 等 confound 存在。结论只能是 multi-region 对 teacher-forced single-union grounding 有稳定关联，不能把它宣称为唯一原因。

Artifact type 分桶未生成假数据：release refs 没有 Physics/Distortion/Structure 字段，状态为 unresolved。

### TF→G0/G1 与解释文本

- mean `TF−G0 IoU = 0.204497`；mean `TF−G1 = 0.206805`。
- token-edit similarity：ρ=-.0106, p=.724（无关联）；exact-prefix overlap：ρ=.0745, p=.013；length ratio：ρ=.0643, p=.033。后两者虽因 n=1104 达统计显著，但 effect 极小且方向不能作因果解释。
- area ratio：ρ=.1434；artifact count：ρ=-.1440；components：ρ=-.1674；largest-component share：ρ=.1700；均 p<2e-6，但 effect 仍属弱相关。
- repetition 22 例的 mean gap .2334，对其余 .2039；verdict error 仅 8 例，mean gap .3036。样本很少，不宜泛化。

`TF−G0` gap 与 surface edit similarity 没有关联，不能简单说“文字越像 GT，mask 就越好”。更准确的观察是：TF 能在 large/simple region 获益，而 autoregressive `[SEG]` hidden state 没能继承这一收益；surface text metric 不能充分描述 hidden-state trajectory。

## 8. 四象限分析与可视化

诊断阈值固定为 TF IoU 0.5 / G0 IoU 0.2，不作为 paper metric：

| Group | 条件 | n | fraction |
|---|---|---:|---:|
| A | TF high / G0 high | 149 | 13.50% |
| B | TF high / G0 low | 187 | 16.94% |
| C | TF low / G0 low | 620 | 56.16% |
| D | TF low / G0 high | 148 | 13.41% |

每组确定性选择 20 例，共 80 例。每个 panel 包含 Original、GT、TF、G0、G1 overlay，以及 content、area、refs、components、IoUs、GT/G0 explanation；同时保存四张 raw masks。B 组直接展示“mask branch 在 GT context 下会定位、generated context 却转向错误区域”，C 组展示“即便 GT context 也失败”。所有 panel 位于 `outputs/phase2b_legion_parity/visualizations/`。

## 9. Best 与 Last：checkpoint 选择权衡

| Split / metric | step2500 best | step5000 last | last−best |
|---|---:|---:|---:|
| val total loss | 1.529054 | 1.585654 | +.056600 |
| val CLS accuracy | .983273 | .985986 | +.002712 |
| val LM accuracy | .984177 | .985986 | +.001808 |
| val TF mean FG IoU | .325395 | .347640 | +.022245 |
| internal CLS accuracy | .972373 | .977355 | +.004982 |
| internal LM accuracy | .968750 | .976902 | +.008152 |
| internal TF mean/global FG IoU | .344065/.388522 | .366661/.414560 | +.022596/+.026038 |
| internal G0 mean/global FG IoU | .139568/.133436 | .160473/.154279 | +.020905/+.020843 |
| internal G1 mean/global FG IoU | .137260/.132652 | .161632/.169498 | +.024372/+.036846 |

Last 的 detection 与 localization 都略好，但 validation total loss 变差，说明 unified objective 的 selector trade-off 是约 **2–4 foreground-IoU points** 的次要因素，不足以解释原先错误比较产生的约 40-point 表观差距。所有 2208 detection 与每种 1104 Fake localization 的 sample ID 已通过数量、唯一性和 manifest 集合一致性门禁。

正式 baseline 永远保持 step2500；本节不使用 test 重选 checkpoint。

## 10. 三层差距结论

### A. 表观指标差距

占主导。`.1396 foreground IoU` 对 `.5462 fg/bg mIoU` 是无效比较；同 prediction 转为 per-image fg/bg mIoU 后 G0=.5443。

### B. 数据集与协议差距

真实存在：official split 与 frozen split 完全不同；LEGION 为 Fake-only phrase-level multi-mask、专门 artifact prompt、GranD init、512 train grounding、0.4/0.2 mask weights；Phase 2A 为 Real/Fake unified、single union mask、FullScope init、1024、2.0/0.5，并联合 classification。Released LEGION training entry point与 paper 还有 batch/epoch 控制流矛盾。

### C. 真实模型性能差距

在真正同 split/GT/threshold 下，Phase 2A G0 official-1000 的 global fg/bg mIoU=.545930、global FG F1=.295168；相对论文 .5462/.2990 的候选差为 **−.000270/−.003832**（即 −0.027/−0.383 percentage points）。因此当前证据不支持“大模型性能 gap”。然而 paper exact metric code 与 final weights 不可得，无法把该残差严格解释为 model-only gap；严谨结论是 **exact real gap unidentifiable, closest observable candidate approximately zero**。Phase 2A 自身 TF→G0 的 mean FG IoU gap=.204497 则明确存在，属于生成上下文下的 grounding generalization 问题，而不是 paper-parity gap。

## 11. 24 个必须回答的问题

1. Table 2 mIoU：论文定义到 fg/bg 两类 mean（C）；exact per-image/global aggregation 未公开。
2. Public repo validation 是否完全一致：否；repo 是 foreground-only phrase-pair giou/ciou。
3. F1 exact：unresolved，公开 loc validation 无 F1。
4. Phase 2A Mean IoU 能否直接比较：不能。
5. 同口径 internal：TF/G0/G1 per-image fg/bg mIoU=.6536/.5443/.5427；global=.6766/.5425/.5418。
6. official test 是否进 train：0。
7. contamination：无。
8. current fake test 与 official test 交集：0。
9. 同图 annotation：11,046/11,046 text/ref/union-mask pixel exact。
10. supervision representation：LEGION mean 2.674 phrase masks/official-test image；Phase 2A 固定 1 union mask。
11. refs 越多是否下降：TF 从 1-ref .3697 降至 4+ .2880；G0/G1 不单调，存在 confound。
12. mask 越小是否下降：明显，0–1% TF/G0=.2779/.0968；但大 mask 的 G0 仍低。
13. disconnected 越多是否下降：TF 从 1-component .3785 降至 4+ .2840；G0/G1 不单调。
14. TF≈.34、G0≈.14 的统计相关因素：surface edit similarity 不是；area、components、largest-component share 为弱但显著相关；核心是 TF benefit 未传递到 generated hidden trajectory。
15. B 组：187/1104=16.94%。
16. C 组：620/1104=56.16%。
17. explanation drift 是否显著相关：token edit 否（p=.724）；prefix/length 仅极弱相关，不支持强因果说法。
18. step2500 vs step5000：见 §9。
19. checkpoint trade-off：是次要组成；last 的 internal G0/G1 mean FG IoU 提升 .0209/.0244，但 val total loss 变差 .0566，不能解释约 40-point 表观 gap，也不能据此替换 baseline。
20. supervision exposure：Phase 2A exact 50,000 union-mask draws/8,836 unique；LEGION paper exact unresolved，paper+script nominal inference 33,744 image draws。
21. initialization：不完全一致，FullScope vs GranD-Pretrained。
22. resolution/preprocess：global/SAM normalization 同源；train ResizeLongestSide 1024 vs 512；LEGION infer 为1024。
23. loss recipe：Phase 2A text+cls+2.0 BCE+0.5 Dice, LR3e-4；LEGION 1.0 CE+0.4 BCE+0.2 Dice, LR1e-4。
24. true localization gap：exact model-only gap 因 paper code/final weights缺失不可识别；最接近的 official/global 候选为 G0 mIoU −.000270、F1 −.003832，近似为零。独立存在的 internal TF−G0 mean FG IoU gap 为 .204497。

## 12. 下一阶段候选（本阶段不实施）

诊断首先落在 **Case A**：metric parity 后没有观测到显著的 LEGION gap，因此不应因 `.1396 vs .5462` 重构 localization；下一独立授权阶段可保留 Phase 2A baseline，进入 NPR/SRM forensic enhancement 的受控消融。其次是 **Case C**：针对明确存在的 TF→G0 gap，单独研究 explanation-conditioned grounding generalization。phrase-level multi-`[SEG]` 与 localization-focused objective/checkpoint trade-off 只作为后续隔离变量，不在本阶段实施，也不能事后替换 step2500 baseline。

候选只记录在 `outputs/phase2b_legion_parity/future_hypotheses.md`；Phase 2B 没有对模型作任何相应修改。

## 13. 产物、修改文件与测试

Artifacts root：`outputs/phase2b_legion_parity/`，包含要求的 `source_audit/`、`split_audit/`、`metric_parity/`、`gap_analysis/`、`visualizations/`、`optional_legion_checkpoint/`。每项 JSON 都保留 split、threshold 或 unresolved reason。

Phase 2B 修改/新增：

- `scripts/phase2b_legion_parity_audit.py`：SHA256 split/annotation audit、unified metric suite、exposure 与 gap bins/correlations/quadrants；
- `scripts/phase2b_visualize_cases.py`：best checkpoint bounded inference 与 80 panels/raw masks；
- `scripts/phase2b_finalize.py`：完成度 gate、last/official rescore 与 best-last 汇总；
- `scripts/phase2a_final_evaluate.py`：只增加 expected checkpoint、external manifest/root 与 checkpoint SHA256 参数，未改变 forward/generation/metric；
- `tests/test_phase2b_legion_parity.py`；
- 本文档与 Phase 2B output artifacts。

最终 regression：`CUDA_VISIBLE_DEVICES='' ... pytest -q tests` → **88 passed, 5 warnings**。Warnings 是既有 dependency deprecation、预期的 negative mismatch test 与 Transformers generation-config warning；没有 test failure。
