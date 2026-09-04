# P1 Reusable Evaluation Baselines

## 1. 用途与基线身份

本文集中保存 P1 已完成且可用于后续模型对比的 classification 与 segmentation/localization 结果。它是协议化基线目录，不是把不同数据集、prompt、target 或 aggregation 混在一起的统一排行榜。

正式 P1：

- checkpoint：`/data/yz/groundingLMM_official/checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt`
- optimizer step / epoch：`3500 / 7`
- SHA256：`fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326`
- selector：minimum validation total loss；internal test / official1000 / external data 均未参与选模
- canonical user prompt：`Determine whether this image is authentic and explain the forensic evidence.`
- prompt SHA256：`32f0856f718fb3e19f34b552892636fb6c2613adbae4afb8826dea89f670654d`
- generation：greedy，`do_sample=false`，`num_beams=1`，`max_new_tokens=400`
- segmentation threshold：mask logit `> 0.0`；无 threshold sweep

## 2. 快速选择规则

| 对比目标 | 首选 P1 基线 |
|---|---|
| 开发阶段 deployable localization | DEV canonical G0，N=1,106 Fake，mean FG IoU `0.148233` |
| 开发阶段 language controllability | 同一 DEV 的 G0 / Phrase-only / TF-PHRASE：`0.148233 / 0.245798 / 0.342928` |
| 冻结内部最终分类 | Internal test direct batch=1 CLS，N=2,208，Accuracy `0.983696` |
| 冻结内部最终定位 | Internal test Fake G0，N=1,104，mean FG IoU `0.166414` |
| SynthScars official 最终定位 | Official1000 G0，N=1,000 Fake，mean FG IoU `0.229544` |
| 分类鲁棒性 | Internal test direct batch=1 的 Original/JPEG/Gaussian 表 |
| 定位鲁棒性 | 同一 internal/official population 的 canonical G0 corruption 表 |
| OOD 分类 | LOKI / RAISE / AIGI-test fixed CLS 表，分别解释 |
| External localization | LOKI 229-image G1 bbox-derived表；AIGI-test 861-Fake G1 pixel-mask 表 |

## 3. Classification baselines

### 3.1 DEV validation diagnostic

Internal validation 共 2,212 张（1,106 Real + 1,106 Fake）。P1 generation-derived classification accuracy 为 `0.986438`。

该值来自 Phase3D.1 validation comparison，只适合相同 evaluator 的开发对照；正式部署式 classification 应优先使用后面的 direct batch=1 internal-test 表。

### 3.2 Internal test — canonical standalone CLS

Population：2,208 = 1,104 Real + 1,104 Fake。Fake 为正类。

| Route | N | Accuracy | Precision | Fake recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Fixed CLS head | 2,208 | 0.983696 | 0.988117 | 0.979167 | 0.983621 | 0.998389 |
| LM verdict | 2,208 | 0.984149 | 0.986351 | 0.981884 | 0.984113 | 0.998407 |

CLS–LM agreement：`0.996830`。正式 classification 数值采用 direct batch=1；历史 batch=8 cache 只用于训练/selector，不替代该表。

### 3.3 Official1000 — Fake-only classification

Population：1,000 Fake。由于没有 Real 负类，Accuracy 等于 Fake recall，ROC-AUC、specificity 和 FPR 不可定义。

| Route | N | Accuracy / Fake recall | F1 |
|---|---:|---:|---:|
| Fixed CLS head | 1,000 | 0.980000 | 0.989899 |
| LM verdict | 1,000 | 0.976000 | 0.987854 |

CLS–LM agreement：`0.996000`。

### 3.4 Internal test classification robustness — direct batch=1

JPEG quality factor 为 70/80；Gaussian σ=5/10 定义在 RGB 0–255 像素尺度，并在所有模型预处理之前施加。

| Condition | N | CLS Accuracy | Precision | Fake recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Original | 2,208 | 0.983696 | 0.988117 | 0.979167 | 0.983621 | 0.998389 |
| JPEG70 | 2,208 | 0.948822 | 0.919560 | 0.983696 | 0.950547 | 0.993137 |
| JPEG80 | 2,208 | 0.959239 | 0.931857 | 0.990942 | 0.960492 | 0.995937 |
| Gaussian5 | 2,208 | 0.953804 | 0.988304 | 0.918478 | 0.952113 | 0.994462 |
| Gaussian10 | 2,208 | 0.934783 | 0.981928 | 0.885870 | 0.931429 | 0.988826 |

### 3.5 Official1000 classification robustness — Fake-only

| Condition | N | Accuracy / Fake recall | F1 | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Original | 1,000 | 0.980000 | 0.989899 | 0.019138 | 0.026941 |
| JPEG70 | 1,000 | 0.982000 | 0.990918 | 0.014786 | 0.020255 |
| JPEG80 | 1,000 | 0.988000 | 0.993964 | 0.008036 | 0.012462 |
| Gaussian5 | 1,000 | 0.909000 | 0.952331 | 0.078767 | 0.097036 |
| Gaussian10 | 1,000 | 0.867000 | 0.928763 | 0.106192 | 0.130279 |

仍然是 Fake-only；这些 Accuracy 只能解释为 Fake recall。

### 3.6 External OOD classification — fixed CLS head

协议：canonical prompt、fixed CLS evaluator、BF16、batch=8、classification threshold `0.5`。外部集未用于训练、selector 或 calibration。

| Dataset | Composition | N | Accuracy / specificity | Precision | Fake recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| LOKI | mixed Real/Fake | 2,217 | 0.543076 | 0.751656 | 0.344723 | 0.472670 | 0.653859 |
| RAISE | Real-only | 998 | specificity 0.993988 | N/A | N/A | N/A | N/A |
| AIGI-test | 870 Real + 861 Fake | 1,731 | 0.733102 | 0.739496 | 0.715447 | 0.727273 | 0.818418 |

RAISE fixed CLS 为 TN=992、FP=6，FPR=`0.006012`。Real-only population 不能报告 Fake precision/recall/F1 或 ROC-AUC。

### 3.7 External OOD classification — LM verdict diagnostic

| Dataset | N | LM Accuracy / specificity | LM F1 | LM ROC-AUC | CLS–LM agreement |
|---|---:|---:|---:|---:|---:|
| LOKI | 2,217 | 0.548038 | 0.484037 | 0.649589 | 0.981507 |
| RAISE Real-only | 998 | specificity 0.992986 | N/A | N/A | 0.998998 |
| AIGI-test | 1,731 | 0.725014 | 0.720657 | 0.811107 | 0.983824 |

LM verdict 是生成式诊断，不能静默替换 fixed CLS 主结果。

## 4. Segmentation/localization baselines

### 4.1 DEV validation — current canonical comparison baseline

Population：internal validation Fake，N=1,106。Target 为 official SynthScars annotation-derived per-image all-reference union。Canonical invalid G0 共 28 个，正式策略为 IoU=0。

| Condition | N | mean FG IoU | mean FG F1 | global FG IoU | global FG F1 |
|---|---:|---:|---:|---:|---:|
| G0 | 1,106 | 0.148233 | 0.210697 | 0.146524 | 0.255597 |
| Authoritative Phrase-only | 1,106 | 0.245798 | 0.343492 | 0.245605 | 0.394354 |
| TF-PHRASE | 1,106 | 0.342928 | 0.452020 | 0.389014 | 0.560130 |

这是 Phase4F/4H 当前模型比较应使用的 canonical P1 三条件基线。

### 4.2 DEV validation — Phase3C.0 diagnostic protocol

| Condition | Scope | mean FG IoU | mean FG F1 | mean fg/bg mIoU |
|---|---:|---:|---:|---:|
| Canonical G0 | 1,106 | 0.148233 | 0.210697 | 0.544399 |
| Authoritative Phrase-only | 1,106 | 0.247362 | 0.345261 | 0.584177 |
| TF-PHRASE | 1,106 | 0.342928 | 0.452020 | 0.652260 |

边界：Phase3C.0 Phrase-only `0.247362` 与当前 canonical artifact `0.245798` 来自不同历史实现，后续新模型默认应对齐当前 canonical 表。

### 4.3 Internal test localization

Population：1,104 Fake；target 为 SynthScars per-image all-reference union；threshold 为 mask logit `>0`。

| Condition | mean FG IoU | mean FG F1 | mean fg/bg mIoU | global FG IoU | global FG F1 | global fg/bg mIoU | SEG trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 0.166414 | 0.233295 | 0.553925 | 0.162135 | 0.279029 | 0.553051 | 0.983696 |
| Phrase-only | 0.260755 | 0.362305 | 0.589662 | 0.249761 | 0.399694 | 0.588549 | N/A |
| TF-PHRASE canonical（fresh rerun） | 0.360798 | 0.475815 | N/A | 0.413693 | 0.585266 | N/A | N/A |

旧 `0.360137` TF 行来自错误 legacy user prompt，已按用户指令丢弃，不得作为后续基线。

### 4.4 Official1000 localization

Population：1,000 Fake；相同 SynthScars target 和 threshold。

| Condition | mean FG IoU | mean FG F1 | mean fg/bg mIoU | global FG IoU | global FG F1 | global fg/bg mIoU | SEG trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 0.229544 | 0.319323 | 0.571646 | 0.236949 | 0.383118 | 0.576302 | 0.968000 |
| Phrase-only | 0.332896 | 0.448494 | 0.616986 | 0.396181 | 0.567521 | 0.654930 | N/A |
| TF-PHRASE canonical（fresh rerun） | 0.437453 | 0.563571 | N/A | 0.504498 | 0.670653 | N/A | N/A |

旧 `0.437609` TF 行来自错误 legacy user prompt，已按用户指令丢弃，不得作为后续基线。

补充 TF-OLD mean FG IoU=`0.250854`。TF-OLD 指旧 assistant target protocol（无 `Target regions:`），不等于 Phase3D.2 曾出现的 legacy user-prompt bug。

### 4.5 G0 localization robustness

| Population | Condition | N | P1 mean FG IoU |
|---|---|---:|---:|
| Internal test Fake | Original | 1,104 | 0.166414 |
| Internal test Fake | JPEG70 | 1,104 | 0.167208 |
| Internal test Fake | JPEG80 | 1,104 | 0.164906 |
| Internal test Fake | Gaussian5 | 1,104 | 0.156419 |
| Internal test Fake | Gaussian10 | 1,104 | 0.149851 |
| Official1000 Fake | Original | 1,000 | 0.229544 |
| Official1000 Fake | JPEG70 | 1,000 | 0.232847 |
| Official1000 Fake | JPEG80 | 1,000 | 0.234617 |
| Official1000 Fake | Gaussian5 | 1,000 | 0.222148 |
| Official1000 Fake | Gaussian10 | 1,000 | 0.205977 |

未来模型复跑 corruption 时，所有分支必须看到同一 corrupted image；不得把 clean forensic evidence/cache 与 corrupted P1 image path 混用。

### 4.6 LOKI external localization

Population：LOKI `open_ended_vqa.json` 中 229 张 fully-synthetic valid images。GT 为 `problems.regional[].region=[x,y,w,h]` 填充矩形后按图 union；不是像素级人工 artifact mask。

| Protocol | N | mean FG IoU | mean FG F1 | global FG IoU | global FG F1 | global fg/bg mIoU | SEG trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| G1 structural GT `[FAKE]` prefix | 229 | 0.076893 | 0.126401 | 0.077554 | 0.143945 | 0.425447 | 0.960699 |

G1 更接近 known-Fake localization，但仍不是 LEGION direct artifact-localization prompt 的逐字节复现。

### 4.7 AIGI-test external localization — G1 only

Population：AIGI-test 861 张带非空 released pixel mask 的 Fake；870 张 Real 不进入定位评估。按用户修正，本数据集只评估 G1，不评估 G0、Phrase 或 TF。G1 使用 canonical user prompt，并把 GT `[FAKE]` 作为 assistant continuation prefix 后执行 greedy generation；`max_new_tokens=400`、batch=4、seed=3407、mask logit threshold `>0`。844/861 形成唯一有效 q-seg；其余 17 个按固定 failure policy 记空预测、IoU=0。

| Model | N | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 861 | 0.185580 | 0.063742 | 0.250676 | 0.132414 | 0.233862 |
| Phase4H-D R1 | 861 | 0.200430 | 0.081218 | 0.272542 | 0.115063 | 0.206380 |

R1 − P1 的 paired mean FG IoU delta=`+0.014850`，bootstrap 95% CI=`[-0.001455, +0.031145]`，W/T/L=`412/76/373`，Wilcoxon p=`0.047558`。Primary mean-IoU bootstrap CI 跨 0，因此结论是 **R1 点估计高于 P1，但统计稳健的 mean-IoU improvement 未获支持**。mean FG F1 delta=`+0.021867`，95% CI=`[+0.003683, +0.040229]`，但 global FG IoU/F1 低于 P1；这些 aggregation 必须并列报告，不能只保留有利指标。

R1 与 P1 共用完全相同的 G1 trajectory/q-seg、样本顺序、image features、mask、geometry 与 threshold；R1 checkpoint 为 Phase4H-D selected epoch9，SHA256=`9b38ef1e62c86c74287895e681ec8ebb96b724e3bae52622d52b61bca7ddf5a5`。冻结 source hash 前后完全一致；没有训练、选模或调参。

## 5. Historical prompt caveat

Phase3D.2 原 TF evaluator 曾误用 legacy localization user question。相同 P1 step 0 的 legacy TF IoU 为 `0.340823`，修正为 canonical matched prompt 后为 `0.342928`。后续模型默认使用 canonical matched `0.342928`；旧 `0.340823` 只保留为历史 protocol-mismatch 记录，不作为当前 baseline。

## 6. 后续模型对比硬规则

1. 必须先固定 checkpoint SHA256、population、sample-ID set/order、prompt、target、geometry、threshold、batch contract 和 aggregation。
2. 只有相同 population、相同 target 与相同 evaluator 的逐样本结果才能计算 paired delta、bootstrap CI、W/T/L 和 Wilcoxon。
3. Classification 正式结果使用 direct batch=1；不得用 batch=8 cache 的 boundary-sensitive数值替换。
4. G0 是 deployable localization primary；Phrase-only、TF-PHRASE、TF-OLD 是 language/oracle diagnostics，不能替代 G0。
5. Official1000 是 Fake-only：classification Accuracy 只能解释为 Fake recall，不能声称 balanced robustness、specificity 或 FPR。
6. SynthScars pixel-union、LOKI bbox-union、per-image mean 与 global-pixel aggregation必须分别报告，禁止直接相减。
7. AIGI-test localization 当前只认可 G1；不得补写或推导 G0/Phrase/TF，且其 released pixel-mask target 不与 SynthScars/LOKI 数值直接相减。
8. internal test 与 official1000 保持封存，只有明确授权后才能给新模型运行；读取本文历史数值不等于重新访问数据集。
9. 不允许基于新模型结果回调 threshold、gamma、gate、checkpoint 或 prompt。

## 7. 推荐的新模型对比表模板

每个新模型应在同一 protocol 下填表，不适用项写 `NOT RUN`，不能用其他 split 的结果代填。

| Population / condition | P1 | Candidate | Paired delta | Bootstrap 95% CI | W/T/L | Wilcoxon | Protocol parity |
|---|---:|---:|---:|---|---|---:|---|
| DEV G0 mean FG IoU | 0.148233 |  |  |  |  |  |  |
| DEV Phrase-only mean FG IoU | 0.245798 |  |  |  |  |  |  |
| DEV TF-PHRASE mean FG IoU | 0.342928 |  |  |  |  |  |  |
| Internal test CLS Accuracy | 0.983696 |  |  |  |  |  |  |
| Internal test G0 mean FG IoU | 0.166414 |  |  |  |  |  |  |
| Official1000 Fake recall | 0.980000 |  |  |  |  |  |  |
| Official1000 G0 mean FG IoU | 0.229544 |  |  |  |  |  |  |
| AIGI-test G1 mean FG IoU | 0.185580 |  |  |  |  |  |  |

## 8. Machine-readable source index

- DEV canonical P1 equivalence：`outputs/phase4f_language_preserving_rectification/preflight/p1_equivalence_audit.json`
- DEV Phase3C.0：`outputs/phase3c0_residual_diagnosis/`
- Internal/official P1 classification and G0/Phrase/TF：`outputs/phase3a_phrase_grounding/evaluation/`
- TF-OLD / TF-PHRASE cross protocol：`outputs/phase3a1_paired_control/evaluation/tf_cross_eval/`
- Internal/official robustness：`outputs/phase3c2_p3_robustness/`
- External CLS/LM verdict：`outputs/phase3a1_paired_control/evaluation/external_classification/p1/`
- LOKI G1 localization：`outputs/phase3a1_paired_control/evaluation/external_g0/loki/p1/`
- AIGI-test P1/R1 G1 localization：`outputs/p1_r1_aigi_localization/results.json`
- AIGI-test G1 frozen protocol：`outputs/p1_r1_aigi_localization/protocol.json`
- Phase3D.2 prompt mismatch audit：`outputs/phase3d2a_spatial_attribution_audit/train_eval_consistency_audit.json`
- Phase3D.2 canonical matched reevaluation：`outputs/phase3d2b_matched_spatial_reevaluation/`
## 9. Phase4H-D R1 完整同条件对比

R1 checkpoint：`/home/yz/groundingLMM_official/outputs/phase4hd/r1/selected_checkpoint.pt`，epoch 9，SHA256 `9b38ef1e62c86c74287895e681ec8ebb96b724e3bae52622d52b61bca7ddf5a5`。本节仅补齐本文件保留的 P1 协议；已丢弃的 G0 Phrase Repair、LOKI G0 free-authenticity，以及 AIGI G0/Phrase/TF 均未恢复。

R1 不改 classification/LM verdict 路径。`outputs/p1_r1_reusable_matrix/classification_invariance.json` 已审计 R1 checkpoint 只含 CSCU utility 与 Phase4F rectifier state，因此第 3.1–3.7 节的 R1 classification 值对 P1 为 **EXACT REUSE**，不是新推理结果。

旧 Internal/Official TF 产物使用过错误的 legacy user prompt，按用户指令丢弃。下表 TF 行是 P1 与 R1 使用正确 canonical prompt 的 fresh同条件重跑；不得再引用旧 TF 点值。

| Population / protocol | P1 mean FG IoU | R1 mean FG IoU | R1−P1 delta [bootstrap 95% CI]; W/T/L; Wilcoxon |
|---|---:|---:|---|
| DEV G0 | 0.148233 | 0.195477 | existing Phase4H-D selected-checkpoint result |
| DEV Phrase | 0.245798 | 0.208660 | existing Phase4H-D selected-checkpoint result |
| DEV TF canonical | 0.342928 | 0.215622 | existing Phase4H-D selected-checkpoint result |
| Internal test G0 | 0.166414 | 0.206178 | +0.039764 [+0.028923, +0.050908]; 536/208/360; p=8.04076e-13 |
| Internal test Phrase | 0.260755 | 0.216566 | -0.044189 [-0.059002, -0.029453]; 428/39/637; p=4.70988e-08 |
| Internal test TF canonical (fresh P1/R1) | 0.360798 | 0.229128 | -0.131670 [-0.146073, -0.117545]; 278/61/765; p=1.60481e-63 |
| Official1000 G0 | 0.229544 | 0.286588 | +0.057044 [+0.045424, +0.068729]; 603/64/333; p=1.10303e-22 |
| Official1000 Phrase | 0.332896 | 0.312487 | -0.020409 [-0.037349, -0.003773]; 497/1/502; p=0.406424 |
| Official1000 TF canonical (fresh P1/R1) | 0.437453 | 0.319176 | -0.118277 [-0.133849, -0.103092]; 339/2/659; p=8.49489e-41 |
| Internal G0 JPEG70 | 0.167208 | 0.206310 | +0.039102 [+0.027906, +0.050140]; 536/215/353; p=1.00336e-12 |
| Internal G0 JPEG80 | 0.164906 | 0.201084 | +0.036178 [+0.024963, +0.047725]; 539/202/363; p=6.31811e-12 |
| Internal G0 Gaussian5 | 0.156419 | 0.192187 | +0.035768 [+0.025404, +0.046693]; 524/254/326; p=3.11511e-13 |
| Internal G0 Gaussian10 | 0.149851 | 0.187749 | +0.037898 [+0.027425, +0.048703]; 492/281/331; p=7.99668e-13 |
| Official G0 JPEG70 | 0.232847 | 0.289210 | +0.056363 [+0.044295, +0.068876]; 593/51/356; p=8.01116e-21 |
| Official G0 JPEG80 | 0.234617 | 0.286972 | +0.052355 [+0.040400, +0.064337]; 596/48/356; p=2.63313e-19 |
| Official G0 Gaussian5 | 0.222148 | 0.277030 | +0.054883 [+0.042871, +0.066860]; 555/113/332; p=1.98046e-20 |
| Official G0 Gaussian10 | 0.205977 | 0.265002 | +0.059024 [+0.047514, +0.070824]; 554/141/305; p=1.43725e-23 |
| LOKI G1 | 0.076893 | 0.061839 | -0.015054 [-0.027823, -0.002607]; 80/35/114; p=0.00800584 |
| AIGI-test G1 | 0.185580 | 0.200430 | +0.014850 [-0.001455, +0.031145]; 412/76/373; p=0.0475583 |

所有新增定位行均使用同一 sample order、target、geometry 与 mask-logit `>0` threshold；G0/G1 复用冻结 P1 trajectory，corruption 行逐图验证 corrupted RGB SHA256，所有 frozen source 前后 tensor hash 必须一致。完整逐样本记录与统计见 `outputs/p1_r1_reusable_matrix/results.json`。

<!-- PHASE5A4_LEGION_RETRAINED_LOKI_START -->
## 10. Phase 5A-4 LEGION-retrained LOKI localization

Population 为冻结 LOKI 229-image localization scope，GT 与 §4.6 相同：原分辨率 filled bbox union。LEGION 使用官方 image-only L-FREE prompt、free generation、`[SEG]`→SAM、mask logit `>0`、multiple masks union；无 `[SEG]`/空预测在 full-N 主结果中计零。checkpoint 是本项目 internal-train 重训的 Stage-1 merged LE，而不是 public intermediate `legion_LE`。

| Model / protocol | N | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 | valid SEG+mask |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 G1 known-Fake | 229 | 0.061839 | 0.018219 | 0.103666 | 0.045873 | 0.087722 | 220/229 |
| LEGION public intermediate L-FREE | 229 | 0.098816 | 0.059680 | 0.160061 | 0.116530 | 0.208735 | 229/229 |
| LEGION-retrained L-FREE | 229 | 0.079178 | 0.024776 | 0.127911 | 0.124314 | 0.221137 | 226/229 |

R1−LEGION-retrained 的逐图 descriptive mean FG IoU delta=`-0.017339`，bootstrap 95% CI=`[-0.035711, +0.000245]`，W/T/L=`97/32/100`，Wilcoxon p=`0.4076144`。

该行是 **cross-protocol diagnostic**，不是严格同输入条件主 baseline：R1 使用 structural GT `[FAKE]` prefix 的 G1，LEGION 使用不含 GT authenticity/phrase/explanation 的官方 L-FREE。因此 paired statistics 只描述同一图像/GT/evaluator 下的数值差异，不将差异归因为纯模型效应。

LEGION-retrained−public intermediate 是严格同 L-FREE 输入与 evaluator 的配对比较：mean FG IoU delta=`-0.019638`，bootstrap 95% CI=`[-0.038732, -0.000349]`，W/T/L=`68/37/124`，Wilcoxon p=`0.0010422682`。公开权重是 official released intermediate，不是论文 final checkpoint。

LEGION source commit=`d21535dd45f6fea509337a83095966f0b86ac924`；Stage-1 checkpoint canonical SHA256=`6b66fd51f8ea0b26a1930084f858010efc99faa52c04c0802e1666801e304844`；machine result=`outputs/phase5a4_legion_retrained_loki/results.json`。
<!-- PHASE5A4_LEGION_RETRAINED_LOKI_END -->

<!-- PHASE5A4_LEGION_RETRAINED_CONTROLLED_START -->
## 11. Phase 5A-4 LEGION-retrained controlled evaluation

### 11.1 Official1000 localization

三方使用相同的 1,000-image SynthScars population、pixel-union GT、原分辨率 evaluator、logit `>0` 和 full-N failure policy。R1 保留 canonical G0；两种 LEGION 保留官方 image-only L-FREE prompt。

| Model / protocol | N | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 | valid SEG+mask |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 G0 | 1000 | 0.286588 | 0.232139 | 0.393974 | 0.267042 | 0.421521 | historical frozen coverage |
| LEGION public intermediate L-FREE | 1000 | 0.223234 | 0.161411 | 0.321147 | 0.241450 | 0.388980 | 999/1000 |
| LEGION-retrained L-FREE | 1000 | 0.196195 | 0.133239 | 0.286687 | 0.200490 | 0.334014 | 998/1000 |

R1−LEGION-retrained mean FG IoU delta=`+0.090392`，95% CI=`[+0.076050, +0.104308]`，W/T/L=`649/32/319`，Wilcoxon p=`3.0807735e-35`。两者属于 condition-level image-only free-generation parity，但 prompt/architecture 不逐 token 相同。

LEGION-retrained−public intermediate 使用严格相同 L-FREE/evaluator：mean FG IoU delta=`-0.027038`，95% CI=`[-0.040090, -0.013927]`，W/T/L=`387/33/580`，Wilcoxon p=`2.6826542e-08`。

### 11.2 Classification

R1 classification 路径对 P1 是结构性 EXACT REUSE。LEGION 使用官方 Stage-2 CLIP CLS→prediction_head，官方标签 Real=1/Fake=0；下表统一按 Fake-positive 口径报告，threshold=0.5。Internal 保留历史 direct batch=1，AIGI-test 保留历史 batch=8。

| Dataset / model | N | Accuracy | Precision | Fake recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Internal R1/P1 exact reuse | 2208 | 0.983696 | 0.988117 | 0.979167 | 0.988225 | 0.983621 | 0.998389 |
| Internal LEGION-retrained | 2208 | 0.986413 | 0.981183 | 0.991848 | 0.980978 | 0.986486 | 0.999258 |
| AIGI-test R1/P1 exact reuse | 1731 | 0.733102 | 0.739496 | 0.715447 | 0.750575 | 0.727273 | 0.818418 |
| AIGI-test LEGION-retrained | 1731 | 0.816869 | 0.780992 | 0.878049 | 0.756322 | 0.826681 | 0.899467 |

Accuracy delta（LEGION-retrained−R1/P1）：Internal `+0.002717`，McNemar p=`0.45138083`；AIGI-test `+0.083767`，McNemar p=`7.1946242e-21`。

LEGION source commit=`d21535dd45f6fea509337a83095966f0b86ac924`；Stage-1 SHA256=`6b66fd51f8ea0b26a1930084f858010efc99faa52c04c0802e1666801e304844`；Stage-2 SHA256=`f33fda9ddf0e22bcd9bca8dcc998d8a9c0c6421f6bd6a46573044c6cc7365739`；machine result=`outputs/phase5a4_legion_retrained_controlled/results.json`。
<!-- PHASE5A4_LEGION_RETRAINED_CONTROLLED_END -->
