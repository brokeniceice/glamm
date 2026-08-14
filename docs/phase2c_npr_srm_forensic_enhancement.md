# Phase 2C：NPR / SRM 取证特征增强

## 1. 核心结论

Phase 2C 完成了预先注册的 classification-first residual late fusion 实验。Phase 2A primary checkpoint 永久保持为 step 2500 / epoch 5，selector 为 minimum validation total loss；Phase 2A 和 NPR/SRM expert 均未继续训练。

实验结果为正向但有限。三种 fusion head 都提高了 frozen internal test 的分类性能。B3（NPR+SRM）在 internal test 上最好：Accuracy 从 0.972373 提高到 0.980978（+0.008605，净纠正 19 个错误），F1 从 0.972411 提高到 0.980961（+0.008550）；paired McNemar exact two-sided p-value 为 0.001319。B1 和 B2 各净纠正 16 个样本。External evaluation 中，B3 将 RAISE real-only FPR 从 0.02906 降到 0.00501，并将 official SynthScars fake recall 从 0.973 提高到 0.979；在 LOKI 上，AUC、AUPRC 和 Accuracy 提高，但 fake recall 仍然较低且略低于 B0，因此不能宣称获得了全面的 OOD robustness。

本实验不支持 localization improvement 结论。Canonical G0/G1/TF 输出按设计保持完全一致，Joint IoU 只能通过 classification gate 变化，并且三个 fusion variant 都略有下降。正确的后续决策是把 B3 保留为 classification-only 候选，目前不训练 V2 cross-attention；除非未来补充更广泛的 OOD 验证，否则研究优先级应回到已确认的 TF→G0 grounding gap。

## 2. Phase 2B 结论与实验动机

Phase 2B 证明 Phase 2A foreground IoU 0.139568 与 LEGION 报告的 fg/bg mIoU 属于不同指标，因此原先看起来约 40 个百分点的差距很大程度来自 metric mismatch。Phase 2B.1 随后修正了 comparison scope：Phase 2A official-1000 的 `0.545930/0.295168` 是 overall 结果，而 LEGION 的 `0.5462/0.2990` 是 Object 类别结果。由于缺少 official-1000 的 authoritative per-image content labels 和 Table 2 精确聚合实现，类别级精确对齐仍未解决。

Phase 2C 不依赖 Phase 2A 与 LEGION Table 2 的精确数值对齐。Frozen NPR/SRM ablation、validation selection、internal/external results 和结论均独立于错误的 overall-vs-Object subtraction。完整说明见 [Phase 2B.1 修正文档](phase2b1_legion_category_parity_correction.md)。

独立存在的 TF→G0 mean foreground-IoU gap 仍为 0.204497。Phase 2C 不处理这个问题，而是验证低层 forensic evidence 能否补充 frozen semantic detector，同时保持 language generation 和 grounding 不变。

## 3. 冻结的 baseline、数据与实验协议

- Phase 2A checkpoint：`checkpoints/phase2a_unified_baseline/single/best/checkpoint/mp_rank_00_model_states.pt`
- Phase 2A SHA256：`07250fe4e82dee3b1a69c2c3b65311404757e6a7ca4e12adb17b4a845304c072`
- Frozen modules：LLM/LoRA、vision tower、multimodal projector、text hidden projector、SAM encoder/decoder、embeddings、LM head 和已有 classification head
- 数据：不变的 Phase 2A train/validation/test manifests，Real/Fake 维持 1:1；没有增加训练数据
- Optimization objective：仅 classification cross-entropy
- Effective global batch：20；AdamW，LR 1e-3，weight decay 1e-4
- Training budget：严格 1500 optimizer steps，每 250 steps validation
- Model selection：B1/B2/B3 分别按 minimum validation classification loss 选择
- External datasets 仅在全部 checkpoint 冻结后评测，不参与选择

Phase 2C validation cache 使用符合部署语义的 fixed prompt-only `[CLS]` context，其 B0 Accuracy 为 0.977848。该上下文与 Phase 2A 训练期间早先报告的 teacher-forced validation context 不同；Phase 2C 所有 validation comparison 都使用同一 prompt-only B0 logits。Internal test 的 B0 也与正式 Phase 2A 输出完成校验：2208 个 prediction 全部一致，最大 probability difference 为 1.19e-7。

## 4. NPR 来源与实现

NPR 来自 CVPR 2024 论文 *Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection*。官方仓库为 `https://github.com/chuangchuangtan/NPR-DeepfakeDetection`，审计版本为 commit `781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a`。

本项目复用 `npr_expert/official_npr_srm.py`：先计算 nearest-neighbor down/up residual，再通过 ResNet bottleneck stem/layer1/layer2 和 global average pooling。输入处理为 Resize(256)、CenterCrop(224)、ToTensor 与 ImageNet normalization，输出 feature dimension 为 512。Phase 2C 只增加 pre-classifier feature API，历史 forward path 保持不变。

## 5. SRM 来源与实现

SRM 是项目历史版本中的 NPR enhancement，并非 NPR 官方仓库的一部分。它使用 9 个 frozen 5x5 RGB high-pass filters，以及小型 Conv/BN/ReLU backend（9→32→128→512）和 global average pooling。Filter 与 backend 均冻结，本阶段没有重新设计 SRM architecture。

共享的历史 expert checkpoint 为 `checkpoints_stage1/official_npr_srm_20260802_062826/best.pth`，SHA256 为 `4100c43edf3fa3adf3f65f1057507aae830f518db5d6a3e35973fedefd8e0cf8`，历史训练数据是 AIGI-Holmes train/validation。该 checkpoint 的 shared classifier 原本在 NPR+gated-SRM 上训练，因此下文 standalone score 是把 frozen shared head 分别作用于单一 branch 的诊断结果，而不是独立训练的单专家最优值。Fusion 使用 classifier 之前的 branch features。

Frozen expert 共 2,068,386 个参数：NPR backbone 1,437,248，SRM backend/gate 630,625，shared classifier 513；Phase 2C 中 trainable expert params 为 0。完整 hash 与 preprocessing 记录位于 `outputs/phase2c_forensic_fusion/expert_audit/provenance.json`。

## 6. 冻结专家的独立性能审计

所有结果均来自同一 frozen prompt-only validation split（n=2212）。

| Detector | Accuracy | Precision | Recall | F1 | ROC-AUC | AUPRC | FPR@95TPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLaMM B0 | 0.977848 | 0.977416 | 0.978300 | 0.977858 | 0.997453 | 0.997529 | 0.008137 |
| NPR branch | 0.793400 | 0.769742 | 0.837251 | 0.802079 | 0.873592 | 0.851918 | 0.483725 |
| SRM branch | 0.801537 | 0.759533 | 0.882459 | 0.816395 | 0.872997 | 0.829832 | 0.378843 |
| 历史 NPR+SRM | 0.834539 | 0.808333 | 0.877034 | 0.841284 | 0.909067 | 0.887456 | 0.342676 |

两个 expert 单独作为 detector 时都明显弱于 GLaMM，因此不应替换原 detector。

## 7. 错误互补性

GLaMM 在 validation 上有 49 个错误。NPR 和 SRM 各自能正确分类其中 34 个（69.39%），历史 joint expert 能纠正 36 个（73.47%）。NPR 与 GLaMM 重叠错误数为 15，Jaccard 0.03055；SRM 与 GLaMM 同样重叠 15 个，Jaccard 0.03171；历史 joint expert 与 GLaMM 重叠 13 个，Jaccard 0.03234。

另一方面，NPR 会在 GLaMM 原本正确的样本上引入 442 个错误，SRM 引入 424 个，历史 joint expert 引入 353 个。较小的 error overlap 证明存在可用的 complementary evidence，而大量 expert-only errors 说明应该采用 residual fusion，而不能直接替换 GLaMM。

## 8. Fusion architecture

所有 raw feature 均冻结。Semantic input 是已有的 fixed prompt-only forensic `[CLS]` hidden state；NPR 和 SRM 分别提供 512-dimensional pre-head feature。每种输入先投影到预先固定的 `d_fuse=256`，进行 normalization，再 concat 并通过小型 MLP 输出两个 residual logits：

`logits_final = logits_base.detach() + alpha * delta_logits`

`alpha` 是 trainable scalar，严格初始化为 0。B1 输入 semantic+NPR，B2 输入 semantic+SRM，B3 输入 semantic+NPR+SRM。Forensic feature 不进入 CLIP visual tokens、LLM sequence、text grounding projection、`[SEG]` 或 SAM。

## 9. Zero-init parity 与 feature cache 校验

训练前，三个 variant 在完整 validation set 上的最大 logit absolute difference 都是 0，prediction 完全一致。Feature cache 保存 sample ID、Phase 2A/prompt/image/expert/preprocessing hash 和 code commit。Train cache 包含 17,672 个唯一、平衡且保持 manifest 顺序的样本。

对随机 32 个样本进行 cached-vs-online comparison：base/LM logits 完全一致；FP16 cached semantic/NPR/SRM feature 的 max absolute error 小于 0.01，minimum cosine similarity 至少为 0.99999988。Real/Fake 使用完全相同且 deterministic 的 preprocessing pipeline。

## 10. Trainable parameter 与 checkpoint selection 审计

| Variant | Selected step | Min val cls loss | 新增 trainable params | Phase 2A trainable | Expert trainable |
|---|---:|---:|---:|---:|---:|
| B1 NPR | 1500 | 0.047788 | 1,321,219 | 0 | 0 |
| B2 SRM | 1250 | 0.046635 | 1,321,219 | 0 | 0 |
| B3 NPR+SRM | 1250 | 0.045306 | 1,518,595 | 0 | 0 |

每次训练的 optimizer parameter IDs 都严格等于 fusion module 的 trainable parameter IDs。三个实验都完成 1500 steps，没有出现 NaN 或 instability，因此没有启动备用 LR sanity check。250/500/750/1000/1250/1500 checkpoints，以及每步 loss、alpha 和 logit norm 均已保存。

## 11. Validation 结果

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | AUPRC | Δ Acc | 纠正 / 新增错误 / 净纠正 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 Phase 2A | 0.977848 | 0.977416 | 0.978300 | 0.977858 | 0.997453 | 0.997529 | — | — |
| B1 +NPR | **0.985533** | 0.990859 | 0.980108 | **0.985455** | 0.998730 | 0.998753 | +0.007685 | 25 / 8 / +17 |
| B2 +SRM | 0.985081 | 0.989954 | 0.980108 | 0.985007 | 0.998797 | 0.998765 | +0.007233 | 24 / 8 / +16 |
| B3 +NPR+SRM | 0.984629 | 0.990842 | 0.978300 | 0.984531 | **0.998904** | **0.998888** | +0.006781 | 22 / 7 / +15 |

B1 的 validation Accuracy/F1 最好，B3 则具有最低 selection loss 和最佳 ranking metrics。每个 model 始终只按各自预注册的 minimum validation loss 选择；没有依据 test 或 external results 重新选择 variant。

## 12. 冻结 internal test 结果

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | AUPRC | Δ Acc | 纠正 / 新增错误 / 净纠正 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 Phase 2A | 0.972373 | 0.971093 | 0.973732 | 0.972411 | 0.996282 | 0.996122 | — | — |
| B1 +NPR | 0.979620 | 0.980054 | 0.979167 | 0.979610 | **0.998141** | **0.998019** | +0.007246 | 21 / 5 / +16 |
| B2 +SRM | 0.979620 | 0.981802 | 0.977355 | 0.979573 | 0.997778 | 0.997606 | +0.007246 | 26 / 10 / +16 |
| B3 +NPR+SRM | **0.980978** | **0.981851** | **0.980072** | **0.980961** | 0.998005 | 0.997846 | **+0.008605** | 26 / 7 / **+19** |

B1/B2/B3 的 absolute F1 gain 分别为 +0.007200、+0.007163、+0.008550；exact paired McNemar p-value 分别为 0.002494、0.011331、0.001319。这些提升幅度有限，但分别对应 16、16、19 个净纠正样本，而不是仅一两个样本。B3 在 test Accuracy/F1 上优于两个单 branch，但并非所有 ranking metric 都最好，因此只能说明 joint feature 有一定价值，不能宣称全面占优。

Calibration 同样改善：B0 Brier 为 0.021394，B1/B2/B3 分别为 0.015691/0.016283/0.015389；ECE 从 B0 的 0.012965 变为 0.012217/0.012611/0.012902。

## 13. 冻结 external/OOD 结果

| Model | LOKI Acc / F1 / AUC / AUPRC | RAISE real accuracy / FPR | Official SynthScars fake recall / mean fake prob |
|---|---:|---:|---:|
| B0 Phase 2A | 0.56743 / 0.52548 / 0.65489 / 0.73865 | 0.97094 / 0.02906 | 0.973 / 0.96718 |
| B1 +NPR | **0.57736** / **0.53127** / 0.66209 / 0.74969 | 0.99098 / 0.00902 | 0.976 / 0.97422 |
| B2 +SRM | 0.57465 / 0.52253 / 0.68568 / 0.76070 | 0.99399 / 0.00601 | 0.978 / 0.97266 |
| B3 +NPR+SRM | 0.57465 / 0.52589 / **0.68929** / **0.76242** | **0.99499 / 0.00501** | **0.979 / 0.97509** |

LOKI 包含 2217 个样本。B3 的 Accuracy、AUC、AUPRC 分别提高 0.00722、0.03441、0.02376，但 fake recall 从 0.40319 降到 0.39711；B1 的 recall 不变。因此 fusion 主要改善 ranking 和 precision，而不是 OOD sensitivity。

RAISE 包含 998 张 real-only 图像，三个 fusion variant 都显著降低 false positive，没有产生新的误报问题。Official SynthScars 包含 1000 张 fake-only 图像，B1/B2/B3 分别多识别 3/5/6 张 fake image。所有 external set 都未影响 checkpoint 或 hyperparameter。

## 14. CLS–LM agreement

所有 variant 的 LM Accuracy 都严格保持 0.968750。CLS–LM agreement 从 B0 的 0.994565 降到 B1 0.986413、B2 0.982790、B3 0.983243。原因是 classifier 接收了 forensic evidence 并得到改善，而独立冻结的 LM verdict 没有接收该信息。这是需要明确报告的 system-level tradeoff，不能被 aggregate classifier metric 掩盖。

## 15. Joint E2E 传递与 localization invariance

| Model | Joint mean FG IoU | Joint global FG IoU |
|---|---:|---:|
| B0 Phase 2A | **0.139568** | **0.133461** |
| B1 +NPR | 0.139514 | 0.133360 |
| B2 +SRM | 0.139376 | 0.133408 |
| B3 +NPR+SRM | 0.139514 | 0.133360 |

Joint evaluation 直接复用 canonical Phase 2A G0 generation 和 mask，只把 fusion decision flip 应用于 classification gate。三个 fusion variant 的 gate pass rate 都为 0.992754，但被 gate 影响的具体样本不同。因此 Joint localization 是极小幅下降，而不是改善。

Real-model regression 证明训练后仍严格不变：LM token logits max-abs difference 为 0；G0 generated token IDs 相同；`[SEG]` position 相同（101）；G0 mask logits max-abs difference 为 0；G1 token IDs 和 mask logits 相同；TF mask logits max-abs difference 为 0。Classification logits 是唯一允许改变的输出。因此 G0/G1/TF 指标严格等于 Phase 2A，没有 generation/localization graph contamination。

## 16. Content 与 source breakdown（B3）

| 内容类别 | n | Accuracy | Fake recall | F1 |
|---|---:|---:|---:|---:|
| Animal | 304 | 0.983553 | 0.986842 | 0.983607 |
| Human | 1162 | 0.983649 | 0.982788 | 0.983635 |
| Object | 388 | 0.984536 | 0.989691 | 0.984615 |
| Scene | 354 | 0.966102 | 0.954802 | 0.965714 |

Scene 仍是表现最弱的内容类别。

| 数据来源 | n | Real accuracy / Fake recall | 错误数 | Mean fake probability |
|---|---:|---:|---:|---:|
| COCO2017 real | 201 | 0.990050 | 2 | 0.01068 |
| FFHQ real | 162 | 1.000000 | 0 | 0.00839 |
| OpenImagesV7 real | 459 | 0.965142 | 16 | 0.03594 |
| PASS real | 242 | 0.991736 | 2 | 0.01207 |
| iNaturalist real | 40 | 1.000000 | 0 | 0.00239 |
| SynthScars fake | 1104 | 0.980072 | 22 | 0.97492 |

OpenImagesV7 的 internal real-source FPR 最高，为 0.034858，并占 B3 20 个 real false positives 中的 16 个。

## 17. Source shortcut 诊断

B3 fake confidence 与 width、height、pixel count、aspect ratio、JPEG quantization proxy 的未校正 Spearman correlation 分别为 −0.1827、0.0652、−0.1545、−0.4286、0.7717。JPEG 图像（n=1757）的 mean fake probability 为 0.4701，fake fraction 为 0.4644；PNG 图像（n=450）分别为 0.6074 和 0.6400。

这些统计只是 diagnosis，不代表 causal effect：format、compression、source、geometry 与 class 在 frozen manifest 中高度混杂。较大的 JPEG-proxy correlation 和 OpenImages error concentration 表明 source/compression shortcut risk 仍然不可忽视，即便 Real/Fake preprocessing 完全对称。本阶段没有删除样本或修改 split。

Feature centroid 同样只用于描述：test 上 semantic/NPR/SRM class-centroid L2 distance 为 21.903/8.979/1.998，cosine similarity 为 0.674/0.935/0.719，不能作为性能证据。

## 18. 参数量与 latency overhead

Latency 使用 64 张图像、batch size 1，并排除 preprocessing。Phase 2A 为 170.976 ms/image。

| Model | 新增参数 | Expert ms | Fusion ms | 估计总耗时 ms | 增幅 |
|---|---:|---:|---:|---:|---:|
| B1 +NPR | 1,321,219 | 2.5421 | 0.0034 | 173.521 | +1.489% |
| B2 +SRM | 1,321,219 | 0.4340 | 0.0033 | 171.413 | +0.256% |
| B3 +NPR+SRM | 1,518,595 | 3.1454 joint | 0.0041 | 174.125 | +1.842% |

由于没有适用于完整 mixed stack 的可靠 profiler，本阶段没有估算 FLOPs。Feature extraction 与 fusion-head training 通过已验证 cache 分开执行；wall-time 未被正式计时，因此不进行事后推测。

## 19. Phase 2C 科学问题的回答

1. NPR standalone Accuracy/F1 为 0.793400/0.802079；SRM 为 0.801537/0.816395。
2. 二者都有 complementary error：虽然 overall 较弱，但各自能纠正 GLaMM 49 个 validation errors 中的 34 个。
3. B1 和 B2 在 internal test 上都优于 B0；B3 的 internal test Accuracy/F1 最佳，并在多数 external ranking/specificity metric 上最好，但 B1 的 validation Accuracy/F1 和 LOKI Accuracy/F1 最好。
4. B3 internal gain 为 Accuracy +0.008605、F1 +0.008550，净纠正 19 个样本。
5. OOD gain 是混合结果：LOKI ranking/precision 和 RAISE specificity 提高，但 LOKI fake recall 没有提高。
6. RAISE false positive 减少，没有恶化。
7. CLS–LM agreement 降低 0.00815–0.01178，而 LM output 保持不变。
8. Joint localization 没有提高，mean FG IoU 最多下降 0.000192。
9. G0/G1/TF 以及所有检查过的 generation/mask tensors 均严格不变。
10. B1/B2 新增 1.32M trainable params，B3 新增 1.52M；最大 latency increase 为 1.84%。
11. Source/compression shortcut risk 清晰可见，在 deeper integration 前需要更广泛且受 source 控制的数据验证。

## 20. V2 gate 与下一阶段建议

Classification-only fusion 已证明存在实际的 complementary signal，因此 B3 值得保留为可选的 frozen classification component。该结果满足“设计未来 V2”的弱前提，但目前不足以支持直接训练 V2：LOKI sensitivity 仍低，source shortcut 仍可能存在，CLS–LM agreement 下降，Joint localization 也没有改善。

如果未来更广泛的 OOD 实验排除了这些问题，候选方案是把 projected spatial NPR/SRM feature 通过 zero-initialized gated residual cross-attention 注入 semantic visual tokens。该模型会改变 explanation 与 grounding path，必须作为新的 controlled phase，而不能视为 Phase 2C 的自动延续。

对应原指令给出的选项，当前建议是 **B + E**：保留 classification-only B3 作为领先候选，然后优先研究 TF→G0 autoregressive grounding-generalization gap。在收到新的明确指令和更充分的 OOD/source-controlled gate 证据前，不 unfreeze expert、不把 forensic feature 注入 LLM/SAM path，也不启动 V2。

## 21. 产物与可复现信息

- 配置：`configs/phase2c_npr.yaml`、`configs/phase2c_srm.yaml`、`configs/phase2c_npr_srm.yaml`
- Fusion module：`model/forensic_fusion.py`
- 主流程：`scripts/phase2c_forensic_fusion.py`
- Cache verifier：`scripts/phase2c_verify_cache.py`
- Internal finalizer：`scripts/phase2c_finalize.py`
- External evaluator：`scripts/phase2c_external_evaluate.py`
- Invariance 与 latency：`scripts/phase2c_output_invariance.py`、`scripts/phase2c_latency.py`
- 全部输出：`outputs/phase2c_forensic_fusion/`
- Regression tests：`tests/test_phase2c_forensic_fusion.py`

Phase 2A step-2500 checkpoint 仍是 primary baseline。Phase 2C 没有修改 grammar、`[SEG]`、classification head 之外的 loss definition、generation、grounding 或 checkpoint selection policy。
