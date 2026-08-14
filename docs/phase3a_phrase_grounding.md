# Phase 3A：Phrase-Aligned 自回归定位增强

## 1. 执行摘要

Phase 3A 只改变 Fake 语言目标：在完整 forensic explanation 与唯一 `[SEG]` 之间加入来自 `refs.phrase` 的 `Target regions:` 字段。模型结构、union mask、损失、学习率、训练步数、selector、G0 prompt、generation config 与阈值均未改变。用户明确要求不重训 C0，因此训练层面的结论是 **UNPAIRED_TRAINING_COMPARISON**；历史 Phase 2A step 2500 同时作为 C0 与历史参考，不能把全部差异严格归因为 phrase。

正式 P1 checkpoint：step 3500 / epoch 7，selector=`min validation total loss`，official test 在 selector 冻结后才运行。

## 2. 继承的 Phase 2D.1 证据与假设

Phase 2D.1 已确认 G0 与 TF 使用共同的 full-sequence localization downstream，差异已编码在 `[SEG]` predictor representation 中。Phase 3A 检验：显式学习在 `[SEG]` 前生成与 union mask 对应的 authoritative phrase，能否把 TF 条件下的定位能力迁移到真实自由生成 G0。

## 3. 数据与 phrase 审计

仍使用 Phase 2A 的统一 Real/Fake train/val/internal-test split；official1000 仅作冻结后的最终测试。审计原始数据：train Fake=8836，val Fake=1106，internal test Fake=1104；空 phrase 均为 0。多 phrase 按 annotation 顺序、空白归一化、精确重复去重、分号连接；mask 始终沿用原 union mask，不生成 proxy phrase 或新 mask。

## 4. C0/P1 协议与精确差异

C0 是历史 Phase 2A step-2500，不重训。P1 单卡、micro batch=10、gradient accumulation=2、effective global batch=20、5000 optimizer steps、seed=3407、LR=3e-4、linear warmup=100、BF16、ZeRO-2。`max_length=1536` 保持不变；仅对极少数 P1 超长样本保护完整 `Target regions ... [SEG]`，历史 C0 逻辑未改。机器可读 diff 状态：`PASS`。

## 5. Checkpoint 选择与训练纪律

P1 在完整 5000 steps 后按冻结的最小 validation total loss 选择 step 3500。official test 未用于训练、继续训练、checkpoint 选择、prompt/template/threshold/generation 调参。

## 6. Internal evaluation

P1 internal G0 mean foreground IoU=0.166414；TF-full=0.360137；phrase-only oracle=0.260755。G0 是 deployable 主指标，后两者仅为 oracle diagnostic。

## 7. Official1000 G0 主结果

Historical/C0 G0 mean foreground IoU=0.187885、mean foreground F1=0.268184。P1 G0 mean foreground IoU=0.229544、mean foreground F1=0.319323、mean fg/bg mIoU=0.571646；global-pixel foreground IoU=0.236949、foreground F1=0.383118、fg/bg mIoU=0.576302。固定阈值始终为 mask logit > 0，无 sweep。

## 8. Paired effect 与统计

Official1000 同图 paired G0 IoU 差=+0.041659，95% bootstrap CI=[+0.027269, +0.055877]，win/tie/loss=526/77/397，Wilcoxon p=2.76389e-08。这是真实 paired evaluation effect，但训练 run 未 paired，因果等级受限。

## 9. Generated phrase 质量与 mask 关联

target presence=97.2000%，normalized exact match=0.7000%，mean token F1=0.314318；phrase-F1 与 G0-IoU Spearman rho=0.212487（p=1.13158e-11）。这里使用确定性归一化 token overlap，不使用 LLM evaluator。

## 10. Severe failure 恢复

历史 severe 定义固定为 TF IoU>=0.70 且 G0 IoU<=0.30。历史 severe=119；其 mean G0 IoU 从 0.067323 变为 0.234881；P1 在相同阈值下 severe=61。

## 11. G0 与 oracle diagnostics

P1 official G0=0.229544，TF-full=0.437609，phrase-only=0.332896。TF-G0 gap=0.208065。不能用 oracle 指标替代 G0 成功判断。

## 12. Classification non-regression 与 explanation 审计

Internal paired CLS accuracy：C0=0.972373，P1=0.983696，McNemar exact p=0.000346001。LM verdict accuracy：C0=0.968750，P1=0.984149。原始 generation 全量保留，qualitative 页面同时列出 C0/P1 explanation、P1 phrase 与 P1 mask overlay，用于检查 explanation 丢失、重复字段与 hallucination。

## 13. CERTAIN

- 数据/template/config 审计通过；phrase 来自 authoritative `refs.phrase`，mask identity 不变。
- P1 在冻结 selector 和固定 G0/threshold 下的直接测量值、逐图 raw mask/token/phrase 产物与 paired evaluation effect 如上。
- official test 未用于训练或 checkpoint 选择。

## 14. SUPPORTED BUT NON-CAUSAL

- P1 与历史 C0 的差异支持 phrase-aligned protocol 的有效性判断，但因用户决定不重训 paired C0，训练随机性无法完全排除。
- phrase 质量与 mask IoU 的相关性属于机制关联，不是单样本因果证明。

## 15. UNRESOLVED 与 Phase 3B 建议

- 未完成相同初始化、相同时间窗的 C0/P1 paired retraining，因此严格 phrase-only causal gain 未识别。
- C0 历史 official 产物没有 raw spatial logits/mask（Phase 3A 规范建立前生成），其空间产物无法事后补造；P1 已完整保存。
- 不自动启动 Phase 3B。是否进入下一阶段应依据 G0 effect、severe recovery、classification non-regression 与残余 TF-G0 gap共同决定。

## 16. Artifact inventory 与测试

- selector：`outputs/phase3a_phrase_grounding/selection/`
- internal/official evaluation：`outputs/phase3a_phrase_grounding/evaluation/`
- raw spatial predictions：各 mode 的 `spatial/`
- phrase/severe/statistics/classification：对应子目录
- qualitative：`outputs/phase3a_phrase_grounding/qualitative/index.html`
- 正式中文报告：`docs/phase3a_phrase_grounding.md`
- 回归测试结果记录于最终 manifest；正式评测 schema 包含 logits、binary mask、tokens、phrase、TP/FP/FN/TN。

## 17. Experiment discipline

`official_test_used_for_training: false`  
`official_test_used_for_checkpoint_selection: false`  
`proxy_phrase_labels_created: false`  
`threshold_sweep_performed: false`  
`forensic_fusion_used: false`  
`multi_seg_training_used: false`
