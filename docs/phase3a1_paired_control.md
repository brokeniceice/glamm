# Phase 3A.1：配对 C0 确认与 TF 协议交叉评测

## 1. 执行摘要

本阶段没有重训或修改 P1。我们新训练了一个与 P1 在单卡执行环境、seed、优化器、学习率、batch、5000 steps、数据与 selector 上匹配的 C0；唯一语义差异是 C0 沿用历史 Fake target，不含 `Target regions:`。主结论门为 **PHRASE_EFFECT_STRONGLY_CONFIRMED**。

## 2. 为什么需要 Phase 3A.1

Phase 3A 的 P1 比 historical Phase 2A checkpoint 更强，但两者来自不同训练轨迹，因此历史 +0.041659 只能称为 checkpoint effect，不能全部归因于 phrase。本阶段用 new C0 缩小该混淆，并用 TF 2×2 拆开 checkpoint effect 与 test-time context effect。

## 3. 初始化审计

P1 原始 step-0 snapshot 不可恢复，状态保持 `BASE_CHECKPOINT_MATCH_BUT_RANDOM_INIT_UNPROVEN`。相同代码与 seed 重建出的 P1/C0 初始化，以及正式 new C0 启动初始化，全量参数与可训练参数哈希一致；这支持 intended initialization identity，但不升级为原始 P1 step-0 已逐字节恢复。

## 4. 训练环境差异与 matched C0 定义

new C0 与 P1 都是单卡、micro-batch=10、gradient accumulation=2、effective batch=20、bf16、ZeRO-2、seed=3407、LR=3e-4、warmup=100、5000 optimizer steps、max_length=1536。用户要求的 CUDA allocator cache 高水位保留是非语义运行策略，不改变模型、数据、loss 或 selector。

## 5. C0 训练与 selector

训练完整到 step 5000，按 `min validation total loss` 冻结 step 2500 / epoch 5，val total loss=1.509045。official1000 在 selector 冻结后才启动，未参与选模或 threshold 调整。

## 6. Historical / new C0 / P1 official1000 G0

| 模型 | mean FG IoU | mean FG F1 | mean fg/bg mIoU |
|---|---:|---:|---:|
| Historical Phase 2A | 0.187885 | 0.268184 | 0.551843 |
| New paired C0 | 0.179786 | 0.251918 | 0.548158 |
| P1 | 0.229544 | 0.319323 | 0.571646 |

## 7. Phrase-only paired effect

P1 − new C0：mean FG IoU=+0.049758，median=+0.014324，bootstrap 95% CI=[+0.034534, +0.064341]，win/tie/loss=569/89/342，Wilcoxon p=5.36112e-15。FG F1 差=+0.067405，fg/bg mIoU 差=+0.023487。

## 8. Severe failure

固定 historical severe 119 个样本上三模型的 mean G0 IoU 为：{"historical_phase2a": 0.06732322918465485, "new_paired_c0": 0.14833139453713656, "p1": 0.23488066973443691}。协议显式 severe count 为：{"historical_tf_old": 119, "new_c0_tf_old": 105, "p1_tf_old": 27, "p1_tf_phrase": 101}。P1 的 TF-OLD 与 TF-PHRASE severe 数必须分开解读。

## 9. Classification non-regression

Internal 2208：new C0 CLS accuracy/F1=0.977355/0.977190；P1=0.983696/0.983621。new C0 LM accuracy=0.975543，P1=0.984149；CLS-LM agreement 分别为 0.996377/0.996830。McNemar 详见机器可读 artifact。

## 10. 2×2 TF cross-evaluation

| checkpoint | TF-OLD mean FG IoU | TF-PHRASE mean FG IoU |
|---|---:|---:|
| Historical Phase 2A | 0.408905 | 0.340545 |
| P1 | 0.250854 | 0.437609 |

仅增加 test-time authoritative phrase：Historical B−A=-0.068361，P1 D−C=+0.186754。同 TF protocol 的 checkpoint effect：TF-OLD C−A=-0.158051，TF-PHRASE D−B=+0.097064。TF 是 oracle diagnostic，不替代 G0。


## 10.1 补充诊断：new paired C0 × TF-PHRASE

本补充实验只运行冻结的 new paired C0（step 2500 / epoch 5）× 既有 TF-PHRASE × official1000；相对 P1 TF-PHRASE，唯一关键模型变量是 checkpoint。未重训 P1/C0，未调整 threshold、phrase、mask、SAM 或 evaluator。

new C0 TF-PHRASE：mean FG IoU=0.337845，median FG IoU=0.285047，mean FG F1=0.452692，mean fg/bg mIoU=0.630824。按 FG IoU≤0.30 定义的 severe failure 为 519。

| checkpoint | TF-OLD mean FG IoU | TF-PHRASE mean FG IoU | PHRASE − OLD |
|---|---:|---:|---:|
| New paired C0 | 0.400811 | 0.337845 | -0.062966 |
| P1 | 0.250854 | 0.437609 | +0.186754 |

new C0 的 paired FG IoU 差：mean=-0.062966，median=-0.023110，bootstrap 95% CI=[-0.075255, -0.051087]，win/tie/loss=364/7/629，Wilcoxon p=2.30587e-22。FG IoU≤0.30 severe count：TF-OLD=406，TF-PHRASE=519。phrase context 对 new C0 并非普遍有益；相反，P1 显示出对 phrase-conditioned context 的特异适应。

matched interaction `(P1 PHRASE − P1 OLD) − (C0 PHRASE − C0 OLD)`=+0.249720，sample-level paired bootstrap 95% CI=[+0.228733, +0.271278]。该 TF 结果是 mechanism/oracle diagnostic，与 G0 matched-control 主结论分开解读。

## 11. Historical gain decomposition

Historical→new C0 与 new C0→P1 只作 controlled decomposition，不能宣称为严格可加的因果方差分解。对应数值见 `statistics/historical_gain_decomposition.json`。

## 12. 证据边界

CERTAIN：P1 与 historical checkpoint 的 G0 差异存在；本次 official scope、阈值与 evaluator 一致；P1 未重训；new C0 selector 在 official test 前冻结；TF-OLD/TF-PHRASE 已交叉运行。

SUPPORTED BUT NON-CAUSAL：确定性重建初始化一致；new C0 提供了更接近 P1 的受控参照。

UNRESOLVED：原始 P1 step-0 snapshot 不存在，原始运行的逐字节初始化身份与不可见系统轨迹不能事后证明。因此即使门为 strongly confirmed，也不能声称完整 historical +0.041659 都由 phrase 导致。

## 13. Phase 3B gate

P1 的 TF-PHRASE−G0 gap=+0.208065。Phase 3B 建议：**推荐作为候选，但本阶段未启动**。无论结论如何，本阶段均未启动 Phase 3B。

## 14. Artifact、测试与实验纪律

完整 artifact 位于 `outputs/phase3a1_paired_control/`：audit、训练日志与 checkpoints 引用、selection、G0、classification、severe、TF 2×2、statistics、qualitative HTML。正式报告同时放在 `docs/` 与 outputs/reports。本次全量回归为 147 passed、3 skipped、5 个预期 warning。未使用新 phrase label、新 mask、NPR/SRM、SAM 修改、新 loss、threshold sweep 或 test-set 选模。
