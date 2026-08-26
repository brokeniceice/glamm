# Phase 4G-0 最终报告

## 结论

Phase 4F 已证明 forensic evidence 对 canonical G0 有真实、图像特异和空间特异的正贡献，但其固定 feature rectification 只保留约 17.28% 的 P1 autonomous–oracle gap。文献审计表明，最成熟且与当前问题最匹配的下一步不是再加一个 scalar gate，而是把 frozen P1 与 frozen forensic prediction视为两个空间 expert，用经校准的 evidential uncertainty 与 pixel conflict 做 adaptive discount/late fusion。

```yaml
PRIMARY_FAILURE_MODE_OF_PHASE4F: BOTH_PLAUSIBLE
BEST_LITERATURE_MECHANISM_FAMILY: conflict-guided evidential reliability-aware dense late fusion
PRIMARY_CANDIDATE: PCERF
SECONDARY_CANDIDATE: AHBFR
RELIABILITY_GRANULARITY: hybrid
TRAINING_BALANCE_REQUIRED: YES
P1_LANGUAGE_PATH_PRESERVED: YES
ORIGINAL_SAM_DECODER_PRESERVED: YES
FORENSIC_GEOMETRY_PRESERVED: YES
DEPLOYMENT_COMPATIBLE: YES
PHASE4G1_FULL_TRAINING_JUSTIFIED: NO
```

当前的 `NO` 不否定 PCERF 路线。它表示 Phase 4G-0 足以支持“实现与校准 preflight 的方案审阅”，但 reliability 尚未在本项目上证明与 source error 稳定相关，因此此刻不能直接授权 full training。只有公式 parity、invariants 与 reliability preflight 全通过，并获得新的明确授权后，才能把该字段在 Phase 4G-1 协议中改为 YES。

## Architecture decision

### PRIMARY：PCERF

TMC/QMF/ECoLaF/TruFor 血统；冻结 P1 与 forensic expert，训练 pixel evidential heads 与 conflict discount。它把 Phase 4F 的不可逆 feature override 转成可观察的专家权重，并在 forensic vacuous 时 exact recover P1。

### SECONDARY：AHBFR

CMX channel/spatial rectification + MAG norm cap + uncertainty map。它保留原 SAM decoder和几何，但仍在 S64 内注入 residual，表示/优化 dominance 风险高于 Primary。

### REJECTED FOR FIRST STUDY：TSST

TokenFusion 的位置保持与 token选择成熟，但硬替换、language reliability 语义不匹配和 24→64 支持问题使其不适合首轮。另拒绝 Phase 4F warm start、uncalibrated scalar gate 与 over-composed 模块堆叠。

## 可靠性结论

部署 gate 只能看 canonical trajectory 自然产生的 q、P1 logits/S64、forensic logits/F24 与两预测冲突。LLM token entropy、softmax margin、forensic activation 都只是候选输入；只有通过 ECE/Brier/NLL、uncertainty–error correlation、QMF负相关和 permutation test，才可称为 reliability。

## 文献缺口

FakeShield、ForgeryGPT、SIDA 和 Propose-and-Rectify 均推进了 explainable IFDL 或 forensic-to-SAM/MLLM coupling，但没有显式解决 language-conditioned grounding 与 forensic evidence 的 calibrated conflict routing。TruFor、Omni-IML、ECoLaF 与 UMFNet 分别提供 forensic reliability、sample-adaptive forensic utility、conflict discount 与 pixel uncertainty 的关键先例。

## 停止状态

Phase 4G-0 已完成并停止。未实现候选、未训练、未做 validation performance probe、未调 gamma、未启动 Teacher/KD，internal test 与 official1000 未使用。等待人工审阅 `11_candidate_architectures.md`、`13_primary_architecture_freeze_proposal.md` 与 `14_phase4g1_training_proposal.md`。
