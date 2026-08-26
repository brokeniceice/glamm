# Phase 4G-0 问题定义

## 冻结起点

本阶段只做文献研究、机制审计、架构综合与可实现性审计，不训练、不做 validation IoU 架构搜索，也不使用 internal test 或 official1000。内部问题只由 Phase 4F 冻结结果定义。

| 路径 | G0 | Phrase-only | TF-full |
|---|---:|---:|---:|
| P1 | 0.148233 | 0.245798 | 0.342928 |
| Phase 4F FORENSIC-RECT | 0.171614 | 0.193573 | 0.205262 |

Phase 4F 在 canonical G0 上相对 P1 增加 0.023381，95% CI 为 `[0.014357, 0.032572]`；matched/cross-image/spatial-shuffle/zero/off 分别为 0.171614/0.095691/0.126938/0.149094/0.148233。由此只能确认：正确图像、正确位置的 forensic evidence 被真实利用，且对部署条件 G0 有正贡献。不能据此声称所有语言条件均获益。

P1 的 oracle gap 为 `0.342928 - 0.148233 = 0.194695`。Phase 4F 虽仍满足 `G0 < Phrase < TF`，但 oracle gap retention 仅 17.28%。因此问题不是“forensic residual 是否有用”，而是：

> 当稀疏的 language-conditioned grounding 与稠密 forensic spatial evidence 的质量不同甚至冲突时，模型能否用部署时可得、经校准的可靠性信号，在合适粒度上决定信任谁、信任多少？

## 两类不可混淆的主导问题

### Representation dominance

前向过程中固定/全局式 `S' = S + gamma R(F)` 对所有语言条件持续注入 forensic residual。Phase 4F 的 TF 大幅下降与这一机制相符，但尚无 matched adaptive-gate 实验，因此状态为 **LIKELY**，不是已证因果。

### Optimization dominance

G0-only mask loss 可能优先奖励更快降低损失的 forensic path，使 gamma、投影或融合参数越来越依赖 forensic evidence。Phase 4F 中 gamma 增长提供了线索，但没有模态分解的梯度、损失贡献和学习速度审计，因此状态为 **UNTESTED**。

最终归因：`PRIMARY_FAILURE_MODE_OF_PHASE4F = BOTH_PLAUSIBLE`。其中表示主导证据更强，优化主导仍需未来审计。

## 概念边界

- G0、Phrase、TF 是同一 language grounding trajectory 的不同质量条件，不是三个训练模态。
- 真正的信息源是 language-conditioned SAM 与 dense forensic evidence。
- G0 是部署 endpoint；Phrase/TF 是 oracle diagnostic，不能作为 gate 身份输入。
- future gate 不得使用 GT、权威 phrase、TF 标识、mask/polygon 或未来评测条件。
- 目标是 G0 与 language controllability 的 Pareto 改善，不是机械恢复某个 TF 单点。

## 内部机制假设

| 假设 | 状态 | 证据边界 |
|---|---|---|
| H1：Phase 4F G0 增益来自 forensic 对不完美 language grounding 的补偿 | **SUPPORTED** | G0 paired CI 为正，且 matched 显著优于 cross/shuffle/zero/off |
| H2：TF 回退来自不能随 language reliability 调节的固定/全局 rectification | **LIKELY** | 与现象和结构一致，但未做 adaptive matched intervention |
| H3：reliability-aware fusion 可保留 G0 补偿并减少可靠语言下的不必要介入 | **UNTESTED** | 文献支持可迁移性，不是本项目实证 |
| H4：训练期 modality imbalance 独立促成 Phase 4F trade-off | **UNTESTED** | 尚无分支贡献/梯度审计 |

## 本阶段判定标准

Primary architecture 必须同时满足：原 P1 language path 可恢复、原 SAM decoder 保留、forensic 几何保留、reliability signal 部署可得、gate 非 TF 身份代理、冲突处理有原论文依据、每一新增机制均能映射到一个已观察问题，并能设计清晰的反事实消融。

