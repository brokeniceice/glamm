# Modality Imbalance 与 Optimization Dominance 审计

## OGM-GE

OGM-GE 将 joint classifier 拆出 modality-specific contribution，按正确类别 softmax contribution 的比率估计哪一模态学习更快，再缩放 dominant modality 的梯度。作者同时指出梯度缩放会降低 SGD noise，因而加入动态 Gaussian noise 作为 Generalization Enhancement。原论文在 audio-visual classification/event localization 与不同 optimizer 上验证。

可迁移结论：应监控 source-specific loss contribution、gradient magnitude 和 learning speed，而不能只看融合后的总 loss。不可直接照搬之处：本项目是 binary spatial localization，P1 expert 冻结，且 language/forensic 并非两个同构可训练 encoder；OGM 的正确类别贡献公式不能未经验证直接套用。

## PMR 与 MMPareto

PMR 用 class prototype 的 non-parametric classifier 评价各模态学习进度，对慢模态施加 prototype cross-entropy，并在早期用 prototype entropy regularization减轻 dominant modality 抑制。它需要稳定类别 prototype；本项目只有 foreground/background 且 spatial distribution 高度不均衡，直接迁移风险较大。

MMPareto 指出 multimodal 与 targeted unimodal objective 之间可能存在 gradient conflict，并寻找对所有目标共同下降的 Pareto gradient。它提醒“添加 auxiliary expert loss”不必然无害。若 Phase 4G-1 同时使用 fused、language reliability、forensic reliability 三类 loss，应记录 pairwise gradient cosine；发生持续负冲突时才考虑预注册的 Pareto arm。

## Phase 4F 的归因状态

| 审计项 | 现有证据 | 状态 |
|---|---|---|
| forward forensic residual 过强 | TF 从 0.342928 降至 0.205262；固定全局 gamma | LIKELY |
| forensic 参数更快降低 G0 loss | 未保存 per-source counterfactual loss trajectory | UNTESTED |
| language branch 被 under-optimize | P1 language/SAM branch 冻结，严格说不是 OGM 原定义的 under-optimization | NOT ESTABLISHED |
| fusion/gate 参数向 forensic shortcut 收敛 | gamma 增长是线索，但无梯度因果证据 | UNTESTED |

## Phase 4G-1 的最低训练平衡要求

`TRAINING_BALANCE_REQUIRED = YES`，但含义不是默认启用 OGM-GE。Primary 采用冻结双 expert + 分别校准的 evidential heads，从结构上避免更新 P1/forensic backbone；必须记录：

- language evidence、forensic evidence、fusion output 的分项 loss；
- reliability 与实际 pixel/image loss 的相关性；
- 各 trainable head 的 gradient norm；
- gate/discount 分布、all-language/all-forensic 占比；
- fused loss 与两个 reliability loss 的 gradient cosine。

只有预注册阈值显示持续 optimization dominance 时，才允许启用单独的 OGM-style 或 Pareto matched arm；不把它塞进 Primary 默认实现。

