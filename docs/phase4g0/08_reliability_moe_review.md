# Reliability、Conflict 与 MoE Routing 综述

## Evidential reliability

TMC 将每个 view 的 evidence 参数化为 Dirichlet distribution，用总 evidence 形成 uncertainty，并通过 Dempster–Shafer 组合。它提供了比 raw softmax 更明确的 epistemic/evidential表示，但原任务是分类。

ECoLaF 把这一思路推进到 semantic segmentation：每个传感器输出 pixel-wise evidence，依据模态间 conflict 自适应 discount，再 late fuse。其消融显示去掉 adaptive discounting 后模型更依赖单一模态，在 sensor failure 下显著退化。对本项目的“两个空间专家冲突”是最直接的 **DIRECT/ADAPTED** 证据。

2026 UMFNet 将每像素 feature 建模为 Gaussian latent distribution，以 local uncertainty 寻找跨模态一致区域，再用 uncertainty-derived confidence map调节 RGB-T 融合。它直接支持 pixel-wise uncertainty 与 spatial fusion，但目标是 unaligned RGB-T salient object detection，且 architecture 较新，作为 **ADAPTED** 补强证据。

## Quality-aware dynamic fusion

QMF 给出可检验条件：权重必须与对应 source loss 负相关。Predictive Dynamic Fusion进一步用 mono/holo confidence 和 relative calibration。二者共同表明：

1. reliability head 的输出不能仅看数值范围；
2. 必须测试 calibration 和与 correctness 的关系；
3. static-weight 与 reliability permutation 是必要反事实。

## MoE routing

通用 MoE 证明 trainable router 可以按输入结构路由，并需要 load balancing 以避免 expert collapse。但 language-conditioned SAM 与 forensic evidence 已经是两个有明确语义的 expert，不需要扩大成 top-k 大型 MoE。通用 MoE 在此只提供 **INSPIRATIONAL** 的 router-collapse 警示，不作为 Primary 核心血统。

## 为什么选择 evidential late fusion 为 Primary family

- 与 dense segmentation 的 ECoLaF 任务层级最接近；
- 两个专家可以冻结，P1 与 forensic specialization 不互相改写；
- pixel-wise conflict、discount 和 source ablation 都可直接观测；
- 当 forensic evidence 被 discount 到 vacuous 时，可 exact recover P1 logits；
- 避免 Phase 4F 在 feature representation 内不可逆地覆盖 language path。

限制同样明确：late fusion 可能无法利用 feature-level互补；evidence calibration失败时整个机制没有可信基础；24×24 forensic logits 上采样可能限制边界精度。因此 Phase 4G-1 必须先做 reliability preflight，而不能直接用 full training 验证架构。

