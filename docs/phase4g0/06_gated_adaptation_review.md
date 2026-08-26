# Gated / Bounded Multimodal Adaptation 综述

## MAG

MAG 将 pretrained lexical vector `Z_i` 与 visual/acoustic feature 联合生成模态 gate，再形成 nonverbal displacement `H_i`。关键不是普通 sigmoid，而是对位移做 norm-based bound：

`Zbar_i = Z_i + α H_i`

`α = min(β ||Z_i||_2 / ||H_i||_2, 1)`。

因此 `||αH_i||` 不超过由原 representation 范数和 β 决定的范围。原 BERT/XLNet 主体不改，只附加 MAG，且原论文比较了插入层位置。对本项目最直接的迁移是 `S64 + bounded forensic displacement`，证据等级 **ADAPTED**。

MAG 解决的是 residual magnitude，不直接估计输入可靠性；β 在原文经交叉验证，不能在 Phase 4G-0 被偷换成 validation IoU 架构搜索。未来必须预注册 β 或只把它作为训练安全上界。

## GMU 与 MMTM

GMU 使用互补 gate 在模态 hidden 间插值，说明 sample-conditioned modality contribution 是成熟概念，但原工作缺少 dense geometry、calibration 和 pretrained-path exact recovery，对 Primary 仅为 **INSPIRATIONAL**。

MMTM 通过 joint squeeze + modality-specific excitation 对 CNN channel 重标定，兼容已有 pretrained branches，但没有 spatial conflict 和可靠性校准，适合作为 CMX channel path 的旁证。

## UNO、QMF 与 Predictive Dynamic Fusion

UNO 对 modality-specific segmentation probability 做 uncertainty scaling，并用 noisy-or 融合；还提出 data-dependent spatial temperature scaling。它直接说明 dense task 中 uncertainty 可以是空间化的，但其 uncertainty 在仿真 degradation 场景验证，迁移等级 **ADAPTED**。

QMF 的关键理论准则是：dynamic weight 若与对应 unimodal loss 负相关，才可能优于 static fusion。其 energy-based uncertainty 在原分类任务上只有中等相关性，论文也把现实 uncertainty estimation 列为限制。因此本项目必须先验证 `corr(weight, localization loss) < 0`，而不是把 confidence 名称当作可靠性。

Predictive Dynamic Fusion进一步区分 mono-confidence 与 holo-confidence，并用 relative calibration 修正 Co-Belief。它支持“单源可靠性 + 联合可靠性”比单一 confidence 更完整，但仍是决策级分类迁移。

## 结论

MAG 是防止 Phase 4F residual 无界放大的成熟部件；QMF/PDF 提供 gate 正确性的检验标准；UNO 提供 spatial uncertainty 先例。三者不能机械堆叠：Primary 采用 evidential late fusion时不需要 MAG；只有 Secondary feature rectification 需要 MAG norm cap。

