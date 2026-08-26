# Cross-Modal Feature Rectification 综述

## CMX 的原始机制

CMX 在每个双流 backbone stage 先做 Cross-Modal Feature Rectification Module，再做 Feature Fusion Module。对同尺寸 `RGB_in, X_in ∈ R^{H×W×C}`：

1. 对两流分别做 global average/max pooling，拼接后经 MLP+sigmoid，拆成两个 channel weight `W_RGB^C, W_X^C ∈ R^C`。
2. 拼接两流 feature，经两个 1×1 conv 与 sigmoid，得到 spatial weight `W_RGB^S, W_X^S ∈ R^{H×W}`。
3. 交叉校正：一流的权重控制另一流特征；输出为输入加 channel 与 spatial residual：

`RGB_out = RGB_in + λ_C (W_X^C ⊙ X_in) + λ_S (W_X^S ⊙ X_in)`，X 流对称。

论文默认 `λ_C = λ_S = 0.5`。消融显示只用 channel 或只用 spatial 均弱于二者结合；CM-FRM 与 FFM 同时使用优于单独使用。其证据等级对本项目为 **ADAPTED**：任务是对齐传感器的语义分割，且原机制是双向、两条可训练 backbone，并未显式证明 reliability 与错误相关。

## 对 Phase 4F 的价值与限制

价值：

- channel 与 spatial 双粒度直接覆盖 Phase 4F 的 global gamma 不足。
- residual 形式兼容 pretrained backbone。
- spatial map 保持几何，对 24×24 forensic grid 到 64×64 SAM grid 有迁移价值。

限制：

- sigmoid attention 是 feature calibration，不是经校准 uncertainty。
- CMX 是对称双流；本项目 language-conditioned SAM 与 forensic evidence 不对称。
- CMX 的固定 λ 仍可能造成全局过度介入。
- 原 FFM 会引入新的 cross-attention/fusion decoder 路径，不能直接替代 P1 SAM decoder。

## MMTM 与 TruFor

MMTM 通过多模态 squeeze 后为各流生成 channel excitation，支持不同空间尺寸和 pretrained unimodal branches，但没有 spatial gate、显式 uncertainty 或 conflict handling，故只能支撑 channel recalibration，证据为 **ADAPTED**。

TruFor 将 RGB 与 Noiseprint++ 通过 transformer feature calibration 融合，并额外输出 localization reliability map。它证明 forensic segmentation 中“预测 mask + spatial reliability”是有任务先例的；但其 reliability map 服务于自身 anomaly map/detection，并没有研究 MLLM language guidance 与 forensic evidence 的冲突。

## 迁移结论

CMX 适合作为 **Secondary feature-level candidate** 的结构血统，但不能单独成为 Primary。若迁移，必须改为非对称：language path 只控制 forensic injection，P1 的 `text_hidden_fcs`、SAM prompt encoder 和 SAM mask decoder 全部冻结保留；并用 MAG 式 norm cap 与独立可靠性校准补足 CMX 没有解决的 dominance 问题。

