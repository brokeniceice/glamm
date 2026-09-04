# CMX mechanism review

Primary source：[CMX, IEEE T-ITS 2023](https://arxiv.org/abs/2203.04838)，官方实现见 [RGBX Semantic Segmentation](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)。

## 原生机制

CMX 在双流 backbone 的多个 stage 使用 Cross-Modal Feature Rectification Module（CM-FRM）。channel path 对两模态做 global average/max pooling 后联合生成 channel weights；spatial path 对 channel-compressed 双模态 feature 生成 spatial weights。每一流都用另一流的加权 feature 做 residual rectification。随后 Feature Fusion Module（FFM）先交换 cross-attention 的长程 context，再做 mixed channel embedding。

论文 ablation 说明不能把机制缩写成一层 gate：仅 CM-FRM、仅 FFM 分别优于简单平均，二者合用更强；channel-only 或 spatial-only 均弱于完整 rectification。CMX 的价值是“联合 feature interaction”，不是 calibrated reliability 的现成证明。

## 对当前问题的可迁移部分

- joint channel descriptors：可表达 forensic channel 与 language/SAM context 是否相容；
- aligned spatial rectification：同坐标 L/F 局部关系直接进入 utility path，理论上对 spatial shuffle 敏感；
- context exchange：允许 `q_seg` 条件下的 long-range semantic agreement；
- CM-FRM+FFM 的完整性提醒：Candidate A3 不能退化成 `sigmoid(1x1conv(F))`。

## 不可直接照搬部分

CMX 是对称、可训练双 backbone，并让 rectified features 继续进入下一 stage；本项目的 P1 language path 是冻结 anchor。Primary 不在 `S64` 内覆盖 P1 representation，而只在旁路 context/utility branch 吸收 CMX interaction。若把 CMX residual 注入 `S64`，那属于 Secondary AHBFR，必须加 MAG 式 norm bound。

CMX attention/rectification weights不能直接命名为 reliability 或 utility；它们只提供 joint interaction feature，最后必须经过显式 `U_F` 定义与 future validity audit。
