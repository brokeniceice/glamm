# Phase 4F 证据到机制映射

## 冻结证据图

| Phase 4F 观察 | 可确认结论 | 不能确认 | 所需成熟机制 |
|---|---|---|---|
| G0 +0.023381，CI 全正 | forensic compensation 在 G0 有效 | 所有 language conditions 均有效 | quality-aware dynamic fusion |
| matched > cross-image | 使用 image-specific evidence | 学到了可解释置信度 | per-sample/per-pixel reliability audit |
| matched > spatial-shuffle | 使用正确空间几何 | 已定位语言错误区域 | geometry-preserving spatial fusion |
| forensic > CLIP +0.013631，CI 全正 | 当前 G0 endpoint 有 forensic specialization | forensic 普遍优于 CLIP | matched evidence-source arm |
| TF 0.342928 → 0.205262 | P1 oracle capability 被削弱 | fixed residual 是唯一原因 | bounded/asymmetric/conflict-aware fusion |
| gamma 最终约 0.05 | 全局介入强度被优化放大 | 存在已证 gradient dominance | norm cap + train-time contribution audit |
| Phrase、TF 仍高于 G0 | language sensitivity 未完全消失 | oracle gap 得到充分保留 | Pareto rather than scalar selection |

## 问题到论文机制

| 当前问题 | 文献机制 | 迁移等级 | 采用方式 |
|---|---|---|---|
| 固定 gamma 对所有位置/样本相同 | CMX channel+spatial rectification；MAG norm-bounded shift | ADAPTED | 作为 feature-level Secondary，不直接照搬双流 backbone |
| 可靠 language 不应被持续覆盖 | QMF 权重与 unimodal loss 负相关准则；UNO/PDF uncertainty calibration | ADAPTED | 先证明 reliability 与错误相关，再允许动态融合 |
| forensic 必须保持正确位置 | TokenFusion residual positional alignment；CMX spatial rectification | ADAPTED | 使用原图归一化坐标，不打乱 24×24→64×64 对应 |
| 两源冲突时不知道信谁 | ECoLaF conflict discounting；TMC evidential uncertainty | DIRECT/ADAPTED | Primary 使用 pixel-wise evidential late fusion |
| forensic 可能在训练中占优 | OGM-GE、PMR、MMPareto | ADAPTED | 先做贡献/梯度审计；不把分类公式未经验证地直接硬套 |
| gate 可能恒为 0/1 | TMC evidential loss、QMF uncertainty-loss correlation、TokenFusion auxiliary pruning loss | ADAPTED | 用可验证可靠性目标而非任意 entropy soup |
| frequency/forensic 对不同样本有双刃剑效应 | Omni-IML sample-specific modal gate | DIRECT（取证任务） | 支持 sample-adaptive 必要性，但其 decoder replacement 不采用 |
| forensic localization 本身有不确定性 | TruFor reliability map、UMFNet pixel uncertainty | DIRECT/ADAPTED | reliability head 必须输出可审计空间图 |

## 为什么不能仅加一个 sigmoid scalar

Phase 4F 已经显示问题同时具有 sample、spatial、conflict 和 optimization 四个层面。单一 `sigmoid(MLP)` scalar 无法说明某个区域为何信任 forensic、无法保留位置冲突信息，也无法验证其权重是否与真实错误负相关。文献支持的最低充分形式至少应具有 pixel-wise reliability/conflict；feature-level 方案还需 channel/spatial rectification 与有界位移。

## 初始化判断

未来默认从 **frozen P1 + pretrained Phase 4C-A forensic adapter** 新建融合/可靠性模块。Phase 4F epoch-9 仅作为证据与比较点，不作默认 warm start，因为其表示已表现出 forensic-biased trade-off，warm start 会混淆“架构是否有效”和“是否继承旧表征”。

