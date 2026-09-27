# R1 演进汇总：P1+R1 → C1+new R1 → 后续改进路线

更新日期：2026-09-24。本文汇总已落盘的 Phase4H、Phase6E、Phase6F、Phase6G.0–G.20 与 Phase6H.1 记录；只报告对应阶段实际验证过的结论，不把不同数据集、不同目标函数或线性探针的数值串成一条性能曲线。

后续补充：[Phase6E.3 C1 原生分阶段 R1](phase6e3_c1_native_staged_r1.md)已按 C1 Rectifier → C1 Utility → C1 joint 顺序完成训练，并完成 [Official1000 逐图配对比较](phase6e3_c1_native_vs_original_c1_r1_official1000.md)。其 mean FG IoU 为 **0.318875**，原 C1+new R1 为 **0.319867**，配对均值差 **−0.000992**、95% CI `[−0.004768,+0.002833]`；global FG IoU 则为 **0.350363 vs 0.343538**。外部定位 OOD 正在卡 0 和卡 2 上评测，尚无最终汇总。下文“最终候选”指此前模型选择时的结论，Phase6E.3 的证据另按新报告解读。

## 先看结论

1. **已完成的正式主线**：P1+old R1 在 Official1000 canonical G0 的 mean FG IoU 为 **0.286588**；C1+old R1 为 **0.300479**；C1+new R1 为 **0.319867**。C1 专属适配相对 C1+old R1 提升 **0.019389**，且 global FG IoU 从 **0.283588** 升至 **0.343538**。因此当时冻结的正式 R1 候选是 **C1+new R1**。[P1/R1 复用矩阵](../outputs/p1_r1_reusable_matrix/results.json)、[6E.1](phase6e1_c1_old_r1_transfer.md)、[6E.2](phase6e2_c1_specific_r1.md)
2. **后续证据源路线**：block11+17 融合在独立空间探针上明显优于单层，但接入完整 R1 后，internal validation mean IoU 虽从 **0.202488** 到 **0.210203**，paired 95% CI 跨零，global IoU 反而从 **0.222581** 降至 **0.211857**。新增 Official1000 固定评测：G.7 mean IoU **0.319427**，与正式 new R1 的 **0.319867** 接近，global IoU **0.306826** 低于 **0.343538**；G.1 mean IoU **0.323908**，但配对增益 CI 跨零且 global IoU **0.337060** 仍低于正式 new R1。均不支持替换正式候选。[6G.2 结果](../outputs/phase6g2_multilevel_attention/phase6g2a/summary.json)、[官方固定评测](phase6g_selected_official1000.md)
3. **后续纠正/Utility 路线**：在各自匹配的内部验证中，零初始化 side correction、单层 Translator、重启后分阶段优化，以及保留来源上下文的 Utility 均提供了机制证据；但 G.19/G.20 的最终内部验证仍未构成对正式 C1+new R1 的直接、完整替代验证。[6G.12](phase6g12_full_rectifier_side_correction.md)、[6G.14](phase6g14_single_matched_translator.md)、[6G.16](phase6g16_type2_correction_aware_utility.md)、[6G.20](phase6g20_evidence_correction_interaction.md)
4. **Phase6H.1 对原始 C 的空间监督**：A0 通过 Phase6E.2 复现关口；A1−A0 internal DEV mean IoU **−0.001407**，95% CI **[−0.004073,+0.001313]**，未达到预注册的 +0.010 门槛，判定 **STOP**。后续单独授权的 A1 Official1000 mean/global IoU 为 **0.316889/0.333648**，历史 6E.2 参考为 **0.319867/0.343538**；A0 没有单独跑 Official1000，不把官方差值当作 A1−A0 因果效应。[6H.1 内部报告](phase6h1_c_spatial_supervision.md)、[A1 官方报告](phase6h1_a1_official1000.md)
5. **main new R1 与 I2 最终选择**：严格核对同一 Official1000 的 1000 张样本、顺序、GT、`[SEG]` 有效性、训练配方与冻结哈希后，main−I2 mean IoU **+0.009480**，配对 bootstrap 95% CI **[+0.002486,+0.016898]**；global IoU **+0.042436**，配对图像 bootstrap CI **[+0.027650,+0.058477]**。保留 main 为论文 C1+R1 定位模型，I2 作随机初始化消融；逐图胜/平/负为 499/25/476，不能宣称多数图像均改善。[最终选择与限制](phase6e2_main_vs_i2_final_selection.md)

## 口径与名称

- **最终 C1 分类架构是 C1-center**：C1 的冻结 RINE-Q2 证据经两层 GELU projector 注入 GLaMM，联合训练的 H2 分类头输出 Fake/Real logit margin；最终决策使用冻结 internal-TRAIN 均值 `−9.190834884678093` 作中心化，`margin > mean` 判 Fake（等价于 `z_C > 0`）。这里的 C1+new R1 定位主线沿用这个 C1 分类架构；new R1 只改变 `[SEG]` 之后的定位路径。后续 OOD 分类汇总应使用 **C1-center** 主指标，raw C1 和 D1 仅按其各自实验角色标注。[C1 模型](phase6d3_c1_rine_conditioned_p1.md)、[决策边界](phase6d6_decision_boundary_disentanglement.md)
- **old R1** 指 P1 阶段选出的 Utility + Rectifier；其权重只作用于 `[SEG]` query 形成之后的 SAM 图像特征/掩码路径，不改 P1 分类或文本生成权重。**new R1** 指 Phase6E.2 用 C1 query/cache 重新适配并由 internal-validation canonical G0 选出的 epoch 8 版本。C1 改变了 `[SEG]` query，不能把 P1 与 C1 的裸分割值视为同一模型中的单模块增益。[复用矩阵](../outputs/p1_r1_reusable_matrix/results.json)、[6E.1](phase6e1_c1_old_r1_transfer.md)、[6E.2 selector](../outputs/phase6e2_c1_specific_r1/selector.json)
- **new R1 的训练来源边界**：main 臂从 P1 路线的 Phase4F epoch 9 Rectifier 和 Phase4H-C A2 epoch 3 Utility 初始化，再固定 C1、用 C1 query 联合训练；没有加载 P1+old R1 的 Phase4H-D 最终权重。因此“C1 专属”指 C1 条件下的联合适配，不能写成“Rectifier、Utility 从 C1 起分别预训练后再联合训练”。I1 只随机重置 Utility，I2 同时随机重置两支；它们均直接在 C1 下联合训练，仍未补齐 C1 原生分阶段预训练的对照。[初始化溯源](../outputs/phase6e2_c1_specific_r1/initialization_provenance.json)、[I1](phase6e2_i1_random_utility.md)、[I2](phase6e2_i2_random_init.md)
- **正式对比**指 Official1000 canonical G0、1000 张、已冻结 selector；**后续 G 阶段**主要为 internal validation Fake、1106 张，包含 16 张无有效 `[SEG]` 的按协议计分样本。两者的绝对 IoU 不可横向比较。G.10–G.15 等 Rectifier 阶段的 IoU 是该阶段的分割目标，不是最终 R1 掩码；独立 1×1 探针只测表示可解码性。[6F.1](phase6f1_c1_newr1_coverage_attribution.md)、[6G.8](phase6g8_evidence_to_correction_gain_attribution.md)
- 下表的「支持」仅对应表中具体对照；“CI 跨零”表示没有稳定正增益证据，并不证明严格等价。新旧 checkpoint、训练初始化、数据群体、评估模式或目标不同的行只作阶段参照。

## 一、P1+R1 到 C1+new R1：正式链路

| 阶段 | 具体改动 | 结果 | 结论 |
|---|---|---|---|
| Phase4H-C/D：形成 old R1 | 保持 P1/CLIP/SAM 冻结；先学习直接门控 `S_adapt=S64+U_F·Δ_F`，随后在匹配 R0 中只训 Utility、在 R1 中同时解冻原 Rectifier。 | internal DEV G0：R0 **0.179644**，R1 **0.195477**，差 **+0.015833**，95% CI `[0.009060,0.022501]`。但 R1 的 Phrase/TF 对 R0 分别 **−0.013830/−0.033274**。 | Rectifier 解冻提高 G0，同时有条件依赖与 oracle 模式退化，不能宣称所有模式改善。[4H-C](phase4hc/report.md)、[4H-D](phase4hd/report.md) |
| P1 → P1+old R1 | 在 P1 query 后加入 old R1；保持 P1 生成/分类路径。 | Official1000 G0 mean/global IoU **0.229544/0.236949 → 0.286588/0.267042**；paired mean IoU **+0.057044**，CI `[0.045424,0.068729]`。Internal G0 **0.166414 → 0.206178**，但这是另一群体。 | old R1 对 G0 定位有效，P1 分类可精确复用。[复用矩阵](../outputs/p1_r1_reusable_matrix/results.json) |
| P1 → C1，暂不换 R1 | C1 更换上游生成/query；冻结 old R1 接到 C1，测可迁移性。 | Official1000 G0：裸 P1 **0.229544**、裸 C1 **0.206240**；P1+old R1 **0.286588**、C1+old R1 **0.300479**。C1+old R1 对裸 C1 的 mean IoU 增益 **+0.094239**；`[SEG]` 触发率 **0.968 → 0.992**（P1 对 C1）。 | C1 改变 base localization 和 query；old R1 仍可迁移，因此**不以“必须重训”作为前提**，但可继续测 C1 专属适配。[6E.1](phase6e1_c1_old_r1_transfer.md) |
| C1+old R1 → C1+new R1 | 沿用 C1 query、原 dense CLIP/Adapter/SAM 路径；在 C1 数据上重新适配 Utility/Rectifier，DEV G0 选 epoch 8，之后才访问 Official1000。 | Official1000 mean IoU **0.300479 → 0.319867**（**+0.019389**）；mean F1 **0.415358 → 0.440385**；global IoU **0.283588 → 0.343538**；global F1 **0.441867 → 0.511393**；SEG 触发率同为 **0.992**。 | **C1 专属 new R1 是本链路冻结的正式候选。**[6E.2](phase6e2_c1_specific_r1.md) |
| 6E.2 初始化归因 I1/I2 | I1：随机 Utility + Phase4F epoch9 Rectifier；I2：Utility 与 Rectifier 均随机；其余 C1/cache、10 epoch、selector 匹配。 | Official1000 mean IoU：old R1 **0.300479**，I2 **0.310387**，I1 **0.317788**，主 new R1 **0.319867**。main−I2 配对差 **+0.009480**，CI `[+0.002486,+0.016898]`；global IoU 差 **+0.042436**。 | C1 条件下的训练本身有效；旧参数初始化非唯一有效路径。严格配对比较支持保留 main 作为最终 C1+R1 定位模型，I2 作为随机初始化消融；I1 与主臂差距小，不能把全部收益归因于预训练初始化。[最终选择](phase6e2_main_vs_i2_final_selection.md)、[I1](phase6e2_i1_random_utility.md)、[I2](phase6e2_i2_random_init.md) |

### old R1 的条件边界

同一 P1/R1 复用矩阵的 Official1000 中，canonical Phrase mean IoU **P1 0.332896 → R1 0.312487**，TF **0.437453 → 0.319176**；对应 global IoU 分别 **0.396181 → 0.332786**、**0.504498 → 0.329902**。这解释了为什么 R1 的主张应限定在实际 G0 生成模式，不能用 oracle Phrase/TF 的值为 R1 背书。[复用矩阵](../outputs/p1_r1_reusable_matrix/results.json)

## 二、6F：从视野覆盖率寻找瓶颈

| 阶段 | 具体改动 | 结果 | 结论 |
|---|---|---|---|
| 6F.0/6F.1 诊断 | 审计 CLIP `ResizeShortest336→CenterCrop336` 与真实 GT coverage，并按 coverage 归因 C1→new R1 的收益；没有训练。 | 1106 张中 212 张存在非方形 crop，15 张 GT coverage=0。内部验证 C1→new R1 overall mean IoU **0.134698 → 0.202488**（**+0.067790**）；coverage=1 的 1018 张 gain **+0.072507**，coverage<0.5 的 23 张 gain **−0.013587**。 | 低覆盖区确是失败来源之一，但大多数样本 GT 已全覆盖；不能预设 full-FOV 会提高总体均值。[6F.0](phase6f0/c1_newr1_forensic_evidence_bottleneck_audit.md)、[6F.1](phase6f1_c1_newr1_coverage_attribution.md) |
| 6F.2/6F.3 全视野接口与冻结回放 | 构造 multi-tile/full-FOV 证据，仅改变取证覆盖，C1、Adapter、Rectifier、Utility、SAM 均冻结。 | internal validation mean IoU **0.202488 → 0.202649**，差 **+0.000161**，CI `[-0.001587,0.002000]`；coverage `(0,0.5)` 子组 **+0.051518**，coverage=1 子组 **−0.001171**。 | 低覆盖组有信号，但零样本整体增益不稳定；原 R1 针对 center crop 分布学习，不能用冻结回放单独否定 full-FOV。[6F.2](phase6f2_full_fov_forensic_interface_audit.md)、[6F.3](phase6f3_full_fov_frozen_replay.md) |
| 6F.4 匹配重训 | full-FOV 下分别从 I2 随机起点联合训练、以及分阶段训练 Rectifier/Utility，冻结 C1/CLIP/Adapter/SAM。 | internal mean IoU：I2 **0.204031**、staged **0.203822**，相对 center-crop new R1 **0.202488** 的 paired CI 均跨零；Official1000：I2 **0.317837**、staged **0.310626**，均低于正式 new R1 **0.319867**。 | full-FOV 尚未给出总体替换证据；保留覆盖率子组诊断，不升级为主路线。[6F.4 汇总](phase6f4_full_fov_training_results.md)、[I2](phase6f4_full_fov_i2.md)、[staged](phase6f4_full_fov_staged.md) |

## 三、6G.0–G.8B：换取证层与 Adapter

以下「probe」是冻结特征上训练的独立空间头；「final R1」才是完整 C1+R1 输出。除特别指出，均为 internal validation 1106 张。

| 阶段 | 具体改动 | 结果（mean FG IoU） | 结论 |
|---|---|---|---|
| G.0 | 对 CLIP block5/11/17/22 做匹配线性 probe。 | **0.094461 / 0.151869 / 0.154149 / 0.143842**。 | 中间层 11/17 比当前 block22 更易解码，值得进入真实 R1 对照。[G.0](phase6g0_multilevel_dense_clip_audit.md) |
| G.1 | 仅以 block17 替代 block22，重训匹配 Adapter 与 R1。 | final R1 **0.202488 → 0.205624**，差 **+0.003136**，CI `[-0.004078,0.010368]`。 | 单层替换不稳定；probe 排名不能直接代表完整 R1。[G.1](phase6g1_block17_single_layer_replacement.md) |
| G.2A | 融合 block11+17，做匹配 probe。 | block17 **0.154149 → 融合 0.186741**，差 **+0.032592**，CI `[0.026037,0.039106]`。 | 多层互补在表示层成立。[G.2A](../outputs/phase6g2_multilevel_attention/phase6g2a/summary.json) |
| G.3/G.3A/G.4 | 在融合特征上测试 spatial-prior Adapter、位置对齐 stem、多尺度局部 Adapter。 | G.3 原 Adapter **0.184992**、spatial prior **0.185957**（+0.000965）；G.3A stem 单独 **0.004547**；G.4 多尺度 **0.184364** 对原版 **0.184992**。 | spatial prior 有很小的 probe 收益，但零 rescue；stem 单独很弱，多尺度容量未带来稳定优势。[G.3](phase6g3_forensic_adapter_audit.md)、[G.3A](phase6g3a_spatial_prior_stem_diagnostic.md)、[G.4](phase6g4_multiscale_local_forensic_adapter.md) |
| G.5 | 在 block11+17 上再加 block22。 | probe **0.186741 → 0.183542**，差 **−0.003199**，CI 跨零。 | 没有证据支持额外 block22；后续融合固定 11+17。[G.5](phase6g5_block11_17_22_fusion_probe.md) |
| G.6/G.7 | 同样的 11+17 证据：G.6 从初始化联合训 Adapter+Rectifier+Utility；G.7 先训 Adapter、再 Rectifier、Utility、joint。 | final R1：原版 **0.202488**；G.6 **0.207724**（差 +0.005236，CI 跨零）；G.7 **0.210203**（差 +0.007715，CI `[-0.000619,0.016106]`）。G.7 global IoU **0.211857**，低于原版 **0.222581**。 | 完整 R1 尚无稳定增益；分阶段的 mean 提升不能遮盖 global 退化。[G.6](phase6g6_joint_from_init.md)、[G.7](phase6g7_block11_17_staged_r1.md) |
| G.8/G.8A/G.8B | 冻结整条链，逐节点 probe 与输入/输出对照。 | G.8：融合证据 probe gain **+0.014796**，到 `Δ_F` 仅 **+0.003107**（CI 跨零）；G.8A 11+17 Adapter input→output **0.178937→0.179411**（+0.000475，CI 跨零）；G.8B 单纯 1024→256 projection 与完整 Adapter 都没有稳定优于原 source。 | 主要损失发生于证据向纠正提案的转换；在这组融合证据上，原三层 Adapter 的额外专门化也未得到支持。[G.8](phase6g8_evidence_to_correction_gain_attribution.md)、[G.8A](phase6g8a_adapter_input_output_probe.md)、[G.8B](phase6g8b_multilevel_evidence_interface_audit.md) |

### G.1 与 G.7 的后续 Official1000 固定评测

在看候选的 Official1000 结果前，按 internal DEV 冻结 G.7 和 G.1 各自的第 8 轮完整 R1 检查点；1000 张 Fake 与历史 Phase6E.2 逐样本 ID/顺序相同。G.7 的 DEV mean IoU 最高；G.1 的 DEV mean/global IoU 均小幅高于同口径基线。两者 DEV 配对 IoU CI 都跨零。官方测试没有用于改选 epoch 或调参。[完整结果与证据](phase6g_selected_official1000.md)

| Official1000 canonical G0 | mean FG IoU | mean FG F1 | global FG IoU | global FG F1 |
|---|---:|---:|---:|---:|
| 历史 C1+new R1 | 0.319867 | 0.440385 | 0.343538 | 0.511393 |
| G.7 block11+17 staged | 0.319427 | 0.433966 | 0.306826 | 0.469575 |
| G.1 block17 staged | 0.323908 | 0.442676 | 0.337060 | 0.504181 |

G.7−历史 new R1 的配对 mean IoU 为 **−0.000440**，95% CI **[−0.011432,+0.010667]**；G.1−历史 new R1 为 **+0.004041**，CI **[−0.004930,+0.013288]**。两者都没有稳定的 mean IoU 正增益，且 global IoU 均下降；因此继续保留历史 C1+new R1 为当前有完整外部证据的基准。

## 四、6G.9–G.15：Rectifier / Translator 的改进与归因

这里的数值是各轮匹配的 **Rectifier 阶段目标**；G.13 明确训练 Utility，但仍不能与 G.7 的 final R1 绝对值混算。

| 阶段 | 具体改动 | 结果（internal mean FG IoU） | 结论 |
|---|---|---|---|
| G.9 | 对 Rectifier 的输入、K/V、cross-attention R2、输出投影、residual/support 逐节点 probe。 | raw R2 的 A1−A0 probe gain **+0.034692**，到输出投影降为 **+0.014240**；后续 fusion/Utility 更不稳定。 | 定位为 Rectifier 内部转换瓶颈；不能从中直接决定删除投影。[G.9](phase6g9_rectifier_internal_conversion_audit.md) |
| G.10 | A0 保留 `out_proj→projection`；A1 去 `out_proj`；A2 两者都 bypass；同时测任务目标和 `Δ_F` probe。 | 任务目标 **0.181244 / 0.179407 / 0.173795**；A2 对 A0 **−0.007448**，CI 全负，即使 A2 的 `Δ_F` probe 更高。 | 投影会改变 probe 可解码性，但直接 bypass **损害任务分割**；保留任务目标作为选择依据。[G.10](phase6g10_post_attention_projection_bypass.md) |
| G.11 | 冻结原 main Rectifier，仅在 raw R2 上加零初始化 1×1 side correction。 | main-only **0.181244 → main+side 0.185546**，差 **+0.004302**，CI `[0.000917,0.007823]`，但 Wilcoxon p=0.304。 | side path 有 paired mean/CI 支持，但统计判据并非全通过；只归因受控侧路，不外推为完整 R1 改善。[G.11](phase6g11_zero_init_correction_side_path.md) |
| G.12 | 对照冻结 main+side 与 joint main+side。 | main-only **0.181244**、冻结 main+side **0.185546**、joint **0.185791**；joint−冻结 **+0.000246**，CI 跨零。 | jointly 解冻没有额外稳定收益；优先冻结 main 训练 side。[G.12](phase6g12_full_rectifier_side_correction.md) |
| G.13 | 冻结 main+side，独立训练 Utility。 | 不带 Utility **0.185546 → 带 Utility 0.192480**，差 **+0.006935**，CI `[0.004509,0.009336]`。 | Utility 能放大这条纠正路线的收益，但 gate 与 side 单样本收益相关性接近零，不能称其已正确判断每一处 side 增益。[G.13](phase6g13_utility_on_main_side_rectifier.md) |
| G.14 | 将两个连续仿射 Translator 精确合并为一个、从函数等价起点进行 fresh matched training。 | fresh factorized **0.179351**、single **0.181284**，差 **+0.001933**，CI `[0.000383,0.003445]`。 | 单层可优化得略好；两者函数类相同，增益归于优化/参数化，不是新增表达能力。[G.14](phase6g14_single_matched_translator.md) |
| G.15 | 基础单层先训 5 epoch，再用零 residual 训 5 epoch；加入同起点、同重启、直接更新单层的 C1/C2 控制。 | 直接连续 A0 **0.181284**；冻结 upstream 的 residual A1 **0.179813**；upstream 共适配 A2 **0.184964**；匹配直接更新 C2 **0.183544**。A2−C2 **+0.001420**，CI `[-0.000044,0.002754]`。 | 分阶段/共适配有收益，但在严格 residual 隔离对照下 CI 跨零；不能断言历史 side gain 源自两分支架构。最终仿射图可解析折叠为单层。[G.15](phase6g15_residual_staged_optimization.md) |

## 五、6G.16–G.20：Utility 如何使用纠正提案

这些试验固定上游 C1、block11+17 融合、G.15 折叠后的 Translator、SAM，仅变 Utility 信息流。这里的 **C** 是真正注入量：`S_adapt = S64 + U_F·C`；**Srect** 是 `S64+C`。G.16 的 Type-I/II、G.17 的 H0/H1、G.19/20 的 A0/A1 使用不同架构或独立训练，只有各自表内的成对对照有因果解释。[G.16](phase6g16_type2_correction_aware_utility.md)、[G.18](phase6g18_utility_context_architecture_audit.md)

| 阶段 | 具体改动 | 结果（internal mean/global FG IoU） | 结论 |
|---|---|---|---|
| G.16 Type-II | 保持 Translator 冻结，比较原来源感知 Type-I Utility 与直接看 `(S64,C,q_seg)`、容量匹配的 Type-II。 | 无 Utility B0 **0.184905/0.205176**；Type-I A0 **0.189357/0.203636**；Type-II A1 **0.184320/0.199347**。A1−A0 mean **−0.005037**，CI 全负。 | 只看纠正结果/基础状态丢失来源上下文；原取证来源信息仍必要。[G.16](phase6g16_type2_correction_aware_utility.md) |
| G.17 Type-III | 在旧来源上下文 Utility 的输出 logit 末端加零初始化 correction residual；同架构 H0 喂零 C、H1 喂真实 C。 | H0 **0.186963/0.200287**、H1 **0.187527/0.200678**；H1−H0 **+0.000563**，CI `[-0.000180,0.001323]`。 | 真实 C 的末端附加价值未稳定成立；不能由此否定更早的上下文交互。[G.17](phase6g17_type3_incremental_correction.md) |
| G.18 结构审计 | 梳理 `F24,z_F24` 的来源上下文与 C/Srect 的信息角色，提出在 64×64 对齐后、CMX 前分别编码证据与提案，再合成 ForensicContext。 | **无训练数值**；给出匹配 C/Srect、零 C 和简单投影控制的实验定义。 | 这是待实验的架构假设，不是已证实增益。[G.18](phase6g18_utility_context_architecture_audit.md) |
| G.19 结构化上下文 | 固定其余组件，只把同一结构化 ForensicContext 的第三路输入设为 C 或 Srect；两臂各 **462,299** 可训练参数，初始 trainable hash 一致。 | C **0.190478/0.202057**；Srect **0.190244/0.201339**；C−Srect **+0.000234**，CI `[-0.000669,0.001138]`。 | 两种组织没有稳定差异；尚缺同架构零输入臂，不能单凭 G.19 证明 C 的增量信息收益。[G.19](phase6g19_r1_c_vs_srect.md) |
| G.20 交互算子 | 复用 G.19 的 C 简单 concat/3×3 merge；仅换成局部 Evidence-query/Correction-key-value cross-attention。 | 简单 merge **0.190478/0.202057**；cross-attention **0.189764/0.204623**；后者 mean 差 **−0.000714**，CI `[-0.002109,0.000713]`，global 指标反而更高。 | 预定 mean-IoU 选择标准下，复杂 attention 没有稳定优势，保留简单结构；记录 global 方向相反，不能称所有指标一致。[G.20](phase6g20_evidence_correction_interaction.md) |

## 六、6H.1：原始 C 的训练期空间监督

A0 只复现 Phase6E.2；A1 在同一原图中对最终注入 SAM 的 `C` 增加 1×1 auxiliary mask head 与 λ=0.1 的 BCE+Dice loss，主 final-mask selector 不变。A0/A1 原始 Utility、Rectifier 初始 state hash 相同；A1 仅多 257 个可训练参数。A0 的第 8 轮 DEV mean FG IoU **0.203351**，相对原始 6E.2 selector **0.202732** 差 **+0.000619**，逐轮样本顺序、有效曝光数及 optimizer updates 全部匹配，复现审计 **PASS**。[完整协议与轨迹](phase6h1_c_spatial_supervision.md)

| internal DEV canonical G0 | A0 | A1 | A1−A0 |
|---|---:|---:|---:|
| mean FG IoU | 0.203351 | 0.201945 | −0.001407 |
| global FG IoU | 0.230169 | 0.224800 | −0.005369 |

A1−A0 的严格配对 mean-IoU 95% CI **[−0.004073,+0.001313]**，预注册要求的增益 ≥+0.010 与 CI 下界 >0 均未满足，结论 **STOP**。A1 auxiliary mask 的 mean FG IoU 为 **0.164076**，不能据此宣称最终掩码改善，也不能在 A0 没有对应辅助头时量化其相对可解码性增益。

用户随后单独授权 A1 selected checkpoint 访问 Official1000：A1 mean/global FG IoU **0.316889/0.333648**，历史 Phase6E.2 为 **0.319867/0.343538**；配对 mean IoU 差 **−0.002978**，CI **[−0.006306,+0.000408]**。这项外部结果与内部 STOP 方向相容，但历史模型不是本次重新训练的 A0，故官方差值只作参考，不替代内部 matched 因果对照。[A1 官方固定评测](phase6h1_a1_official1000.md)

## 综合判断与当前边界

- **已冻结且有 Official1000 直接证据的主候选仍为 C1+new R1**。G.1/G.7 两个完整 R1 路线的同协议 Official1000 测试均未显示稳健总体替换收益；6H.1 的内部预注册判定为 STOP，A1 单独官方测试也没有显示优势。其他 G 系列仍主要是内部瓶颈与局部设计实验。不能把 G.19 的内部 **0.190478** 同 6E.2 的官方 **0.319867** 比大小。[6E.2](phase6e2_c1_specific_r1.md)、[G.19](phase6g19_r1_c_vs_srect.md)、[G.1/G.7 官方固定评测](phase6g_selected_official1000.md)、[6H.1](phase6h1_c_spatial_supervision.md)
- **优先保留的机制发现**：多层 CLIP 证据确有信息互补；但证据到 `C` 的转换会损失增益。side correction 可以改善 Rectifier 目标，Utility 对该 side 路线有额外正收益。单仿射 Translator 提供更简单的等函数类实现。来源上下文不能被仅含 C 的 Utility 替掉。[G.2A](../outputs/phase6g2_multilevel_attention/phase6g2a/summary.json)、[G.8](phase6g8_evidence_to_correction_gain_attribution.md)、[G.13](phase6g13_utility_on_main_side_rectifier.md)、[G.14](phase6g14_single_matched_translator.md)、[G.16](phase6g16_type2_correction_aware_utility.md)
- **尚未解决的问题**：如何把 block11+17 的 probe 增益稳定转成最终掩码且不损失 global IoU；如何在保留来源上下文的同时证明 C 在前段结构化上下文中的**增量**作用；以及 full-FOV 的低 coverage 组收益能否在不损失多数 coverage=1 样本时转化为总体收益。G.19 尚无同架构 zero-C 因果对照；G.20 仅比较交互算子。[G.7](phase6g7_block11_17_staged_r1.md)、[G.19](phase6g19_r1_c_vs_srect.md)、[G.20](phase6g20_evidence_correction_interaction.md)

## 后续尚未完成的工作

| 问题 | 当前证据 | 若继续，需要先完成什么 |
|---|---|---|
| side correction 能否替换正式 new R1 | G.11/G.13 在各自内部对照有正增益，但起点与阶段目标不同；没有与正式 6E.2 完整 R1 的匹配替代对照。 | 先定义同一 C1、同一原始 R1 起点的完整 A0/side A1，对最终掩码做内部 DEV 成对比较和 global 安全检查；通过后才考虑新的外部确认。 |
| C 对结构化 Utility 是否有独立增量价值 | G.19 的 C 与 Srect 差异不稳定，且缺同架构 zero-C 臂。 | 在同架构、同初始化、同训练/selector 的条件下补 zero-C 控制；先停留在 internal DEV。 |
| 低 coverage 样本的改善如何不损失整体 | 6F 的 full-FOV 低覆盖子组有收益，但匹配重训的 Official1000 总体未超过 new R1。 | 先提出固定、可验证的子组触发或无害性机制，并在内部验证覆盖率/误触发；不由现有官方子组结果直接调阈值。 |
| 更广的泛化或语言安全性 | G.1/G.7 和 6H.1 A1 只完成 canonical G0 的本次固定官方测试；未做这些候选的 Phrase/TF、OOD 或 internal test。 | 当前两条 G 路线没有稳健总体增益，6H.1 预注册 STOP，暂不把扩展评测当作必须完成的当前任务。 |

本轮授权的两条历史路线 Official1000 评测、Phase6H.1 A0/A1 收尾和文档归档已完成；没有仍在运行的相关作业。上表是新的研究问题，不是本轮漏跑的步骤。继续训练、增加 Official1000 候选或进入 Phase6H.2 需另立明确实验范围。

本文是现有结果的汇总，不发起新训练、评估或 checkpoint 选择。仓库中个别 G.16–G.20 文档/脚本仍为未跟踪工作文件，本汇总按当前可读结果记录，不把其状态误写成已并入正式主线。
