# Phase6E.3：C1 原生分阶段 R1 vs 原 C1+new R1

Official1000 canonical G0，完整 1000 张，逐图 ID、顺序、GT 前景像素数和 `[SEG]` 有效性一致。差值方向为新方案减原方案。

| 指标 | C1 原生分阶段 R1 | 原 C1+new R1 | 差值 |
|---|---:|---:|---:|
| Mean FG IoU | 0.318875 | 0.319867 | -0.000992 |
| Mean FG F1 | 0.439764 | 0.440385 | -0.000620 |
| Global FG IoU | 0.350363 | 0.343538 | +0.006825 |
| Global FG F1 | 0.518917 | 0.511393 | +0.007524 |

逐图配对 mean FG IoU 差的 95% bootstrap CI：`[-0.004768, +0.002833]`；胜/平/负 `464/25/511`，Wilcoxon 双侧 `p=0.212470`。
图像级配对重采样的 global FG IoU 差 95% CI：`[+0.000393, +0.013079]`。
判定：`NO_STABLE_UNIDIRECTIONAL_GAIN`。按预定主指标 mean FG IoU，C1 原生分阶段路线没有稳定优于原方案；global FG IoU 的配对区间为正，应单独报告。两指标方向不同，不据此宣称全面改进。

本轮阶段选择只使用 internal DEV；Official1000 已参与历史候选选择，因此这里是匹配对照，不能称为全新独立测试。两方案的预训练来源与总训练量不同，该对照估计的是整条训练路线的效果，不是单个预训练步骤的因果效应。分类保持 C1-center，本对照只检验定位。

历史参考：C1+old R1（Phase6E.1 冻结迁移）的 mean/global FG IoU 为 `0.300479/0.283588`；C1 原生分阶段方案相对它的配对 mean FG IoU 差为 `+0.018396`，95% CI `[0.009915107994727659, 0.02699616334977931]`。这不是上表“原 C1+new R1”的主比较。

逐图、哈希和完整配对统计见[JSON](../outputs/phase6e3_c1_native_staged/official1000_paired_vs_original_c1_r1.json)。
