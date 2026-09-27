# Phase6G 历史 R1 候选 Official1000 固定评测

本次在访问候选 Official1000 前，依据 internal DEV canonical G0 冻结 G.7 与 G.1 两个完整 R1 检查点；均为第 8 轮。G.7 的 DEV mean IoU 最高，但 global IoU 下降；G.1 的 DEV mean/global IoU 均略高于同口径 6E.2 回放基线。两者 DEV paired mean-IoU CI 都跨 0，故这次是验证性评测，不把 DEV 点估计当成稳健增益。

Official1000 为相同的 1000 张 Fake、C1 canonical G0、全引用掩码并集、logit 阈值 0；无 `[SEG]` 计零，多掩码取像素级 logit 最大值。候选与历史 Phase6E.2 逐样本 ID、顺序完全相同。没有用 Official1000 调参数或重选 epoch。

| 模型 | mean FG IoU | mean FG F1 | global FG IoU | global FG F1 | SEG trigger |
|---|---:|---:|---:|---:|---:|
| 历史 C1+new R1 | 0.319867 | 0.440385 | 0.343538 | 0.511393 | 0.992000 |
| G.7 block11+17 staged | 0.319427 | 0.433966 | 0.306826 | 0.469575 | 0.992000 |
| G.1 block17 staged | 0.323908 | 0.442676 | 0.337060 | 0.504181 | 0.992000 |

| 严格配对差值 | mean IoU Δ | IoU bootstrap 95% CI | 胜/平/负 | mean F1 Δ | F1 bootstrap 95% CI |
|---|---:|---|---:|---:|---|
| G.7 − 历史 6E.2 | -0.000440 | [-0.011432, +0.010667] | 504/18/478 | -0.006419 | [-0.019049, +0.006305] |
| G.1 − 历史 6E.2 | +0.004041 | [-0.004930, +0.013288] | 543/17/440 | +0.002291 | [-0.007677, +0.012706] |
| G.7 − G.1 | -0.004481 | [-0.013332, +0.004414] | 472/21/507 | -0.008710 | [-0.018792, +0.001464] |

## 解释边界

这两次评测只检验已冻结的完整 R1 路线。判断总体收益时同时查看 mean、global 与严格配对区间；若方向不同，明确记录 trade-off。G.13 的 side/Utility 阳性结果属于另一匹配对照，本次未推断它在正式 C1+new R1 上的外部增益。

证据：`outputs/phase6g_selected_official1000/selection.json`、各臂 `protocol.json`、`predictions.jsonl`、`results.json` 与 `joint_results.json`。没有访问 internal test 或 OOD。
