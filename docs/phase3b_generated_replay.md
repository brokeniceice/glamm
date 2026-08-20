# Phase 3B：固定策略 Generated-Context Replay 的配对实验

## 1. P1 source checkpoint / hash

两臂均从冻结 P1 step 3500 / epoch 7 继续：`/data/yz/groundingLMM_official/checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt`；P1 SHA256=`fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326`。两臂使用 fresh、相同的 AdamW/scheduler 状态，模型可训练权重 zero-step hash 另见 audit。

## 2. Replay cache eligibility

内部 train Fake=8836，eligible=8690（98.3477%）；唯一资格规则是恰有一个、且前方存在 predictor token 的 `[SEG]`。NoSEG 不补写、不纠正、不进入辅助分支。cache SHA256=`6ab4c428748e426132edaf6570ee38f12bbc8c1a2e0d44b089152563fd0990ed`。

## 3. B0 selected step

B0 选择 step 750 / epoch 3，checkpoint SHA256=`63061c3580a73dc34645fabdd6220fe7775e0c902ff389dd7adb46392e3ac007`。

## 4. B1 selected step

B1 选择 step 1500 / epoch 6，checkpoint SHA256=`8b719cea43fcac4c53c176badef78b4afe7852fc9480d465cb4616ecaf5e3eee`。

## 5. Validation selector values

| Arm | val Fake G0 mean FG IoU | mean FG F1 |
|---|---:|---:|
| B0 gold replay | 0.167508 | 0.233930 |
| B1 generated replay | 0.169167 | 0.235068 |

Selector 仅使用 internal val Fake G0；internal test、official1000 和 TF 均未参与选模。

## 6. Internal test B0/B1 G0

| Arm | mean FG IoU | mean FG F1 | mean fg/bg mIoU |
|---|---:|---:|---:|
| B0 | 0.171717 | 0.240630 | 0.557497 |
| B1 | 0.164684 | 0.233652 | 0.557213 |

## 7. Official1000 B0/B1 G0

| Arm | mean FG IoU | mean FG F1 | mean fg/bg mIoU |
|---|---:|---:|---:|
| B0 | 0.232592 | 0.325042 | 0.573264 |
| B1 | 0.233874 | 0.327673 | 0.575125 |

## 8. Paired B1−B0 statistics

Official FG IoU mean=+0.001283，median=+0.000097，bootstrap 95% CI=[-0.011733, +0.014439]，win/tie/loss=504/53/443，Wilcoxon p=0.305802。Internal mean=-0.007033，95% CI=[-0.017795, +0.003848]。

## 9. Classification non-regression

Internal CLS accuracy B1−B0=-0.006341，预注册容忍下限 −0.005，pass=False。完整 CLS/LM/F1/agreement 见 `statistics/classification_non_regression.json`。

## 10. SEG / phrase / generation length

完整 internal 与 official1000 的 `[SEG]` trigger、target presence、exact、token F1、长度对照见 `generation_diagnostics/comparison.json`；no-generation-collapse=False。

## 11. B0/B1 TF-PHRASE

Official TF-PHRASE mean FG IoU：B0=0.416212，B1=0.414324。TF 是 oracle diagnostic，不替代 G0。

## 12. TF−G0 gap / delta-gap

Official gap：B0=+0.183620，B1=+0.180450，delta-gap=-0.003170。负 delta-gap 仅作为 exposure-gap 缩小的 secondary evidence。

## 13. Fixed severe119

固定 historical severe-119（实际匹配 119）：P1/B0/B1 mean FG IoU=0.234881/0.245792/0.203932；B1−B0 mean=-0.041860，95% CI=[-0.09459616648734187, 0.010152931112541525]。

## 14. Replay subgroup analysis

按冻结 P1 在对应 eval 样本上的生成属性进行 post-hoc 分组，结果见 `statistics/replay_subgroups.json`。这些结果全部是 exploratory，未用于 selector 或调参；重点比较 `phrase_low_quality` 且 `replay_eligible` 子群，但不可将训练 cache 属性错误地逐样本套到 test。

## 15. Artifact paths

完整 artifact 位于 `outputs/phase3b_generated_replay/` 的 cache、audit、training、selection、evaluation、statistics、severe119、generation_diagnostics、qualitative、reports。checkpoint 实体位于 `/data/yz/groundingLMM_official/checkpoints/phase3b_generated_replay/`。

## 16. Regression tests

测试结果写入 `reports/regression_tests.txt`。核心 Phase 3B 契约与全部既有 regression 均须通过后才标记 COMPLETE。

## 17. Final gate

**`GENERATED_REPLAY_HARMFUL`**。

本阶段未使用 NPR/SRM、FOCAL、SAM 架构修改、新 decoder、新 phrase/mask、mask refinement、threshold sweep、动态 replay、on-policy generation、scheduled sampling、consistency loss、TF distillation 或 test-set 选模。Phase 3C 未启动。
