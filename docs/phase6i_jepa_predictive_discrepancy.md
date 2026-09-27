# Phase6I-JD：冻结 I-JEPA 预测误差诊断

状态：**COMPLETE STOP**；结论：**NOT SUPPORTED**；下一阶段门槛：**STOP**。

> JEPA predictive discrepancy does not reliably distinguish artifact regions from matched normal regions.

## 人群与方法

主分析为 canonical internal DEV 的 1106 张 Fake，805 张形成合法 A/B 配对，301 张显式 invalid。
同一 internal validation 清单的 1106 张 Real 为次级 C 对照来源；其中与有效 A/B 样本对应的 805 张进入 C 统计，不改变主 DEV 人群和顺序。
沿用 Phase6I 官方 ViT-H/16@448 checkpoint。encoder、target encoder、predictor 全冻结；target latent 来自完整图像，predictor 仅接收排除 target patch 后的 context tokens。
目标块面积允许 118–156/784 patches（实际合法整数形状为 120–156），长宽比 0.75–1.5；正常块与 artifact 块宽高完全相同，GT 膨胀 2 patches 后避让。A 的多个块合计覆盖至少 50% 原图 GT，否则标记 invalid。每图先平均各块，再按图配对；统计单位是图像。
primary 为 cosine distance；normalized L2 和 raw L2 为预定次级指标。

## 主结果：Artifact A 与同图正常 B

| 指标 | Mean A | Mean B | A−B | 配对 bootstrap 95% CI | 胜/平/负 |
|---|---:|---:|---:|---|---:|
| cosine_distance | 0.318149 | 0.348453 | -0.030304 | [-0.035676, -0.024937] | 304/0/501 |
| normalized_l2 | 0.753094 | 0.789981 | -0.036887 | [-0.043855, -0.029930] | 310/0/495 |
| raw_l2 | 24.753754 | 25.697598 | -0.943844 | [-1.136531, -0.750677] | 321/0/484 |

## 次级对照与稳健性

| cosine distance 比较 | Mean difference | bootstrap 95% CI |
|---|---:|---|
| A−C（非配对） | -0.011599 | [-0.018054, -0.005274] |
| B−C（非配对） | +0.018705 | [+0.012460, +0.024781] |

| GT 面积分组 | 全部 / 有效 | A−B cosine | 95% CI |
|---|---:|---:|---|
| area_le_1pct | 414 / 382 | -0.044517 | [-0.051764, -0.037167] |
| area_1_to_5pct | 440 / 348 | -0.020626 | [-0.029263, -0.012361] |
| area_5_to_20pct | 208 / 74 | -0.003082 | [-0.020936, +0.014125] |
| area_gt_20pct | 44 / 1 | +0.016837 | 不可估计（仅 1 张） |

配对标准化效应 d=-0.3891；双侧配对置换 p=0.000100。
预注册 GO 四项检查：`{"mean_delta_cosine_ge_0p02": false, "paired_ci_lower_gt_zero": false, "win_rate_gt_0p55": false, "artifact_increment_gt_synthetic_global": false}`。

## 五个问题的回答

1. **Artifact 比同图正常区域误差更高吗？** 没有。主指标 A−B 为 **−0.030304**，配对 95% CI **[−0.035676, −0.024937]**，方向与假设相反。
2. **逐图是否稳定？** 805 张有效图中 A>B 为 304 张，A<B 为 501 张，配对效应量 d=−0.3891。负向差异并非由少数异常值造成；中位差为 −0.024037。
3. **Synthetic 正常区域是否比 Real 正常区域难预测？** 在本次非配对次级分析中 B−C=**+0.018705**，95% CI **[+0.012460, +0.024781]**。Real 与 Fake 来源构成不同，因此这个差值只能描述当前样本，不能单独归因为 syntheticness。
4. **小面积伪影有效吗？** 没有。GT≤1% 的 382 张有效图 A−B=**−0.044517**；1%–5% 的 348 张为 **−0.020626**，两组 CI 都在 0 以下。
5. **能证明 artifact-specific contextual inconsistency 吗？** 不能。A 的误差低于 B，而 B 高于 C；四项预注册 GO 条件均未满足。当前路线判定 **STOP**。

## 人群、有效性与解释边界

主 DEV 1106 张全部是 Fake；同源 Real 只用于次级 C。无效 301 张逐图保留：裁剪后无 GT 15 张，合法块覆盖不足 149 张，无安全正常对照 108 张，原图 GT 覆盖不足 29 张。有效率 805/1106；尤其 GT>20% 仅 1/44 张有效，不能对大伪影总体作结论。

818 个有效 A 块中，GT-positive patches 平均只占目标块 **18.96%**（中位 **15%**）。这是官方 JEPA 训练尺度的块遮挡与细小伪影之间的分辨率限制，可能稀释伪影信号；本诊断仍然否定当前冻结协议下的预设 GO 条件。B/C 的语义类别没有严格匹配，不能把 B−C 当作纯粹的生成来源因果效应。

样本审计 SHA256：`sample_ids=1695664e127eaa2e0e50cf3ab05821741914e57bd81a6ff7d8f200d63db18c38`；`sample_order=2145bad0303e6aa22e32a11bf207978ad948b9abac8a8ef1286b4932f6ebb1d2`；`gt_mask=5d8b0e1d7f62eb6b6324362057295566247b9feeb8a1b52148bb769d979bb218`；`real_fake_label=ddfe53fecbd548b828c3f039dba3757b5ef5ca971b8317745534a15c5915adb6`。原图尺寸哈希与 Real 清单哈希见 `sample_manifest.json`。

泄漏审计确认 target/context patch 索引不相交；修改被遮挡目标像素后，predictor 输出最大绝对变化为 **0**（Fake 与 Real 各抽验一个有效区域）。其余区域遵循相同索引隔离实现；完整审计见 `leakage_audit.json`。

模型没有训练或更新 R1；此实验不能推出 JEPA 改善 R1。未访问 Official1000、internal test 或外部 OOD。
完整人群哈希、区域、逐区域/逐图数值和 leakage audit 位于 `outputs/phase6i_jepa_discrepancy/`。
