# Phase 6E.2：main new R1 与 I2 的最终模型选择

更新日期：2026-09-24。比较对象是 **C1+main new R1**（A2 epoch 3 Utility + Phase4F epoch 9 Rectifier 初始化，再联合训练）与 **C1+I2 R1**（两支均以 seed 3407 随机初始化，再联合训练）。两臂推理结构相同；复杂度差异在训练初始化流程。

## 结论

按预先使用的 **Official1000 mean foreground IoU** 主指标，选择 **main new R1** 作为当前论文的 C1+R1 定位模型。main−I2 的逐图配对均值差为 **+0.009480**，10,000 次图像级配对 bootstrap 95% CI 为 **[+0.002486, +0.016898]**。global IoU 同向增加 **+0.042436**，图像级配对 bootstrap CI 为 **[+0.027650, +0.058477]**。因此现有证据支持保留分阶段初始化；I2 是有效的简化训练消融，但其定位结果较低。

该优势体现在**均值和像素汇总量**，不表现为逐图普遍胜出：IoU 的 main 胜/平/负为 **499/25/476**，配对差中位数为 **0**，Wilcoxon 双侧 `p=0.319422`。应表述为“平均定位质量更高”，不能表述为“多数图像均改善”或“随机初始化无效”。

## 可比性审计

| 项目 | main | I2 | 判定 |
|---|---|---|---|
| 冻结上游 | 同一 C1 epoch 5 / step 2500，SHA256 `85af4e0c…cb132c6ff` | 相同 | PASS |
| 新 R1 初始化 | A2 epoch 3 Utility + Phase4F epoch 9 Rectifier | 两支随机，seed 3407 | 唯一设计差异 |
| 联合训练 | 同一配方，10 epoch；逐轮样本顺序、有效/无效曝光和 optimizer updates 一致 | 相同 | PASS |
| 训练范围 | Utility **371,803** + Rectifier **329,985** 参数；C1/LoRA/H2/FRC/RINE 等冻结；冻结哈希前后一致 | 相同 | PASS |
| selector | internal validation canonical G0 mean IoU；epoch **8**，DEV **0.202732** | 同规则；epoch **10**，DEV **0.201710** | PASS；所选轮次由同一规则独立产生 |
| Official1000 | SynthScars Fake-only、原图、canonical G0、1000 张 | 相同 | manifest SHA256 `fff3c383…2d7bf5d3edbe8c700`；ID/顺序完全一致 |
| 输出/GT | 992 张有效 `[SEG]`、8 张按无 `[SEG]` 计零；每图 `tp+fn` 与 I2 一致 | 相同 | PASS |
| 分割口径 | 全部 GT mask 取并集；多个 `[SEG]` mask 逐像素 logit max；logit 阈值 0 | 同一评测脚本 | PASS |

两臂的 checkpoint SHA256 分别为 `c067eb2240af9d5fd61fc5dc367338ed585fcc6a5d5d0f982df0fa886302e603` 与 `41f25e2c37002c5fa400074cc22f71ecb8e255c0a4b10fd9287812d5f0574a07`；逐样本记录 SHA256 分别为 `f1489da001fd0c74a3cb3290ca5006e7d16ab9021250eb8f630a3d7b12b1b080` 与 `0bf5eec7ac265dfacbf0f57833d045244c3bf3877288a6d64d842422f2f0dff4`，均与各自 selector / 结果文件登记一致。两臂选模时均声明未使用 Official1000 或 internal test。训练配置、群体与冻结模块哈希一致；由于选择 epoch 不同，这里估计的是**初始化方案连同相同训练与选模规则**的最终效果差异，不是固定 epoch 的纯初始化效应。

I2 曾产生一份 **1,581 行的并发冲突文件** `official1000_concurrent_conflict_1581rows.jsonl`。本次统计只使用其后完成的、与官方 manifest 一一对应的 1000 行正式文件；两文件 SHA256 不同，冲突文件未进入比较。main 的旧 `worker_status.json` 记录了 2026-09-14 20:07 的 worker 失败；正式 main 逐样本文件完成于当日 18:00，当前结果摘要中的记录 SHA256 与文件完全一致，1000 行可独立重算出相同指标。旧 worker 状态不作为本次统计的完成证据。

## Official1000 结果

差值统一为 **main−I2**。mean 按 1000 张图平均；global 先汇总 1000 张图的 TP/FP/FN 再计算，因此对大掩码更敏感。

| 指标 | main new R1 | I2 | 差值 |
|---|---:|---:|---:|
| Mean foreground IoU（主指标） | **0.319867** | 0.310387 | **+0.009480** |
| Mean foreground F1 | **0.440385** | 0.427922 | **+0.012463** |
| Global foreground IoU | **0.343538** | 0.301102 | **+0.042436** |
| Global foreground F1 | **0.511393** | 0.462842 | **+0.048552** |
| `[SEG]` 触发率 | 0.992 | 0.992 | 0 |

| 配对统计，1000 张 | main−I2 均值 | 95% bootstrap CI | 胜/平/负 | Wilcoxon 双侧 p |
|---|---:|---:|---:|---:|
| Foreground IoU | **+0.009480** | **[+0.002486, +0.016898]** | 499/25/476 | 0.319422 |
| Foreground F1 | **+0.012463** | **[+0.004655, +0.020589]** | 499/25/476 | 0.167744 |

CI 使用 seed 3407、10,000 次图像级成对重采样。global IoU 的 CI 在每次成对抽样后重新汇总 TP/FP/FN。Wilcoxon 检验的是逐图配对差的秩分布，与“平均差是否大于零”并非同一统计量；这里没有将其不显著误写成两臂等效。

## 论文表述边界

- **最终选择**：在已评测的 C1+R1 候选中，main new R1 的平均与 global 定位表现均优于 I2；采用 main，保留 I2 作为随机初始化联合训练消融。两臂推理参数规模相同，因此理由是定位收益与训练流程成本之间的权衡。
- **不可扩展的结论**：这里没有多随机种子训练方差，也没有证明所有数据集或每张图均受益；Official1000 是 SynthScars Fake-only 定位集，不能推断真图误报或分类性能。
- **测试集使用**：这次根据 Official1000 结果作最终模型选择，论文应如实披露；若将其称为完全未参与模型选择的独立最终测试，会高估该指标的证据强度。需要独立最终确认时，应使用新的未参与选择的测试集，不再按 Official1000 调整模型。

可复核产物：[配对审计 JSON](../outputs/phase6e2_c1_specific_r1/main_vs_i2_paired.json)、[审计脚本](../scripts/phase6e2_main_vs_i2_paired.py)、[main 原结果](../outputs/phase6e2_c1_specific_r1/phase6e2_c1_specific_r1.json)、[I2 原结果](../outputs/phase6e2_c1_specific_r1/i2/phase6e2_i2_random_init.json)。运行：`/home/yz/miniconda3/envs/glamm_official/bin/python -m scripts.phase6e2_main_vs_i2_paired`。
