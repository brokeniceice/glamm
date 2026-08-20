# Phase 3C.2 — P3（P1 + NPR/SRM B3）与官方1000鲁棒性

## 1. 协议

P3 在冻结 Phase 3A P1（step 3500 / epoch 7，SHA256 `fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326`）上复用 Phase 2C B3 的 classification-only residual late-fusion 结构。NPR/SRM checkpoint 冻结，P1、LLM、LoRA、CLIP、SAM、NPR、SRM 均未训练；只训练 P3 fusion head。selector 仅使用 internal validation classification loss，internal test 和 official1000 不参与选模。

P3 selected step=1250，checkpoint SHA256=`5b955a15800bfbe7c7d6984fd3e669d2d46e95e5a750c7cc35d58c8a5028d279`。canonical G0 不经过 classification gate，因此 P3 与 P1 逐样本共用完全相同的生成和掩码；另行报告 classification-gated G0 作为系统级指标。

## 2. Internal test 分类

P1 Accuracy=0.982790，P3=0.983243，P3−P1=+0.000453；F1 变化=+0.000493。paired McNemar：base-correct/P3-wrong=8，base-wrong/P3-correct=9，net=+1，exact p=1。

## 3. Official1000 原始条件

P1 fake recall/accuracy=0.979000，P3=0.981000，变化=+0.002000。canonical G0 mean FG IoU（P1=P3）=0.229544；classification-gated G0：P1=0.226972，P3=0.227774，变化=+0.000802。

## 4. Official1000 五条件鲁棒性

JPEG70/80 表示 JPEG quality factor；Gaussian5/10 表示 RGB 0–255 像素尺度的确定性高斯噪声 σ=5/10（即约 5/255、10/255）。扰动在所有模型预处理之前施加，GT mask geometry 不变。

| 条件 | 模型 | 分类准确率/假召回 | canonical G0 FG IoU | gated G0 FG IoU | Δ分类 vs 原始 | Δcanonical IoU vs 原始 |
|---|---|---:|---:|---:|---:|---:|
| original | C0 | 0.974000 | 0.194407 | 0.192237 | +0.000000 | +0.000000 |
| original | P1 | 0.980000 | 0.229544 | 0.226972 | +0.000000 | +0.000000 |
| original | P3 | 0.984000 | 0.229544 | 0.228383 | +0.000000 | +0.000000 |
| jpeg70 | C0 | 0.979000 | 0.200190 | 0.197488 | +0.005000 | +0.005783 |
| jpeg70 | P1 | 0.982000 | 0.232847 | 0.231592 | +0.002000 | +0.003303 |
| jpeg70 | P3 | 0.988000 | 0.232847 | 0.232618 | +0.004000 | +0.003303 |
| jpeg80 | C0 | 0.987000 | 0.201196 | 0.199343 | +0.013000 | +0.006790 |
| jpeg80 | P1 | 0.988000 | 0.234617 | 0.233158 | +0.008000 | +0.005073 |
| jpeg80 | P3 | 0.996000 | 0.234617 | 0.234591 | +0.012000 | +0.005073 |
| gaussian5 | C0 | 0.934000 | 0.186136 | 0.180362 | -0.040000 | -0.008271 |
| gaussian5 | P1 | 0.909000 | 0.222148 | 0.217700 | -0.071000 | -0.007396 |
| gaussian5 | P3 | 0.919000 | 0.222148 | 0.218903 | -0.065000 | -0.007396 |
| gaussian10 | C0 | 0.921000 | 0.186774 | 0.179925 | -0.053000 | -0.007633 |
| gaussian10 | P1 | 0.867000 | 0.205977 | 0.200986 | -0.113000 | -0.023567 |
| gaussian10 | P3 | 0.878000 | 0.205977 | 0.200274 | -0.106000 | -0.023567 |

## 5. 解释边界

- P3 的 canonical G0 与 P1 相等是架构约束，不是 P3 获得了定位增益。
- classification-gated G0 的变化只来自分类决策翻转。
- official1000 为 Fake-only，分类数值是 fake recall/accuracy，不能单独衡量 FPR 或完整 balanced accuracy。
- 本阶段只检验当前 P1 上的 B3 对应方案，不重新解释 Phase 2C 的 B0 结果，也不证明 NPR/SRM 具有像素级定位能力。

完整 artifact：`outputs/phase3c2_p3_robustness/`；P3 checkpoint 实体：`/data/yz/groundingLMM_official/checkpoints/phase3c2_p3/p3`。
