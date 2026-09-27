# Phase 6E.3：从 C1 开始的 R1 分阶段训练

状态：三阶段训练与 Official1000 已完成；[与原 C1+new R1 的逐图配对比较](phase6e3_c1_native_vs_original_c1_r1_official1000.md)已落盘。外部定位 OOD 已在卡 0 和卡 2 上启动，当前状态见 [`supervisor_status.json`](../outputs/final_evaluation/c1_native_staged_r1/supervisor_status.json)。原自动链在 Official1000 入口路径错误后退出；评测随后在卡 2 上修复重跑并完成，没有改动训练或选模检查点。本轮独立于 Phase6E.2，保留历史 checkpoint 与结果，不覆盖旧实验。

## 模型与数据边界

- 上游固定 C1 epoch 5 / step 2500，SHA256 `85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff`。最终分类判定仍是 **C1-center**；R1 只负责 `[SEG]` 之后的定位。SAM、CLIP 取证源、C1、H2、LoRA 和证据源头均冻结。
- 使用已审计的 C1 canonical G0 `[SEG]` query cache，训练 8,836 张，其中 8,741 张 C1 恰有一个有效 `[SEG]`；internal DEV 1,106 张，其中 1,090 张有效，其余按零 IoU 保留。空间张量沿用 Phase4F 缓存，**不读取缓存内的旧 P1 query**。阶段中不访问 Official1000、external OOD 或 internal test 来选 epoch。
- CSCU Utility 可训练部分从 seed 3407 随机初始化；Rectifier 从 Phase4F 同架构随机初始化与其训练前几何比例 `gamma` 出发。[C1 训练有效样本的几何审计](../outputs/phase6e3_c1_native_staged/rectifier/c1_gamma_audit.json)独立选出 `gamma=0.01`，与历史几何审计数值一致；不使用 P1 query 或定位分数。所有阶段从各自选定的 C1 权重接续，不加载历史 P1 训练过的 Rectifier、Utility 或 old R1 权重。

## 三阶段

| 阶段 | 输入与可训练参数 | 训练与选择 | GPU |
|---|---|---|---|
| 1. C1 Rectifier | C1 query、冻结 SAM/取证源；仅 Rectifier 329,985 参数训练 | Phase4F 的 mask loss、AdamW warmup+cosine、10 epoch；DEV G0 mean FG IoU 从 epoch 0–10 选最大，平手选较早 | 1 |
| 2. C1 Utility | 固定阶段 1 所选 Rectifier；仅随机初始化的 CSCU Utility 371,803 参数训练 | Phase4H-C A2 的 segmentation + relative + ranking、AdamW、10 epoch；DEV G0 同规则选模 | 2 |
| 3. C1 joint | 从阶段 1 和 2 所选权重开始，Utility + Rectifier 共 701,788 参数联合训练 | Phase4H-D / Phase6E.2 的相同联合训练目标、AdamW、10 epoch；DEV G0 同规则选模 | 1 |

每阶段核验 C1 cache、样本 ID/顺序、有效样本数、trainable 参数数、冻结模块哈希以及上一阶段所选 checkpoint 的 SHA256。顺序依赖决定前两个训练阶段不能同时开始。

## 最终评测

联合阶段 selector 冻结后，卡 2 运行 SynthScars Official1000 canonical G0，保留全 1,000 张并逐图核对历史 manifest、ID、顺序与 GT；报告与历史 main new R1、C1+old R1 的配对统计。之后卡 1 跑 LOKI→PAL4VST、卡 2 跑 X-AIGD 的 known-Fake G1 定位 OOD。分类 OOD 复用已经核对的 C1-center 逐样本分数，不重复 GPU 推理。

Official1000 已用于历史候选的最终选择，本轮再次使用时必须如实披露；它不能再被描述为从未参与模型选择的独立测试集。本轮不据其结果调整阶段、超参数或 checkpoint。外部 OOD 只做最终报告，不回流训练或选模。

训练脚本：[C1 几何审计](../scripts/phase6e3_c1_gamma_audit.py)、[三阶段](../scripts/phase6e3_c1_native_staged.py)、[Official1000](../scripts/phase6e3_c1_native_official.py)、[外部 OOD](../scripts/final_eval_c1_native_staged_r1.py)、[自动接续](../scripts/phase6e3_c1_native_pipeline.py)。
