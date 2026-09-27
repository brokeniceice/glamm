# Phase6I：I-JEPA 空间特征接入 C1-native R1

状态：**COMPLETE STOP**。两臂三阶段训练与 internal DEV 逐图配对均完成，最终判定见[结果文档](phase6i_ijepa_r1_internal_results.md)。Official1000、internal test 和外部定位 OOD 均未执行。

## 问题与对照

检验冻结 I-JEPA ViT-H/16@448 的 patch 特征，相对既有 `F24` 是否为 Rectifier 提供额外空间信息。主比较是同构、同参数量、同初始化的两臂：

| 臂 | Rectifier 附加输入 `X` | 其他部分 |
|---|---|---|
| A0 | 原冻结 forensic Adapter `F24` 的固定通道重复，`256→1280` | 与 A1 相同 |
| A1 | 官方 I-JEPA target encoder 特征，`1280×28×28` 对齐为 `1280×24×24` | 与 A0 相同 |

两臂计算 `F′ = F24 + α·Conv1×1(LayerNorm(X))`，`α=0` 初始化。Rectifier 原有 `F24` 主路、空间坐标、hard support 和 SAM 注入公式保留。附加支路两臂均为 `330497` 个参数；完整 Rectifier 为 `660482` 个可训练参数。零门控初始输出与原始 Rectifier 完全一致。A0 是容量与继续训练对照，不能用历史 C1-native checkpoint 替代 A0 做因果差值。

## 特征与数据

- I-JEPA 使用 [官方仓库](https://github.com/facebookresearch/ijepa) commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048` 和官方 ImageNet-1K ViT-H/16@448 full checkpoint 的 `target_encoder`。仅冻结推理，不更新其权重。
- 对每张图复用既有 CLIP 几何记录：按 `resized_hw` 缩放、取相同 `336×336` 中心裁剪、缩放到 `448×448`，用 ImageNet mean/std 归一化。28×28 patch grid 双线性对齐至同一裁剪区域的 24×24 网格；验证图像尺寸、cache shard 与样本 ID/顺序。
- C1 epoch 5 / step 2500、C1-center 分类、query cache、SAM、CLIP、Forensic Adapter 和证据源头冻结。TRAIN 8836 张，DEV 1106 张；有效单 `[SEG]` 分别 8741/1090，其余 DEV 样本按原协议零 IoU。绝不读取缓存里的旧 P1 query。
- A0/A1 各自在 C1 上按 Rectifier → Utility → joint 顺序训练。第一阶段仅 Rectifier 与附加支路；第二阶段冻结所选 Rectifier、随机初始化 Utility；第三阶段联合训练。每阶段沿用 [Phase6E.3](phase6e3_c1_native_staged_r1.md) 的损失、优化器、10 epoch、样本顺序和 DEV canonical G0 final-mask mean FG IoU 选模，平手选较早 epoch。
- Utility 输入定义不变。cross-image / spatial-shuffle ranking 只发生在 Utility 来源上下文中；I-JEPA 不进入该分支，因此不存在“正确 I-JEPA 特征留在错误来源负样本里”的路径。

## 预注册判定

选定两臂 joint checkpoint 后，在相同 1106 张 DEV 上逐图回放。主比较 `A1−A0` mean FG IoU；按图像配对 bootstrap 10000 次，报告 95% CI、胜/平/负及 F1。global FG IoU 和 F1、`[SEG]` 有效率为完整副指标，不隐藏反向变化。另报告 GT 面积占原图不超过 5% 和 CLIP 裁剪覆盖小于 80% 两个次级分组，不用它们选模或改变主门槛。

进入后续研究的条件须同时满足：`Δmean FG IoU ≥ +0.010`、配对 CI 下界 `>0`、`Δglobal FG IoU ≥ −0.005`。无论通过与否，本轮到 `COMPLETE_STOP` 为止；**不自动接续 Official1000、外部 OOD 或 internal test**。Official1000 已用于历史选模，未来若单独授权访问，必须标为复用的官方评测集。

## 运行与产物

- GPU 1：A0 三阶段；GPU 2：权重准备与 A1 特征缓存，随后 A1 三阶段。下载期间 A0 可先运行。两臂各自阶段串行，彼此独立。
- 程序：[冻结特征缓存](../scripts/phase6i_ijepa_cache.py)、[匹配训练](../scripts/phase6i_ijepa_staged.py)、[内部配对](../scripts/phase6i_ijepa_finalize.py)、[后台接续](../scripts/phase6i_ijepa_supervisor.py)。
- 状态：`outputs/phase6i_ijepa_r1/pipeline_status.json`；各臂 `arm_status.json`、阶段 selector/provenance；最终 `internal_dev_paired.json`。大 checkpoint 与冻结特征缓存在 `/data/yz/groundingLMM_official`，现有输出不覆盖。

本方案测试的是 **I-JEPA 冻结 encoder 的附加表征**，并未测试 I-JEPA latent prediction error 或重新预训练 JEPA 目标。

## 2026-09-25 启动故障与接续

官方 448 checkpoint 的 `target_encoder` state keys 带 DDP `module.` 前缀。首轮缓存 worker 以无前缀名称严格加载时报错；总控按 fail-closed 规则停掉了当时运行在卡 1 的 A0。故障发生前 A0 已完整写入 Rectifier epoch 0–3 的 checkpoint、history、优化器和调度器状态；epoch 4 仅有中途日志，没有完成 checkpoint。

现已在严格加载前统一去掉官方 state keys 的 `module.` 前缀，并在 CPU 严格载入 389 个 key、GPU 单图前向得到 `[1,784,1280]` 有限特征。A0 从 epoch 3 的完整 checkpoint 恢复，重新执行 epoch 4；检查 epoch 0 初始化哈希、history/epoch 连续性、3315 次 optimizer update、C1 SHA 和优化器/调度器状态。保留首次失败状态文件，不覆盖已完成的 epoch 0–3。恢复后的后台服务是 `phase6i-ijepa-r1-recovered.service`；最终实验范围和 `COMPLETE_STOP` 边界不变。
