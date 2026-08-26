# Phase 3D.2 — Direct Spatial-Path Optimization

## 1. 研究问题与冻结边界

Phase 3D.1 已以 `GATE_REWARD_FORMULATION_NOT_PRIMARY_BOTTLENECK` 结束，因此本阶段不继续 R3/Q2 reward tuning，也不混入 reward、语言交叉熵、分类损失或新 forensic branch。Phase 3D.2 只回答：在语言策略、图像表征和模型结构保持不变时，直接训练现有 spatial grounding pathway，能否显著改善 forensic localization？

所有实验从原始 P1 checkpoint 初始化，SHA256 为 `fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326`。未使用 R3/Q2、Phase 3D.0 rollout 或其他微调 checkpoint。internal test 和 official1000 全程封存，未参与训练、选择或解释。

## 2. 为什么使用 authoritative context

训练采用 teacher-forced authoritative forensic target：`[FAKE] explanation` 加 `Target regions: <authoritative phrase> [SEG]`。冻结 P1 LLM 产生 `[SEG]` hidden state，随后仅由 `text_hidden_fcs` 和 `mask_decoder` 预测 mask。这样可以隔离“给定正确语言证据时空间路径是否可改善”。本阶段明确不使用 P1 自主生成文本再回放监督的 generated replay，避免把语言错误和空间优化混在同一干预中。

## 3. 数据、目标与训练配置

- 训练 split：统一 forensic train，共 17,672 张图，其中 Fake 8,836 张；primary spatial loss 仅使用带非空定位目标的 Fake。
- 实际暴露：1,000 个按冻结 schedule 选定的 Fake image exposures；250 optimizer steps，batch size 4，gradient accumulation 1。
- target：冻结的 SynthScars annotation polygons，经既有 per-image all-ref union、rasterization 和 preprocessing 构造；未重新生成 mask，也未使用 pseudo mask。
- optimizer：AdamW，学习率 `3e-4`，betas `(0.9, 0.95)`，weight decay 0；10-step warmup 后线性衰减；bf16；gradient clip 1.0。
- 唯一目标：direct mask supervision，BCE 权重 2.0、Dice 权重 0.5；reward、language CE、classification loss 均为 0 或 null。
- 训练状态：`COMPLETE`，耗时 473.4 秒，完成 250 steps。

## 4. 可训练参数与冻结参数

总参数量 `8,025,250,866`，实际可训练参数 `21,888,484`：

- `text_hidden_fcs`：17,830,144 参数；
- `mask_decoder`：4,058,340 参数。

LLM base、全部 LoRA、token embedding、LM head、classification head、vision tower、grounding image encoder、mm projector、region encoder 及其他 grounding encoder 均设置为 `requires_grad=False`，且未注册进 optimizer。

## 5. 冻结的 checkpoint selector

Primary selector 为 internal-validation Fake TF-PHRASE mean foreground IoU，tie-breaker 为 mean foreground F1。候选为 step 0/50/100/150/200/250；G0、training loss、internal test、official1000 和 qualitative case 均未参与选择。

| Step | TF FG IoU | IoU vs P1 | TF FG F1 | F1 vs P1 |
|---:|---:|---:|---:|---:|
| 0 | 0.340823 | +0.000000 | 0.449934 | +0.000000 |
| 50 | 0.338428 | -0.002394 | 0.447210 | -0.002724 |
| 100 | 0.332482 | -0.008341 | 0.438323 | -0.011611 |
| 150 | 0.337848 | -0.002975 | 0.445243 | -0.004692 |
| 200 | 0.338030 | -0.002793 | 0.444637 | -0.005297 |
| 250 | 0.338544 | -0.002278 | 0.445345 | -0.004589 |

所有实际训练 checkpoint 都低于 step 0。训练 checkpoint 中最好的是 step 250，TF FG IoU 为 0.338544，相对 P1 为 -0.002278。因此 selector 按预注册规则选择 **step 0**，即原始 P1，而不是训练后的 checkpoint。这是负结果，不是训练后模型与 P1 达到相同结果。

## 6. TF-PHRASE mechanistic result

| Path | P1 FG IoU | Spatial FG IoU | Delta | 95% CI | P1 FG F1 | Spatial FG F1 |
|---|---:|---:|---:|---:|---:|---:|
| TF-PHRASE | 0.340823 | 0.340823 | +0.000000 | [+0.000000, +0.000000] | 0.449934 | 0.449934 |

这里 P3D2-SPATIAL 指 selector 选中的正式 arm；由于 selector 回退到 step 0，配对的 1,106 个 Fake 全部为 tie，差值和 10,000 次 paired bootstrap 置信区间均严格为 0。P1 自身 TF FG IoU 的均值 95% bootstrap CI 为 [0.325063, 0.356939]。结合候选表，直接空间训练没有产生可选择的 validation gain，Gate A 不通过。

## 7. G0 greedy deployment result

| Metric | P1 | Selected P3D2 | Delta / relation |
|---|---:|---:|---:|
| Fake FG IoU | 0.146008 | 0.146008 | +0.000000 |
| Fake FG F1 | 0.207524 | 0.207524 | +0.000000 |
| Classification accuracy | 0.985081 | 0.985081 | +0.000000 |
| Classification F1 | 0.984966 | 0.984966 | +0.000000 |
| Phrase semantic score | 0.637322 | 0.637322 | identical |
| Structure validity | 0.985533 | 0.985533 | identical |
| Malformed rate | 0.014467 | 0.014467 | identical |

G0 的 FG IoU paired-bootstrap 95% CI 为 [+0.000000, +0.000000]。因为 Gate A 未通过且 selector 为 P1，G0 结果只确认正式部署 arm 没有变化，不能解释成训练后的空间模型保持了完全相同的部署性能。

## 8. Invariance 与 gradient audit

- Parameter audit：**PASS**。`text_hidden_fcs` 和 `mask_decoder` 分别发生参数变化，delta norm 为 7.286033 和 2.412869；所有冻结模块 SHA256 前后一致、delta norm 为 0。
- Gradient audit：**PASS**。在 step 1、150、250，只有 `text_hidden_fcs` 与 `mask_decoder` 存在梯度和更新；其余组 grad absent、update norm 0。
- Behavioral audit：**PASS**。16 个 deterministic samples 的 token IDs、文本、verdict、target phrase、SEG 次数与位置、classification logits/prediction、structure validity 均逐项一致。
- 全 validation exact audit：1,106 个 Fake 的 greedy token/text/phrase/SEG，以及 2,212 个样本的 classification prediction/probability 均完全一致。

这些审计证明训练边界正确且允许模块确实被优化；它们不等价于证明 validation localization 得到改善。

## 9. Failure analysis

正式 selected-arm 对比中，`TF and G0 both improve` 为 0，`TF improves but G0 does not` 为 0，`both remain poor` 为 386。由于 selected arm 就是 P1，前两类为 0 是 selector 回退的必然结果；386 个双路径低 IoU 样例仅用于描述残余困难，不能单凭这些样例证明 architecture bottleneck。

## 10. Final route gate

Gate A：**GATE_SPATIAL_PATH_TRAINABILITY_NOT_SUPPORTED**。Gate B：**GATE_SPATIAL_GAIN_DOES_NOT_SIGNIFICANTLY_TRANSFER_TO_G0**。主门：**GATE_EXISTING_SPATIAL_PATH_OPTIMIZATION_INSUFFICIENT**。

这里“Gate A 不支持 trainability”专指未检测到满足预注册显著性条件的 validation localization gain；参数和梯度审计已经证明优化过程在工程意义上实际发生。科学结论是：**在本阶段固定的数据、teacher-forced context、BCE+Dice 目标和 1,000 Fake exposures 下，仅训练现有 `text_hidden_fcs + mask_decoder` 不足以改善 validation forensic localization。** 限制可能来自 frozen representation、image features、language-to-spatial interface、target complexity 或现有 architecture capacity，本阶段无法区分这些解释。

## 11. 结论边界与停止条件

不得将本结果表述为“mask decoder 是唯一或首要瓶颈”，也不得推广为所有直接空间监督都无效。未查看 internal test，未使用 official1000 调参，未修改历史 Phase 3B、3C 或 3D.1 结论。Phase 3D.2 到此停止；未自动启动 FC-only/decoder-only ablation、扩大训练预算、joint language-spatial training、reward+mask、forensic branch 或架构修改。
