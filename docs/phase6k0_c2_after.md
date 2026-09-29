# Phase6K — Frozen C2 Spatial Transfer: After Injection

状态：**DEV 阶段完成；Official1000 按用户要求跳过，自动接续器已停止。**

## 五项训练前门槛：PASS

1. C2 checkpoint SHA256 `4a67e6a87c453d554fa5bd6cf93329ae54c1c2853f0925dc7a63397eef27e8ce`，epoch 7 / step 3500；C1-native joint R1 SHA256 `1446074c8233ccce560f57cde2b609e8600337453ad620fa60cf285a97b90a5b`，selector 和 checkpoint 哈希已验证。
2. A 形状 `[1, 8, 24, 24]`；E 形状 `[1, 512, 24, 24]`；A mass 最大误差 1.1920929e-07；E-context 相对误差 0.001074141。开启 capture 前后 fused token、原始分类 logits、生成 token、q_seg、掩码完全一致（PASS）。
3. TRAIN 8836 张，C2-valid 8682 张；DEV 1106 张，C2-valid 1080 张。原始 ID/顺序及 shard SHA 均已锁定。
4. B0 DEV Mean FG IoU 0.197691、Mean FG F1 0.276334、Global FG IoU 0.227295、Global FG F1 0.370401。
5. A/E step-0 的 F24、z_F24、Rectifier、Utility、adapted SAM embedding、mask logits、loss 与 B0 相等；投影梯度和 Utility 输入梯度非零，所有冻结参数 SHA 未变。

五项通过后，A/E 在卡 0/2 各完成 10 epoch。两个 selector 均只使用内部 DEV Mean FG IoU，均选择 epoch 1。

## 内部 DEV 结果（1106 张）

| 模型 | 选中 epoch | Mean FG IoU | 相对 B0 | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|---:|
| B0：C2 + 冻结 C1-native R1 | — | 0.197691 | — | 0.276334 | 0.227295 | 0.370401 |
| A：8→256 attention-map 投影 | 1 | 0.197797 | +0.000106 | 0.274244 | 0.213951 | 0.352487 |
| E：512→256 evidence-map 投影 | 1 | 0.195894 | −0.001797 | 0.272183 | 0.205096 | 0.340381 |

逐图配对的 Mean FG IoU 差值（相对 B0，2000 次 bootstrap，seed 3407）的 95% 区间：A 为 [−0.002991, 0.003165]，E 为 [−0.005885, 0.002363]。A 的微小 Mean IoU 正差没有稳定证据，且 Global IoU 与两项 F1 均下降；E 的选中模型在所列 DEV 指标上均低于 B0。A/E 后续 epoch 的 Mean IoU 都未超过各自 epoch 1。

## 结束范围

Official1000 未运行，不能据此给出官方测试结论。后台自动接续器已停止，避免卡 2 释放后自行启动评测。完整 DEV 数值与 selector 来源见 `outputs/phase6k0_c2_after/dev_only_summary.json`、`a/selector.json`、`e/selector.json`。
