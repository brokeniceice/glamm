# C1 原生分阶段 R1：Official1000 与外部 OOD

生成时间：`2026-09-24T16:14:14.940395+00:00`。C1 checkpoint SHA256 `85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff`；C1 原生分阶段 R1 SHA256 `1446074c8233ccce560f57cde2b609e8600337453ad620fa60cf285a97b90a5b`。

## 分类 OOD：最终 C1-center

**主分类口径是 C1-center**：使用 C1 的 H2 分类 logit margin，并以已冻结的 internal-TRAIN 均值 `-9.190834885` 为阈值，`margin > mean` 判 Fake。等价地，使用冻结 mean/std 标准化后的 `z_C > 0`。new R1 仅修改 `[SEG]` 之后的定位路径。复用已完成的 Phase6D.5 C1 逐样本分类推理，并与 Phase6D.6 的 C1-center 数值逐项复核；四份结果与当前 final_evaluation manifest 的 SHA256、样本数和顺序匹配。raw C1（margin > 0）仅作为诊断保留在结果 JSON，不进入下表。未在 OOD 上调阈值，也没有重复 GPU 推理。

| 数据集 | N | Accuracy | Precision | Fake recall | TNR | FPR | F1 | ROC-AUC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aigi_holmes | 99999 | 0.908289 | 0.994813 | 0.820856 | 0.995720 | 0.004280 | 0.899501 | 0.979485 | 0.983391 |
| genimage | 100000 | 0.809400 | 0.971014 | 0.637840 | 0.980960 | 0.019040 | 0.769929 | 0.928839 | 0.937480 |
| loki | 2217 | 0.718088 | 0.834623 | 0.655277 | 0.810000 | 0.190000 | 0.734156 | 0.751232 | 0.799730 |
| raise998 | 998 | 0.998998 | 0.000000 | 0.000000 | 0.998998 | 0.001002 | 0.000000 | N/A | N/A |

RAISE998 只有 Real，主要看 TNR/FPR；ROC-AUC 和 AUPRC 不适用。GenImage 的八个生成器分组 C1-center 指标完整保存在结果 JSON 中。ROC-AUC/AUPRC 沿用同一原始 margin 的排序结果；正比例线性中心化不改变排序。

## 定位 OOD

SynthScars 复用本轮 Phase6E.3 C1 原生分阶段 R1 的 Official1000 G0，逐图 ID、顺序和 GT 前景像素数与最终评测清单一致。其余三组是本轮新推理：known-Fake G1 canonical unified prompt，不输入 GT 解释或定位短语。

| 数据集 | 模式 | N | 空 GT | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| synthscars | G0 | 1000 | 0 | 0.318875 | 0.439764 | 0.350363 | 0.518917 |
| loki | G1 | 229 | 0 | 0.087661 | 0.139278 | 0.062531 | 0.117702 |
| xaigd | G1 | 2419 | 247 | 0.083084 | 0.123880 | 0.075038 | 0.139601 |
| pal4vst | G1 | 1441 | 313 | 0.076464 | 0.113540 | 0.106225 | 0.192050 |

无 `[SEG]` 或无有效 query 的样本按 full-N 计零；多个预测 mask 在原图上以 logit 最大值取并集，固定阈值为 0。X-AIGD 的 `labels=[]` 和 PAL4VST 的官方空 GT 都保留；LOKI 的 GT 是 box union。不同 GT 语义与 G0/G1 模式的绝对值不直接横比。

## 数据重叠与使用边界

沿用 final_evaluation 的泄漏审计：`BLOCKED_OVERLAP`；覆盖记录：`ACTIVE`。AIGI-Holmes 与内部训练集存在已记录的 779 个 exact SHA256 重叠，因此该行应连同此限制披露。

分类结果是 C1-center 的冻结后处理复用，不是 new R1 带来的分类增益。定位结果按各数据集固定口径报告；没有在这些 OOD 集上选择 checkpoint、训练参数或阈值。

复核入口：[结果 JSON](../outputs/final_evaluation/c1_native_staged_r1/results.json)、[冻结协议](../outputs/final_evaluation/c1_native_staged_r1/protocol.json)、[评测脚本](../scripts/final_eval_c1_native_staged_r1.py)。
