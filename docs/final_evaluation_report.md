# Final Evaluation Report｜按数据集汇总

更新：2026-09-29。数值保留六位小数；`-` 表示缺少可报告结果或指标不适用。每个数据集单独一表。

## 口径与模型

- **P1-old R1** 指 P1 上的旧 R1；它只修改 `[SEG]` 后的定位路径，分类权重与 P1 相同。外部分类原记录名为 `r1`，P1 行按分类不变性精确复用，并非第二次推理。
- **C1-native R1** 指从 C1 出发，依次预训练 Rectifier、Utility 后联合训练的 Phase6E.3 模型。分类主口径固定为 **C1-center**：C1 H2 logit margin 大于 internal-TRAIN 均值 `-9.190834884678093` 判 Fake；R1 不改变分类。
- **C1-raw** 是同一 C1 分类头在中心化之前的原始判定：H2 Fake−Real logit margin `> 0` 判 Fake。它与 C1-native R1 行使用相同的 C1 权重和逐图分数，只改变判定阈值；作为诊断口径列出，最终 C1 分类架构与主结果仍为 **C1-center**。正比例标准差缩放不改变 ROC-AUC，因此两个口径的 ROC-AUC 相同。
- **C2-raw** 是选定 C2 checkpoint（epoch 7、step 3500）的原始 H2 分类口径，Fake 概率阈值 0.5；定位是未接 R1 的 canonical G0 基线。C2 与历史 C1 的训练设备与预算不同，不能把差异严格归因于交叉注意力。
- **C2-native R1** 是 Phase6P0 从冻结 C2 出发，按 Rectifier、Utility、联合训练三个阶段得到的定位分支；各阶段仅用 internal DEV Mean FG IoU 选轮，最终 joint 选第 8 轮。R1 不改变 C2 分类路径，分类仍按 C2-raw/C2-center 各自的冻结阈值报告。
- **C2-center** 复用同一 C2 checkpoint 和逐图分类分数；内部训练集 17,672 张图像的 H2 Fake−Real logit margin 均值为 `-5.201266923`。固定规则 `margin > mean`，对应 Fake 概率阈值 `0.00547939064`。仅分类判定改变，Official1000 定位与 C2-raw 相同。
- **legion-retrained-match** 为独立重训练的 matched 版本。其 X-AIGD/PAL4VST 结果 JSON 的旧字段仍写 `model: legion_retrained`，但 checkpoint 角色与路径指向 `legion_retrained_match_stage1_LE`。
- **legion-intermediate + aligned Stage-2** 以作者公开 `legion_LE` 第一阶段中间权重为起点，按 `legion-retrained` 的第二阶段配方在本项目数据上训练分类头；分类表使用该完整模型的冻结评测。定位表中的 **legion-intermediate** 仍指原始公开 LE 中间权重，未替换为第二阶段模型。原始公开 LE 单独没有已训练的分类头。
- **RINE-official-224** 使用官方 RINE 模型、OpenAI CLIP `ViT-L/14` 224 权重和官方 224 图像变换，在本项目数据上从随机 RINE 分类头训练一轮；官方发布的 RINE 分类头权重未用于初始化。40 张尺寸不足 224 的 Real 训练图像按预注册规则排除，实际训练 17,632 张。
- **NPR-official-retrained** 指独立 NPR 基线：直接使用固定官方 NPR ResNet 源码，在本项目 17,672 张内部训练图上随机初始化、训练 50 轮，固定最后一轮权重；仅有分类输出。
- **RINE-336-adapted** 指既有 `RINE-on-C1` 独立分类器：官方 RINE Q1/TIE/Q2/head 结构适配冻结的项目 CLIP ViT-L/14@336，在全部 17,672 张内部训练图像上训练一轮。它与最终 C1-center 共享 RINE 来源，但此行是 RINE 自身的分类输出，而非 C1-center 输出。

C1-raw、NPR、公开 LE 第二阶段和两个 RINE 行只用于分类比较。两个 RINE 版本的 CLIP 权重、输入尺寸、图像变换及训练样本范围不同，因此其数值差异不能单独归因于分辨率或轮数。各分类表沿用相同测试清单和 Fake 正类；C1-raw 使用原始 margin 阈值 `0`，C1-center 使用上面注明的冻结中心阈值，其他概率型分类行使用阈值 `0.5`。

## 分类｜内部测试集

### Internal2208（Real 1104，Fake 1104）

| 模型 | N | Accuracy | F1 | ROC-AUC | Fake recall | TNR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 2208 | 0.983696 | 0.983621 | 0.998389 | 0.979167 | 0.988225 | 0.011775 |
| P1-old R1 | 2208 | 0.983696 | 0.983621 | 0.998389 | 0.979167 | 0.988225 | 0.011775 |
| C1-raw | 2208 | 0.995924 | 0.995926 | 0.999550 | 0.996377 | 0.995471 | 0.004529 |
| C1-center | 2208 | 0.991848 | 0.991906 | 0.999550 | 0.999094 | 0.984601 | 0.015399 |
| legion-retrained | 2208 | 0.986413 | 0.986486 | 0.999258 | 0.991848 | 0.980978 | 0.019022 |
| legion-retrained-match | 2208 | 0.989130 | 0.989140 | 0.999241 | 0.990036 | 0.988225 | 0.011775 |
| legion-intermediate + aligned Stage-2 | 2208 | 0.987772 | 0.987821 | 0.999283 | 0.991848 | 0.983696 | 0.016304 |
| RINE-official-224 | 2208 | 0.981431 | 0.981423 | 0.998414 | 0.980978 | 0.981884 | 0.018116 |
| RINE-336-adapted | 2208 | 0.993207 | 0.993246 | 0.999874 | 0.999094 | 0.987319 | 0.012681 |
| NPR-official-retrained | 2208 | 0.949728 | 0.950336 | 0.988020 | 0.961957 | 0.937500 | 0.062500 |
| C2-raw | 2208 | 0.995924 | 0.995929 | 0.999692 | 0.997283 | 0.994565 | 0.005435 |
| C2-center | 2208 | 0.995018 | 0.995038 | 0.999692 | 0.999094 | 0.990942 | 0.009058 |

C1-raw 内部行与冻结的 2,208 条 C1 分数逐样本复核，TP/TN/FP/FN = `1100/1099/5/4`；C1-native R1 行按 C1-center 阈值复算为 `1103/1087/17/1`。两行都是同一分数的后处理，没有重新推理。

## 分类｜外部 OOD

### AIGI-Holmes TestSet（Real 50,000，Fake 49,999）

| 模型 | N | Accuracy | F1 | ROC-AUC | Fake recall | TNR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 99999 | 0.831168 | 0.797995 | 0.956480 | 0.666953 | 0.995380 | 0.004620 |
| P1-old R1 | 99999 | 0.831168 | 0.797995 | 0.956480 | 0.666953 | 0.995380 | 0.004620 |
| C1-raw | 99999 | 0.836468 | 0.804649 | 0.979485 | 0.673593 | 0.999340 | 0.000660 |
| C1-center | 99999 | 0.908289 | 0.899501 | 0.979485 | 0.820856 | 0.995720 | 0.004280 |
| legion-retrained | 99999 | 0.888109 | 0.875091 | 0.976110 | 0.783896 | 0.992320 | 0.007680 |
| legion-retrained-match | 99999 | 0.867959 | 0.848672 | 0.975100 | 0.740515 | 0.995400 | 0.004600 |
| legion-intermediate + aligned Stage-2 | 99999 | 0.883919 | 0.869728 | 0.974483 | 0.774995 | 0.992840 | 0.007160 |
| RINE-official-224 | 99999 | 0.765968 | 0.695293 | 0.985019 | 0.534031 | 0.997900 | 0.002100 |
| RINE-336-adapted | 99999 | 0.884399 | 0.869632 | 0.991609 | 0.771135 | 0.997660 | 0.002340 |
| NPR-official-retrained | 99999 | 0.782098 | 0.734501 | 0.909928 | 0.602832 | 0.961360 | 0.038640 |
| C2-raw | 99999 | 0.846418 | 0.818738 | 0.983475 | 0.693714 | 0.999120 | 0.000880 |
| C2-center | 99999 | 0.876909 | 0.859907 | 0.983475 | 0.755555 | 0.998260 | 0.001740 |

**泄漏提示：**此 TestSet 与原 internal-TRAIN 存在 779 个 exact SHA256 重叠；RINE-official-224 排除 40 张小图后，其实际训练清单仍有 775 个 exact 重叠。这些样本未从评测中删除。

### GenImage held-out（Real 50,000，Fake 50,000）

| 模型 | N | Accuracy | F1 | ROC-AUC | Fake recall | TNR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 100000 | 0.641980 | 0.455317 | 0.871324 | 0.299280 | 0.984680 | 0.015320 |
| P1-old R1 | 100000 | 0.641980 | 0.455317 | 0.871324 | 0.299280 | 0.984680 | 0.015320 |
| C1-raw | 100000 | 0.706470 | 0.587849 | 0.928839 | 0.418660 | 0.994280 | 0.005720 |
| C1-center | 100000 | 0.809400 | 0.769929 | 0.928839 | 0.637840 | 0.980960 | 0.019040 |
| legion-retrained | 100000 | 0.729160 | 0.637779 | 0.931478 | 0.476880 | 0.981440 | 0.018560 |
| legion-retrained-match | 100000 | 0.676750 | 0.530303 | 0.926752 | 0.364960 | 0.988540 | 0.011460 |
| legion-intermediate + aligned Stage-2 | 100000 | 0.705230 | 0.591431 | 0.926150 | 0.426700 | 0.983760 | 0.016240 |
| RINE-official-224 | 100000 | 0.581940 | 0.307045 | 0.888290 | 0.185240 | 0.978640 | 0.021360 |
| RINE-336-adapted | 100000 | 0.773820 | 0.712641 | 0.963910 | 0.560920 | 0.986720 | 0.013280 |
| NPR-official-retrained | 100000 | 0.667250 | 0.560209 | 0.755341 | 0.423860 | 0.910640 | 0.089360 |
| C2-raw | 100000 | 0.720790 | 0.616180 | 0.942834 | 0.448240 | 0.993340 | 0.006660 |
| C2-center | 100000 | 0.768740 | 0.703627 | 0.942834 | 0.549040 | 0.988440 | 0.011560 |

八个生成器的分组结果保存在各模型原始分类结果中；此处只列完整 held-out 总体指标。

### LOKI 分类（Real 900，Fake 1,317）

| 模型 | N | Accuracy | F1 | ROC-AUC | Fake recall | TNR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 2217 | 0.542625 | 0.472973 | 0.654220 | 0.345482 | 0.831111 | 0.168889 |
| P1-old R1 | 2217 | 0.542625 | 0.472973 | 0.654220 | 0.345482 | 0.831111 | 0.168889 |
| C1-raw | 2217 | 0.625169 | 0.585949 | 0.751232 | 0.446469 | 0.886667 | 0.113333 |
| C1-center | 2217 | 0.718088 | 0.734156 | 0.751232 | 0.655277 | 0.810000 | 0.190000 |
| legion-retrained | 2217 | 0.583672 | 0.559847 | 0.681164 | 0.445710 | 0.785556 | 0.214444 |
| legion-retrained-match | 2217 | 0.561119 | 0.513257 | 0.680319 | 0.389522 | 0.812222 | 0.187778 |
| legion-intermediate + aligned Stage-2 | 2217 | 0.576906 | 0.547297 | 0.679072 | 0.430524 | 0.791111 | 0.208889 |
| RINE-official-224 | 2217 | 0.505187 | 0.398904 | 0.621919 | 0.276386 | 0.840000 | 0.160000 |
| RINE-336-adapted | 2217 | 0.678845 | 0.678410 | 0.777690 | 0.570235 | 0.837778 | 0.162222 |
| NPR-official-retrained | 2217 | 0.562923 | 0.531658 | 0.645753 | 0.417616 | 0.775556 | 0.224444 |
| C2-raw | 2217 | 0.641407 | 0.613890 | 0.756006 | 0.479879 | 0.877778 | 0.122222 |
| C2-center | 2217 | 0.676590 | 0.673051 | 0.756006 | 0.560364 | 0.846667 | 0.153333 |

### RAISE998（Real 998，Fake 0）

| 模型 | N | Accuracy | F1 | ROC-AUC | Fake recall | TNR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 998 | 0.993988 | - | - | - | 0.993988 | 0.006012 |
| P1-old R1 | 998 | 0.993988 | - | - | - | 0.993988 | 0.006012 |
| C1-raw | 998 | 1.000000 | - | - | - | 1.000000 | 0.000000 |
| C1-center | 998 | 0.998998 | - | - | - | 0.998998 | 0.001002 |
| legion-retrained | 998 | 0.988978 | - | - | - | 0.988978 | 0.011022 |
| legion-retrained-match | 998 | 0.992986 | - | - | - | 0.992986 | 0.007014 |
| legion-intermediate + aligned Stage-2 | 998 | 0.991984 | - | - | - | 0.991984 | 0.008016 |
| RINE-official-224 | 998 | 1.000000 | - | - | - | 1.000000 | 0.000000 |
| RINE-336-adapted | 998 | 0.998998 | - | - | - | 0.998998 | 0.001002 |
| NPR-official-retrained | 998 | 0.990982 | - | - | - | 0.990982 | 0.009018 |
| C2-raw | 998 | 1.000000 | - | - | - | 1.000000 | 0.000000 |
| C2-center | 998 | 0.998998 | - | - | - | 0.998998 | 0.001002 |

该集只有 Real；主要阅读 TNR/FPR。Fake recall、F1 和 ROC-AUC 无法作双类解释，统一记为 `-`。

同一 C1 分数下，raw 口径的 Internal2208 Accuracy 比 center 高 `0.004076`；AIGI-Holmes、GenImage、LOKI 三个双类 OOD 集分别低 `0.071821`、`0.102930`、`0.092919`。RAISE998 的 raw FPR 为 `0`，center 为 `0.001002`。这些差异只由固定决策阈值改变产生。

## 定位｜官方1000

### SynthScars Official1000（官方 polygon union，空 GT 0）

| 模型 | 条件 | N | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---|---:|---:|---:|---:|---:|
| P1 | G0 | 1000 | 0.229544 | 0.319323 | 0.236949 | 0.383118 |
| P1-old R1 | G0 | 1000 | 0.286588 | 0.393974 | 0.267042 | 0.421521 |
| C1-raw | G0 | 1000 | 0.206240 | 0.291229 | 0.190361 | 0.319838 |
| C1-native R1 | G0 | 1000 | 0.318875 | 0.439764 | 0.350363 | 0.518917 |
| legion-retrained | L-FREE | 1000 | 0.196195 | 0.286687 | 0.200490 | 0.334014 |
| legion-retrained-match | L-FREE | 1000 | 0.177485 | 0.259904 | 0.210989 | 0.348457 |
| legion-intermediate | L-FREE | 1000 | 0.223234 | 0.321147 | 0.241450 | 0.388980 |
| C2-raw | G0 | 1000 | 0.241224 | 0.334328 | 0.233257 | 0.378278 |
| C2-native R1 | G0 | 1000 | 0.300914 | 0.417736 | 0.326449 | 0.492215 |

P1、P1-old R1、C1-raw、C1-native R1、C2-raw、C2-native R1 为 canonical G0；三个 LEGION 模型为官方 L-FREE。legion-retrained-match 使用的旧格式 manifest SHA 不同，但 1,000 个 sample ID 与顺序和冻结 Official1000 完全一致。C1-native R1 的 Official1000 曾用于历史候选比较，这里是已冻结结果汇总，不能称为全新独立测试。

在自身架构路线上，P1-old R1 相对 P1 的 Mean FG IoU 增加 `0.057044`；C1-native R1 相对 C1-raw 增加 `0.112635`；C2-native R1 相对 C2-raw 增加 `0.059690`。C2 这组同底座逐图配对提升的 bootstrap 95% CI 为 `[0.048098, 0.071275]`，W/T/L 为 `628/38/334`；Global FG IoU 增加 `0.093192`。C2-raw 的 Mean FG IoU 高于 C1-raw `0.034984`，但 C2-native R1 仍低于 C1-native R1 `0.017961`；跨 C1/C2 训练路线的差值不能归因于单个架构模块。

C2-native R1 的生成轨迹与实时 C2 前向输出逐图对齐冻结的 C2-raw 记录。R1 所用独立 BF16 SAM 解码路径在阈值边缘与实时 C2 存在像素差异，其未修改基线 Mean FG IoU 为 `0.241206`；相对该同解码路径基线，C2-native R1 的配对 Mean FG IoU 增量为 `0.059708`。Official1000 未用于 C2-native R1 训练或选轮；该数据集的历史使用仍限制独立测试解释。

## 定位｜外部 OOD

### LOKI229（687 个官方区域框在 229 张 Fake 图像上取 union；空 GT 0）

| 模型 | 条件 | N | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---|---:|---:|---:|---:|---:|
| P1 | G1 | 229 | 0.076893 | 0.126401 | 0.077554 | 0.143945 |
| P1-old R1 | G1 | 229 | 0.061839 | 0.103666 | 0.045873 | 0.087722 |
| C1-native R1 | G1 | 229 | 0.087661 | 0.139278 | 0.062531 | 0.117702 |
| legion-retrained | L-FREE | 229 | 0.079178 | 0.127911 | 0.124314 | 0.221137 |
| legion-retrained-match | L-FREE | 229 | 0.097270 | 0.154727 | 0.119893 | 0.214115 |
| legion-intermediate | L-FREE | 229 | 0.098816 | 0.160061 | 0.116530 | 0.208735 |
| C2-raw | G1 | 229 | 0.059698 | 0.100194 | 0.043595 | 0.083548 |

legion-retrained-match 使用的旧格式 LOKI manifest SHA 不同，但 229 个 sample ID 与顺序和冻结清单一致。LOKI 的框 union 与 X-AIGD/PAL4VST 的像素伪影 GT 语义不同。

### X-AIGD labeled_test（官方人类感知伪影 polygon union；空 GT 247）

| 模型 | 条件 | N | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---|---:|---:|---:|---:|---:|
| P1 | G1 | 2419 | 0.071161 | 0.109794 | 0.070236 | 0.131253 |
| P1-old R1 | G1 | 2419 | 0.080213 | 0.119900 | 0.059704 | 0.112680 |
| C1-native R1 | G1 | 2419 | 0.083084 | 0.123880 | 0.075038 | 0.139601 |
| legion-retrained | L-FREE | 2419 | 0.077187 | 0.119954 | 0.079918 | 0.148008 |
| legion-retrained-match | L-FREE | 2419 | 0.084703 | 0.130600 | 0.094309 | 0.172363 |
| legion-intermediate | L-FREE | 2419 | 0.083862 | 0.129184 | 0.090972 | 0.166773 |
| C2-raw | G1 | 2419 | 0.063434 | 0.098394 | 0.061191 | 0.115326 |

官方 `labels=[]` 样本保留为全零 GT。论文级 X-AIGD 类别无关比较应优先报告 Global FG IoU/F1，逐图均值为补充。

### PAL4VST test（官方像素伪影 mask；空 GT 313）

| 模型 | 条件 | N | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---|---:|---:|---:|---:|---:|
| P1 | G1 | 1441 | 0.061279 | 0.094792 | 0.052402 | 0.099585 |
| P1-old R1 | G1 | 1441 | 0.097387 | 0.138952 | 0.094196 | 0.172175 |
| C1-native R1 | G1 | 1441 | 0.076464 | 0.113540 | 0.106225 | 0.192050 |
| legion-retrained | L-FREE | 1441 | 0.050836 | 0.081036 | 0.045726 | 0.087454 |
| legion-retrained-match | L-FREE | 1441 | 0.051007 | 0.082446 | 0.055633 | 0.105402 |
| legion-intermediate | L-FREE | 1441 | 0.059880 | 0.095279 | 0.061973 | 0.116714 |
| C2-raw | G1 | 1441 | 0.052712 | 0.081433 | 0.051530 | 0.098009 |

官方空 GT 样本全部保留；逐图均值包含这些样本。

## 解释边界与结果来源

- 定位中 P1、P1-old R1、C1-native R1、C2-raw 在外部 OOD 使用 canonical known-Fake G1；LEGION 三模型使用 image-only L-FREE。跨模型提示条件不同，数值并非完全 matched 的同提示对照。统一报告完整 N、固定 mask logit 阈值 `> 0`；无 `[SEG]` 或无有效 mask 样本按零分计。
- Mean FG 指逐图平均；Global FG 根据全数据集 TP/FP/FN 汇总。X-AIGD 与 PAL4VST 含空 GT，不能只看逐图均值，也不能把不同 GT 定义的数据集直接横比。
- 泄漏审计状态为 **BLOCKED_OVERLAP**，已记录的继续评测 override 为 **ACTIVE**。外部清单相对原 internal-TRAIN 共 779 个 exact 重叠、793 对 pHash 近重叠；exact 重叠全在 AIGI-Holmes。RINE-official-224 的实际训练集排除 40 张小图后，AIGI-Holmes exact 重叠为 775。pHash 分布：AIGI-Holmes 784、GenImage 7、LOKI 分类 1、PAL4VST 1。AIGI-Holmes 与 GenImage 之间还有 35 对 pHash 近重叠。
- 结果及逐样本来源：[原始四模型汇总](../outputs/final_evaluation/final_results.json)、[C1-native R1 汇总](../outputs/final_evaluation/c1_native_staged_r1/results.json)、[legion-retrained-match 汇总](../outputs/legion_retrained_match_evaluation/results.json)、[C1 内部测试逐样本分数](../outputs/phase6d5_decision_integration/raw_scores/internal_test.jsonl)、[C1 内部测试排序指标](../outputs/phase6d5_decision_integration/results.json)。数据集 manifest SHA256 与 revision 见原始四模型汇总的 `integrity` 字段；[C1 原生 R1 配对分析](phase6e3_c1_native_vs_original_c1_r1_official1000.md)另行报告。
- C1-raw 外部四集的完整 TP/TN/FP/FN、ROC-AUC 与 manifest SHA256 见[Phase6D.5 全量分类 OOD 结果](../outputs/phase6d5_full_classification_ood/results.json)；[Phase6D.6 决策边界记录](phase6d6_decision_boundary_disentanglement.md)将其与 C1-center 放在相同样本上比较。内部测试复用上一条的 2,208 条冻结分数及 C1 原始阈值指标。
- C2-raw 的内部测试、四组外部分类 OOD 与 Official1000 定位使用同一选定 checkpoint；逐图结果、清单与 checkpoint 哈希见[C2 阶段性评测汇总](../outputs/phase6j0_c2/final_evaluation/results.json)。C2-raw 外部定位 OOD 的 G1 逐样本结果和可供后续 C2 接 R1 复用的冻结特征缓存见[C2 定位 OOD 汇总](../outputs/phase6j0_c2/final_evaluation/localization/summary.json)。
- C2-native R1 的 Official1000 完整逐图结果、checkpoint 哈希、与 C2-raw 及独立 BF16 解码基线的配对统计见[Phase6P0 Official1000 结果](../outputs/phase6p0_c2_native_staged_r1/official1000_results.json)和[阶段报告](phase6p0_c2_native_staged_r1.md)。目前该模型没有定位 OOD 结果。
- C2-center 使用冻结 internal-TRAIN 均值校准，内部测试和 OOD 逐图分数精确复用 C2-raw；没有在测试/OOD 上选阈值。计数、清单及 checkpoint 哈希见[C2-center 汇总](../outputs/phase6j0_c2/center/results.json)。
- NPR 分类结果来源：[NPR 总结果](../outputs/npr_official_retrain/results.json)与[NPR 重训练记录](npr_official_retrain_results.md)。
- 新增分类结果来源：[公开 LE + aligned Stage-2 汇总](../outputs/legion_public_le_stage2_evaluation/results.json)、[官方 RINE 224 重训练汇总](../outputs/rine_official224_retrain/results.json)、[RINE 336 外部 OOD 汇总](../outputs/phase6b7_rine_ood/results.json)、[RINE 336 内部测试汇总](../outputs/phase6d5_decision_integration/results.json)。训练与评测身份分别见[公开 LE 第二阶段记录](legion_public_le_stage2_evaluation.md)、[官方 RINE 224 记录](rine_official224_retrain_results.md)和[RINE 336 记录](phase6b7_rine_ood_results.md)。
