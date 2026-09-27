# RINE 官方 224 配置：本项目数据重训练与分类评测

状态：**COMPLETE**（2026-09-26）。本实验使用 RINE 官方源码和 OpenAI CLIP `ViT-L/14` 的 **224px** 权重，在项目内部训练集上从随机分类头开始训练一轮；随后固定 epoch 1 checkpoint，评测内部测试集和四个外部数据集。没有载入官方发布的 RINE 分类头权重，也没有用 OOD 数据选 checkpoint、调阈值或校准。

## 1. 来源和复现口径

| 项目 | 冻结值 |
|---|---|
| RINE 官方源码 | `external/RINE_official`，commit `9b7fd5857cc205d0412be6aeee0d7611b95bd620` |
| OpenAI CLIP 官方源码 | commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |
| CLIP 官方权重 | `ViT-L-14.pt`，SHA256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836` |
| RINE 模型实现 | 直接导入官方 `src.models.Model`，取全部 24 层 `ln_2` 输出的 CLS token |
| RINE 4-class 配置 | `q=2`、`D'=1024`、`ξ=0.2`，Q1/TIE/Q2/head 共 6,323,201 个可训练参数 |
| 训练变换 | 直接导入官方 `get_transforms()[0]`：模糊/JPEG 增强、`RandomCrop(224)`、随机水平翻转、CLIP 归一化；**无 resize** |
| 验证/测试变换 | 直接导入官方 `get_transforms()[1]`：`CenterCrop(224)`、CLIP 归一化；**无 resize** |
| 训练 | seed 0，batch 128，Adam，学习率 `1e-3`，一轮、138 步；`BCEWithLogitsLoss(sum) + 0.2 ×` 官方 `SupConLoss`；CLIP 冻结 |
| 决策 | epoch 1 固定；Fake 为正类，sigmoid 概率阈值 0.5 |
| 训练 checkpoint | `outputs/rine_official224_retrain/checkpoint_epoch1.pt`，SHA256 `569da47fd7e5b0f511298b2e9924269c206c3453b7e15fa4b92e57ef9d931182` |

项目侧只实现 manifest 读取、训练循环、断点续评和结果保存：[执行脚本](../scripts/rine_official224_retrain.py)。官方模型、图像变换、SupCon 源码均未修改。完整训练前冻结协议和文件哈希见 [`protocol.json`](../outputs/rine_official224_retrain/protocol.json)。

**内部训练集边界：**原 manifest 有 17,672 张，其中 40 张 Real 图像宽或高小于 224，官方 `RandomCrop(224)` 对这些图像会报错。为保持训练变换原样，本实验预先列出并排除这 40 个 ID，实际训练 **17,632 张**（Fake 8,836、Real 8,796）。验证集 2,212 张、内部测试集 2,208 张和所有外部数据集均完整保留。因训练清单和预训练 CLIP 与 336 适配版不同，两版本差异不能单独归因于输入分辨率。

## 2. 内部测试与外部 OOD

数值保留六位小数。RAISE998 只有 Real，AUC、Fake recall 和 F1 不适用。

| 数据集 | N | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Internal validation | 2,212 | 0.981013 | 0.998251 | 0.974684 | 0.987342 | 0.012658 | 0.980892 |
| **Internal test** | **2,208** | **0.981431** | **0.998414** | **0.980978** | **0.981884** | **0.018116** | **0.981423** |
| AIGI-Holmes TestSet | 99,999 | 0.765968 | 0.985019 | 0.534031 | 0.997900 | 0.002100 | 0.695293 |
| GenImage held-out | 100,000 | 0.581940 | 0.888290 | 0.185240 | 0.978640 | 0.021360 | 0.307045 |
| LOKI classification | 2,217 | 0.505187 | 0.621919 | 0.276386 | 0.840000 | 0.160000 | 0.398904 |
| RAISE998 | 998 | 1.000000 | - | - | 1.000000 | 0.000000 | - |

各数据集的 [`results.json`](../outputs/rine_official224_retrain/evaluation/) 和逐图 `predictions.jsonl` 位于 `outputs/rine_official224_retrain/evaluation/<dataset>/`；[总结果](../outputs/rine_official224_retrain/results.json)和[完成状态](../outputs/rine_official224_retrain/status.json)均为 `COMPLETE`。

### GenImage 八个生成器

| 生成器 | N | 224 Accuracy | 224 ROC-AUC | 224 Fake recall | 224 FPR | 336 适配版 Accuracy | 336 适配版 ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| ADM | 12,000 | 0.527583 | 0.723023 | 0.079500 | 0.024333 | 0.643667 | 0.926686 |
| BigGAN | 12,000 | 0.504083 | 0.903471 | 0.027500 | 0.019333 | 0.890333 | 0.992310 |
| GLIDE | 12,000 | 0.578583 | 0.910346 | 0.179000 | 0.021833 | 0.887500 | 0.989863 |
| Midjourney | 12,000 | 0.694583 | 0.950854 | 0.409833 | 0.020667 | 0.786167 | 0.964630 |
| SD v1.4 | 12,000 | 0.631500 | 0.941763 | 0.282000 | 0.019000 | 0.841167 | 0.985078 |
| SD v1.5 | 16,000 | 0.632938 | 0.945675 | 0.285750 | 0.019875 | 0.851812 | 0.985990 |
| VQDM | 12,000 | 0.496417 | 0.801603 | 0.016000 | 0.023167 | 0.565667 | 0.895332 |
| Wukong | 12,000 | 0.572833 | 0.910518 | 0.168833 | 0.023167 | 0.698250 | 0.960630 |

## 3. 与现有 RINE-on-C1 @336 适配版对照

下表复用既有 [336 适配版内部测试记录](phase6d5_decision_integration.md)及[四项 OOD 冻结结果](phase6b7_rine_ood_results.md)，没有重复推理。两版本在外部四集使用完全相同的 manifest SHA256；内部测试的 2,208 个 sample ID、顺序和标签逐项一致。

| 数据集 | 224 Accuracy | 336 Accuracy | 224 ROC-AUC | 336 ROC-AUC | 224 Fake recall | 336 Fake recall | 224 FPR | 336 FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Internal test | 0.981431 | 0.993207 | 0.998414 | 0.999874 | 0.980978 | 0.999094 | 0.018116 | 0.012681 |
| AIGI-Holmes | 0.765968 | 0.884399 | 0.985019 | 0.991609 | 0.534031 | 0.771135 | 0.002100 | 0.002340 |
| GenImage | 0.581940 | 0.773820 | 0.888290 | 0.963910 | 0.185240 | 0.560920 | 0.021360 | 0.013280 |
| LOKI | 0.505187 | 0.678845 | 0.621919 | 0.777690 | 0.276386 | 0.570235 | 0.160000 | 0.162222 |
| RAISE998 | 1.000000 | 0.998998 | - | - | - | - | 0.000000 | 0.001002 |

严格官方 224 版在内部测试和三个混合类别 OOD 集上，Accuracy、ROC-AUC 与 Fake recall 均低于共享 CLIP@336 的适配版；RAISE998 的误报从 1/998 降至 0/998。224 版在 GenImage 的多个生成器上仍有较高 AUC，但固定 0.5 阈值下 Fake recall 很低。这个比较同时改变了 CLIP 权重、输入尺寸、训练/评测变换以及 40 张训练图的纳入情况，**不能作为“单独提高分辨率”的因果消融**。

## 4. 完整性与解释边界

- 外部四个 manifest SHA256 与既有 336 适配版完全一致；每个结果的逐图行数、sample ID 和顺序均与本次清单吻合。
- AIGI-Holmes 与原内部训练 manifest 存在 779 张文件 SHA256 完全重叠；在本次排除 40 张小图后，实际训练清单与其仍有 **775 张**重叠。AIGI-Holmes 指标必须附带此泄漏提示，不能称为无重叠的独立 OOD 测试。
- 本实验只评测真假**分类**，没有定位输出。RINE 官方发布的分类头权重没有参与训练初始化。
- 本次遵循官方 4-class 已发布超参数，不在项目 validation/test/OOD 上搜索 `q`、`D'`、`ξ`、阈值或 epoch；训练数据换成项目内部数据是任务要求中唯一的数据域替换，另外预先排除了官方随机裁剪无法处理的 40 张小图。
