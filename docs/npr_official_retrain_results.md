# NPR 官方架构：本项目数据重训练与分类评测

状态：**COMPLETE**。使用官方 NPR 源码的随机初始化分类网络，在本项目内部训练清单上单独重训 50 轮；固定第 50 轮权重后完成内部测试和四个外部 OOD 数据集。

## 来源与协议

- 官方论文：[Tan et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.html)；[官方代码仓库](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)，固定 commit `781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a`。
- 直接导入官方 `networks.resnet.resnet50(pretrained=False, num_classes=1)`：nearest-neighbor 下/上采样差分作为输入，截断 ResNet50 的 layer1/layer2 后全局池化和单 logit head。训练从随机权重开始，未载入作者发布的 NPR 权重、ImageNet 权重或项目历史 NPR+SRM 联合权重。
- 内部训练 17,672 张（Real/Fake 各 8,836）；内部验证 2,212 张；训练 batch 32、Adam 学习率 `2e-4`、BCEWithLogitsLoss、50 轮，按官方代码每 10 轮衰减 0.9。双卡 DDP 每卡 batch 16、同步 BatchNorm，保持全局 batch 32；此执行方式与官方单卡脚本有浮点/批统计差异。
- 训练图像变换：Resize 到 256×256、RandomCrop 224、随机水平翻转、ImageNet 归一化；验证和测试：Resize 到 256×256，不裁剪、不翻转、相同归一化。这些是官方数据代码中的有效变换；项目 adapter 只负责按既定 manifest 读取和记录。
- 仅第 50 轮 checkpoint 用于结果；Fake 为正类，官方 `sigmoid > 0.5` 决策。内部测试和 OOD 没有参与选择 checkpoint、阈值或超参数。官方训练脚本每轮调用其测试集；本实验为保持测试隔离，省略该训练中测试调用。
- 执行在第 6 轮结束的完整 checkpoint 处恢复为独立后台进程；模型和 Adam 状态接续，从第 7 轮起随机裁剪/翻转序列与不中断训练可能不同。50 轮预算、数据清单及选模规则未变。
- [冻结协议与 manifest SHA256](../outputs/npr_official_retrain/protocol.json)、[checkpoint 身份](../outputs/npr_official_retrain/checkpoint_identity.json)、[逐图结果目录](../outputs/npr_official_retrain/evaluation/)、[总结果](../outputs/npr_official_retrain/results.json)。

## 分类结果

| 数据集 | N | Accuracy | F1 | ROC-AUC | Fake recall | TNR | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Internal validation | 2212 | 0.953436 | 0.954038 | 0.992093 | 0.966546 | 0.940325 | 0.059675 |
| Internal2208 | 2208 | 0.949728 | 0.950336 | 0.988020 | 0.961957 | 0.937500 | 0.062500 |
| AIGI-Holmes TestSet | 99999 | 0.782098 | 0.734501 | 0.909928 | 0.602832 | 0.961360 | 0.038640 |
| GenImage held-out | 100000 | 0.667250 | 0.560209 | 0.755341 | 0.423860 | 0.910640 | 0.089360 |
| LOKI classification | 2217 | 0.562923 | 0.531658 | 0.645753 | 0.417616 | 0.775556 | 0.224444 |
| RAISE998 | 998 | 0.990982 | - | - | - | 0.990982 | 0.009018 |

RAISE998 只有 Real；F1、ROC-AUC、Fake recall 不适用。

### GenImage 分生成器

| 生成器 | N | Accuracy | ROC-AUC | Fake recall | FPR |
|---|---:|---:|---:|---:|---:|
| adm | 12000 | 0.492167 | 0.346766 | 0.078333 | 0.094000 |
| biggan | 12000 | 0.756500 | 0.903108 | 0.599667 | 0.086667 |
| glide | 12000 | 0.658417 | 0.839217 | 0.406000 | 0.089167 |
| midjourney | 12000 | 0.746750 | 0.872896 | 0.583333 | 0.089833 |
| sdv4 | 12000 | 0.764417 | 0.901406 | 0.622667 | 0.093833 |
| sdv5 | 16000 | 0.770375 | 0.906415 | 0.624750 | 0.084000 |
| vqdm | 12000 | 0.459917 | 0.419626 | 0.012000 | 0.092167 |
| wukong | 12000 | 0.655083 | 0.805698 | 0.397167 | 0.087000 |

## 解释边界

- 这是**单独重训练的 NPR 检测器**；项目历史 NPR+SRM 的共享分类头只能作特征/分支诊断，不能替代此基线。
- AIGI-Holmes TestSet 与原内部训练清单已有 779 个文件 SHA256 完全重叠；NPR 使用全部原训练图，这 779 个重叠会影响该集的独立 OOD 解释。评测保留完整 99,999 张以与其他模型同清单对比。
- NPR 仅作真假分类，不输出定位 mask。其随机初始化、CNN 取证输入和 256 像素测试变换与 CLIP/RINE/C1 不同，跨模型性能差异不是单因素消融。
