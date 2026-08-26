# 文献检索与证据协议

## 检索范围

检索覆盖五类强制方向：cross-modal feature rectification、token selective fusion、gated/bounded adaptation、modality imbalance、reliability/uncertainty/conflict/MoE routing；另单独核查 image forensics 与 explainable IFDL。

主要检索源为 CVF Open Access、ACL Anthology、PMLR/ICML、OpenReview/ICLR、NeurIPS Proceedings 与 arXiv 官方稿。架构冻结只引用原论文或官方项目页，博客不进入核心证据链。

时间分层：

- 经典机制：GMU、MAG、MMTM、UNO、TMC。
- 2022–2024：CMX、TokenFusion、OGM-GE、PMR、QMF、EAU、MMPareto、Predictive Dynamic Fusion。
- 2025–2026：ECoLaF、UMFNet、Omni-IML，以及近期 explainable IFDL。

## 标准抽取字段

每篇记录：paper、venue/year、task、modalities、fusion location、gate granularity、reliability source、fusion equation、bounded residual、bidirectionality、conflict handling、training balancing、auxiliary loss、frozen-backbone compatibility、spatial geometry、正式消融、失败模式、Phase 4F relevance、candidate usage。

## 证据等级

- **DIRECT**：原论文在与该条机制相同的任务层级明确实验验证，例如 ECoLaF 对 dense segmentation 的 pixel-wise conflict discount。
- **ADAPTED**：机制成熟且有消融，但需跨任务或跨表示迁移，例如 MAG 从 token sentiment 迁移到 SAM spatial feature。
- **INSPIRATIONAL**：概念相似但不足以支撑核心架构，例如通用 MoE router 对本项目的启发。

Primary 不得主要建立在 INSPIRATIONAL 证据上。

## 检索式主题

使用的主题组合包括：`cross-modal feature rectification segmentation`、`multimodal token fusion positional alignment`、`bounded multimodal adaptation gate`、`modality imbalance gradient modulation`、`quality-aware dynamic fusion`、`evidential conflict multimodal segmentation`、`uncertainty-aware pixel fusion`、`forgery localization reliability map`、`MLLM forensic rectification`。

## 纳入与排除

纳入：给出明确 forward、reliability/gate 生成、训练目标或消融的原论文。排除为架构核心：只有摘要口号、没有可复现机制、依赖 GT/任务身份作为 inference gate、或仅在无空间结构的分类中展示任意 softmax confidence 且无校准审计的方案。

## 校准审计原则

softmax/LLM token probability 只是一种候选信号，不自动等于 localization correctness。未来若使用任何 confidence，必须在冻结 train-calibration fold 上报告 reliability–error correlation、ECE/Brier/NLL、分布偏移敏感性和 reliability permutation test；不通过则不能进入正式融合。

## 原始来源索引

完整 URL、年份、venue 与标准字段见 `outputs/phase4g0/paper_inventory.csv`。核心来源包括 [CMX](https://arxiv.org/abs/2203.04838)、[TokenFusion](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Multimodal_Token_Fusion_for_Vision_Transformers_CVPR_2022_paper.html)、[MAG](https://aclanthology.org/2020.acl-main.214/)、[OGM-GE](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.pdf)、[QMF](https://proceedings.mlr.press/v202/zhang23ar.html)、[ECoLaF](https://openaccess.thecvf.com/content/WACV2025/html/Deregnaucourt_A_Conflict-Guided_Evidential_Multimodal_Fusion_for_Semantic_Segmentation_WACV_2025_paper.html) 与 [UMFNet](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Uncertainty-Aware_Modality_Fusion_for_Unaligned_RGB-T_Salient_Object_Detection_CVPR_2026_paper.html)。

