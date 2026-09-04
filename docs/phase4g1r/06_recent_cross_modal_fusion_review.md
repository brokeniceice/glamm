# 2023–2026 targeted cross-modal fusion review

本次只保留与 conditional compatibility、spatial mismatch、missing/noisy modality直接相关的一手工作。完整字段见 `outputs/phase4g1r/literature_inventory.csv`。

| 工作 | 直接启示 | 不可越界的限制 |
|---|---|---|
| [CMNeXt, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Delivering_Arbitrary-Modal_Semantic_Segmentation_CVPR_2023_paper.html) | arbitrary/missing modality segmentation 说明统一 geometry 与缺失模态路径是成熟问题 | 没有显式 conditional utility |
| [Dynamic Multimodal Fusion, CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/MULA/html/Xue_Dynamic_Multimodal_Fusion_CVPRW_2023_paper.html) | noisy RGB-D 下 input-dependent fusion 优于固定融合的经验先例 | 非 language-conditioned，未给可干预 utility |
| [Missing Modality Robustness, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Maheshwari_Missing_Modality_Robustness_in_Semi-Supervised_Multi-Modal_Semantic_Segmentation_WACV_2024_paper.html) | missing-modality robustness 与显式 late fusion control 必须报告 | semi-supervised setting，不解决 compatibility |
| [Keep the Balance, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Cai_Keep_the_Balance_A_Parameter-Efficient_Symmetrical_Framework_for_RGBX_Semantic_CVPR_2025_paper.html) | 指出 global cross-modal correlation 可能引入 noise，并用 dynamic sparse fusion 限制交互 | 对称 PEFT 双流不兼容冻结 P1 anchor |
| [AMDANet, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhong_AMDANet_Attention-Driven_Multi-Perspective_Discrepancy_Alignment_for_RGB-Infrared_Image_Fusion_and_ICCV_2025_paper.html) | discrepancy-aware alignment 支持在 compatibility feature 中显式使用差异/一致性 | RGB-IR fusion/segmentation setting不同，不能当 utility 因果证据 |

综合证据支持“选择性、局部、联合 interaction”，不支持无约束 global attention 或把任意 attention map命名为 utility。空间 shuffle 要求促使 Primary 使用 original-normalized 同坐标局部 correspondence；cross-image 要求促使其同时看 `q_seg/S64/z_L` 与 F context。两项都必须从 forward dependency 产生，而不是依赖模型自行猜测 condition label。

Standalone Candidate C（仅 cross-attention）不进入候选：cross-attention 是成熟的 context exchange mechanism，但现有直接证据不足以说明其输出如何成为独立、可校准、可干预的 utility。
