# Phase 4E-0 — 文献清单

访问日期：2026-08-25。优先依据正式论文正文与官方仓库；标为“迁移推断”的内容是本项目综合判断，不是原论文的直接结论。

## MLLM grounding lineage

| 工作 | 发表 | 与本项目相关的核心机制 | 官方实现 | 迁移判断 |
|---|---|---|---|---|
| [LISA](https://openaccess.thecvf.com/content/CVPR2024/papers/Lai_LISA_Reasoning_Segmentation_via_Large_Language_Model_CVPR_2024_paper.pdf) | CVPR 2024 | `[SEG]` hidden 投影到 SAM prompt space；联合 language/mask loss | [仓库](https://github.com/dvlab-research/LISA) | 当前 lineage；single bottleneck 的审计起点 |
| [GLaMM](https://arxiv.org/abs/2311.03356) | CVPR 2024 | grounded conversation、region feature、多 grounded phrase、SAM grounding encoder | [仓库](https://github.com/mbzuai-oryx/groundingLMM) | 直接代码 lineage；多个 `[SEG]` 不等于单目标的 coordinated codebook |
| [PixelLM](https://openaccess.thecvf.com/content/CVPR2024/papers/Ren_PixelLM_Pixel_Reasoning_with_Large_Multimodal_Model_CVPR_2024_paper.pdf) | CVPR 2024 | segmentation codebook、multi-scale image feature、lightweight pixel decoder、target refinement loss | [仓库](https://github.com/MaverickRen/PixelLM) | multiple target/scale token 与非单一 SAM prompt decoder 的强先例 |
| [PSALM](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04964.pdf) | ECCV 2024 | condition prompt、mask token、Mask2Former multi-scale decoder | [仓库](https://github.com/zamling/PSALM) | decoder-side conditional query 的强先例 |
| [OMG-LLaVA](https://arxiv.org/abs/2406.19389) | NeurIPS 2024 | universal segmentation encoder/decoder；image/object/pixel/prompt token；text-to-vision projection | [仓库](https://github.com/lxtGH/OMG-Seg) | 双向 LLM–perception interface 先例 |
| [VisionLLM v2](https://papers.nips.cc/paper/2024/file/81a60d18e010b27b36cd465c6604b915-Paper-Conference.pdf) | NeurIPS 2024 | super link 把 task information 与 gradient 传给 task decoder | 论文项目链接 | 保留 specialist dense decoder，而非把全部证据塞进 language token |

## Explainable image forensics

| 工作 | 状态 | 机制 | 相关性与边界 |
|---|---|---|---|
| [ForgeryGPT](https://arxiv.org/abs/2410.10238) | official arXiv | FL-Expert + mask encoder；mask-aware forgery extractor；分阶段 mask–text alignment | 支持 forensic/mask information 显式进入 MLLM；未定位到可核验的官方完整实现，因此不作代码结论 |
| [FakeShield](https://openreview.net/pdf/c2e03abe9dfa3b923b99b40e14a5d9dfde0668a3.pdf) | ICLR 2025 | DTE-FDM explanation/domain tag，随后 MFLM/SAM localization | 支持 language proposal 引导定位，但不证明本项目 dense representation 会被利用 |
| [FakeShield 官方代码](https://github.com/zhipeixu/FakeShield) | 已发布 | MFLM 仍使用 GLaMM-style text-to-SAM sparse prompt；DTE 与 MFLM 分模块 | 说明“forensic system”本身不会自动消除 single-query path |
| [Propose and Rectify](https://arxiv.org/abs/2508.17976) | official arXiv，2025 | MLLM proposal、multi-scale forensic rectification、enhanced SAM segmentation | decoder/image-embedding rectification 的最直接先例；较新，仅作支持证据而非唯一依据 |

## Dense 与 query distillation

| 工作 | 发表 | 蒸馏对象 | 相对 Phase 3F/3G 的修正 |
|---|---|---|---|
| [Structured KD](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.pdf) | CVPR 2019 | pixelwise output、pairwise relation、holistic structure | 不把空间图压成单个 cosine vector |
| [Channel-Wise KD](https://openaccess.thecvf.com/content/ICCV2021/papers/Shu_Channel-Wise_Knowledge_Distillation_for_Dense_Prediction_ICCV_2021_paper.pdf) | ICCV 2021 | channel-wise spatial distribution 的 normalized KL | 传递尺度归一化的 salient spatial distribution |
| [CIRKD](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Cross-Image_Relational_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2022_paper.pdf) | CVPR 2022 | pixel–pixel 与 pixel–region cross-image relation | 支持关系教师信号；memory bank 首轮非必需 |
| [Knowledge Review](https://arxiv.org/abs/2104.09044) | CVPR 2021 | cross-stage feature review、ABF、hierarchical contextual loss | 支持 multi-level decoder feature transfer |
| [DETRDistill](https://openaccess.thecvf.com/content/ICCV2023/papers/Chang_DETRDistill_A_Universal_Knowledge_Distillation_Framework_for_DETR-families_ICCV_2023_paper.pdf) | ICCV 2023 | matched prediction logit、target-aware feature KD、query-prior assignment | 支持 query set matching，而非 raw indexwise cosine |
| [KD-DETR](https://arxiv.org/abs/2211.08071) | CVPR 2023 | query detector 的 consistent distillation point | teacher/student query 不同位置时的补充先例 |

## 文献质量结论

没有一篇论文可以整套照搬。可辩护的迁移是成熟机制的证据驱动组合：PixelLM/PSALM 的 multiple conditional queries、VisionLLM v2/OMG-LLaVA 的 specialist decoder linkage、Propose-and-Rectify 的 forensic dense rectification，以及 structured/query KD。创新点应是它们针对 autonomous explainable forensic grounding 的新组合，而不是宣称发明任一基础组件。

## 核心机制精读卡

### Q1：segmentation token / query interface

- **LISA**：一个 segmentation-token causal state 通过 MLP 进入 SAM，language CE 与 BCE/Dice mask loss 共同优化。它证明 special-token-as-mask-prompt 可行，但没有对单 token 容量作充分性证明。
- **GLaMM**：扩展为 grounded conversation、region feature 与多个 phrase/`[SEG]` association；每个 occurrence 仍形成独立 prompt/mask，因此“multi occurrence”不等于同一 mask 的 query codebook。
- **PixelLM**：segmentation codebook 包含面向不同 visual scale 的 learnable token；public recipe 公开 `seg_token_num=3`、`image_feature_scale_num=2`。codebook hidden 与 multi-scale image feature 在 lightweight pixel decoder 相遇，target refinement/overlap 约束多个 target 的区分。
- **PSALM**：input schema 显式区分 image、task instruction、conditional prompt、mask token；Mask2Former lineage 的 learned queries、multi-scale deformable pixel decoder、Hungarian matching 与 deep supervision 构成统一 segmentation formulation。
- **OMG-LLaVA**：universal segmentation model 同时产生 pixel/object/perception-prior token；visual→LLM projector 与可选 text→vision projector 建立双向接口，pixel-level state 不完全依赖普通 CLIP patch token。
- **VisionLLM v2**：Super Link 让 MLLM task token 与 task-specific decoder 通信，并把 decoder loss gradient 传回 task interface；其关键启示是 specialist decoder state 和梯度通道，不是一个更宽的 hidden MLP。

### Q2：forensic evidence interface

- **ForgeryGPT**：Mask-Aware Forgery Extractor 由 FL-Expert 与 Mask Encoder 组成；FL-Expert 使用 object-agnostic forgery prompt 和 vocabulary-enhanced vision encoder 提取 multi-scale fine-grained clue，再把 mask-aware feature 对齐到语言空间。Stage 1 冻结 vision encoder/LLM，仅做通用 image-text projector alignment；Stage 2 用 Mask-Text Alignment 数据做 region/mask-text alignment；Stage 3 做 IFDL task-specific instruction tuning。这说明普通 CLIP alignment 不被视为足以自动发现 subtle artifact。
- **FakeShield**：DTE-FDM 先利用 domain tag 区分 Photoshop、DeepFake、AIGC 等 domain 并生成 detailed textual tampering description；MFLM/TCM 再把长文本与视觉 condition 对齐，产生 segmentation intent 交给 SAM。两个模块分别训练。与本项目 P1 相同点是 text-guided SAM localization，不同点是它显式提供上游 detailed description/domain cue；其官方 MFLM 仍没有证明 dense forensic lattice 直达 decoder。
- **Propose and Rectify**：proposal stage 由 forensic-adapted MLLM 给出 semantic analysis 与初始区域；FRM 用多个 specialized filter 的 multi-scale forensic feature 验证/修正 proposal；ESM 对 forensic 与 SAM image embedding 做 alignment，并用语义/取证 interaction（包括 difference/product 类 discrepancy）形成 spatial/channel enhancement 后再解码。它直接对应“4C-A positive + Reader negative”，支持在 SAM image path 保留 forensic 2D cue。

### Q3：teacher 应传什么

- **Structured KD** 同时传 pixel output、pairwise feature relation 与 holistic structure，说明 dense prediction 的 target 不应退化成单 vector distance。
- **Channel-Wise KD** 把每个 channel 的 spatial response 归一化为概率分布并做 temperature KL，适合迁移“关注哪里”。
- **CIRKD** 用 pixel-to-pixel、pixel-to-region 和 cross-image relation 迁移结构；本项目首轮不引入 memory bank，但保留 relation 思路。
- **Knowledge Review** 通过 ABF 连接不同 stage，并用 hierarchical contextual loss 做 cross-level review，反驳“必须同层同维一一对齐”的默认假设。
- **DETRDistill/KD-DETR** 先解决 teacher/student query correspondence，再蒸馏 prediction、target-aware feature 或 consistent query point；因此 TF state 更适合作为 teacher query/behavior source，而不是未经 assignment 的 vector target。
