# Phase 4E-0 — 代码架构审计

## 本地实现

| 组件 | 文件 / symbol | 输入 → 输出 | P1 lineage 中的状态 | 交互 / 迁移点 |
|---|---|---|---|---|
| causal SEG state | `model/GLaMM.py::extract_seg_predictor_hidden` | `[B,L,4096]+[B,T] → list[Nseg,4096]` | 随 LLM/LoRA policy 而变 | 正确选择 `[SEG]` 前一 state |
| text projection | `_initialize_text_projection_layer` | `4096→4096→256` | 相应 recipe 可训练 | 当前压缩瓶颈 |
| SAM prompt path | `_generate_and_postprocess_masks` | `[Nseg,256] → [Nseg,1,256]` sparse prompt | prompt encoder 冻结 | 不接收 forensic lattice |
| SAM image path | `get_grounding_encoder_embs` | image → `[B,256,64,64]` | canonical P1 冻结 | rectification/fusion 候选点 |
| mask decoder | SAM mask decoder | image grid + prompt → mask logits | 随 recipe | replacement/augmentation 候选点 |
| forensic adapter | `clip_forensic_adapter.py::CLIPSpatialArm` | `[B,1024,24,24] → [B,256,24,24]` | projection + 3 local residual blocks | 应保留的 positive dense representation |
| minimal Reader | `evidence_reader.py` | q `[B,1,256]`、F `[B,576,256] → q'` | Reader + scalar gate | 把 dense evidence 压入一个 q |
| position Reader | `position_aware_evidence_reader.py` | fixed 2D lattice + 单层 cross-attention residual | Reader + beta | 修正后仍 matched≈cross/shuffle |

## 官方实现核对

### PixelLM

官方 `model/PixelLM.py` 构造 `MaskDecoderMultiScale`，支持 `seg_token_num` 与 `image_feature_scale_num`；codebook state 与 image feature 在 lightweight decoder 内相遇。这与仅向 GLaMM answer 增加多个独立 `[SEG]` 不同。可迁移单元是 coordinated query/codebook bank 与 multi-scale decoder interaction。

### PSALM

官方 `llava_phi.py` 实例化 `NUM_OBJECT_QUERIES` 个 learned `seg_query`、multi-scale deformable pixel decoder 与 Mask2Former transformer predictor，并分别投影 segmentation query、`[SEG]` state 与 class name。可迁移单元是 condition-aware query set + dense pixel decoder；不照搬其 panoptic task schema。

### OMG-LLaVA

官方 `omg_llava.py` 把 universal segmentation visual encoder 投影进 LLM，并可启用 `projector_text2vision`；visual decoder 可显式训练或冻结。可迁移单元是 bidirectional LLM↔perception linkage 与保留 decoder query/prior embedding。

### FakeShield

官方 MFLM `model/GLaMM.py` 仍把 projected language hidden 作为 SAM sparse `text_embeds`。其更大的 DTE-FDM→MFLM 系统支持 staged forensic reasoning，但 MFLM 代码不能证明 single-query SAM prompting 已解决 dense evidence utilization。

## 官方仓库逐项记录

| Paper / repo | Relevant files / class | token construction 与 input→output | projector / decoder / fusion | trainable policy 与 loss | 可迁移点 |
|---|---|---|---|---|---|
| LISA / [repo](https://github.com/dvlab-research/LISA) | `model/LISA.py`, `LISAForCausalLM` | `[SEG]` state `[N,4096]→prompt [N,256]→mask` | text MLP + SAM prompt/mask decoder | LoRA、text MLP、SAM mask decoder；CE+BCE+Dice | lineage baseline；不复制 single bottleneck |
| GLaMM / [repo](https://github.com/mbzuai-oryx/groundingLMM) | `model/GLaMM.py`, `GLaMMForCausalLM` | multi occurrence 各自提取 causal state；region feature 另行进入 LLM | `text_hidden_fcs` + SAM；每 mask 独立 | LoRA/projector/mask decoder 依 recipe；CE+BCE+Dice | 保留 P1/canonical geometry，替换下游接口 |
| PixelLM / [repo](https://github.com/MaverickRen/PixelLM) | `model/PixelLM.py`, `MaskDecoderMultiScale` | `seg_token_num` codebook hidden + multi-scale CLIP feature → masks | 4096→256 MLP；multi-scale SAM-like lightweight decoder | codebook/projector/decoder；mask loss + target refinement/overlap | `K` query codebook、multi-scale dense interaction |
| PSALM / [repo](https://github.com/zamling/PSALM) | `llava_phi.py`, `seg_query`, `MSDeformAttnPixelDecoder`, Mask2Former predictor | learned object queries、condition/`[SEG]`/class projections + feature pyramid → query masks | deformable pixel decoder + transformer decoder | segmentation decoder/projectors；Hungarian matching、mask/class losses、deep supervision | query assignment 与 condition-aware dense decoder |
| OMG-LLaVA / [repo](https://github.com/lxtGH/OMG-Seg) | `omg_llava.py`, `OMGSegVisualEncoder`, projector modules | image/object/pixel/prior query 经 visual→LLM projector；可由 text→vision projector 回传 | universal segmentation encoder-decoder 与双向 projector | 可 freeze visual encoder/decoder，LoRA 可选；language+segmentation task loss | bidirectional task link、保留 perception query |
| FakeShield / [repo](https://github.com/zhipeixu/FakeShield) | `DTE-FDM`, `MFLM/model/GLaMM.py` | DTE textual description/domain cue；MFLM `[SEG]` state→SAM prompt | staged modules；MFLM 保留 GLaMM SAM path | staged instruction/localization training；CE+mask losses | language proposal 可作 teacher/context；不要误称其 MFLM 已做 dense fusion |

ForgeryGPT 没有定位到可核验的官方完整实现；只根据论文记录其 FL-Expert、object-agnostic prompt、vocabulary-enhanced encoder、mask encoder 与三阶段 mask-text alignment，不虚构文件/class。Propose and Rectify 以论文架构证据支持 rectification，当前不把未核验代码细节写成实现事实。

## 拟议代码边界（仅设计）

```text
ForensicGroundingDecoder(
  g0_state: [B,4096],
  sam_grid: [B,256,64,64],
  forensic_grid: [B,256,24,24]
) -> queries [B,K,256], multi_level_features,
     spatial_attention, mask_logits [B,1,H,W]
```

建议冻结 proposal 为 `K=4`：一个由 G0 初始化的 semantic anchor 加三个 learnable residual slot。这不是经验最优值，而是首次 formal study 中足以检验 coordinated multi-token behavior、同时接近 PixelLM 三 token 公开 recipe 的最小 query set。实际 spatial shape 必须由 runtime tensor 得到，不得把旧式 575 offset 复制到新 decoder。

## 实现风险

- generated 与 teacher-forced path 必须共用 causal-state helper；
- Detection/G0/TF prompt 必须显式固定为 canonical；
- SAM 与 CLIP inverse geometry 在计算任何 loss 前做 exact audit；
- wrong-image 与 spatial-content shuffle 必须固定 position lattice；
- 不再使用 zero scalar gate，以免 BF16/gradient suppression；
- multiple slot 必须有 assignment 或 shared union-mask aggregation，防止 duplicate/empty shortcut。
