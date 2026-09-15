# Phase 6F.0 — C1 + new R1 Forensic Dense-Evidence Bottleneck Audit

## A. Executive conclusion

本轮是只读架构、代码、checkpoint 与既有 internal-validation artifact 审计；没有训练、模型推理、test/OOD 访问或 checkpoint 选择。

1. **当前 C1 + new R1 确实受 dense forensic evidence 的硬性可见域限制，但尚不能证明 `F_forensic` 是当前唯一或主要性能瓶颈。** 非方形输入的 CLIP center crop 会不可恢复地删除 crop 外视觉信息；Rectifier 与 Utility 只能在剩余 support 内工作。另一方面，当前尚无同一 internal-validation sample 的 C1 与 selected new-R1 prediction 联表，因此还不能检验 coverage 与 `delta IoU`、inside/outside-support FN 的关系。
2. **Coverage 值得作为第一优先级，但第一步应是诊断，不是 multi-tile 正式训练。** 既有 validation mask/geometry cache 的 metadata-only 计算显示：1106张Fake中212张发生非方形 crop；精确 CLIP crop 的 GT coverage 均值为0.960715，当前 patch-center support hull 的同口径均值为0.938424；15张 coverage=0，另有23张位于 `(0,0.5)`。
3. **Multi-level CLIP 有合理实验空间，但没有已有 dense layer-level 对照证据。** 当前 spatial route只保存/使用 penultimate hidden `[-2]` patch tokens。RINE的24层CLS结果只能提供研究动机，不能证明 dense patch fusion有效。
4. **现在不建议增加 adapter capacity。** Phase4C-A 已证明3-block adapter显著优于 raw/projection-only；后续没有实验替换或推翻该 selected adapter，而且 current new R1 训练明确冻结它。应先区分“未采到信息”和“单层遗漏信息”，再研究加工能力。
5. 建议路线为：

   `coverage attribution diagnostic -> controlled full-FOV acquisition -> matched single/multi-level probe -> adapter capacity only if needed`

最终建议：**`INSUFFICIENT_EVIDENCE_NEED_DIAGNOSTIC`**，方向上执行 **`PROCEED_COVERAGE_FIRST`**，但这里的“proceed”仅指先完成预注册的 validation read-only attribution diagnostic。

## B. Historical evidence table

| 阶段 | 比较与关键结果 | 已支持结论 | 对当前路线的约束 |
|---|---|---|---|
| Phase2C | NPR/SRM classification residual fusion有有限正收益；localization保持不变/略降 | NPR/SRM存在分类互补性，不构成dense定位改善证据 | 不重启NPR/SRM backbone zoo |
| Phase3C.1 | frozen linear probe mean FG IoU：SAM 0.098538、CLIP 0.143842、NPR 0.003808、SRM 0.003588、FOCAL 0.048959 | 在当时相同probe协议下CLIP是最强frozen spatial source | CLIP继续作为anchor；不是“所有未来表示中绝对最优”的证明 |
| Phase4A FEPN | FEPN 0.068252 vs matched CLIP 0.143842；delta -0.075590，95% CI `[-0.085346,-0.065996]` | FEPN学到global classification evidence，但dense localization明显弱于CLIP | 不用FEPN替换CLIP；优先specialize CLIP |
| Phase4C-A | raw 0.143842；projection-only 0.138264；3-block adapter 0.171424；adapter-raw +0.027581，adapter-proj +0.033160 | local adapter提供真实增益，且不是单纯1x1 projection效果 | selected adapter是当前合理基线；不能把后续容量增长与信息增益混为一谈 |
| Phase4C-A failure | adapter zero-IoU 28.93%，IoU<0.1 53.16%；562/284/260 win/tie/loss | adapter不是普遍解决方案，仍有persistent misses与over-localization | 需要定位failure来源，但该记录未区分crop内外 |
| Phase4F | P1/FORENSIC-RECT/CLIP-RECT G0：0.148233/0.171614/0.157983 | adapted forensic feature比projection-only feature更适合rectification | selected Phase4C-A forensic adapter进入后续R1链路 |
| Phase4H-C | utility训练使用 `S64 + U_F * Delta_F`；A2为seed3407随机utility并在DEV G0选择epoch3 | learned utility可决定位置性使用forensic correction | evidence缺失区域不能被utility凭空恢复 |
| Phase4H-D | rectifier unfreeze：R1 0.195477 vs R0 0.179644，delta +0.015833，CI `[+0.009060,+0.022501]` | 在同一evidence下适配rectifier有效 | Rectifier仍受输入support/representation上限约束 |
| Phase6E.1 | old R1可迁移到C1；Official1000 C1 0.206240 -> C1+oldR1 0.300479 | C1 `[SEG]` query改变但旧dense route仍有用 | 当前baseline必须是C1，不退回P1 |
| Phase6E.2 | selected new R1 epoch8；internal-val G0 0.202732；Official1000 C1+newR1 0.319867，较C1+oldR1 +0.019389 | C1-specific utility/rectifier适配进一步有效 | 主架构冻结为C1+new R1；本轮只审计evidence输入 |
| Phase6B RINE | 24个CLIP block的CLS，经Q1/TIE/Q2做image-level分类 | classification存在使用层级信息的动机 | 不能直接复制到dense patch route；TIE层权重变化很小，不能当作dense层选择证据 |

历史链路因此是：

`NPR/SRM/FOCAL/SAM/CLIP comparison -> CLIP selected -> FEPN replacement dense failure -> CLIP specialization -> forensic adapter positive -> Rectifier -> Utility Gate -> old R1 -> C1-specific new R1 -> current C1 + new R1`

仓库中没有发现后续实验替换 Phase4C-A selected forensic adapter。Phase6E.2 的 frozen source state训练前后hash均为 `9e8f3853...b50e9`，配置仍指向 Phase4C-A epoch4 selected checkpoint（文件SHA256 `725dd44e...078f7`）。

## C. Current architecture trace

### C.1 在线部署/Official1000真实路径

```text
raw RGB
  |-- C1 global_enc_image
  |     -> C1 GLaMM CLIP vision tower
  |     -> selected penultimate patch tokens [B,576,1024]
  |     -> reshape H_clip [B,1024,24,24]
  |     -> frozen Phase4C-A CLIPSpatialArm
  |          projection 1024->256 (1x1)
  |          3 x LocalForensicBlock
  |          -> F_forensic [B,256,24,24]
  |          -> z_F24/dense logits [B,1,24,24]
  |
  |-- C1 grounding_enc_image
  |     -> frozen SAM image encoder
  |     -> S_base/raw [B,256,64,64]
  |
  |-- C1 canonical G0 generation
        -> actual generated [SEG] hidden(s)
        -> C1 text_hidden_fcs
        -> q_seg [T,256]

geometry_for(CLIP/SAM) -> clip_coordinates [B,576,2]
                        -> sam_coordinates [B,4096,2]
F_forensic + S_base + coordinates
  -> GeometryAwareSAMRectifier
  -> S_rect, Delta_F=S_rect-S_base, attention, semantic support

S_base/q_seg/base mask logits + F_forensic/z_F24
  -> CSCU utility branch
  -> U_F on forensic support

S_adapt = S_base + U_F * Delta_F
  -> frozen SAM prompt encoder + mask decoder
  -> low-res mask -> inverse SAM geometry -> original-space mask
```

### C.2 文件、tensor与状态

| 步骤 | 文件；class/function | tensor/shape | 状态与checkpoint | 几何/重采样/信息损失 |
|---|---|---|---|---|
| CLIP preprocessing | `datasets/`中的`CLIPImageProcessor`；几何复刻于 `tools/phase3c1.py::geometry_for('clip')` | RGB -> `[B,3,336,336]` | C1 vision tower frozen | shortest edge resize到336后center crop 336；crop外像素不可恢复 |
| CLIP patch extraction | `model/llava/model/multimodal_encoder/clip_encoder.py::CLIPVisionTower.feature_select`；在线见 `scripts/phase6e2_official_finalize.py::evaluate` | hidden `[-2]`, remove CLS，`[B,576,1024]` | frozen CLIP ViT-L/14@336 | 所有block patch resolution均为24x24；本路径只保留一层 |
| raw grid | `scripts/phase3c1_cache.py::clip_grid` | `H_clip=[B,1024,24,24]` | train/val来自Phase3C.1 immutable cache；在线实时提取 | 无额外空间插值，只reshape |
| forensic adapter | `model/clip_forensic_adapter.py::CLIPSpatialArm.forward` | `F0/F_forensic=[B,256,24,24]`, logits `[B,1,24,24]` | Phase4C-A epoch4 selected；完全frozen/eval | 3个3x3 depthwise block的理论局部RF为7x7 patch（约98x98 crop pixels），无multi-scale |
| CLIP coordinates | `tools/phase4e1.py::clip_coordinates` | patch centers `[B,576,2]` | deterministic | 使用每个14x14 patch中心，坐标归一到resize后完整图 |
| SAM image embedding | `model/GLaMM.py::get_grounding_encoder_embs`；`tools/phase4f.py::load_sam_runtime` | raw/S64 `[B,256,64,64]` | SAM image encoder来自C1；prompt/mask runtime沿用冻结P1/Phase4F权重 | ResizeLongestSide1024 + right/bottom pad；后续映射到原图 |
| support | `model/sam_forensic_rectifier.py::GeometryAwareSAMRectifier.semantic_support` | `[B,4096]` bool | 无参数 | 以patch-center坐标的min/max矩形定义；比真实336 crop边界每侧保守半个patch |
| rectifier | 同文件 `GeometryAwareSAMRectifier.forward`；内部`CrossAttentiveSemanticRectification` | evidence `[B,256,24,24]`; attention `[B,8,4096,576]`; residual/S_rect `[B,256,64,64]` | new R1中329,985参数经过C1-specific训练 | F24先flatten；geometry-aware cross attention；support外不能得到correction |
| utility | `scripts/phase4g1q_conditional_utility.py::utility_forward` | aligned forensic/SAM grids，`U_F`最终映射到`[B,1,64,64]` | new R1中371,803参数经过C1-specific训练；source evidential heads frozen | `_map_forensic`把24x24证据映射到64x64；support外vacuous/zero |
| gate/adaptation | `scripts/phase4ha_utility_gated_rectification.py::gate_to_sam_grid/gated_embedding` | `S_adapt=[B,256,64,64]` | 无新增模块 | `gate *= support`；support外严格回到S_base |
| decoder | `model/sam_forensic_rectifier.py::FrozenP1SAMPath.forward` | q `[T,256]`, embedding `[B,256,64,64]` -> mask `[T,1,256,256]` | frozen prompt encoder/mask decoder | bilinear/inverse geometry到original，logit>0 |

### C.3 命名澄清

“F24”不是current实现中唯一明确的tensor名字：

- Phase3C.1 spatial cache中的 `features` / Phase4F store读出的raw tensor是 **`H_clip: [B,1024,24,24]`**。
- `CLIPSpatialArm(..., return_features=True)`输出的真正 forensic representation是 **`F_forensic: [B,256,24,24]`**。
- Utility接口历史字段 `batch['F24']` 在Phase4G/4H/6E在线路径中装入的是 **`F_forensic`**，不是raw 1024-channel CLIP grid。
- `z_F24`是adapter dense head的 **`[B,1,24,24]` logits/evidence**。

后续报告应使用 `H_clip[-2]`、`F_forensic`、`z_forensic` 三个名字，避免把它们统称为F24。

### C.4 C1与RINE的边界

C1内部的RINE-Q2通过FRC token进入LLM并影响生成的`[SEG]` query；它不是new R1的dense evidence。new R1的dense route仍来自GLaMM CLIP的单层patch tokens。RINE的24层CLS cache/aggregation不能直接供当前rectifier使用。

## D. Bottleneck audit

### D.1 Coverage/FOV

代码确认 `ResizeShortest336 -> CenterCrop336`。对于非方形图，crop外区域完全没有CLIP patch token；adapter、rectifier与utility都无法恢复这些像素的图像证据。support外的最终路径不是“未知值插值”，而是通过 `gate * support` 精确退回`S_base`。

本轮对既有 internal-validation Fake 1106 的 mask/geometry cache做了metadata-only统计；没有执行模型forward：

| 统计 | exact 336 crop | current patch-center hull |
|---|---:|---:|
| mean GT coverage | 0.960715 | 0.938424 |
| median | 1.000000 | 1.000000 |
| P5 | 0.675811 | 0.536141 |
| minimum | 0 | 0 |

精确crop coverage分桶：`1.0: 1018`，`[0.9,1): 16`，`[0.5,0.9): 34`，`(0,0.5): 23`，`0: 15`。212张发生非方形crop；aspect ratio>=1.2的210张平均coverage仅0.794764。当前patch-center hull相对真实crop平均再损失0.022290 GT coverage；316/1106张受到该保守边界影响，loss的P95为0.125189。

因此存在两个不同问题：

1. **acquisition loss**：真实center crop没有观察crop外图像；
2. **support implementation loss**：`semantic_support()`使用首末patch中心而不是patch footprint边界，使crop内边缘再被判为unsupported。

第二项更像可审计的geometry实现选择，并不需要multi-tile才能修复；但在验证其与历史权重兼容前不应直接修改。

当前不能回答coverage是否与`delta IoU`正相关，也不能回答new R1的FN是否集中在unsupported region，因为Phase6E.2 selector只保存aggregate validation metric，没有保存selected epoch的逐样本validation prediction/FN map。Official1000记录不能用于架构选择，本报告不拿它替代validation诊断。

### D.2 Resolution

24x24 token grid对应336 crop上的14x14 patch。adapter输出仍是24x24；rectifier通过4096x576 cross attention将其作用到SAM 64x64 grid。虽然attention可以产生64x64 residual，但无法创造patch内已丢失的细粒度取证信号。细小artifact、窄边缘和高频纹理可能因此受限。历史linear probe/adapter结果证明24x24有用，却没有做分辨率对照，所以“resolution是主要瓶颈”仍是未验证假设。

### D.3 Layer representation

当前选择hidden `[-2]`来自历史GLaMM/CLIP vision-tower默认，而不是一次early/mid/late dense spatial layer search。仓库未发现保存early/mid/late patch grids的可复用dense cache；现有Phase3C.1 cache只保存selected `[-2]`，Phase6B RINE cache保存/提取的是多层CLS而非完整patch grid。

ViT-L/14各transformer block的patch token count与hidden width保持576/1024，因此不同层在token lattice、CLIP crop、position embedding和坐标系统上天然对齐，不需要空间resize；需要控制的是各层feature normalization与projection。

Multi-level的最小充分实现不是RINE-TIE复制，而是共享容量的dense probe：每层先用相同、低参数的1024->256 projection/normalization，使用固定或极少参数的层加权求和，再进入完全相同的3-block adapter。为防止容量混淆，单层arms必须拥有相同projection/adapter参数量，多层arm新增参数最好只是一组层标量或channel-shared convex weights；另设“重复同一层三次”的matched-capacity/null control。

### D.4 Adapter capacity

current new R1确实使用Phase4C-A selected adapter，且Phase6E.2中始终 `eval().requires_grad_(False)`；训练manifest的source hash before/after一致，optimizer只含utility与rectifier。因此new R1训练从未反向更新adapter。

3个local blocks的理论RF是7x7 patches，但它没有dilation、multi-scale branch或跨全图context。它可能限制细粒度/跨尺度artifact变换；然而现有证据只能说明adapter仍有failure，不能把failure归因于capacity。简单`3->6 blocks`同时改变参数量、RF和优化难度，不是有效归因实验。

### D.5 Geometry/resampling

- CLIP和SAM使用不同预处理坐标系，但都通过original-normalized coordinates对齐。
- CLIP patch centers用于cross-attention locality与support；SAM token centers基于ResizeLongestSide1024 geometry。
- `F_forensic`本身不先插值到64x64供Rectifier使用；Rectifier直接对24x24 evidence做geometry-aware attention。
- Utility内部另用`_map_forensic`对24x24 feature/evidence映射到64x64。
- 输出低分辨率mask经bilinear resize和inverse SAM geometry返回原图。
- 不可恢复损失首先发生在CLIP center crop，其次是14x14 patch化；之后的插值只能重采样现有信息。

## E. Recommended experiments

实验树保持最小化，任何正式训练均需下一阶段单独授权。

### E0 — Validation coverage attribution diagnostic（首选，read-only）

- **Control/Candidate**：同一样本的 frozen C1 与 selected C1+new R1。
- **Only difference**：是否启用selected new R1；不改权重。
- **Data**：internal validation Fake 1106；禁止test/OOD/Official1000参与判断。
- **操作**：保存逐样本original-space C1/new-R1 logits或至少binary masks；从现有geometry构造两张mask：真实CLIP crop footprint与当前patch-center hull support。
- **输出**：GT coverage；C1/new-R1 IoU与delta；C1/new-R1 FN inside/outside；按指定coverage bins报告；Spearman与分层bootstrap；另按aspect ratio分层。
- **Go**：低coverage组的delta显著更低，或new-R1 FN在outside-support显著富集，并且该关系在控制GT面积/aspect ratio后仍存在。
- **Stop**：coverage与delta/FN无稳定关系；此时不启动multi-tile，转E2 layer probe。

### E0b — Support-boundary semantics diagnostic（与E0同批、无训练）

- 比较current center-hull support、patch-footprint support和exact crop footprint；只重新统计support/FN，不改变模型输出。
- 若性能归因主要来自半patch边界，而非crop外区域，先设计geometry-only controlled correction；不得把它写成multi-tile收益。

### E1 — Controlled full-FOV acquisition（仅E0支持后）

- **A0**：current single center crop -> frozen current adapter -> frozen current new R1。
- **A1 diagnostic**：overlapping tiles覆盖完整resize后图像，每tile仍336、同一frozen CLIP和同一frozen adapter；将token连同各自original-normalized坐标送入Rectifier。重叠token先保留，不做不可解释的feature平均。
- **Only difference**：evidence acquisition/FOV与token数量。
- **第一步frozen replay**：C1、CLIP、adapter、rectifier、utility、SAM全部冻结，观察是否存在zero-shot gain与内存/时延代价。注意current rectifier/utility是在576-token分布上训练，zero-shot失败不能直接否定full-FOV。
- **若需训练**：最小方案只重训new R1的rectifier+utility；adapter可先冻结，因为每tile的输入分布与训练时336 crop一致。只有frozen-adapter方案显示明显representation不足时，才允许重新训练adapter。
- **Data/selector**：同Phase6E.2 internal train；仅internal validation G0 selector；loss/epoch/LR不变。
- **Go**：相对A0的paired mean IoU/F1 CI下界>0，global指标不降，且低coverage/高aspect组获益更大；同时高coverage=1组不得显著退化。
- **Stop**：仅增加FP/时延、低coverage组无特异改善，或增益只来自改变support边界。

### E2 — Matched dense layer probe（E0不支持coverage，或E1后仍有support内failure）

- **A0**：current hidden[-2]。
- **A1/A2**：预注册一个middle层和一个earlier层；每个均使用相同1024->256 projection + 同构3-block adapter，参数量/训练recipe相同。
- **A3**：三层各做共享或严格匹配的projection后，以少量convex layer weights融合，再进同一adapter。
- **Capacity control**：三路都输入hidden[-2]或固定平均的duplicate-layer null，匹配A3的计算/参数接口。
- **Frozen**：CLIP、C1主体、SAM；第一阶段只训练matched dense probe/adapter，不动rectifier/utility，以直接测representation。
- **Data/selector/eval**：复用Phase4C-A internal train/validation与mask loss、selector；不得使用current test/OOD。
- **Go**：A3超过最佳单层且超过duplicate-layer control，paired CI下界>0；否则不宣称multi-level information有效。
- **下一步**：只有probe成立后，才把selected evidence source接入同构new-R1 retraining。

### E3 — Adapter capacity matched ablation（最后）

- 前提：full-FOV后仍有大量support内FN；multi-level probe无充分解释；feature诊断显示current `F_forensic`分辨能力不足。
- 保持输入层/FOV、输出256x24x24、loss、参数预算和训练数据一致。
- 比较current 3xDW3x3、matched-parameter dilated/local-context variant，以及matched-parameter multi-scale variant；用通道宽度调整匹配总参数，而非直接3->6。
- selector仍是internal validation dense mask metric；接入new R1前先证明standalone evidence improvement。

## F. Final recommendation

**Primary verdict：`INSUFFICIENT_EVIDENCE_NEED_DIAGNOSTIC`**

**Route recommendation：`PROCEED_COVERAGE_FIRST`**

理由：代码已经证明coverage存在不可恢复的信息缺失，metadata-only统计也证明该问题对一个非小的validation子集很严重；同时current support还存在patch-center hull额外保守约2.23个百分点平均GT coverage的问题。但缺少C1/new-R1逐样本validation输出，尚不能证明这些coverage损失正在主导new R1的实际FN或限制其增益。因此下一步必须先做E0/E0b，不应直接训练multi-tile，也不应先改adapter。

Multi-level CLIP排第二是合理的：历史没有dense layer comparison，且相同24x24 lattice使matched experiment成本可控。Adapter capacity排第三同样合理：已有adapter相对raw/projection-only的正向因果证据，而coverage和单层选择是更早发生的信息瓶颈。

本轮到此停止；未启动任何训练、模型评测或外部数据访问。
