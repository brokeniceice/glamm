# Phase 6H.0 — Phase6E.2 原始 C1+new R1 基线审计

日期：2026-09-24。状态：**计算图与 checkpoint 审计通过；6H.1 尚未训练**。本步只读取代码、配置与已有产物，没有启动 GPU 训练或访问新的 Official1000 样本。后续若进入训练，按本任务指定使用物理 GPU 1 和 2，并保持 A0、A1 的历史训练顺序与各自独立输出。

## 1. 精确起点和结果锚点

| 项目 | 经核对的原始 Phase6E.2 main 值 |
|---|---|
| C1 config | [`configs/phase6d3_c1_rine_conditioned_p1.yaml`](../configs/phase6d3_c1_rine_conditioned_p1.yaml)，C1 checkpoint epoch 5 / step 2500 |
| C1 checkpoint | [`checkpoints/phase6d3_c1/best/checkpoint/mp_rank_00_model_states.pt`](../checkpoints/phase6d3_c1/best/checkpoint/mp_rank_00_model_states.pt)，SHA256 `85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff`；本次重新计算 SHA256 一致 |
| R1 数据、取证与 SAM 基础配置 | [`configs/phase4f_language_preserving_rectification.yaml`](../configs/phase4f_language_preserving_rectification.yaml)；Phase6E.2 实际训练超参数以 [`phase6e2_c1_specific_r1_train.py`](../scripts/phase6e2_c1_specific_r1_train.py) 引入的 [`phase4hd_rectifier_unfreeze_control.py`](../scripts/phase4hd_rectifier_unfreeze_control.py) 常量及 [`initialization_provenance.json`](../outputs/phase6e2_c1_specific_r1/initialization_provenance.json) 为准。**不能把 Phase4F YAML 内的 `5e-5`、cosine scheduler 误当作 Phase6E.2 训练参数。** |
| R1 初始 Utility | Phase4H-C A2 epoch 3：[`outputs/phase4hc/a2/selected_checkpoint.pt`](../outputs/phase4hc/a2/selected_checkpoint.pt)，SHA256 `bb13999348c6cff336c037feb8d712bfa2b4494cb6cac0aa3070907ecf02e9fd` |
| R1 初始 Rectifier | Phase4F forensic_rect epoch 9：`/data/yz/groundingLMM_official/checkpoints/phase4f_language_preserving_rectification/forensic_rect/epoch_9.pt`，SHA256 `875165ab1dfc40787a850f34826a0d50f6e0984d719f85bd0aac2e194f909095`；本次重新计算 SHA256 一致 |
| 不得用作 6H.1 初始化 | Phase4H-D old R1 selected checkpoint SHA256 `9b38ef1e62c86c74287895e681ec8ebb96b724e3bae52622d52b61bca7ddf5a5`；Phase6E.2 main 明确记录 `loaded_as_initialization=false` |
| Phase6E.2 最终 selector | [`selector.json`](../outputs/phase6e2_c1_specific_r1/selector.json)：internal-validation canonical G0 mean FG IoU，平分取较早 epoch，选 **epoch 8**；所记 DEV mean IoU **0.2027321013** |
| 最终 selected checkpoint | [`selected_checkpoint.pt`](../outputs/phase6e2_c1_specific_r1/selected_checkpoint.pt)，SHA256 `c067eb2240af9d5fd61fc5dc367338ed585fcc6a5d5d0f982df0fa886302e603`；本次重新计算 SHA256 一致。checkpoint 自报 epoch 8、optimizer updates 8840、C1 SHA256 与预期一致；`utility_state` 与 `rectifier_state` 可严格载入原始 `CSCULF` 和 `GeometryAwareSAMRectifier`，所有 key 匹配。 |
| 正式结果 | [`phase6e2_c1_specific_r1.json`](../outputs/phase6e2_c1_specific_r1/phase6e2_c1_specific_r1.json)：Official1000 canonical G0 中 C1+old R1 mean/global IoU **0.300479/0.283588**、mean/global F1 **0.415358/0.441867**；C1+new R1 对应 **0.319867/0.343538**、**0.440385/0.511393**；SEG trigger **0.992**。这里只用于识别历史结果，不作本轮新评估。 |

**内部验证锚点有两个不同来源。** 原始 6E.2 epoch 8 selector/checkpoint 记录 mean FG IoU `0.2027321013`；后续 [6F.1 归因](phase6f1_c1_newr1_coverage_attribution.md) 和 [6F.3 冻结回放](phase6f3_full_fov_frozen_replay.md) 对同名 new R1 报告 `0.202488`。两者相差约 `0.000244`，不能宣称是同一次精确重复。**6H.1 A0 的训练复现检查应首先对原 6E.2 selector 协议与 `0.2027321013`，同时单列后续冻结回放口径；若未查清差异，不把 `0.202488` 写成原 selector 值。**

## 2. C1 query/cache 和样本资格

[`phase6e2_c1_g0_cache.py`](../scripts/phase6e2_c1_g0_cache.py) 使用冻结 C1（上表 SHA256）在统一 canonical G0 prompt 下生成真实 `[SEG]` 投影 query。训练期从 `/data/yz/groundingLMM_official/cache/phase6e2_c1_specific_r1/{train,val}/shard_*.pt` 按样本顺序加载 `q_seg`、`valid`、`seg_count`；[`cache/status.json`](../outputs/phase6e2_c1_specific_r1/cache/status.json) 为 `COMPLETE`：train 8836 张，其中恰一 `[SEG]` 8741；validation 1106 张，其中恰一 `[SEG]` 1090。其余分别 95/16 张无 `[SEG]`，validation 按原协议记零。原 [`c1_query_dependency_audit.json`](../outputs/phase6e2_c1_specific_r1/c1_query_dependency_audit.json) 确认空间样本顺序精确一致，历史 P1 query 不被消费。

训练中 [`c1_language_batch`](../scripts/phase6e2_c1_specific_r1_train.py) 从 cache 读取 `q_seg`，用冻结 SAM 产生其 `z_L`；这不是从旧 P1 `q_seg` 继承，也不更新 C1/LLM。Official1000 的历史 evaluator 通过同一冻结 C1 checkpoint 生成 query；它仅作为历史确认程序定位，不在 6H.0 执行。[`phase6e2_official_finalize.py`](../scripts/phase6e2_official_finalize.py)

## 3. 原始 dense evidence 和模块所有权

| 模块/数据 | 原始计算图与训练状态 | 核对依据 |
|---|---|
| CLIP dense source | Phase3C.1 缓存的 ViT-L/14@336 `hidden[-2] = hidden_states[23]`，即 zero-based **block 22**；去 CLS 后 576 patch tokens，`[B,1024,24,24]`。CLIP 冻结，无 block11/17 融合。 | [4C-A 源审计](../scripts/phase4c_a_prepare.py)、[6G.0 层语义审计](phase6g0_multilevel_dense_clip_audit.md)、[Phase4F store](../tools/phase4f.py) |
| Forensic Adapter | Phase4C-A 选中的 `1024→256` 投影 + 3 个 `LocalForensicBlock`；`F_forensic` 形状 `[B,256,24,24]`。本阶段只读取其冻结特征，**不训练 Adapter 或 dense head**。checkpoint SHA256 `725dd44e0c867360e7a963eb13d8312187fff185849bd52de6b78d4b27a078f7`，本次重新计算一致。 | [Adapter 模型](../model/clip_forensic_adapter.py)、[Phase4F evidence loader](../tools/phase4f.py)、[基础配置](../configs/phase4f_language_preserving_rectification.yaml) |
| Rectifier | 原始完整 `GeometryAwareSAMRectifier`（归一化、Q/K/V cross attention、`out_proj`、`projection`、`gamma`、hard spatial support），从 Phase4F epoch 9 初始化后**整个模块训练**，329,985 trainable parameters。 | [Rectifier 模型](../model/sam_forensic_rectifier.py)、[6E.2 trainer](../scripts/phase6e2_c1_specific_r1_train.py) |
| Utility | 原始 `CSCULF`，其 language/forensic evidential source heads 冻结，language/forensic context、交互及 `U_F` head 按原方案训练；371,803 trainable parameters。Utility 读取 `S64,q_seg,z_L,F24,z_F24,geometry`，**不直接读取 `C`**。 | [Utility loader](../scripts/phase4hc_direct_utility_arms.py)、[Utility forward](../scripts/phase4g1q_conditional_utility.py)、[6E.2 初始化记录](../outputs/phase6e2_c1_specific_r1/initialization_provenance.json) |
| SAM | `S64` 来自既有冻结空间 cache；P1 prompt encoder + mask decoder 载入冻结 `p1_sam_runtime.pt`，本轮仅做原始 mask 解码，不训练。 | [SAM runtime](../tools/phase4f.py)、[FrozenP1SAMPath](../model/sam_forensic_rectifier.py) |
| C1 / LLM / 分类 / 文本 | C1、LoRA、H2/classification head、FRC projector、RINE、text generation 均冻结；R1 checkpoint 仅含 Utility/Rectifier state，没有上游分类或生成权重。因 R1 位于 query 形成之后，A0/A1 的训练不能改变分类或文本生成路径。 | [C1 query cache](../scripts/phase6e2_c1_g0_cache.py)、[6E.2 checkpoint](../outputs/phase6e2_c1_specific_r1/selected_checkpoint.pt) |

原始有效训练 recipe 来自 `hd.config_contract()`：seed **3407**、10 epochs、batch **8**、AdamW lr **1e-4**/weight decay **1e-4**、gradient clip **1.0**、**无 scheduler**、无 gradient accumulation（每个有效 batch 一次 `optimizer.step()`）、固定 `Random(seed+1009×epoch)` 样本顺序、cyclic cross negative、seed 3407 的固定 576 patch shuffle。loss 为 `L_seg + L_relative + L_ranking`；`L_seg=2×BCEWithLogits+0.5×soft Dice`，ranking 中 cross/shuffle 各占 0.5。主 selector 只看 DEV canonical G0 final-mask mean IoU，不看 auxiliary 指标、Phrase/TF 或 Official1000。[配置契约](../scripts/phase4hd_rectifier_unfreeze_control.py)、[trainer 原损失与 step](../scripts/phase6e2_c1_specific_r1_train.py)、[mask loss](../tools/phase4f.py)

## 4. 真正注入的 correction `C` 在哪里

原始代码没有独立命名为 `C` 的训练局部变量。对应关系为：

```text
s64                          = Phase4FStore 中的 SAM image embedding [B,256,64,64]
value = rectifier(s64, F_forensic, sam_coords, clip_coords, valid)
p4f = value["image_embeddings"] = Srect                         [B,256,64,64]
C   = p4f.float() - s64.float()  (语义上即 Rectifier residual) [B,256,64,64]
u   = utility_forward(model, batch)["U"]                        [B,1,64,64]
gate = gate_to_sam_grid(u, sam_coords) * support
S_adapt = gated_embedding(s64, p4f, gate)
        = S64 + gate * (Srect - S64)
final_mask = frozen_SAM(q_seg, S_adapt)
```

Rectifier 的原始 `residual` 在 [`CrossAttentiveSemanticRectification.forward`](../model/tf_fdg.py) 中经 `gamma` 和 hard support 后加到 `semantic`；[`GeometryAwareSAMRectifier.forward`](../model/sam_forensic_rectifier.py) 将 rectified grid 返回为 `image_embeddings`，同时返回 `residual`/`support`。训练 [`phase4f_batch`](../scripts/phase6e2_c1_specific_r1_train.py) 将 `value["image_embeddings"]` 命名为 `p4f`；[`gated_embedding`](../scripts/phase4ha_utility_gated_rectification.py) 用 `(p4f−s64)` 实际注入 SAM。**6H.1 auxiliary head 应从这一同一张 `p4f−s64` 计算图读取 `C`，保持对 Rectifier 的梯度；不可用 Srect、R2、F24 或 Utility 输出代替。** Utility 的 `U` 是在读取 C 之前由源上下文产生的，因此额外监督只能通过 Rectifier 改变 C，不改变原 Utility 输入定义。

## 5. G.9–G.20 后续结构差异及 6H-A 隔离

Phase6E.2 main 的导入链只使用原始 `scripts/phase4*` 与 `tools/phase4f.py`、`model/clip_forensic_adapter.py`、`model/sam_forensic_rectifier.py`、`model/csculf.py`；这些文件在当前 `git status` 中没有本地修改。原始 selected checkpoint 的 15 个 Rectifier state keys 只对应原有 norm、Q/K/V、`out_proj`、`projection`、`gamma`，在 CPU 上对原始模型 `strict=True` 载入成功。因此存在于单独 G 阶段脚本/产物中的差异**尚未进入 6E.2 原图**。

| G 阶段 | 后续差异 | 6H-A 约束 |
|---|---|---|
| G.9 | Rectifier 节点转换诊断，无正式原图替换 | 只用 6E.2 原始 `GeometryAwareSAMRectifier` |
| G.10 | `out_proj` / `projection` bypass | 两层均按原始 checkpoint 保留 |
| G.11–G.12 | `R2` 侧路 `side_projection`、main+side 联训 | 不实例化 side path；原始 15-key state 无 side 参数 |
| G.13 | side-path 上重新训练的 Utility | 只加载 Phase4H-C A2 初始化和原 CSCULF |
| G.14–G.15 | single Translator、零 residual、折叠 Translator | 保留原双仿射 `out_proj→projection` |
| G.16–G.17 | Type-II 直接看 C / Type-III late residual Utility | 保留原始来源上下文 `CSC U_F`，不输入 C |
| G.18 | 结构化上下文设计审计 | 不采用其提案 |
| G.19 | C/Srect 结构化 `ForensicContext` | 不加载其模型或 checkpoint |
| G.20 | Evidence–Correction 局部 cross-attention | 不加载该算子或 checkpoint |

G.1–G.8B 的 block17、block11+17、spatial stem、多尺度 Adapter 等也全部不在原图中。6H.1 A0/A1 必须直接复用 6E.2 main 的 source checkpoint、构造器和数据 cache，**不可按“最新 G 代码”自动替换**。这是一项代码和 checkpoint 级确认；A0 实际数值复现、逐 batch 权重/数据不变性与 A0/A1 trainable manifest 尚需在 6H.1 前置检查中再验证。

## 6. 审计关口结论与下一步边界

**6H.0 结论：原始 Phase6E.2 计算图可从代码、config、初始化来源及 epoch 8 selected checkpoint 确认；允许准备 6H.1 的匹配复现。** 本报告不声明 6H.1 已开始或通过。下一个独立步骤必须先保存 A0/A1 trainable 参数 manifest，确认 A1 只增加 `Conv2d(256,1,1)` 的 weight/bias，并用 A0 在 GPU 1 上复现 6E.2；随后才可在 GPU 2 上运行 A1。训练及验证不得覆盖任何既有 6E/6F/6G 产物。是否进入 6H.2 取决于 6H.1 的预注册 paired final-mask gate；在此之前不运行 Official1000。
