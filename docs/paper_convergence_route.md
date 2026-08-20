# Explainable Image Forensics with Multimodal Large Language Models

## Canonical Research Plan

**状态：CANONICAL（自 2026-08-20 起）**

本文档是后续论文收束、实验设计与路线授权的唯一 canonical research plan。它不覆盖、不重写 Phase 2/3 的历史 artifact 或历史 gate。若本文档、自然语言任务说明与已完成的机器可读 artifact 存在冲突，以机器可读 artifact 为事实来源，并在本文档中记录冲突与修正；任何新结果必须通过独立 artifact 和预注册 gate 更新路线，不能静默改写本文件中的历史事实。

当前只授权 **Phase 3D.0 — Evidence-Aware Reward / Rollout Preflight**。本授权不包含训练或模型更新。

## 1. 研究问题与论文主线

本论文研究一个可统一完成以下任务的可解释图像取证 MLLM：

1. Real/Fake authenticity detection；
2. forensic explanation；
3. artifact / target-region phrase generation；
4. pixel-level evidence localization。

核心科学问题是：现有 forensic MLLM 即使具备较强的语义推理与 segmentation 能力，在 autonomous generation 下，语言取证结论仍可能与像素级 forensic evidence 明显脱节。本文将该现象称为：

> **Autonomous–Oracle Forensic Grounding Gap**

后续方法与分析必须围绕 `language reasoning ↔ forensic evidence ↔ spatial grounding` 的一致性展开。论文不能被描述成彼此无关的分类模块与分割模块的简单拼接。

## 2. 数据与 mask provenance audit

### 2.1 Annotation provenance

当前定位标注来源统一表述为 **LEGION / SynthScars official segmentation annotations**，具体为 SynthScars 官方发布包中的：

- `train/annotations/train.json`，审计 SHA256 `94100c0b381f4b24745d75e5cba65872686ccf9e37af8e9fd669349e3509cd27`，11,236 条 annotation；
- `test/annotations/test.json`，审计 SHA256 `3bb55380f9ad75d76c26ad9a18035296be4ab2a7255b5e2fa11b131a2417e7e4`，1,000 条 annotation。

原始 JSON 是 annotation-centric schema：每个外层 annotation id 对应 `img_file_name`、`caption` 与非空 `refs`；每个 ref 使用 `sentence`、`explanation`、`bbox` 和 **`refs[].segmentation`**。当前发布数据中的 `segmentation` 是一组 polygon；adapter 将其保留为 manifest 中逐 ref 的 `polygons`，同时保留 annotation id、ref id、原图文件名与原图尺寸。来源与冻结审计见 [SynthScars adapter](../dataset/forensics/synthscars.py) 和 `outputs/data_audits/synthscars_image_grouped_v1/summary.json`。

本文不得将当前标注写为 GroundingDINO、SAM 或 LLM 生成的 pseudo mask，也不得写成原始 LISA 数据生成路线；这些词只有在显式讨论已废弃历史方案时才能出现。Phase 3D 及后续实验不得重新生成 mask。

### 2.2 Our model supervision representation

`SynthScarsAdapter` 先按 image identity 聚合官方 annotation，保留所有 source annotation/ref。`UnifiedForensicsDataset._fake_union_mask` 再执行以下固定 target construction：

1. 按 ref 读取官方 polygon；
2. 若同一视觉身份存在尺寸/编码变体，将 polygon 从 source image size 缩放到 canonical image size；
3. 使用 COCO polygon rasterization 生成逐 ref binary mask；
4. 对该图全部 ref masks 做布尔 OR；
5. 产生单图、单 `[SEG]` 的非空监督 mask。

其准确名称是：

> **derived union target from official SynthScars annotations**

这是本项目的 supervision representation，不是新标注，也不是 pseudo mask。当前 construction 保持冻结；除非未来另行预注册实验，不得改变。

### 2.3 Evaluation unit 与 aggregation

必须始终分开报告：

- **A. annotation provenance**：official LEGION/SynthScars annotation；
- **B. model supervision representation**：本项目的 per-image all-ref union target；
- **C. evaluation aggregation**：如 foreground IoU、foreground F1、fg/bg mIoU，以及 global/per-image aggregation。

本项目的 derived union target 尚未被验证为与 LEGION paper 的 phrase-level multi-mask evaluation unit 完全相同。不同 metric、内容类别、split、target representation 或 aggregation 的数字不得直接相减。尤其 Phase 2A 的 foreground IoU `0.139568` 不能与 LEGION 的 fg/bg mIoU `0.5462` 直接比较；LEGION Table 2 的 Object 子集也不能与本项目 official-1000 overall 结果直接比较。详见 [Phase 2B.1 类别对齐修正](phase2b1_legion_category_parity_correction.md)。

## 3. 当前基础系统与实现边界

当前系统以 forensic-adapted GLaMM 为基础：

```text
Image
  ↓
Vision / multimodal representation
  ↓
LLM
  ├─ [CLS] → Real/Fake classification
  ├─ explanation
  ├─ Target regions phrase
  └─ [SEG]
       ↓
grounding representation
       ↓
      SAM
       ↓
      mask
```

`[CLS]` 是 Ours 的分类实现，不是外部 baseline 必须复制的结构。外部 baseline 只需为统一 benchmark 做最小且自然的适配。

## 4. 已冻结的观察、诊断与路线 gate

### 4.1 Phase 2D / 2D.1：Autonomous–Oracle gap

已完成结果支持以下 observation：canonical autonomous G0 localization 显著低于 teacher-forced/oracle localization；`[SEG]` 不生成不是主要原因；在冻结 downstream 的条件下，TF context 形成的 predictor representation 能得到明显更好的 mask。因此 language-conditioned `[SEG]` representation 是 localization 差异的重要 mediator，而 downstream segmentation capacity 并非完全不存在。

这些 diagnostic intervention 支持 mechanism interpretation，但不构成严格因果识别。历史事实与边界见 [Phase 2D](phase2d_grounding_gap.md) 和 [Phase 2D.1](phase2d1_trace_replay.md)。

### 4.2 Stage-I / Phase 3A–3A.1：Phrase-Aligned Forensic SFT

历史 fake target 为：

```text
[FAKE] explanation [SEG]
```

P1 fake target 为：

```text
[FAKE] explanation
Target regions: <authoritative phrase> [SEG]
```

Stage-I 的目的不是机械增加一句文本，而是在 SFT 阶段显式建立 forensic target semantics 与 localization representation 的对应关系。正式工作名为 **Phrase-Aligned Forensic SFT**。

Phase 3A.1 matched C0 的冻结结果为：C0 G0 `0.179786`，P1 G0 `0.229544`，P1−C0 `+0.049758`，bootstrap 95% CI `[+0.034534,+0.064341]`，gate 为 `PHRASE_EFFECT_STRONGLY_CONFIRMED`。该 matched protocol 支持 P1 对 deployable G0 localization 的稳定正效应；但原始 P1 step-0 不可用，不能把 deterministic reconstruction 当作 byte identity，也不能把整个历史增益表述为严格因果效应。P1 是后续 language-policy experiment 的 primary checkpoint。详见 [Phase 3A.1](phase3a1_paired_control.md)。

### 4.3 Phase 3B：Rejected alternative

Phase 3B 测试 fixed-policy generated-context replay：在 P1 自身生成的 context 下，再以 GT mask 施加 auxiliary segmentation supervision。机器可读最终 gate 为：

> `GENERATED_REPLAY_HARMFUL`

因此停止 replay fraction/lambda tuning、fixed-policy generated replay、通过额外 mask loss 强迫模型适应错误自产文本，以及未经新证据授权的 scheduled-sampling 扩展。该结果可作为 rejected / ineffective alternative，说明 autonomous grounding problem 不能简单通过“让 segmentation 适应自产错误 context”解决。完整边界见 [Phase 3B](phase3b_generated_replay.md)。

### 4.4 Phase 3C.0：Residual bottleneck diagnosis

在冻结 P1 下比较：A=canonical G0；B=保留 generated explanation、仅以 authoritative phrase 替换 generated Target regions；C=authoritative phrase-only；D=TF-PHRASE。

- B−A：n=1,076，mean FG IoU `+0.108989`，bootstrap 95% CI `[+0.095485,+0.122814]`；
- gate：`GATE_LANGUAGE_PHRASE_ERROR_SUPPORTED`；
- C−B：mean `−0.012476`，95% CI `[-0.021230,-0.003543]`，因此不支持 `generated explanation contamination`；
- 476 个 persistent failures 支持当时的诊断授权 `GATE_DOWNSTREAM_SPATIAL_PREFLIGHT_AUTHORIZED`。

B 是 `CONTROLLED_MULTI_FACTOR_DIAGNOSTIC`：phrase replacement 还可能改变 tokenization、sequence length、`[SEG]` position 与 hidden trajectory。`+0.108989` 不能写成严格的 phrase-semantic causal effect。当前不优先 explanation deletion、short-CoT optimization 或 context-decoupling training。详见 [Phase 3C.0](phase3c0_residual_localization_diagnosis.md)。

### 4.5 Phase 3C.1：Frozen spatial preflight

在 frozen SAM、CLIP、NPR、SRM、FOCAL spatial features 上分别只训练 `Conv2d(C,1,1)` linear probe。机器可读最终 gate 为：

> `GATE_SPATIAL_PREFLIGHT_NOT_SUPPORTED`

best existing GLaMM probe 是 CLIP；NPR/SRM/FOCAL 未显示 complementary superiority；五个 source 均未通过完整自身 negative-control gate。因此不授权 frozen NPR/SRM/FOCAL spatial fusion，也不授权为了论文结构直接添加 forensic branch。这不证明空间问题不存在，只说明当前测试的 frozen spatial representations 没有提供足够证据支持直接 forensic spatial fusion。详见 [Phase 3C.1](phase3c1_frozen_spatial_evidence_probe.md)。

### 4.6 Phase 3C.2：classification-only 旁证，不并入定位创新

已完成 artifact 还包含 P3，即在冻结 P1 上复用 NPR/SRM 的 classification-only residual late fusion。P3 的 canonical G0 与 P1 逐样本相同是架构约束；其变化只发生在分类及 classification-gated system metric，不能解释为 localization gain，也不能证明 NPR/SRM 具有像素级定位能力。该结果不改变 Phase 3C.1 的 spatial gate，也不授权 frozen spatial fusion。详见 [Phase 3C.2](phase3c2_p3_robustness.md)。

## 5. 收束后的核心方法

论文第一核心方法线为：

> **Progressive Evidence-Aligned Forensic Post-training**

它由两阶段组成：

- **Stage-I — Phrase-Aligned Forensic SFT**：在 supervised target 中建立 forensic target semantics 与 localization representation 的显式对应；
- **Stage-II — Evidence-Aware Policy Optimization**：直接优化 autonomous forensic trajectory 在真实性、target phrase、pixel evidence 与输出结构上的一致性。

Stage-II 的创新不是“使用 GRPO”。GRPO、PPO 或其他 policy optimizer 只是候选 optimization backend；真正的研究问题是如何定义 evidence-aware reward，使 deployable autonomous trajectory 沿 `classification ↔ target semantics ↔ pixel evidence ↔ structure` 同步改善。

Phase 3C.0 的 Phrase Repair 恢复了显著 localization performance，构成 Stage-II 的直接 motivation：若模型能自主生成更接近 authoritative forensic target semantics 的 trajectory，则存在较大的 deployable improvement space。此处是由诊断结果支持的方法动机，不是对未来训练收益的保证。

## 6. Stage-II 候选 reward signals

候选 reward decomposition 为：

- `R_cls`：真实性判断正确性；
- `R_phrase`：Target-regions phrase 与 authoritative forensic target 的一致性；
- `R_ground`：该 trajectory 对应的 predicted mask 与 SynthScars-derived union target 的一致性；
- `R_struct`：verdict / `Target regions:` / `[SEG]` 满足输出协议；
- `R_align`：phrase correctness 与 mask evidence 的联合一致性。

free-form explanation lexical similarity 暂不进入 primary reward。解释存在大量合法 paraphrase，简单 BLEU、ROUGE 或 token overlap 容易形成错误 reward；explanation 先作为 audit target，只有在以后获得可靠的 evidence-grounded reward 时才单独预注册加入。

## 7. Phase 3D.0 — 当前唯一授权实验

**目标：Evidence-Aware Reward / Rollout Preflight。** 在真正进行 policy optimization 前，判断 P1 stochastic rollouts 是否存在足够的质量差异，以及候选 evidence-aware reward 能否可靠区分好的与坏的 forensic trajectory。

冻结边界：

- P1 completely frozen；
- no optimizer；
- no backward；
- no GRPO / PPO / DPO；
- no model update；
- 不生成或修改 mask；
- 不改变 frozen split、GT provenance、target construction、evaluator、threshold 或历史 selector。

本阶段只允许 rollout、离线 reward 计算、相关性/排序/失效模式审计及预注册 gate 所需的只读评估。具体采样参数、reward normalization、组合权重、选择 population、统计检验与 pass/fail threshold 必须在运行前形成独立预注册 artifact；不得在观察 test 结果后反向调 gate。只有 Phase 3D.0 gate 通过，才允许提出 Phase 3D.1 policy training 的新授权申请；通过本文件不能自动启动 Phase 3D.1。

## 8. 第二架构创新：假设状态与边界

`Forensic Evidence Prompt Network (FEPN)` 的原始目标是产生 global forensic evidence 与 dense forensic evidence，共同服务 classification、language reasoning、grounding 与 segmentation。当前状态必须标记为：

> `RESEARCH_HYPOTHESIS_NOT_AUTHORIZED`

Phase 3C.1 不支持直接复用 frozen NPR/FOCAL/SRM spatial feature 的简单路径。若未来重启 FEPN，必须针对当前 LEGION/SynthScars annotation-derived target 进行 task-aligned learning，目标应是 rich dense forensic representation：

```text
Image → task-aligned forensic encoder → F_f ∈ R^(H×W×D)
                                      ├─ global authenticity evidence
                                      ├─ LLM evidence conditioning
                                      └─ grounding / segmentation
```

仅实现 `image → forgery expert → binary mask → mask encoder → LLM` 的创新边界可能过弱。不得声称 NPR/FOCAL 已提供 artifact mask evidence，除非新实验直接支持。该路线目前不启动，论文也不强制必须包含“两个模块”。

## 9. External baseline protocol

外部 baseline 的公平原则是：same benchmark data、same allowed supervision、same GT provenance、same frozen splits、same evaluator、same test metrics，且 test 不参与调参。各 baseline 应尽量沿用官方 architecture、preprocessing、optimizer 与 recipe，只做完成 benchmark 所必需的 minimal adaptation。

外部 baseline 不必复制 Ours 的 `[CLS]`、LR、optimizer、LoRA 或 training steps。例如，若 LISA 只比较 localization，则保留其自身 `[SEG]` mechanism；若统一 benchmark 要求 Real/Fake classification，则增加最小且自然的 classifier，并透明报告实现与训练差异。

## 10. Internal ablation protocol

Ours 内部比较必须严格 matched。Stage-II 至少考虑：

1. P1；
2. P1 + matched extra-SFT continuation；
3. P1 + simple best-of-N / rejection-sampling SFT；
4. P1 + proposed policy optimization。

如果 backend 最终选择 GRPO，不得只比较 P1 与 P1+GRPO，否则无法排除额外训练的收益。每个 arm 至少报告 optimizer updates、image exposures、generated rollouts、effective batch、GPU hours 与 selector；数据、允许监督、初始化、评估协议及选择 population 的差异必须显式列出。

## 11. 当前停止与授权矩阵

| 路线 | 当前状态 | 重新开启条件 |
|---|---|---|
| Phase 3B replay fraction/lambda tuning | STOPPED | 新证据 + 新预注册 |
| fixed-policy generated replay / scheduled sampling extension | STOPPED | 新证据 + 新预注册 |
| frozen NPR/SRM/FOCAL spatial fusion | NOT AUTHORIZED | 新的 task-relevant spatial evidence + 新预注册 |
| explanation/context deletion training | NOT PRIORITIZED | 支持 context contamination 的新证据 |
| 为论文结构硬加 forensic branch | NOT AUTHORIZED | 独立科学假设与 gate |
| FEPN / task-aligned dense forensic encoder | RESEARCH_HYPOTHESIS_NOT_AUTHORIZED | 独立预注册与明确授权 |
| Phase 3D.0 reward/rollout preflight | AUTHORIZED | 保持本文件第 7 节冻结边界 |
| Phase 3D.1 policy training | NOT AUTHORIZED | Phase 3D.0 gate 通过 + 新明确授权 |

## 12. 当前论文 story 与贡献边界

当前完整 story 为：

1. **Observation**：Autonomous–Oracle Forensic Grounding Gap；
2. **Diagnosis**：autonomous target semantics 不可靠；简单 generated replay 无效；当前 frozen forensic spatial features 也未获支持；
3. **Method Stage-I**：Phrase-Aligned Forensic SFT；
4. **Method Stage-II**：Evidence-Aware Policy Optimization（候选，须先通过 Phase 3D.0）；
5. **Optional architecture**：Task-Aligned Dense Forensic Evidence Prompt Network，仅在以后独立授权时加入。

在 Stage-II 尚未训练和验证前，不得将其写成已完成贡献。若 Stage-I + Stage-II 最终形成完整且显著的技术方法，它们本身即可构成论文核心贡献；不得为了表面结构完整而虚构或强加第二个模块。

## 13. Canonical maintenance rules

- 新实验开始前，必须声明其对应本文件的哪一条科学问题、冻结边界与授权状态。
- 新 artifact 必须记录 checkpoint/数据/代码 provenance、selector population、metric 定义、aggregation、测试集隔离与 gate。
- 新结论只追加，不回写历史 artifact；若修正旧解释，保留旧数字并明确指出被修正的推论。
- 任何跨实验数值比较先核对 dataset、split、category、target representation、metric、aggregation 与 threshold。
- 诊断 intervention、相关性与 probe 结果保持保守表述，不自动升级为严格因果结论或架构有效性证明。
- 未列为 `AUTHORIZED` 的训练、模型更新、mask 生成、architecture fusion 或自动下一阶段均须再次获得明确授权。
