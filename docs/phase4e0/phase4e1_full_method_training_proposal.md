# Phase 4E-1 — Full Method Training Proposal（仅设计，未授权执行）

## 目标与边界

一次性完整训练 TF-FDG，并通过预注册 ablation 与 causal controls 判断三项贡献：multi-query capacity、forensic spatial use、structured TF teacher。不得从 512-step screening 推断架构成败；不得使用 internal test 或 official1000 做开发、selector、loss weight 或 checkpoint 选择；不得 threshold tuning。

## 冻结输入与数据

- canonical P1 checkpoint 固定；Detection/G0/TF prompts 显式固定为 canonical。
- train population：17,672 forensic images；target 是 LEGION/SynthScars official polygons 派生的 per-image all-ref union mask，不称 pseudo mask、不重新生成。
- validation：现有 internal validation；Fake mean FG IoU 为 selector primary。Real 样本只用于 classification/non-regression，不伪造 localization target。
- 可缓存 frozen P1 G0/TF causal states、SAM feature 与 4C-A feature；cache 必须保存 sample id、prompt hash、checkpoint hash、transform metadata、tensor shape/dtype 与全量 manifest equality。cache 是计算优化，不改变协议。

## 训练模块

| 模块 | 初始化 | Teacher warm-up | Student training |
|---|---|---:|---:|
| P1 LLM/LoRA、旧 `text_hidden_fcs` | canonical P1 | frozen | frozen |
| SAM image encoder / prompt encoder | canonical P1 | frozen | frozen |
| selected 4C-A adapter | Phase 4C-A selector | frozen | frozen |
| semantic/forensic pyramid adapters | new | trainable | trainable |
| K=4 QueryGenerator | new + P1 semantic anchor | teacher copy trainable | student copy trainable |
| 4-layer DualPathDecoder + mask head | new | trainable | initialized from teacher, trainable |
| teacher snapshot | warm-up selector | — | frozen / stop-gradient |

主实验不解冻 P1/4C-A，避免把 Stage-II gain 混成新的 language SFT 或 representation retraining。若主实验成功，解冻 adapter 只能作为单独 ablation，不能事后替换主结果。

按冻结规格估算，QueryGenerator 约 `5.25M` 参数，四层 256D dual-path decoder、两个 pyramid adapter 与 mask head 合计约 `7–13M`，总 trainable parameter 预计 `12–18M`；正式实现后必须从参数表报告精确值并核对 trainable whitelist。该规模远小于 7B LLM，但显著大于此前 one-layer Reader，支持使用多 epoch dense-decoder budget，同时无需解冻 P1。

## 完整预算

公开 PixelLM/PSALM 等 dense decoder recipe 使用多 epoch、多任务规模训练；本项目数据为 17,672 张，过去 250/512-step recipe 只覆盖极少 exposure，而 4C-A 已采用 10 epoch 才建立 representation learnability。因而冻结以下正式预算：

### Stage T — TF teacher warm-up

- `max_epochs=5`，约 `88,360` image exposures；
- effective batch `8`（每卡 micro-batch 1；GPU 数和 gradient accumulation 只为达到固定 global batch，不改变 optimizer semantics）；
- 约 `2,209` optimizer steps/epoch，合计约 `11,045` steps；
- AdamW，decoder/query LR `2e-4`，weight decay `0.05`；linear warmup 5%，cosine decay 到 `2e-5`；
- BF16，FP32 loss accumulation 与 gradient norm；global grad clip `1.0`；
- loss：`L_mask = BCE + Dice`；不使用 KD；
- checkpoint：epoch `0/1/2/3/4/5`；以 validation TF mean FG IoU 选择唯一 frozen teacher，TF selector 仅选择 teacher，不作为 deployable 方法成绩。

### Stage S — G0 student full training

- 从 selected teacher decoder 初始化，使用独立 student QueryGenerator；
- `max_epochs=10`，约 `176,720` image exposures；effective batch `8`；约 `22,090` optimizer steps；
- AdamW：new QueryGenerator/decoder LR `1e-4`，pyramid adapters `5e-5`，weight decay `0.05`；5% warmup + cosine decay；
- BF16、FP32 loss、grad clip `1.0`；无 poor-validation early stopping；只允许 NaN、gradient explosion、implementation failure、severe collapse safety stop；
- checkpoint：epoch `0–10`；primary selector 为 validation Fake canonical G0 mean FG IoU，若数值完全相同以 Detection accuracy non-regression、再以较早 epoch tie-break；
- 不用 cross/shuffle performance 做 checkpoint selector，避免 causal control 变成调参集。

预算是完整 architecture study 的一次冻结 recipe，不因中途 validation 表现改变 epochs/LR/loss weight。

## Loss 规格

student 每图先用 Hungarian cost（teacher/student query mask-logit Dice + query-relation cost；cost stop-gradient）建立 slot assignment。所有项先按 image 归一化，再组合：

```text
L_total = 1.0 L_mask
        + 0.20 L_relation
        + 0.50 L_attention
        + 0.25 L_feature
        + 0.50 L_logit
```

- `L_mask`：authoritative union mask 的 BCE + Dice；student signal `[B,1,h,w]`。
- `L_relation`：matched query 的 L2-normalized Gram `[B,K,K]` Smooth-L1；不使用 raw-hidden cosine。
- `L_attention`：最后两 decoder layer、两个 dense path 的 spatial distribution KL，temperature `T=2`，乘 `T²`。
- `L_feature`：两层 decoder feature 经独立 1×1 adapter 和 channel normalization 后 HCL/L1。
- `L_logit`：teacher/student continuous mask logit 的 temperature soft BCE；不 threshold。
- `L_language=0`：P1 frozen；只报告 language CE/Detection audit。

系数来自 dense/query KD 中“GT 为主、normalized auxiliary 为辅”的常用比例原则，并在开始前一次性冻结；不得以 validation sweep。必须先做只验证 shape、finite、gradient routing 与 geometry 的 implementation audit，该 audit 不产生 checkpoint selector 或效果 gate。

## Main experiment

只训练一次完整 TF-FDG。报告 selected validation checkpoint 的：

- canonical Detection；
- canonical G0 mean/global FG IoU；
- Phrase-only 与 TF-full 作为 oracle analysis，不参与 selector；
- paired bootstrap CI、W/T/L、area/ref-count/source/baseline-difficulty stratification；
- direct batch=1 正式 threshold-boundary 指标。

只有 full method 在 validation 上建立相对 P1 的正 G0 CI，才可请求另行授权 final held-out evaluation。否则冻结 P1，不扩大预算。

## Mandatory ablations

所有 ablation 使用同一 train population、order、optimizer family、exposure、selector 和 evaluator；各执行一次，不做连续 idea search。

| Arm | 唯一移除/替换 | 回答问题 |
|---|---|---|
| Full TF-FDG | 无 | 完整方法 |
| `-teacher KD` | 移除所有 KD，保留 GT mask | teacher behavior 是否贡献；对应 Candidate B |
| `-forensic branch` | 移除 forensic path，参数预算用同宽 semantic adapter 对齐 | 4C-A dense evidence 是否贡献；对应 Candidate A |
| `K=1` | query codebook 改为单 query，decoder 宽深不变 | single-query bottleneck 是否成立 |
| `-rectification` | forensic 只作独立 decoder cross-attention，不调节 SAM feature | hybrid 中 rectification 是否必要 |
| `mask-logit KD only` | 仅保留 `L_logit` | multi-level KD 是否优于最简单 behavior KD |

若 full 未相对 P1 建立正 CI，仍完成最关键的 `-teacher KD`、`-forensic`、`K=1` 以解释失败；其余 ablation 可按预注册资源规则停止，但不得换新架构。

## Causal controls

在 selected full checkpoint 上固定参数评测：

1. matched correct-image forensic feature；
2. deterministic cross-image derangement；
3. spatial-content shuffle，position lattice 固定；
4. zero forensic；
5. raw CLIP projection 替换 4C-A adapter feature。

Forensic utilization gate 同时要求：matched−cross 与 matched−shuffle paired 95% CI lower bound `>0`。Forensic specialization 还要求 matched forensic−matched raw-CLIP lower bound `>0`。raw G0 gain 不能替代这些机制 gate。

## Oracle analysis

selected checkpoint 固定后，在同一 validation population 测 canonical G0、Authoritative Phrase-Only、TF-full。目标不是把 oracle 当部署结果，而是判断：

- G0 上升、TF 不变：主要闭合 autonomous interface gap；
- G0/TF 同升：decoder/forensic path 也增强；
- TF 升而 G0 不升：student distillation 失败；
- 三者不升：完整设计未形成有效优化/表示增益。

## Safety、审计与停止规则

- 开始前验证 sample-set equality、mask provenance、transform/inverse-transform、causal `[SEG]` indexing 与 prompt hash；
- 每个 stage 的 epoch0 必须与对应初始化行为一致；
- audit parameter/gradient/behavior，确认 teacher stop-gradient、student trainable set、forensic intervention 确实改变 continuous logits；
- safety stop 仅限 NaN、连续 gradient explosion、implementation failure 或 severe collapse；不得因早期 validation 差而提前停；
- 不进行 threshold sweep、checkpoint module swap、额外 loss search 或 post-hoc selector；
- Phase 4E-1 必须由用户明确授权后才能执行。本文件没有启动任何训练或评测。
