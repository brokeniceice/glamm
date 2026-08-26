# Phase 4E-1 — TF-FDG Full Method Training Proposal v2

> 本文件替代 v1 作为未来执行协议；v1 保留为历史。当前未授权、未启动 Phase 4E-1。

## 1. 研究问题

一次完整 method study 回答：

1. training-only TF teacher transfer 是否改善 deployable canonical G0 grounding？
2. 4C-A spatial forensic representation 是否被 correct-image、correct-location 地使用？
3. K=4 coordinated queries 是否优于同 topology 的 K=1？

不做 architecture screening、threshold tuning 或连续小 idea search。

## 2. Population 与 target

- localization train population：8,836 frozen train Fake；Real 不进入 mask optimizer。
- validation selector/qualification population：1,106 validation Fake。
- target：official SynthScars ref polygons；M≤4 可逐 ref supervision，M>4 union-only；最终始终保留 per-image all-ref union。
- 不生成 pseudo label，不重新标注。
- internal test 与 official1000 封存。

## 3. Formal topology

每个有 teacher 的 arm 都是：

```text
Stage T: TF hidden → arm-specific Teacher(QG+pyramids+rectification+decoder)
          ↓ selector: epoch 0–5 validation TF mean FG IoU only
          ↓ frozen qualification audit
Stage S: exact-copy Teacher weights → Student
         teacher(h_TF) stop-gradient supervises student(h_G0)
```

P1/CLIP/SAM/4C-A evidence encoder frozen。student 不接收 TF token/phrase；teacher 不参与 inference。

## 4. Stage T — Teacher warm-up

- Fake N=8,836；effective batch=8；最后一个 partial batch=4；
- 5 full epochs，1,105 steps/epoch，5,525 steps，44,180 exposures；
- optimizer：AdamW，new modules LR `2e-4`，weight decay `0.05`；5% linear warmup，cosine decay至 `2e-5`；
- BF16 module compute，FP32 mask/union loss与 grad norm；global grad clip 1.0；
- objective：M≤4 时 `L_slot + L_union`，M>4 时 `L_union`；
- checkpoints：epoch 0/1/2/3/4/5；selector 只用 validation Fake canonical TF mean FG IoU；
- 不因 poor validation early stop；仅 NaN、repeated gradient explosion、implementation failure、catastrophic collapse 可 safety stop。

## 5. Teacher qualification gate

selected teacher 固定后，direct batch=1：

1. Teacher TF vs frozen P1 TF：mean FG IoU、paired delta/CI、W/T/L；按 hardened architecture 中 `-0.010/-0.030` margin 得 `ADEQUATE/WEAK/FAILED`。
2. matched/cross-image/spatial-content-shuffle/zero forensic：得到 image/spatial-specific state。

输出：

```text
TEACHER_MASK_CAPABILITY: ADEQUATE / WEAK / FAILED
TEACHER_IMAGE_SPECIFIC_USE: TRUE / FALSE / INCONCLUSIVE
TEACHER_SPATIAL_SPECIFIC_USE: TRUE / FALSE / INCONCLUSIVE
```

qualification 不重选 teacher。若 spatial use 非 TRUE，assignment 关闭 attention cost、student `L_attention=0`；若 mask WEAK，`L_logit=0`；若 mask FAILED，不启动 Full KD student并停止报告 teacher failure，不搜索 architecture/K/loss。

## 6. Stage S — G0 student

- teacher QG、semantic/forensic pyramid、rectification、decoder、mask head逐 tensor复制到 student；hash exact 后开始；
- 10 full epochs，1,105 steps/epoch，11,050 steps，88,360 Fake exposures；
- AdamW：QG/decoder/head LR `1e-4`，pyramids/rectification LR `5e-5`，weight decay 0.05；5% warmup + cosine；
- BF16 compute、FP32 loss/grad norm、grad clip 1.0；
- checkpoints epoch 0–10；selector 只用 validation Fake canonical G0 mean FG IoU；完全同值时依次以 canonical Detection non-regression、较早 epoch tie-break；
- Phrase-only/TF、cross/shuffle、slot diversity 不参与 selector；
- 无 poor-validation early stop，预算不因中途曲线改变。

保留 5+10 epoch，因为这是约 11.25M trainable dense decoder 的 formal study、decoder 从新初始化、4C-A 历史需要 multi-epoch exposure，且 PixelLM/PSALM 类 decoder 采用完整训练；不是 512-step screening。

## 7. Frozen loss 与 assignment

### Region slot loss

`M≤4`：detached Hungarian cost `1.0 DiceCost + 1.0 BCECost`；matched BCE+Dice，unmatched empty。`M>4`：union-only，不发明 grouping。

### Teacher/student assignment

```text
C_ij = DiceCost(slot_prob_S_i, slot_prob_T_j)
     + mean_abs_logit(slot_logit_S_i, slot_logit_T_j)
     + λ_attn JS(forensic_attention_S_i, forensic_attention_T_j)
```

cost stop-gradient；teacher spatial use TRUE 时 `λ_attn=0.25`，否则 0。Hungarian permutation 建立后才计算 relation/attention/feature/logit KD。

### Student objective

```text
L = 1.0 L_mask + 0.20 L_relation + 0.50 L_attention
  + 0.25 L_feature + 0.50 L_logit
```

qualification 可把 attention/logit weight 按上述预注册规则置零，不能改成其他数值。`L_relation` 为 normalized query Gram SmoothL1；`L_attention` 为 layers 3/4 original-coordinate forensic attention 的 T=2 KL；`L_feature` 为 layers 2/4 normalized query feature L1；`L_logit` 为同一 32×32 continuous union logit soft BCE。

numerical audit 已用 train-only deterministic batch验证 weighted KD/mask gradient ratios 全在 `[1e-5,1e3]`；训练中不再调权重。

## 8. Pre-execution mandatory audits

- manifest count/uniqueness/hash 与 exact Fake-set equality；
- ref polygon/phrase/id schema；
- canonical Detection/G0/TF prompt hashes；
- `[SEG]` 前一 causal state extraction exact；
- CLIP/SAM original-coordinate geometry及 crop/pad validity；
- teacher/student copy hash、teacher stop-gradient、student 无 TF input；
- `L_mask/L_relation/L_attention/L_feature/L_logit` 分别 backward 的 gradient matrix；
- raw/weighted loss与 gradient scale；
- all K slot finite、非 implementation-identical；
- parameter whitelist；无 formal optimizer update发生在 audit 中。

## 9. Formal arms

### Tier 1 — 必须

| Arm | Teacher topology | Student init / KD | 唯一主要变量 | 因果问题 |
|---|---|---|---|---|
| Full TF-FDG | K4 + forensic + rectification | teacher copy + all qualified KD | — | full method |
| `-teacher entirely` | 无 teacher | matched fresh initialization；无 KD | 整条 TF transfer absent | 对应 DPFG/Candidate B |
| `-forensic` | no-forensic K4 teacher | no-forensic teacher copy + KD | forensic path absent on both sides | dense forensic path 是否贡献 |
| `K=1` | K1 teacher | K1 teacher copy + KD | query count | multi-query 是否贡献 |
| `FULL-CLIP` | raw/CLIP-projection evidence teacher | 同 source student + KD | evidence source only | forensic specialization 是否 downstream transfer |

`FULL-CLIP` 与 `FULL-FORENSIC` 使用相同 QG/decoder width/depth、seed、sample order、optimizer、epochs、loss、selector、evaluator；FULL-CLIP 使用 Phase 4C-A matched projection-only 256D CLIP lattice，不做 inference-time swap。

### Tier 2 — Full 有解释价值时

| Arm | Teacher topology | Student |
|---|---|---|
| `-KD losses` | Full teacher照常训练/选择 | teacher exact init，所有持续 KD=0；回答 warm-start 后持续 KD 的增量 |
| `-rectification` | no-rect teacher，保留 forensic cross-attention | matched no-rect student |
| `mask-logit KD only` | Full teacher | teacher init，仅 `L_logit` |

禁止 forensic teacher→no-forensic student、K4 teacher→K1 student等 information/topology leakage。

即使 Full 效果不佳，仍完成 Tier-1 中 `-teacher entirely`、`-forensic`、`K=1` 以解释 full system；不立即发明第四套架构。Tier-2 可按预注册优先级停止。

## 10. Causal controls 与 claim gates

selected Full checkpoint 固定后运行：matched forensic、deterministic cross-image、spatial-content shuffle、zero forensic。shuffle 只置换 feature content，original-coordinate position/validity lattice固定。

```text
IMAGE-SPECIFIC CLAIM:
matched-cross paired 95% CI lower > 0

SPATIAL-SPECIFIC CLAIM:
matched-shuffle paired 95% CI lower > 0

FORENSIC-SPECIALIZATION CLAIM:
FULL-FORENSIC - FULL-CLIP matched-training paired 95% CI lower > 0
```

inference-time raw-CLIP swap只作 distribution-shift diagnostic，不能替代 matched arm。

## 11. Slot-collapse report

每 epoch/selected checkpoint报告 query cosine、attention JS/cosine、mask soft-IoU、mass、confidence、active count、union contribution，并按冻结 population rule输出：

```text
SLOT_COLLAPSE: TRUE / FALSE / PARTIAL
```

TRUE/PARTIAL 时不得用 K4 failure 否定 multi-query hypothesis；同时报告 K1 matched arm。

## 12. Metrics 与 oracle analysis

selected Student：canonical Detection、G0 per-image mean/global FG IoU、paired bootstrap/W-T-L；Phrase-only 与 TF-full 只作 oracle analysis，不参与 selection。正式 threshold-boundary metric direct batch=1，threshold固定0。

只有 validation Full 相对 P1 建立正 G0 CI，且相关 claim gate 通过后，才可另行请求 final held-out evaluation。Phase 4E-1 本身不自动解封 internal test/official1000。

## 13. Implementation validity 与停止

主 Full run 不设置“小效果 gate”。必须满足 QG/forensic/rectification gradient healthy、all slots finite且非代码恒等、teacher stop-gradient exact、student无TF输入、geometry valid、KD/mask finite。

只允许 NaN、repeated gradient explosion、implementation bug、catastrophic collapse safety stop；否则完整跑完预算。禁止依据结果改 K、loss、decoder depth、threshold 或 epoch。

## 14. Complexity report

正式集成后必须输出：

| Arm | Trainable Params | Frozen Teacher Params | Student Params | FLOPs estimate | Peak Memory |
|---|---:|---:|---:|---:|---:|
| Full | 11,246,081（isolated spec；集成后复核） | 11,246,081 | 11,246,081 | TBD by profiler | TBD |
| K=1 | 约10,458,881；集成后复核 | same topology K1 | same | TBD | TBD |
| -forensic | implementation-derived | matched no-forensic | same | TBD | TBD |
| FULL-CLIP | 与 Full 相同 | 与 Full 相同 | 与 Full 相同 | matched | matched |

复杂度差异必须与性能同时讨论。

## 15. 当前状态

本 proposal 已完成 protocol hardening，但没有运行 teacher/student、validation comparison 或 selector。必须获得用户明确授权才能执行 Phase 4E-1。

