# Phase 4E-0 — 完整 Stage-II 架构候选

评分：1=弱，5=强。`shortcut risk` 分数越高表示风险越低。评分只组织判断，不机械求和。

## Candidate A — Multi-Token Teacher-Guided Grounding（MTTG）

P1 G0 4096D causal state 生成 `K=4` grounding query；query set 与 SAM semantic grid 进入轻量 transformer mask decoder。TF teacher 产生同构 query/decoder behavior，通过 query-relation、attention、feature、logit KD 监督 G0 student。无 forensic branch。

优点：最直接检验 single-query 与 G0–TF gap；与 PixelLM、PSALM、DETRDistill 对应清晰。缺点：完全未使用 4C-A positive evidence；若 SAM semantic feature 缺 forensic cue，teacher 也只能重排不足的 visual state。

## Candidate B — Dual-Path Forensic Grounding（DPFG）

G0 causal state 生成 `K=4` query；SAM semantic grid 与 4C-A forensic grid 分别编码，经 multi-scale cross-attention decoder 融合并输出 union mask。只有 GT mask/language supervision，无 TF KD。

优点：直接保留 dense forensic evidence，causal control 清晰；与 PSALM、VisionLLM v2、Propose-and-Rectify 对应。缺点：不利用稳定复现的 G0–TF gap，弱 G0 query 可能无法选择正确 forensic region。

## Candidate C — Teacher-Guided Forensic Dual-Path Grounding（TF-FDG）

Candidate A 与 B 的受限组合：先以 TF causal state 训练同构 `K=4` query/dual-path decoder 并冻结 teacher snapshot，再由其初始化 G0 student；SAM semantic grid 和 4C-A forensic grid 在 decoder 内交互，teacher 训练 student 时完全 stop-gradient；使用 mask GT 与 multi-level behavior KD。部署只保留 G0、P1 image paths、forensic adapter 和 student decoder。

该组合同时对应三个内部事实：TF gap、4C-A positive representation、Reader failure；组件各自有文献机制和单独 ablation，因此不是无理由堆叠。

## 评分

| Criterion | A MTTG | B DPFG | C TF-FDG |
|---|---:|---:|---:|
| Directly addresses G0–TF evidence | 5 | 2 | 5 |
| Uses 4C-A positive result | 1 | 5 | 5 |
| Preserves spatial forensic information | 1 | 5 | 5 |
| Avoids single-query bottleneck | 5 | 5 | 5 |
| Literature support | 5 | 5 | 5 |
| Mechanistic interpretability | 4 | 5 | 4 |
| Novel combination | 3 | 4 | 5 |
| Compatibility with existing GLaMM | 4 | 4 | 3 |
| Training feasibility | 4 | 4 | 3 |
| Ablation clarity | 5 | 5 | 5 |
| Risk of shortcut | 4 | 3 | 3 |
| Likelihood of meaningful Stage-II gain | 3 | 4 | 5 |

## 非机械决策

Candidate C 的成本和优化风险最高，但它是唯一覆盖全部已确认/likely bottleneck 的候选。A、B 的分数并不差，却各自系统性忽略一个强内部事实。由于本阶段目标是“足够解决问题所需的最小完整系统”，不能因训练更便宜而优先 A/B。

```text
RECOMMENDED_ARCHITECTURE: Candidate C — TF-FDG
SECOND_CHOICE: Candidate B — DPFG（若 teacher KD 的 mandatory ablation 为零贡献）
REJECTED: Candidate A 作为最终 Stage-II；保留为 -forensic ablation
```

## 组件证据卡

```text
COMPONENT: K=4 grounding query codebook
SOURCE PAPER: PixelLM; PSALM
SOURCE MECHANISM: multiple codebook/object queries interact with dense decoder
OUR OBSERVATION: one causal state is compressed to one SAM prompt
WHY: preserves multiple grounding roles and enables relational teacher signal
FALSIFY: matched K=1 is non-inferior to K=4
```

```text
COMPONENT: dual-path SAM/forensic decoder fusion
SOURCE PAPER: PSALM; VisionLLM v2; Propose and Rectify
SOURCE MECHANISM: specialist dense decoder and forensic image-feature rectification
OUR OBSERVATION: 4C-A dense map is learnable; q Reader ignores correct layout
WHY: keeps forensic evidence spatial until mask prediction
FALSIFY: matched≈cross/shuffle or -forensic is non-inferior
```

```text
COMPONENT: multi-level TF behavior KD
SOURCE PAPER: DETRDistill; Structured KD; Channel-Wise KD; Knowledge Review
SOURCE_MECHANISM: assigned query, spatial distribution, decoder feature and logit transfer
OUR OBSERVATION: TF gap is stable; pointwise cosine KD failed
WHY: transfers grounding behavior without assuming shared raw coordinates
FALSIFY: -KD is non-inferior and alignment changes do not track IoU
```
