# Phase 4E-0 — 架构迁移矩阵

| Existing problem | Literature mechanism | Paper | 论文证据 | GLaMM 兼容性 | 本项目证据兼容性 | 成本 | 候选？ |
|---|---|---|---|---|---|---|---|
| G0–TF hidden gap | training-only teacher query set | DETRDistill / KD-DETR | query assignment 与 matched prediction KD | 高：两种 trajectory 均可提取 | 避免 raw pointwise cosine | 中 | 是 |
| single `[SEG]` bottleneck | segmentation codebook/object queries | PixelLM / PSALM | 多 token/query 与 pixel decoder 交互 | 中：替换 causal state 下游 | 直接检验 likely bottleneck | 中 | 是 |
| 4096D teacher knowledge | learned query generator + relational KD | PixelLM / DETRDistill | high-dimensional condition 生成 query set，assignment 后匹配关系 | 高 | 不要求 raw coordinate equality | 低–中 | 是 |
| forensic dense representation | 保留 lattice 到 decoder | PSALM / VisionLLM v2 | specialist dense decoder 消费 spatial feature | 高 | 使用 4C-A positive result | 中 | 是 |
| forensic→language bridge | visual/object token 进入 LLM | OMG-LLaVA / ForgeryGPT | perception prior/mask cue 作为 token | 中 | 先前 token Reader 缺 specificity | 中 | 仅次要 |
| forensic→SAM bridge | image-embedding rectification | Propose and Rectify | forensic module 增强 SAM image feature | 高 | 绕开失败的 q-only compression | 中 | 是 |
| spatial evidence preservation | multi-scale decoder + 2D attention | PixelLM / PSALM | 保留 grid 与 spatial cross-attention | 中–高 | 修复 shuffle-invariant route | 中 | 是 |
| decoder-side multimodal fusion | task-decoder super link | VisionLLM v2 / OMG-LLaVA | task information 与 gradient 进入 specialist decoder | 中 | 对应已诊断 interface problem | 中–高 | 是 |
| teacher–student grounding | assigned query/target-aware KD | DETRDistill | query prior、prediction、feature KD | 高 | teacher=TF，student=deployable G0 | 中 | 是 |
| mask/logit distillation | pixelwise/holistic dense KD | Structured KD | output 与结构共同迁移 | 高 | teacher/student mask 几何一致 | 低 | 是 |
| spatial-attention transfer | channel-wise spatial KL | Channel-Wise KD | normalized spatial distribution 迁移 saliency | 中 | 提供 Phase 3F 缺失的 spatial target | 低 | 是 |
| decoder-feature transfer | review/HCL 与 relational KD | Knowledge Review / CIRKD | multi-level 与 relation transfer | 中 | 不只匹配一个 hidden vector | 中 | 有界使用 |

## 选择结论

矩阵支持 **coordinated grounding queries + dual-path dense decoder + multi-level TF distillation 的 hybrid**。不选择单独的 LLM-side forensic tokens：它会重复已无法建立 correct-image/spatial use 的路线。也不选择单独 SAM rectification：它没有处理 autonomous–oracle query gap。

| 组件 | 来源机制 | 本项目观察 | 预期效果 | 证伪条件 |
|---|---|---|---|---|
| query codebook | PixelLM/PSALM | 单一 projected causal state | 让 slot 分担 semantic/spatial role | matched `K=1` 不劣于 full |
| dense forensic decoder path | PSALM/VisionLLM v2 | 4C-A map 可学但 q Reader 忽略布局 | correct-location evidence 改变 mask | matched−cross 或 matched−shuffle CI 不大于 0 |
| image rectification | Propose and Rectify | SAM semantic embedding 无 forensic specialization | forensic residual 改写 spatial image state | 去掉后无差异且 intervention sensitivity 仍为零 |
| multi-level TF KD | DETRDistill/Structured/Channel-Wise KD | 256D/raw cosine recipe 失败 | 迁移 query relation、attention 与 output behavior | 不优于 mask-only/no-KD ablation |

