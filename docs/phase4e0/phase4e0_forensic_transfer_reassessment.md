# Phase 4E-0 — Forensic Transfer 路线重审

## 已知事实

Phase 4C-A 证明 `[B,256,24,24]` task-aligned forensic representation 在 matched validation probe 下更可解码。Phase 4C-C/4D-1R 证明把这张图通过一个 single-query Reader 汇总回 q，未建立 correct-image 或 correct-position utilization。这是接口失败，不是 representation absence。

## 三个接口 family

### Family A — Forensic tokens into LLM

优点：与 ForgeryGPT/OMG-LLaVA 的 visual/mask token 思路兼容，可影响 explanation 与 query generation。缺点：tokenization 容易丢失高频空间结构，且当前项目的 global token 与 Reader 路线已有 negative evidence。结论：不作为主要接口；最多在后续 ablation 中作为 language-side auxiliary summary。

### Family B — Dual-path decoder fusion

```text
G0 causal state → K grounding queries ─┐
                                      ├→ cross-attentive dense decoder → mask
4C-A forensic grid [256,24,24] ───────┤
SAM semantic grid [256,64,64] ────────┘
```

优点：保留位置，直接对应 PSALM/PixelLM/VisionLLM v2；可用 cross/shuffle 因果 control；不会把 dense map 再压成一个 q。缺点：需要新的 geometry adapter 与完整 decoder training。结论：主要接口。

### Family C — SAM image rectification

把 forensic feature resize/project 到 SAM lattice，以 gated residual 或 cross-attention rectification 修改 SAM image embedding，再送入 mask decoder。Propose and Rectify 是直接先例。优点：复用 SAM mask space；缺点：单独使用不能解决 G0–TF language query gap，且 naive residual 可能走 shortcut。结论：作为 dual-path decoder 内的 semantic/forensic fusion stage，而非独立方法。

## 最佳接口

`BEST_FORENSIC_INTERFACE: HYBRID`

这里的 hybrid 是受限组合：4C-A grid 保持独立进入 dual-path decoder，同时通过一个可消融的 cross-attentive rectification block 调节 SAM feature；不把 forensic grid 输入 LLM，也不引入第三个 encoder。

## Family 对比

| 维度 | A：LLM tokens | B：dual-path decoder | C：SAM rectification |
|---|---|---|---|
| 保留 2D 信息 | 中，受 token 压缩影响 | 高 | 高 |
| 与 P1 兼容 | 中，需改 multimodal input | 高，接在 causal state 下游 | 高，接在 SAM image path |
| 使用 4C-A adapter | 可，但需 tokenization | 直接使用 | resize/project 后使用 |
| 训练复杂度 | 中 | 中–高 | 中 |
| 因果可解释性 | 低–中，容易 language shortcut | 高，cross/shuffle 可直接检验 | 高，但需隔离 semantic residual |
| 新颖性 | 低–中 | 高 | 中 |
| 预期表示容量 | 中 | 高 | 中–高 |
| 本项目选择 | 不作主接口 | 主体 | 作为 B 内可消融 rectification |

## 训练约束与 causal control

- correct forensic feature 为 main condition；
- wrong-image derangement 检验 image-specific use；
- spatial-content shuffle（position lattice 固定）检验 spatial-specific use；
- zero forensic 检验 feature-presence shortcut；
- full 必须同时满足 matched>cross 与 matched>shuffle 的预注册 paired CI，单独 raw G0 gain 不足以宣称 evidence utilization。

## 证伪

若 full architecture 仍 matched≈cross/shuffle，即使 G0 上升，也只能称 query/regularization gain，不得称 forensic grounding。若 forensic−CLIP matched ablation 无差异，则“forensic specialization”不是贡献，但 spatial dual-path mechanism 仍可单独判断。
