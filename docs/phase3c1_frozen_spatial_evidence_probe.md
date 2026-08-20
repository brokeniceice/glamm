# Phase 3C.1 — Frozen Spatial Evidence Probe

## 冻结协议

本阶段仅在 frozen spatial feature 上训练 `Conv2d(C,1,1,bias=True)`。P1、LLM/LoRA、CLIP、SAM、NPR、SRM、FOCAL 均保持 `eval + no_grad`，所有 backbone 参数哈希前后 exact。route selection 只使用 internal val Fake 1106；persistent-476 membership 完全继承 Phase 3C.0。

## Validation 结果

| Source | all-val mean FG IoU | persistent mean FG IoU | persistent median | persistent mean F1 |
|---|---:|---:|---:|---:|
| SAM | 0.098538 | 0.036346 | 0.000161 | 0.062731 |
| CLIP | 0.143842 | 0.054106 | 0.000135 | 0.090311 |
| NPR | 0.003808 | 0.002224 | 0.000000 | 0.003981 |
| SRM | 0.003588 | 0.002807 | 0.000000 | 0.005059 |
| FOCAL | 0.048959 | 0.024495 | 0.000000 | 0.042894 |


## Route gate

`GATE_SPATIAL_PREFLIGHT_NOT_SUPPORTED`

- best existing GLaMM spatial probe：`clip`。
- 通过自身三项 negative controls：none。
- forensic complementary sources：none。

Linear probe 优于 shuffle controls 只说明 feature 中存在可线性解码的空间信息；forensic probe 表现更好也不等于接入 GLaMM 后必然改善 G0。跨 source 是 decodability comparison，不是严格 architecture-matched causal test。

## 停止边界

Phase 3C.1 route gate 冻结后停止。本报告不授权自动新增 NPR/SRM/FOCAL fusion、decoder、GLaMM architecture 修改、P1/B1 tuning 或 GRPO。
