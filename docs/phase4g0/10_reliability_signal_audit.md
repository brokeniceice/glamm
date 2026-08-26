# Reliability Signal Audit

## 候选信号

| 信号 | 部署可得 | leakage | calibration | 额外计算 | 文献先例 | 判定 |
|---|---|---|---|---|---|---|
| LLM `[SEG]` token probability/entropy | 是 | 低 | 高需求；未证明与 mask correctness 相关 | 低 | 一般 confidence literature | 仅辅助，不可单独使用 |
| canonical `[SEG]` 是否成功生成 | 是 | 无 | 不需概率校准 | 无 | 项目真实 failure mode | 作为 hard-validity；invalid 仍计 0，不补 hidden |
| P1 mask-logit margin/entropy | 是 | 低 | 必须温度/可靠性校准 | 无额外 decoder | UNO/QMF 类 | 可作为 language evidence head 输入 |
| P1 spatial self-consistency | 是 | 低 | 需定义 perturbation 与一致性阈值 | 可能多次推理 | uncertainty literature | 计算偏高，Secondary diagnostic |
| phrase specificity/长度 | 是 | 中等 proxy 风险 | 很可能跨域失准 | 低 | 无直接 localization 证据 | REJECTED as core |
| Phase 4C-A forensic dense-logit entropy | 是 | 低 | 必须校准 | 已有 | TruFor/UNO | 可作为 forensic evidence head 输入 |
| learned forensic aleatoric uncertainty | 是 | 训练用 GT 合法、推理无 GT | NLL/ECE/Brier 必须 | 小 head | EAU/UMFNet | 可用，需 preflight |
| language–forensic probability disagreement | 是 | 无 | 本身只表冲突，不识别谁正确 | 低 | ECoLaF | 必须与 source reliability 联用 |
| GT mask/polygon、TF identity、authoritative phrase | 否 | 严重 | 不适用 | — | — | 严禁 |

## Primary 的 inference-available signals

Primary 只允许：

```text
q_seg (canonical generated hidden projected by frozen P1)
P1 low-resolution mask logits
S64 frozen SAM image embedding
F24 frozen forensic adapter feature
forensic dense-head logits
valid-SEG indicator
cross-source conflict derived from the two predictions
```

不输入“当前是 G0/Phrase/TF”的标签。Phrase/TF 条件的不同只能通过其自然产生的 `q_seg`、P1 prediction 与对应 reliability体现。

## 可靠性 preflight（未来 Phase 4G-1，尚未执行）

在固定 train-calibration fold 上，对 language 与 forensic 分别冻结检查：

1. per-pixel NLL、Brier、ECE 与 reliability diagram；
2. per-image predicted uncertainty 对 source FG IoU/error 的 Spearman/Pearson；
3. `corr(weight, source loss) < 0` 的 QMF 条件；
4. reliability score permutation 后融合决策应显著变化，否则 router 只是装饰；
5. corrupted/cross-image/spatial-shuffle evidence 下 forensic uncertainty 应上升或 discount 增大；
6. G0/Phrase/TF 只作诊断分组，不作为校准标签。

若 language 或 forensic reliability 与真实错误无稳定相关，停止 full training，`PHASE4G1_FULL_TRAINING_JUSTIFIED` 由预注册 preflight 改为 NO；不得用 validation IoU 反向挑信号。

## Granularity 结论

- sample-only：稳定但不能处理局部冲突，分数 2/5。
- channel-only：适合 feature recalibration，不直接表达 localization correctness，3/5。
- spatial/token：最贴合 causal spatial evidence，4/5，但需校准与几何支持 mask。
- hybrid sample×spatial：能够同时处理整图质量和局部冲突，5/5；Primary 采用 spatial evidential mass，并报告 sample aggregation，故冻结为 `RELIABILITY_GRANULARITY = hybrid`。

