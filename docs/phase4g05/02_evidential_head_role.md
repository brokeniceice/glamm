# Evidential head 职责与消融接口

## Formal freeze

```yaml
HEAD_REFINEMENT_ALLOWED: YES
SOURCE_CLASS_DIRECTION_PRESERVATION_REQUIRED: NO
SOURCE_INFORMATION_UTILIZATION_AUDIT_REQUIRED: YES
```

PCERF 不再被限制成 frozen expert 的纯 router。两个 head 均可同时学习 source-conditioned posterior refinement、evidence strength 与 pixel uncertainty；最终成功可属于 routing-dominated、refinement-dominated 或 hybrid 三种预注册模式。

## 输入、输出与监督职责

Language head 输入 `S64 + broadcast(q_seg) + down(z_L)`，输出 `E_L [B,2,64,64]`。Forensic head 输入 `F24 + z_F24`，输出 `E_F [B,2,24,24]`。`softplus` 保证非负 evidence；`alpha=e+1` 后产生 posterior、belief 与 uncertainty。

未来 TRAIN-FIT 可合法使用官方 union target监督 source head 与 fused output；GT 只进入 loss，不进入 forward、routing 或 inference metadata。TRAIN-CAL 可用 source correctness/error 作 calibration target；TRAIN-AUDIT 只读评估。三者不得复用 image ID。

## 已实现的 ablation hooks

| Head | Full | feature removal | source-logit removal | source-logit only | 其他 |
|---|---|---|---|---|---|
| Language | `S64+q_seg+z_L` | `no-S64` | `no-z_L` | `z_L-only` | `no-q_seg` |
| Forensic | `F24+z_F24` | `no-F24` | `no-z_F24` | `z_F24-only` | — |

另预留 no-language-stream、no-forensic-stream、static fusion、constant reliability、no conflict discount、reliability permutation 与 forensic→matched-CLIP。

这些 hooks 只证明未来能够识别机制；本阶段未运行 development validation ablation。若未来 Full 的增益可完全由 E_F 单独解释，则论文必须改写为 forensic evidential segmentation with language auxiliary context，不得写 balanced dual-expert fusion。

## 三种已冻结解释

- `ROUTING_DOMINATED`：frozen complementarity强，reliability/conflict消融强，head refinement小。
- `REFINEMENT_DOMINATED`：E_L/E_F source-conditioned refinement贡献主要增益，reliability贡献较弱但非零。
- `HYBRID`：frozen complementarity、head refinement、reliability/conflict均有独立贡献。

如果 reliability/conflict 消融无影响，即使 Full 涨点，也不得把主贡献写成 reliability-aware fusion。

