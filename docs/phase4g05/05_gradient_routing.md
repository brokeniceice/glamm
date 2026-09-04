# Gradient routing audit

## 审计设置

使用冻结 train cache 顺序中的前两个 valid canonical-G0样本，加载真实 P1 SAM runtime 与 Phase 4C-A selected forensic adapter。PCERF 做一次 BCE backward；没有 optimizer step。

## Trainable gradients

| Block | aggregate grad norm |
|---|---:|
| language evidential head | 103.820976 |
| forensic evidential head | 3.132940 |

两个 head 的每个 parameter tensor均有 finite non-zero gradient。loss为 `0.5992680788`。

## Frozen invariance

| Source | grad | state hash before/after |
|---|---|---|
| P1 SAM prompt/mask path | none | `b8d1b369...c1b09`，相同 |
| Phase 4C-A forensic adapter+dense head | none | `9e8f3853...b50e9`，相同 |

checkpoint 文件哈希前后也相同：P1 `fa4856f8...6b326`；Phase 4C-A `725dd44e...78f7`。LLM/text_hidden_fcs/CLIP backbone未加载为 trainable module，输入 cache 均 detach。

```yaml
GRADIENT_ISOLATION: PASS
optimizer_step: false
formal_optimizer_updates: 0
```

完整 per-parameter grad norm见 `outputs/phase4g05/gradient_routing.json`。

