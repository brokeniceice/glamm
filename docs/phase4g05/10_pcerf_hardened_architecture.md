# PCERF hardened architecture

## 状态

`PCERF_ARCHITECTURE_HARDENED = YES`，仅表示公式、tensor graph、fallback、geometry、gradient与future calibration protocol已可执行；不表示 reliability已有效或 full training已授权。

## 完整 tensor graph

```text
canonical h_G0 [B,4096] --frozen text_hidden_fcs--> q_seg [B,256]
image --frozen SAM image encoder/cache--> S64 [B,256,64,64]
q_seg + S64 --frozen prompt encoder + original mask decoder--> z_L [B,1,256,256]

CLIP grid --frozen Phase4C-A selected adapter--> F24 [B,256,24,24]
F24 --frozen dense head--> z_F24 [B,1,24,24]

[S64,broadcast(q_seg),down(z_L)] --LanguageHead--> E_L [B,2,64,64]
[F24,z_F24]                    --ForensicHead--> E_F [B,2,24,24]

E_m --softplus--> e_m --alpha=e+1--> (belief_m,u_m)
(belief_F,u_F) --geometry/support--> original-normalized 256 grid
active L/F masses --official ECoLaF conflict discount + Dempster + DSmP--> p_fused
p_fused --clamp[1e-6,1-1e-6], logit--> z_fused [B,1,256,256]
inactive/unsupported/vacuous/off --identity dispatch--> exact z_L
invalid canonical G0 --formal failure--> score 0
```

## Tensor contract

| Tensor | Shape | dtype | coordinate | state / gradient |
|---|---|---|---|---|
| h_G0 | B×4096 | P1 runtime | token | frozen, no grad |
| q_seg | B×256 | BF16 cache/runtime | non-spatial | frozen/detached |
| S64 | B×256×64×64 | BF16 | SAM grid→original normalized | frozen/detached |
| z_L | B×1×256×256 | P1 dtype | original normalized cell center | frozen; exact fallback source |
| F24 | B×256×24×24 | BF16 | CLIP crop cell center | frozen/detached |
| z_F24 | B×1×24×24 | BF16/FP32 | CLIP crop cell center | frozen/detached |
| E_L | B×2×64×64 | FP32 | language evidential grid | trainable head |
| E_F | B×2×24×24 | FP32 | forensic evidential grid | trainable head |
| masses | B×3×M×256×256 | FP32 | original normalized | differentiable to heads only |
| conflict/discount | B×M×256×256 | FP32 | original normalized | fixed official kernel |
| output | B×1×256×256 | preserve/fp32 | original normalized | active fusion or exact P1 |

## Head 与 fusion职责

两个 head允许改变 source posterior，不要求 source class direction preserved。它们也必须输出可校准 evidence strength/uncertainty。ECoLaF公式固定 `lambda=2`，不训练一个“类似 conflict”的任意 gate；future calibration只允许 protocol中预注册的 positive concentration temperature。

sample reliability为 spatial uncertainty的固定聚合，用于审计；fusion仍在 pixel层，故 granularity为 hybrid。outside CLIP support为 vacuous，不是background certainty。

## Numerical behavior

- frozen cache可为BF16；head与fusion统一FP32；
- evidence由stable softplus产生，不作validation-selected clamp；
- ECoLaF log epsilon `1e-10`，DSmP epsilon `1e-4`；
- 仅最终 probability→logit clamp到 `[1e-6,1-1e-6]`；
- full-conflict extreme test finite；
- empty/vacuous/off不进入 kernel，直接identity；
- both-vacuous返回P1；若未来定义无language source，则formal invalid，不让forensic rescue G0。

## Frozen / trainable

P1 LLM、LoRA、text_hidden_fcs、SAM image/prompt/mask modules、CLIP backbone、Phase4C-A adapter与dense head全部冻结。未加载Phase4F epoch-9 rectifier。当前实现只有两个 evidential head可训练；本阶段只做一次 backward、零 optimizer step。

未来必须使用 `no-z_L/no-q_seg/no-S64/z_L-only/no-z_F24/no-F24/z_F24-only/no-stream/static/constant/no-conflict/permutation/matched-CLIP` hooks做因果消融。没有这些证据不得写dual-expert或reliability-aware主张。

代码与机器 manifest见 `model/pcerf.py`、`outputs/phase4g05/architecture_manifest.json`。

