# CSCU-LF frozen architecture

## Full tensor contract

| Tensor | Shape | coordinate/range | status |
|---|---|---|---|
| `q_seg` | B×256 | token context | frozen input |
| `S64` | B×256×64×64 | original-normalized SAM grid | frozen input |
| `z_L` | B×1×256×256 | canonical P1 logits | frozen exact fallback |
| `F24,z_F24` | B×256×24×24 / B×1×24×24 | CLIP crop | frozen forensic source |
| `L64,Fctx64` | B×64×64×64 | original-normalized | trainable context projections |
| `Lr,Fr` | B×64×64×64 | original-normalized | differentiable joint interaction |
| `Lr*Fr` | B×64×64×64 | unbounded | differentiable; support masked |
| `|Lr-Fr|` | B×64×64×64 | nonnegative | differentiable a.e.; support masked |
| local cosine | B×1×64×64 | [-1,1] | differentiable; support masked |
| `p_L,p_F` | B×2×64×64 | [0,1] | frozen source posterior |
| conflict | B×2×64×64 | official ECoLaF range | frozen formula, differentiable input feature |
| comparison | B×263×64×64 | mixed | explicit concat, support masked |
| `U_F64` | B×1×64×64 | [0,1] | independent intervention node |
| `U_F256` | B×1×256×256 | [0,1] | bilinear geometry-preserving map, support masked |
| source masses | B×3×2×256×256 | singleton+ignorance | frozen source materialization |
| output | B×1×256×256 | logits | adapted fusion or exact z_L |

Utility head为 `Conv3×3 263→64 + GN + GELU + Conv1×1 64→1 + sigmoid`。它同时依赖 L/F、product/difference/cosine、source posteriors与official conflict，不能退化为F-only intrinsic confidence。

## Source and route freeze

G1-C epoch10 evidential heads与T_L/T_F只作为 frozen source posterior materializers；不把其 intrinsic uncertainty作为 utility。`REFINEMENT_INCLUDED=NO`，source predictions/masses不由CSC U branch修改。

Forensic absent/off/vacuous/fully unsupported direct bit-exact `z_L`。Partial support外也逐像素 `z_L`。Invalid canonical G0输出formal invalid/zero policy，forensic不能rescue。Forward signature无GT、phrase、TF identity、polygon或condition label，`NO_ORACLE_LEAKAGE=PASS`。

Implementation：`model/csculf.py`；tests：`tests/test_phase4g1p_csculf.py`；supervisor：`scripts/phase4g1p_freeze_preflight.py`。本阶段没有生成formal checkpoint。
