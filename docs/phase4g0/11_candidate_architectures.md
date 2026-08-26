# Phase 4G 候选架构

## Candidate A（PRIMARY）：PCERF

全名：**Pixel-wise Conflict-guided Evidential Reliability Fusion**。主要血统：TMC（evidential uncertainty）+ QMF（weight–loss correlation criterion）+ ECoLaF（dense segmentation conflict discount）+ TruFor（forensic reliability map）。证据以 DIRECT/ADAPTED 为主。

### Tensor graph

```text
canonical h_G0 [B,4096] FROZEN
  └─ P1 text_hidden_fcs ─> q_seg [B,256] FROZEN

image ─> P1 SAM image encoder/cache ─> S64 [B,256,64,64] FROZEN
q_seg + S64 ─> original prompt encoder + mask decoder
  └─ z_L [B,1,256,256]                  (Language expert)

CLIP grid [B,1024,24,24]
  └─ selected Phase4C-A forensic adapter ─> F24 [B,256,24,24] FROZEN
       └─ selected dense_head ─> z_F24 [B,1,24,24]
            └─ frozen geometry-aware resize ─> z_F [B,1,256,256]
                                                    (Forensic expert)

[S64, broadcast(q_seg), down(z_L)] ─> E_L [B,2,64,64] TRAINABLE
[F24, z_F24]                       ─> E_F [B,2,24,24] TRAINABLE
E_F ─> coordinate-aware resize/support mask ─> [B,2,64,64]
E_L,E_F ─> Dirichlet masses + uncertainty + pixel conflict
         ─> ECoLaF-style adaptive discount + DS combination
         ─> fused mass/probability [B,2,64,64]
         ─> geometry-aware resize ─> output logits [B,1,256,256]
```

### Forward 与可靠性

对二分类 `K=2`，两个 head 输出非负 evidence `e_m = softplus(a_m)`，Dirichlet 参数 `α_m=e_m+1`，总强度 `S_m=Σ_k α_{m,k}`，uncertainty `u_m=K/S_m`。转成 singleton 与 ignorance mass 后，按两个 source 的像素级 conflict `κ_xy` 和自身 uncertainty 做 adaptive discount，再用 Dempster–Shafer rule 组合。

这里的关键不是固定某个新公式，而是忠实实现 ECoLaF 的 conflict-guided discount；Phase 4G-1 实现前须把原论文/官方代码的 mass、discount 与 normalization 单元测试逐项对齐。若无法做到忠实 parity，则 Primary 实现状态为 BLOCKED，不得退化成 `sigmoid(MLP)`。

### Frozen / trainable / gradient routing

- Frozen：P1 全路径、`text_hidden_fcs`、SAM prompt encoder、SAM mask decoder、S64、Phase 4C-A forensic adapter 与 dense head。
- Trainable：两个小型 evidential reliability head 与 ECoLaF discount/calibration parameters。
- 梯度不得进入 P1 或 forensic expert；只更新 reliability/fusion。
- invalid canonical G0 没有合法 q/source prediction时保持 formal localization 0，不用 forensic 单独“救活”，避免改变部署任务定义；其处理规则在 Phase 4G-1 preflight 冻结。

### 为什么不重演 Phase 4F / 4E

- 不把 forensic residual写进 S64，因而消除 feature representation 内的持续覆盖。
- P1 expert 独立存在；forensic discount 为完全不可信/vacuous 时输出 exact P1 prediction。
- 可靠性用 source-specific supervision、calibration 和 permutation 检验，常数 gate 不能靠 fused IoU 掩盖。
- 原 `text_hidden_fcs`、prompt encoder、SAM mask decoder 全部保留；没有 Phase 4E 的新 decoder ceiling。

### 风险

24×24 forensic logits 边界较粗；late fusion可能损失 feature-level互补；evidential uncertainty可能校准失败；DS conflict 在高冲突下需数值稳定审计。

## Candidate B（SECONDARY）：AHBFR

全名：**Asymmetric Hybrid Bounded Feature Rectification**。血统：CMX channel/spatial rectification + MAG norm-bounded shift + UMFNet uncertainty confidence。

### Tensor graph

```text
h_G0 [B,4096] --frozen P1--> q_seg [B,256]
S64 [B,256,64,64] -------------------------------┐
F24 [B,256,24,24] --coord cross-attention--> R64 [B,256,64,64]
P1 coarse logits + q_seg + S64 --> U_L [B,1,64,64]
forensic logits + F24 ----------> U_F [B,1,64,64]
[S64,R64,U_L,U_F] --CMX-like--> W_C [B,256,1,1], W_S [B,1,64,64]
R = (W_C + W_S) ⊙ R64
alpha = min(beta ||S64||_2 / (||R||_2+eps), 1)       (MAG bound)
S' = S64 + alpha * reliability(U_L,U_F) * R
S' + q_seg --original frozen SAM prompt/mask decoder--> mask
```

Trainable：cross-attention/projection、CMX channel/spatial rectification、uncertainty heads；P1 与 forensic source frozen。所有坐标使用原图归一化 cell center，CLIP center-crop 外 support 为 0。

防 dominance：spatial/channel gate 取代 scalar gamma；MAG cap 限制 residual norm；uncertainty supervision防止 gate 恒 1；P1 rectifier-off path 必须 exact。它仍可能因 G0 mask loss把 feature fusion推向 forensic shortcut，故 Secondary 需要更严格的 gradient contribution audit。

## Candidate C（REJECTED FOR 4G-1）：TSST

全名：**Token-Selective Spatial Substitution**，主要血统 TokenFusion。

`S_i` importance 由 `[S_i,q_seg,z_Li]` 预测，低 importance token 用坐标匹配的 forensic token替换，并保留原 SAM positional alignment；原 decoder 保留。

拒绝进入首轮的原因：

1. TokenFusion importance 不等于本项目 language correctness，适配跨度大；
2. 硬替换无 MAG 式位移上界，容易破坏 P1 ceiling；
3. S64 在 decoder 前尚未与 q_seg 交互，score 的 reliability 语义较弱；
4. 24×24→64×64 存在 support 与边界问题；
5. Primary/Secondary 已覆盖 conflict late fusion和bounded feature fusion，首轮再加入会降低因果可识别性。

它不是被否定为无效，只是不进入 Phase 4G-1 Primary study。

