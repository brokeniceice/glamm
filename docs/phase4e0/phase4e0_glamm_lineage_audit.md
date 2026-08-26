# Phase 4E-0 — LISA/GLaMM Lineage 审计

## 论文 lineage

LISA 建立了核心 contract：LLM 生成 segmentation token，取一个 causal hidden state 投影到 SAM prompt space，再以 language 与 mask loss 对齐 reasoning 和 pixel。GLaMM 扩展为 grounded conversation，支持 region-conditioned input 及多 phrase–mask association；但 mask primitive 仍是“projected language state → SAM sparse prompt”。

当前 forensic repository 继承该 primitive。它可以处理一个 answer 中多个 `[SEG]` occurrence，但每个 occurrence 独立产生 prompt/mask。这不等于为同一 target 学习 coordinated multi-token codebook：当前没有 query-set 内交互、assignment、shared mask feature 或 slot relation loss。

## 当前代码信息流

```text
canonical image ──CLIP/LLaVA──> language trajectory [B,L,4096]
                                      │
                             [SEG] 前一 causal state
                                      │ [Nseg,4096]
                             MLP 4096→4096→256
                                      │ [Nseg,256]
                             SAM prompt encoder
                                      │ [Nseg,1,256]
SAM image encoder ────────────────────┼──> SAM mask decoder
        [B,256,64,64]                 └──> low-res mask → inverse geometry
```

`extract_seg_predictor_hidden` 明确遵守 causal indexing：位置 `k` 的 LM state 预测 token `k+1`，所以视觉 token 展开后选择 `[SEG]` **前一位置**的 state，而不是 `[SEG]` token 自身 hidden。

## P1 改变了什么

P1 通过 phrase-aligned forensic supervision 学得更好的 autonomous language-to-grounding mapping，但没有改变上述结构瓶颈。因此 P1 gain 与显著 TF gap 可同时成立：Stage-I 改善送到接口的内容，Stage-II 仍用每个 mask 一个 256D sparse prompt 接收它。

## 相对原 GLaMM 的状态标记

| 路径 | 状态 | 说明 |
|---|---|---|
| causal `[SEG]` predictor state → `text_hidden_fcs` → SAM sparse prompt | `UNCHANGED` | 保留核心 grounding primitive；当前实现明确修正为 `[SEG]` 前一 state |
| SAM image encoder / prompt encoder | `FROZEN` | canonical P1 下保持冻结 |
| mask decoder | `MODIFIED`（按历史 recipe 可训练） | Phase 3D.2 曾与 projection 联合训练，但无 validation gain |
| phrase-aligned target text / canonical forensic prompts | `FORENSIC_EXTENSION` | P1 Stage-I 的核心训练扩展 |
| optional forensic evidence token / Reader hooks | `FORENSIC_EXTENSION` | 后续诊断路径，不属于 canonical P1 gain |
| region encoder / visual prompt | `UNCHANGED/FROZEN` | 继承 GLaMM 能力，但不是当前 union-mask forensic evidence bridge |

```text
WHAT_WE_INHERITED_FROM_LISA:
以 causal segmentation-token state 作为 language-to-mask bottleneck，并通过投影连接 SAM 与 mask loss。

WHAT_LISA_DOES_NOT_SOLVE_FOR_FORENSICS:
它未提供独立 dense forensic representation、correct-image/spatial utilization gate，也未解决 autonomous–oracle behavior distillation。
```

## Single-state 判断

`SINGLE_SEG_BOTTLENECK: LIKELY`

支持 LIKELY 的证据：TF 在固定 image path 下显著改变 mask；直接优化同一 projection/decoder path 无 validation gain；dense forensic information 可学习但 one-query Reader 忽略正确图像和布局；PixelLM、PSALM、OMG-LLaVA 与 VisionLLM v2 均提供保留 query set 与 dense decoder state 的成熟替代。

不能定为 SUPPORTED：尚无 matched 实验只把 one prompt 换成 coordinated multi-token decoder 并固定其余 recipe；现有证据也未证明 token 数量是唯一原因。

## 迁移约束

最终方法应保留 P1 的 deployable G0 trajectory 与 canonical prompt，在 causal state 下游增加 query-set decoder interface；不得改变历史 `[SEG]` indexing、重新生成 mask，或用 diagnostic checkpoint 替换 P1。
