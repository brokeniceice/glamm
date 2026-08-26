# Phase 4E-0 — 内部证据汇总

## 范围与不变量

本文件冻结架构综合前的证据。Phase 4E-0 不执行训练、评测、threshold 选择、checkpoint 选择或 held-out 访问。P1 仍是 canonical checkpoint（step 3500 / epoch 7）。下列定位指标保留原 split、target、threshold 与 aggregation 语义，不跨不兼容协议直接相减。

## 证据账本

| 观察 | 人群 / 协议 | 结果 | 支持的解释 | 不支持的解释 |
|---|---|---:|---|---|
| Autonomous–Oracle gap | Phase 3A official1000 | G0 / Phrase / TF `0.229544 / 0.332896 / 0.437609` | autonomous language 与 phrase 之后的 grounding 都存在可恢复损失 | 存在唯一单模块原因 |
| gap 复现 | Phase 3C.0 internal validation Fake，N=1,106 | `0.148233 / 0.247362 / 0.342928` | language 与 downstream spatial 是 mixed bottleneck | TF 可部署，或 Phrase Repair 是严格单因素因果 |
| Phrase-Aligned SFT | matched Phase 3A.1 official1000 | P1−C0 G0 `+0.049758`，CI `[+0.034534,+0.064341]` | 学到了 autonomous language-to-grounding alignment | 与 context 无关的通用 segmentation 改善 |
| 现有 direct spatial path | Phase 3D.2 matched TF validation | 所有训练 checkpoint 低于 step 0 | 有界 `text_hidden_fcs+mask_decoder` recipe 无 validation gain | 所有 direct spatial supervision 都无效 |
| 256D AOGD | Phase 3F validation | step1000 G0 delta `+0.003782`，CI 跨 0 | projected-vector cosine recipe 无可靠增益 | 所有 teacher–student 方法都失败 |
| raw 4096D preflight | Phase 3G，只有 32 Fake | gap 存在；one-step direction 使 mean IoU 恶化 `0.000809` | raw pointwise cosine 的局部方向无效 | 完整 structured 4096D distillation 已被验证 |
| dense forensic representation | Phase 4C-A validation probe | adapter−raw `+0.027581`，CI `[+0.022586,+0.032726]` | task-aligned dense forensic map 可学习 | 已有 downstream causal use 或 held-out 泛化 |
| simple evidence Reader | Phase 4C-C | matched≈cross-image≈spatial-shuffle | 增益是 generic visual-conditioned query compensation | 正确图像或空间证据检索 |
| corrected position-aware Reader | Phase 4D-1R | gradient、BF16 survival 健康；train/val 均 matched≈cross/shuffle | 修复优化后，one-query residual Reader 仍失败 | 所有 position-aware fusion 都失败 |

## 已确认问题

最强联合诊断是 **information-interface problem**。P1 把 autonomous causal language trajectory 压缩成一个 256D SAM sparse prompt；已学得的 `256×24×24` forensic representation 要么不进入 mask decoder，要么再次被压入同一个 query。系统同时具有有用的 teacher behavior 与 dense evidence，但尚无接口把二者的 relational/spatial structure 保留到 mask prediction。

## 证据等级

1. **CONFIRMED**：autonomous–oracle gap；P1 matched gain；dense forensic representation 可学习；当前 Reader 不具 image/spatial specificity。
2. **LIKELY**：single-query compression 是架构瓶颈；尚无 matched multi-query ablation，因此不能定为 SUPPORTED。
3. **OPEN**：structured teacher signals 与 decoder-side forensic fusion 能否带来 deployable G0 gain。
4. **SEALED**：internal test 与 official1000 不用于 Stage-II development/selection，除非另行授权 final evaluation。

## 由证据导出的设计要求

- language grounding state 必须超越一个独立 sparse prompt；
- dense forensic lattice 必须保留到 segmentation decoder；
- 必须显式训练并验证 correct-image 与 correct-position dependence；
- TF 仅作为训练期 teacher，inference 只允许 deployable G0；
- 必须能做 component ablation 与 cross/shuffle causal control。

