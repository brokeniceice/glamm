# Phase 4G-1 未来训练提案（仅设计，未授权执行）

## 阶段顺序

### G1-P：只读/合成实现 preflight

核对公式 parity、tensor shape、坐标、support、P1 exact recovery、gradient isolation、invalid-G0 与 seal。不得以 validation IoU 选架构。

### G1-C：train-calibration reliability preflight

在冻结 train population 内预留固定 calibration fold；不接触 validation selector，更不接触 internal test/official1000。训练/校准两个 expert evidence head后，检查 ECE/Brier/NLL、uncertainty–error correlation、QMF weight–loss负相关、reliability permutation 与 corruption response。

若 reliability 信号不可靠，停止；不进入 full training。

### G1-F：仅在 preflight 通过后 full training

从 P1 + Phase 4C-A forensic source fresh start。训练 PCERF reliability/fusion parameters，两个 expert backbone frozen。预算、epoch、optimizer 与 population 必须由下一阶段正式协议另行冻结；本文不授权。

## 预注册 arms

1. Full PCERF。
2. Static matched fusion（相同 expert 与参数预算，不用 reliability）。
3. No reliability input / constant evidence。
4. Sample-only reliability（去 spatial）。
5. No conflict discount。
6. No calibration。
7. Forensic expert → matched CLIP expert。
8. Reliability permutation（inference causal control）。
9. matched / cross-image / spatial-shuffle / zero-vacuous / off。
10. P1 language-only 与 forensic-only frozen references。

不得把这些 arms 全部先小预算筛选再挑胜者；它们服务于 Primary 的机制归因。

## Selector：Pareto 而非混合 scalar

validation Fake 全 population，formal canonical G0 保留 invalid `[SEG]` failure 为 0。每个 checkpoint 同时报告：

- x：canonical G0 mean FG IoU；
- y1：Phrase-only；
- y2：TF-full；
- oracle gap retention；
- matched/cross/shuffle causal controls；
- valid-G0-only diagnostic（不得替代 headline）。

先形成非支配 Pareto set，再按预注册 lexicographic rule 选择：

1. 相对 Phase 4F G0 满足 non-inferiority（建议 margin 0.005，需人工冻结）；
2. matched > cross 且 matched > shuffle 的 paired CI 下界为正；
3. 在合格 checkpoint 中最大化 `min(Phrase recovery, TF recovery)`；
4. 仍并列时取更早 checkpoint。

不使用 `G0 + TF` 单一 scalar。

## Success criteria

- Endpoint A：G0 不丢失 Phase 4F 正增益；相对 P1 paired CI 下界仍 >0，且相对 Phase 4F 满足预注册 non-inferiority。
- Endpoint B：matched > cross、matched > shuffle，paired CI 下界 >0。
- Endpoint C：Phrase > G0、TF > G0；oracle gap retention 相对 17.28% 的 paired/bootstrap improvement CI 下界 >0。
- Endpoint D：相对 Phase 4F，G0 non-regression 且 Phrase/TF 至少一个严格恢复、另一个不回退，形成 Pareto improvement。

只有 Endpoint A–D 同时成立，才可声称 reliability-aware fusion 缓解 trade-off。仅 TF 恢复或仅 G0 上升均不够。

## Optimization audit

每 batch/epoch记录 source-specific loss、gradient norm、gradient cosine、evidence strength、uncertainty、conflict、discount、饱和率和实际 optimizer updates。OGM/Pareto gradient modulation不是默认 arm；只有预注册 dominance gate 触发后，获得新授权才做 matched arm。

## Seals

internal test 与 official1000 继续封存。Phase 4G-1 成功也不自动解封，不自动进入 Teacher/KD 或 held-out evaluation。

