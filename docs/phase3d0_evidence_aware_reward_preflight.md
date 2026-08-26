# Phase 3D.0 — Evidence-Aware Reward / Rollout Preflight

## 1. 最终状态

本阶段是 frozen P1 的只读 rollout/reward preflight，不是 GRPO，也没有 optimizer、scheduler、backward 或模型更新。最终 primary gate：**`GATE_EVIDENCE_REWARD_PREFLIGHT_SUPPORTED`**。selected reward：**R3（JOINT_EVIDENCE）**；`GRPO_SIGNAL_READY=true`。Phase 3D.1 **未启动**。

## 2. Frozen model 与数据 provenance

P1：step 3500 / logical epoch 7，checkpoint SHA256 `fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326`。所有正式 runner 均 `eval()`、`requires_grad=False`、`torch.no_grad()`；前后 full-state/trainable-state hash exact（汇总状态 `PASS`）。

定位 GT 来自 LEGION / SynthScars 官方 `refs[].segmentation` polygon。当前 evaluator 使用 **derived union target from official SynthScars annotations**：逐 ref polygon 按原/canonical 尺寸缩放并 rasterize，随后对同图全部 ref mask 做 boolean union。它不是 pseudo mask，也未被证明与 LEGION paper 的 phrase-level evaluation unit 完全相同；无 erosion、dilation、refinement、relabel 或 threshold sweep，固定 `mask_logit > 0`。

reward-dev 仅来自 internal train：512 Real + 512 Fake；confirmation 使用完整 internal validation：1106 Real + 1106 Fake。internal test、official SynthScars1000、RAISE、LOKI、FakeBench 与 external test 均未参与 protocol/reward/gate 选择。

## 3. Sampling protocol 与 rollout diversity

reward-dev 在不读取 IoU、phrase reward、candidate reward 或 validation performance 的条件下比较 A（T=0.8, top-p=0.95）与 B（T=1.0, top-p=0.95），最终冻结 setting **A**，K=8。

- groups with >=2 distinct normalized trajectories：0.506329
- selected reward Fake groups with std>0.05：0.574141
- Fake groups mask range>0.10：0.664557
- Fake groups phrase-F1 range>0.10：0.954792

## 4. Reward definitions

- R0：Fake=`R_mask`，Real=`R_cls`（mask-only control）。
- R1：Fake=`R_phrase_lex`，Real=`R_cls`（phrase-only control）。
- R2：Fake=`0.20 R_cls + 0.10 R_struct + 0.30 R_phrase_lex + 0.40 R_mask`；Real=`0.70 R_cls + 0.30 R_struct`。
- R3：Fake=`0.20 R_cls + 0.10 R_struct + 0.20 R_phrase_lex + 0.30 R_mask + 0.20 R_align`；Real 同 R2；`R_align` 是 phrase lexical F1 与 mask IoU 的 harmonic joint。

`R_phrase_lex` 只表示 lexical phrase agreement，不表示 semantic phrase correctness；free-form explanation reward 未加入。

## 5. Greedy 与 reward ranking

canonical greedy：generated-verdict accuracy=0.985986，structure=0.992767，Fake phrase F1=0.299858，Fake FG IoU/F1=0.148233/0.210697。

| Reward | Pareto correct | tie | violation | top LM acc | top structure | top phrase F1 | top FG IoU | GRPO signal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| R0 | 0.725109 | 0.274891 | 0.000000 | 0.991863 | 0.997288 | 0.307429 | 0.301238 | False |
| R1 | 0.903114 | 0.096886 | 0.000000 | 0.992767 | 0.999548 | 0.492991 | 0.174927 | True |
| R2 | 0.999554 | 0.000446 | 0.000000 | 0.992767 | 1.000000 | 0.448695 | 0.276186 | True |
| R3 | 0.999554 | 0.000446 | 0.000000 | 0.992767 | 1.000000 | 0.439357 | 0.280733 | True |

selected R3 top-vs-greedy paired phrase Δ=0.069749，95% CI=[0.06277900332224712, 0.076670071885782]；mask IoU Δ=0.066250，95% CI=[0.05945922566963932, 0.07311551095189707]。

## 6. Real/Fake non-regression 与 reward risk

selected reward generated-verdict accuracy Δ=0.006781，classification non-regression=True；structure Δ=0.007233，structure non-regression=True。`[CLS]` head 是 frozen system diagnostic，未混入 rollout `R_cls`；primary `R_cls` 始终使用 generated LM verdict。

mask-only diagnostic：`MASK_ONLY_REWARD_RISK_SUPPORTED`。selected reward catastrophic rates：wrong class=0.007233，invalid structure=0.000000，TOP_REWARD_MASK_BAD=0.001808，TOP_REWARD_PHRASE_BAD=0.013562。

## 7. Rollout availability 与人工语义审计

Fake groups 中存在 mask 优于 greedy 的 candidate：0.835443；存在 phrase-F1 优于 greedy的 candidate：0.837251；同时满足 class correct、structure valid、phrase 与 mask 均优于 greedy：0.570524。这些是 rollout-distribution availability upper bound，不是 policy performance。

人工审阅包：`outputs/phase3d0_reward_preflight/human_review/index.html` 与 `phrase_semantic_audit.csv`，unique groups=181。人工字段保持空白，automatic gate 不依赖人工标注。**Lexical phrase reward semantic validity remains limited until human audit.**

## 8. 限制与结论边界

best-of-K trajectory 不是 deployment result。Phase 3D.0 只检验 frozen P1 policy distribution 是否含可利用 variation，以及预注册 reward 能否排序该 variation；它不能证明 GRPO 必然提升性能。correlation 仅是 secondary diagnostic，不能以 reward 与自身 component 的相关性循环证明有效性。

## 9. Final route gate

**`GATE_EVIDENCE_REWARD_PREFLIGHT_SUPPORTED`**。

该 gate 仅授权提出 Phase 3D.1 policy-optimization controlled experiment；不自动启动。 本阶段到此停止，未启动 GRPO/PPO/DPO/RL、rejection-sampling SFT、extra SFT、forensic fusion、FEPN 或架构修改。
