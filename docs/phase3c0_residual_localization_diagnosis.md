# Phase 3C.0：P1 残余定位瓶颈只读诊断

## 结论

本阶段只使用 internal validation 的 1106 张 Fake 图像进行 route selection，冻结模型为 Phase 3A selected P1（step 3500 / epoch 7，SHA256 `fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326`）。未训练、未反向传播、未修改权重，也未使用 internal test、official1000、RAISE 或 LOKI 参与路径选择。

最终 evidence gate：`GATE_LANGUAGE_PHRASE_ERROR_SUPPORTED`、`GATE_DOWNSTREAM_SPATIAL_PREFLIGHT_AUTHORIZED`、`GATE_MIXED_BOTTLENECK`。

## 四条件结果

| 条件 | 范围 n | mean FG IoU | mean FG F1 | mean fg/bg mIoU |
|---|---:|---:|---:|---:|
| A：canonical G0 | 1106 | 0.148233 | 0.210697 | 0.544399 |
| B：G0 Phrase Repair（eligible only） | 1076 | 0.261355 | 0.361021 | 0.600721 |
| C：Authoritative Phrase-Only | 1106 | 0.247362 | 0.345261 | 0.584177 |
| D：TF-PHRASE | 1106 | 0.342928 | 0.452020 | 0.652260 |

注意：B 仅针对 exactly-one usable `[SEG]`、可解析 `Target regions:` 且 authoritative phrase 非空的 primary replacement population，因此 B 的绝对均值不能直接与 A/C/D 的全 1106 均值作非配对比较；Q1/Q2 使用同一 eligible 样本配对。

## 配对问题

- Q1 `B−A`：n=1076，mean=+0.108989，median=+0.044371，bootstrap 95% CI=[+0.095485, +0.122814]，win/tie/loss=745/63/268，Wilcoxon p=3.67391e-56。
- 在 `D>A` 的 positive residual gap 中，B 的 recovery ratio：unclipped mean=-0.005542、median=+0.569962；clipped [0,1] mean=0.526977、median=0.569962。
- Q2 `C−B`：n=1076，mean=-0.012476，95% CI=[-0.021230, -0.003543]，Wilcoxon p=1.58277e-05。
- Q3 `D−C`：n=1106，mean=+0.095566，95% CI=[+0.083262, +0.108364]，Wilcoxon p=7.54903e-51。

B 是 `CONTROLLED_MULTI_FACTOR_DIAGNOSTIC`：phrase replacement 同时可能改变 token 长度、`[SEG]` 位置与 hidden trajectory，所以 B−A 不能被称为严格单因素因果效应。C−B 若为正，只支持 generated explanation/context 可能有害，不证明长文本普遍有害；D−C 不自动等价于 CoT benefit。

## 残余失败

- canonical G0 failure（A IoU≤0.30）：877。
- Language-recoverable（A≤0.30 且 D≥0.70）：97。
- Intermediate：264。
- Persistent（A≤0.30 且 D≤0.30）：516。
- `PERSISTENT_ORACLE_FAILURE`（A、C、D 均≤0.30）：476。
- G0 failures 中 usable `[SEG]` 比例：0.968073；全体 G0_SEG_UNAVAILABLE=28；exploratory PHRASE_INSERTION=2。

这些 persistent 样本只能称为 residual downstream / visual-spatial localization bottleneck candidates；不能据此声称 NPR 或 FOCAL 一定有效。

## Phrase quality 与 representation

eligible population 的 normalized token F1 均值为 0.308218；F1 与 A IoU 的 Spearman ρ=0.1391880815283842，与 B−A 的 ρ=-0.10021700871388481，与 D−A gap 的 ρ=-0.158569275412657。Lexical F1 不等于语义正确；本阶段没有使用 external LLM judge。

4096D hidden 和 256D projected embedding 均已逐样本保存。hidden relative-L2(A,D)=1.066137、(B,D)=0.916000、(C,D)=1.005943；与 IoU recovery 的 Pearson/Spearman 详见 `representation_statistics.json`，只作关联性诊断。

## 审查与可视化

- 人工审查：`outputs/phase3c0_residual_diagnosis/human_review/index.html` 与 `human_review.csv`；四组各最多 50，允许重叠，unique=191。人工字段保持空白，自动 gate 不依赖人类标注。
- 定性可视化：`outputs/phase3c0_residual_diagnosis/qualitative/index.html`；选择规则与 sample IDs 固定在 selection manifest，未手工 cherry-pick。

## Invariance、边界与回归

- canonical G0 trace 对 1106 个样本保存 trace-vs-evaluator identity；聚合审计见 `audit/invariance.json`。
- authoritative phrase 直接来自 `UnifiedForensicsDataset` 当前 construction rule；C/D 复用 `GLaMMForensicsBackend` 的共享模板构造。
- threshold 固定为 mask logit `>0`；不存在 threshold sweep。
- checkpoint 与 model state 前后 exact hash 见 `frozen_model_hash.json`。
- 历史 P1 val G0 artifact 不存在，故标记 `HISTORICAL_P1_VAL_G0_ARTIFACT_UNAVAILABLE`；本阶段以 canonical evaluator 与 no-cache trace 的逐样本 exact comparison 作为当前实现复现审计，不能冒充与不存在的历史 artifact 比对。
- 全量 pytest 结果将在 `reports/regression_tests.txt`，最终 manifest 会记录门禁状态。

## 最终停止边界

本报告只给出诊断 evidence gate。未启动且不自动启动 GRPO/PPO/DPO、context-decoupling training、NPR、FOCAL、fusion 或任何 Phase 3C training。
