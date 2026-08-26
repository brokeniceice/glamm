# Phase 3D.0-S — Independent GPT Semantic Audit

## 1. 最终结论

最终 gate：`GATE_GPT_AUDIT_FAILED`。

**由于 position-order consistency 仅为 0.5660，低于 0.90，本阶段按预注册规则判定 GPT judge 不可靠。以下 preference、correlation 与 subgroup 数值只作为故障诊断，不得用于支持或否定 Q2/MiniLM。**

本阶段只对 Phase 3D.0-R 冻结回答进行独立多模态 GPT 外部验证；没有重新 rollout、没有修改 Q2/MiniLM/reward，也没有启动训练。

## 2. GPT 与数据完整性

- judge：`gpt-5.6-luna`，temperature=0，reasoning effort=none。
- 500 对比较、每对 AB/BA 两次，另有 50 对重复调用，总计 1050 条；失败 0 条。
- 输入包括原图、官方 SynthScars annotation-derived reference overlay、两候选 phrase/explanation/mask overlay。
- GPT 不可见候选来源、reward、MiniLM、IoU、F1 或置信度。

## 3. GPT 自身可靠性

- position-order consistency：0.5660（阈值 0.90，FAIL）。
- repeatability winner agreement：0.9400（阈值 0.90，PASS）。
- repeatability mean absolute score difference：0.3400。
- 原始位置选择：AB 调用 A=315、B=118、TIE=67；BA 调用 A=293、B=136、TIE=71。
- 交换候选后仍选择同一字母：A→A=192，B→B=15。这两种模式会改变真实来源胜者，是直接的位置偏差证据。

## 4. Reward ranking validation

本节因 GPT reliability gate 失败而**不可用于科学结论**，仅记录描述性故障统计。

### Q2 vs Greedy

- Q2 wins=72，Greedy wins=87，tie=6，uncertain=0，order disagreement=135。
- decisive preference rate=0.4528，95% Wilson CI=[0.37748186767165937, 0.5304039980738902]，阈值 0.70，FAIL。

### Q2 vs OLD_R3

- Q2 wins=9，OLD_R3 wins=16，tie=50，uncertain=0，order disagreement=25。
- decisive preference rate=0.3600，95% Wilson CI=[0.20247880774458224, 0.5548150225183514]，阈值 0.60，FAIL。

胜率仅使用 AB 与 BA 在真实来源层面一致、且明确给出 Q2 或 comparator 胜者的 pair；tie、uncertain 和 order disagreement 均不进入分母。

## 5. MiniLM–GPT semantic validation

本节同样因 GPT reliability gate 失败而**不能验证或否定 MiniLM**。

- candidate observations=1000；Pearson=0.1256，95% bootstrap CI=[0.06335675126695083, 0.18649565614470145]。
- Spearman=0.1290，95% bootstrap CI=[0.06737527260360349, 0.18797662171113305]，阈值 0.60，FAIL。
- MiniLM false negative=53；false positive=71。
- unique-trajectory sensitivity Spearman=0.1433。

这里 GPT phrase score 是 AB/BA 对同一真实候选评分的平均值。lexical-high/semantic-low 严格池只有 9 个唯一 mask 轨迹，20 次 failure-oriented 判断含确定性有放回补足；因此同时报告 unique-trajectory sensitivity，不把重复项当作额外独立轨迹。

## 6. Semantic collapse audit

- Q2 GPT phrase mean=2.6450；comparator=2.8150；Δ=-0.1700。
- Q2 score<=1 rate=0.1275；comparator=0.1050。
- no obvious semantic collapse=True。

这是审计前未规定数值公式的定性 gate，本次采用保守的显式操作定义：均值退化不超过 0.25/4 分，且低分率增加不超过 5 个百分点。

## 7. 科学边界与下一步

`GATE_GPT_SEMANTIC_AUDIT_SUPPORTED` 只会授权 Phase 3D.1 policy optimization design，不自动授权或启动 GRPO/PPO/DPO/SFT。`GATE_REWARD_PROXY_WEAK` 表示 GPT 自身可靠但 MiniLM proxy 或 ranking gate 未获支持，应停止并审查 semantic proxy。`GATE_GPT_AUDIT_FAILED` 表示 GPT judge 自身不可靠，不能使用其结论。

本次实际进入 `GATE_GPT_AUDIT_FAILED`，因此停止：不授权 Phase 3D.1，不修改 reward，不补做自动调参，也不把描述性胜率解释为 Q2 性能。

定性页面：`outputs/phase3d0s_gpt_audit/qualitative/index.html`。
