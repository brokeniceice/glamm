# Phase 3D.0-R — Evidence-Aware Reward Reformulation

## 1. Final status

本阶段仅复用 Phase 3D.0 frozen trajectories。用户额外授权 reward-dev 固定 token 的 no-cache mask replay；8192 条文本 rollout 未重新采样，Fake 4096 条 token 序列 exact，P1 前后权重 exact。Phase 3D.1 未启动。

最终 selected reward：**Q2**。`REFORMULATED_POLICY_SIGNAL_READY=true`。Primary gate：**`GATE_REFORMULATED_REWARD_SUPPORTED`**。

## 2. 为什么 OLD_R3 降级

OLD_R3 的 `R_phrase_lex` 对所有词近似等权，存在 function-word inflation 与合理 paraphrase underscoring；harmonic OLD_R_align 又把不可靠 lexical score 与受 mask decoder ceiling 限制的绝对 IoU 耦合，导致 phrase-correct/mask-poor trajectory 被重复处罚。因此 OLD_R3 仅保留为 `HISTORICAL_REWARD_CONTROL`，旧 lexical/align 仅作诊断。

## 3. Frozen semantic scorer

Encoder：`sentence-transformers/all-MiniLM-L6-v2`，revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，eval/no-grad/frozen；完整 internal-train authoritative phrase corpus 计算 IDF，validation/test vocabulary 未使用。`R_phrase_sem=.65 R_key_soft + .35 R_sentence_sem`，并显式处罚预注册 spatial contradiction。Sanity status：**PASS**。

Sanity checks：

```json
{
  "S1_identity": true,
  "S2_case_punctuation": true,
  "S3_stopword": true,
  "S4_spatial": true,
  "S5_high_idf": true,
  "S6_synonym": true
}
```

`R_phrase_sem is an automatic frozen semantic proxy, not human semantic ground truth.`

## 4. Relative grounding

`R_ground_rel=C_range*C_ceiling*mask percentile rank`。只有同组 mask range 足够、且最好 mask 达到一定 ceiling 时，mask 才能影响 language reward；absolute `R_mask` 继续只作诊断。Q1 是 language-only；Q2 给 relative grounding 0.10 权重；Q3 的0.20只作 stress test。

## 5. Reward-dev selection

Q2 conditions：

```json
{
  "lm_nonreg": true,
  "structure_nonreg": true,
  "phrase_nonreg": true,
  "high_ground_mask_help": true,
  "low_ground_phrase_nonreg": true,
  "each_failure_not_above_old": true,
  "aggregate_failure_reduction": true
}
```

Reward-dev 在 validation 之前冻结 selected=Q2。Q3 未获准成为 primary。

## 6. Full validation confirmation

| Metric | Greedy | Q2 top |
|---|---:|---:|
| LM verdict accuracy | 0.9860 | 0.9928 |
| Structure validity | 0.9928 | 1.0000 |
| Fake lexical F1 | 0.2999 | 0.4158 |
| Fake semantic phrase | 0.6381 | 0.7660 |
| Fake key-soft | 0.6295 | 0.7388 |
| Fake sentence-sem | 0.7412 | 0.8164 |
| Spatial contradiction rate | 0.0940 | 0.0000 |
| Absolute FG IoU | 0.1482 | 0.2438 |
| FG F1 | 0.2107 | 0.3329 |

Selected top-vs-greedy `R_phrase_sem` mean Δ=0.0639，95% CI=[0.05844430275178728, 0.06950011678176068]；absolute IoU mean Δ=0.0478，95% CI=[0.04172659245761922, 0.053989893746064825]。

## 7. Key diagnostics

- lexical-low / semantic-high trajectories：948
- lexical-high / semantic-low trajectories：9
- spatial contradiction trajectories：886
- phrase-good / mask-poor trajectories：2069
- phrase-good / ground-responsive trajectories：2092
- DOUBLE_PENALTY_CASE：549
- FUNCTION_WORD_INFLATION strict key-soft proxy：0
- FUNCTION_WORD_INFLATION broad lexical-high/semantic-low proxy：9
- PARAPHRASE_UNDERSCORED proxy：948
- SPATIAL_CONTRADICTION_OVERREWARDED：94

这些 automatic subgroup 是 frozen semantic proxy，不是 human ground truth。

## 8. Policy signal and route

```json
{
  "selected_reward": "Q2",
  "conditions": {
    "groups_multiple_unique": true,
    "fake_reward_std": true,
    "classification_nonreg": true,
    "structure_nonreg": true,
    "phrase_sem_positive_ci": true,
    "no_semantic_phrase_collapse": true,
    "spatial_contradiction_not_worse": true,
    "low_controllability_phrase_nonreg": true
  },
  "fraction_groups_multiple_unique": 0.5063291139240507,
  "fraction_fake_groups_reward_std_gt_0.05": 0.7332730560578662,
  "low_controllability_phrase_delta_vs_greedy": 0.1523565343000123,
  "REFORMULATED_POLICY_SIGNAL_READY": true
}
```

最终 gate：**`GATE_REFORMULATED_REWARD_SUPPORTED`**。该 gate 只授权提出 Phase 3D.1 controlled policy optimization；不自动启动。

## 9. Limitations

MiniLM 是通用 frozen embedding proxy，不是 forensic human semantic judge；显式规则仅可靠覆盖预注册 spatial pairs，negation mismatch 因无 deterministic scope parser 只记录、不强罚。best-of-K 仍不是 deployment result；本阶段不能证明 GRPO 必然提升性能。qualitative 页面为确定性抽样，未人工 cherry-pick，共 99 个展示条目。
