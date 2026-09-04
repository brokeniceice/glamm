# Phase 4H-A Phase4F-Preserving Utility-Gated Rectification

## Frozen protocol

本阶段没有训练、optimizer、可训练模块、gate tuning或sweep。直接复用原Phase4F forensic rectifier selected epoch9，并冻结Phase4G-1S epoch10 `U_F`。不使用CSCU-LF/ECoLaF final-mask fusion。定义为 `S64_adapt = S64 + g * (S64_rect_Phase4F - S64)`，其中 `g=U_F`；先把original-normalized U_F按SAM token coordinates对齐到原Phase4F padded S64 64×64网格。

硬端点：`P1_ENDPOINT_RECOVERY=PASS`；`PHASE4F_ENDPOINT_RECOVERY=PASS`。全1078 valid-G0 embedding逐tensor核验，另有16例decoder bit-exact核验。

## Development results

| Path | G0 | Phrase | TF |
|---|---:|---:|---:|
| P1 | 0.148233 | 0.245798 | 0.342928 |
| Phase4F | 0.171614 | 0.193573 | 0.205262 |
| Adaptive Phase4F | 0.172797 | 0.222843 | 0.255661 |

G0 controls：matched=0.172797，cross=0.145711，shuffle=0.148484，forensic-off=0.148233。

## Paired statistics

- Adaptive vs P1 G0：delta=+0.024564, 95% CI [+0.017816, +0.031539], W/T/L=493/294/319, Wilcoxon p=1.663e-13
- Adaptive vs Phase4F G0：delta=+0.001183, 95% CI [-0.003852, +0.006107], W/T/L=439/292/375, Wilcoxon p=0.0412
- Adaptive vs Phase4F Phrase：delta=+0.029270, 95% CI [+0.023549, +0.035249], W/T/L=636/160/310, Wilcoxon p=6.892e-27
- Adaptive vs Phase4F TF：delta=+0.050398, 95% CI [+0.043837, +0.057183], W/T/L=714/151/241, Wilcoxon p=2.69e-64
- matched−cross：delta=+0.027087, 95% CI [+0.020484, +0.034165], W/T/L=486/295/325, Wilcoxon p=1.42e-14
- matched−shuffle：delta=+0.024313, 95% CI [+0.017958, +0.030975], W/T/L=488/297/321, Wilcoxon p=3.049e-13

## Answers

A. Adaptive G0高于P1：**YES**。  
B. Phrase/TF高于Phase4F：**YES / YES**。  
C. `G0 < Phrase < TF`：**YES**。  
D. matched以paired CI lower>0优于cross/shuffle：**YES**。  
E. forensic-off exact P1：**PASS**。

`PARETO_TRADEOFF_IMPROVEMENT=YES`。若失败，下一步严格限制为分析/重学U_F到Phase4F rectification strength的映射，不回到ECoLaF且不增加模块。

## Final gates

```json
{
  "schema": "phase4ha_gate_summary_v1",
  "EVALUATION_COMPLETE": "YES",
  "P1_ENDPOINT_RECOVERY": "PASS",
  "PHASE4F_ENDPOINT_RECOVERY": "PASS",
  "ADAPTIVE_BEATS_P1_G0": "YES",
  "ADAPTIVE_PHRASE_BEATS_PHASE4F": "YES",
  "ADAPTIVE_TF_BEATS_PHASE4F": "YES",
  "LANGUAGE_ORDER_PRESERVED": "YES",
  "MATCHED_GT_CROSS_SHUFFLE": "YES",
  "FORENSIC_OFF_EXACT_P1": "PASS",
  "FROZEN_SOURCE_INTEGRITY": "PASS",
  "PARETO_TRADEOFF_IMPROVEMENT": "YES",
  "NEXT_STEP": "NONE",
  "INTERNAL_TEST_ACCESSED": "NO",
  "OFFICIAL1000_ACCESSED": "NO"
}
```

到此严格STOP。internal test与official1000未访问。
