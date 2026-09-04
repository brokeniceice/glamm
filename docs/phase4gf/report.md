# Phase 4G-F CSCU-LF Formal Localization Training

## Protocol

Warm-start Phase4G-1S epoch10；CSCU-LF d64/window7/heads4/blocks1保持不变，REFINEMENT_INCLUDED=NO。P1/SAM/CLIP/Phase4C-A/source heads frozen，只训练既有371,803-parameter context/interaction/U branch。全train Fake traversal 8,836/epoch，其中8,690 valid-G0进入optimizer、146 invalid保留accounting且不可rescue，Real=0。Loss严格为 `L_seg + L_relative + L_ranking`，系数1:1:1；L_seg复用2×BCE+0.5×Dice，margin=.1，tau=0.0417200699。AdamW lr1e-4/wd1e-4、batch8、10 epochs、无scheduler/early stopping。

每个epoch只用direct batch=1 DEV G0 selector，invalid IoU=0；highest mean G0、tie取earlier。Phrase/TF/control/utility未参与selection，只在selected checkpoint计算。internal test与official1000未访问。

## Selector

Selected epoch **10**，DEV G0=0.083212。完整十个epoch轨迹见training_history.csv。

## Development results

| Model | G0 | Phrase | TF |
|---|---:|---:|---:|
| P1 | 0.148233 | 0.245798 | 0.342928 |
| Phase4F FORENSIC-RECT | 0.171614 | 0.193573 | 0.205262 |
| Phase4E -teacher | 0.172818 | 0.173424 | 0.178207 |
| CSCU-LF Phase4G-F | 0.083212 | 0.159637 | 0.118920 |

Controls：matched=0.083212，cross=0.073444，shuffle=0.074110，off=0.148132，vacuous=0.148132。

## Paired G0 statistics

- CSCU-LF vs P1：`{"n": 1106, "mean_difference": -0.065021746105512, "median_difference": -0.006360289458050072, "bootstrap_95_ci": [-0.0734920760076785, -0.05687359287598501], "wins": 96, "ties": 392, "losses": 618, "wilcoxon_statistic": 31464.0, "wilcoxon_pvalue": 3.9559318079058825e-68}`
- CSCU-LF vs Phase4F：`{"n": 1106, "mean_difference": -0.08840258882382623, "median_difference": -0.0233680057330136, "bootstrap_95_ci": [-0.09797442948881627, -0.07929217853540972], "wins": 88, "ties": 327, "losses": 691, "wilcoxon_statistic": 27886.0, "wilcoxon_pvalue": 9.71109613230083e-87}`
- matched−cross：`{"n": 1106, "mean_difference": 0.009768119357663627, "median_difference": 0.0, "bootstrap_95_ci": [0.006603594316708165, 0.013086182918200238], "wins": 206, "ties": 717, "losses": 183, "wilcoxon_statistic": 28471.0, "wilcoxon_pvalue": 2.0307966944204404e-05}`
- matched−shuffle：`{"n": 1106, "mean_difference": 0.00910191055691912, "median_difference": 0.0, "bootstrap_95_ci": [0.006053679308338332, 0.012312975575393199], "wins": 203, "ties": 719, "losses": 184, "wilcoxon_statistic": 29155.0, "wilcoxon_pvalue": 0.00014040470159314}`

## Scientific answers

A. Deployable G0是否进一步提高：相对P1为 **NO**，相对Phase4F为 **NO**。  
B. `G0 < Phrase < TF`：**NO**。  
C. 与Phase4F tradeoff：依据同时的G0与language order结果解释，不使用Phrase/TF选模。  
D. matched明显优于cross/shuffle（paired CI lower>0）：**YES**。  
E. off/vacuous direct tensor exact recovery：**PASS**。

## Final gates

```json
{
  "schema": "phase4gf_gate_summary_v1",
  "FORMAL_TRAINING_COMPLETE": "YES",
  "SELECTED_EPOCH": 10,
  "SELECTED_DEV_G0": 0.08321165176109217,
  "LANGUAGE_ORDER_PRESERVED": "NO",
  "CSCULF_BEATS_P1_G0": "NO",
  "CSCULF_BEATS_PHASE4F_G0": "NO",
  "MATCHED_GT_CROSS_SHUFFLE": "YES",
  "VACUOUS_EXACT_RECOVERY": "PASS",
  "SOURCE_HASH_INTEGRITY": "PASS",
  "HELDOUT_FINAL_EVALUATION_JUSTIFIED": "NO",
  "INTERNAL_TEST_ACCESSED": "NO",
  "OFFICIAL1000_ACCESSED": "NO"
}
```

到此严格STOP。未访问internal test/official1000，也未自动启动held-out final evaluation。
