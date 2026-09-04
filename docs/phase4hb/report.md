# Phase 4H-B Adaptive Phase4F Progressive Unfreezing — Stage 1

## Frozen protocol

唯一trainable为2参数bounded mapper `phi(U_F)=sigmoid(logit(clamp(U_F))+Conv1x1(U_F))`；Conv weight/bias均以0初始化。其他P1/SAM/CLIP/Phase4C-A/Phase4F rectifier/Phase4G-1S utility全部冻结。Loss=`L_seg+0.1*L_identity`；AdamW lr1e-4/wd1e-4、batch8、10 epochs、seed3407、无scheduler/early stopping。全train Fake每epoch traversal 8836，8690 valid进入optimizer，146 invalid仅accounting，Real=0。

Epoch0 G0/Phrase/TF=0.172844/0.222852/0.255684；仅作Phase4H-A近恒等baseline，不参与selector。Selector只用epochs1-10 direct batch1 DEV G0，selected epoch=10。

## Development results

| Model | G0 | Phrase | TF |
|---|---:|---:|---:|
| P1 | 0.148233 | 0.245798 | 0.342928 |
| Phase4F | 0.171614 | 0.193573 | 0.205262 |
| Phase4H-A | 0.172797 | 0.222843 | 0.255661 |
| Phase4H-B Stage1 | 0.173483 | 0.220989 | 0.251102 |

Controls：matched=0.173483，cross=0.145389，shuffle=0.148526，off=0.148233。

## Paired comparisons to Phase4H-A

- G0：delta=+0.000685, CI=[+0.000180,+0.001220], W/T/L=415/327/364, p=0.02799
- Phrase：delta=-0.001853, CI=[-0.002477,-0.001257], W/T/L=393/165/548, p=3.424e-08
- TF：delta=-0.004559, CI=[-0.005333,-0.003827], W/T/L=283/159/664, p=1.516e-45
- matched−cross：delta=+0.028094, CI=[+0.021317,+0.035311], W/T/L=493/289/324, p=9.825e-15
- matched−shuffle：delta=+0.024956, CI=[+0.018453,+0.031865], W/T/L=483/292/331, p=9.987e-13

## Mapper and gate diagnostics

Selected mapper weight=+0.06493270，bias=+0.11411364。

- U_F: mean=0.539712, std=0.160502, percentiles={"0": 0.00628662109375, "1": 0.1328125, "5": 0.251953125, "25": 0.484375, "50": 0.51953125, "75": 0.59765625, "95": 0.86328125, "99": 0.94921875, "100": 0.99609375}
- g: mean=0.572485, std=0.158907, percentiles={"0": 0.0070440503768622875, "1": 0.1475962996482849, "5": 0.27732911705970764, "25": 0.5207493305206299, "50": 0.5562639832496643, "75": 0.6338176727294922, "95": 0.8821535706520081, "99": 0.957051694393158, "100": 0.996731162071228}
- mean |g-U_F|=0.032773
- g<=0.01=0.000002; g>=0.99=0.000392
- GATE_COLLAPSE=NO

## Final gates

```json
{
  "schema": "phase4hb_gate_summary_v1",
  "STAGE1_TRAINING_COMPLETE": "YES",
  "SELECTED_EPOCH": 10,
  "SELECTED_DEV_G0": 0.17348278944076714,
  "STAGE1_BEATS_PHASE4HA_G0": "YES",
  "STAGE1_G0_CLEAR_BENEFIT_CI_LOWER_GT_ZERO": "YES",
  "STAGE1_LANGUAGE_ORDER_PRESERVED": "YES",
  "STAGE1_PHRASE_NONDEGRADED": "YES",
  "STAGE1_TF_NONDEGRADED": "YES",
  "MATCHED_GT_CROSS_SHUFFLE": "YES",
  "P1_ENDPOINT_RECOVERY": "PASS",
  "PHASE4F_ENDPOINT_RECOVERY": "PASS",
  "FORENSIC_OFF_EXACT_P1": "PASS",
  "GATE_COLLAPSE": "NO",
  "NEXT_UNFREEZE_STAGE_JUSTIFIED": "YES",
  "CURRENT_BEST_SCHEME": "Phase4H-B Stage1",
  "INTERNAL_TEST_ACCESSED": "NO",
  "OFFICIAL1000_ACCESSED": "NO"
}
```

到此严格STOP。没有自动进入Stage2，internal test与official1000未访问。
