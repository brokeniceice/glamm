# Phase 4H-D Rectifier Unfreeze Paired Control

## Pre-training rectifier audit

`GeometryAwareSAMRectifier(
  (rectification): CrossAttentiveSemanticRectification(
    (semantic_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    (forensic_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    (cross_attention): GeometryAwareCrossAttention(
      (q_proj): Linear(in_features=256, out_features=256, bias=True)
      (k_proj): Linear(in_features=256, out_features=256, bias=True)
      (v_proj): Linear(in_features=256, out_features=256, bias=True)
      (out_proj): Linear(in_features=256, out_features=256, bias=True)
    )
    (projection): Linear(in_features=256, out_features=256, bias=True)
  )
)`

- Rectifier parameters: 329985
- Inputs: `{"image_embeddings": [1, 256, 64, 64], "evidence": [1, 256, 24, 24], "sam_coordinates": [1, 4096, 2], "evidence_coordinates": [1, 576, 2], "evidence_valid": [1, 576]}`
- Outputs: `{"image_embeddings": [1, 256, 64, 64], "residual": [1, 256, 64, 64], "attention": [1, 8, 4096, 576], "support": [1, 4096]}`

## Frozen protocol

共同起点为Phase4H-C A2 selected epoch3。R0训练CSCU utility并冻结原Phase4F rectifier；R1训练相同CSCU utility及完整原Phase4F rectifier。两臂数据顺序、seed、loss、optimizer、LR、epoch、corruption与DEV G0 selector完全相同，唯一差异为rectifier是否进入optimizer。Phrase/TF及controls未参与selector。

## Results

| Arm | Selected epoch | G0 | Phrase | TF |
|---|---:|---:|---:|---:|
| A2 start | 3 | 0.179532 | 0.224736 | 0.252063 |
| R0 | 3 | 0.179644 | 0.222490 | 0.248897 |
| R1 | 9 | 0.195477 | 0.208660 | 0.215622 |

- R1 vs R0 G0: delta=+0.015833, CI=[+0.009060,+0.022501], W/T/L=471/251/384, Wilcoxon p=6.321e-06
- R1 vs R0 Phrase: delta=-0.013830, CI=[-0.021390,-0.006542], W/T/L=416/139/551, Wilcoxon p=0.0001526
- R1 vs R0 TF: delta=-0.033274, CI=[-0.041018,-0.025557], W/T/L=374/141/591, Wilcoxon p=2.696e-17
- R0 vs A2 G0: delta=+0.000113, CI=[-0.000738,+0.000951], W/T/L=402/309/395, Wilcoxon p=0.6179
- R1 vs A2 G0: delta=+0.015946, CI=[+0.009003,+0.022804], W/T/L=479/251/376, Wilcoxon p=9.052e-06

## Gates

```json
{
  "schema": "phase4hd_gate_summary_v1",
  "R0_COMPLETE": "YES",
  "R1_COMPLETE": "YES",
  "IDENTICAL_DATA_ORDER": "PASS",
  "R0_LANGUAGE_ORDER_PRESERVED": "YES",
  "R1_LANGUAGE_ORDER_PRESERVED": "YES",
  "R0_MATCHED_GT_CROSS_SHUFFLE": "YES",
  "R1_MATCHED_GT_CROSS_SHUFFLE": "YES",
  "R0_GATE_COLLAPSE": "NO",
  "R1_GATE_COLLAPSE": "NO",
  "R0_SOURCE_INTEGRITY": "PASS",
  "R1_SOURCE_INTEGRITY": "PASS",
  "R0_RECTIFIER_FROZEN_INTEGRITY": "PASS",
  "R1_RECTIFIER_CHANGED": "YES",
  "RECTIFIER_UNFREEZE_BENEFICIAL": "YES",
  "FORENSIC_DOMINANCE_RISK": "YES",
  "INTERNAL_TEST_ACCESSED": "NO",
  "OFFICIAL1000_ACCESSED": "NO"
}
```

结论只依据R1对R0的paired comparison归因rectifier解冻收益。到此严格STOP，未访问internal test或official1000。
