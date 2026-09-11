# Phase 6B.1b — C1-Exact AIGI-Holmes Sanity Check

## Verdict

`CONFIRMED`

C1-Exact checkpoint-554 与 LEGION-retrained checkpoint-554 在 AIGI-Holmes official TestSet 的同一 99,999 张图上，使用完全相同的顺序、RGB/CLIP preprocessing、逐图 online BF16 CLIP forward、Fake-positive convention 和 threshold=0.5。

| Model | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|
| C1-Exact | 0.888108881 | 0.976109916 | 0.783895678 | 0.992320000 | 0.007680000 | 0.875090704 |
| LEGION-retrained | 0.888108881 | 0.976109916 | 0.783895678 | 0.992320000 | 0.007680000 | 0.875090704 |

- prediction disagreements: `0`
- maximum absolute probability difference: `0`
- maximum absolute logit difference: `0`
- prediction-head tensors bitwise identical: `true`

## Interpretation

C1-Exact fully reproduces LEGION-retrained on AIGI-Holmes. Phase6B C1-L versus LEGION-retrained differences are attributable to the training/numerical recipe, not classifier architecture. Phase6B.2 CLIP + LLM fusion is allowed.

## Protocol and firewall

- C1-Exact: `/data/yz/groundingLMM_official/outputs/phase6b1a_exact_stage2_control/checkpoints/checkpoint-554`
- LEGION-retrained: `/data/yz/myLISA_storage/checkpoints/phase5a3_legion_retrained/stage2_cls/checkpoints/checkpoint-554`
- manifest SHA256: `380325bc0cbba1e80044d0d2dbb40ff2f788865288bb47402c1d4b7967431982`
- CLIP parameter SHA256: `ea25ce94579902eb0a94c9638c0277360f2b93cc156eb20f2fc4a481653afc17`
- batch size: 1, sample order unchanged
- accessed external datasets: AIGI-Holmes official TestSet only
- no training, threshold tuning, calibration, fusion, or other external dataset access

阶段完成后 STOP。
