# Phase 6G.7 — Staged New R1 with Block11+17 Evidence

Status: **COMPLETE STOP**. Stage 1 exactly reused the already completed matched Phase6G.3 A0 adapter pretrain; Rectifier, Utility, and joint stages used the historical matched recipes.

| Arm | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|
| A0 block22 current new R1 | 0.202488 | 0.104586 | 0.283836 | 0.222581 | 0.364117 |
| A1 block11+17 staged new R1 | 0.210203 | 0.105633 | 0.289845 | 0.211857 | 0.349640 |

- paired: `{'n': 1106, 'iou': {'mean_delta': 0.007715002478539033, 'median_delta': 0.0, 'bootstrap_95_ci': [-0.0006194742124850484, 0.01610629445051865], 'wins': 487, 'ties': 185, 'losses': 434, 'wilcoxon_statistic': 194668.0, 'wilcoxon_pvalue': 0.029087123209407716}, 'f1': {'mean_delta': 0.006009363711771219, 'median_delta': 0.0, 'bootstrap_95_ci': [-0.004177208115478557, 0.016372370456543803], 'wins': 487, 'ties': 185, 'losses': 434, 'wilcoxon_statistic': 198043.0, 'wilcoxon_pvalue': 0.07767197600984056}}`
- stage provenance: `{'rectifier': {'status': 'COMPLETE_SELECTED_FROZEN', 'stage': 'rectifier', 'selected_epoch': 5, 'selected_checkpoint_sha256': '69b1af5c0f2008c6d1d4ad5a39fdc33d24d5b97cc469987351a486dda9e0b646'}, 'utility': {'status': 'COMPLETE_SELECTED_FROZEN', 'stage': 'utility', 'selected_epoch': 8, 'selected_checkpoint_sha256': '777bf1cc17e05657259b766e5b331661c68ad6955cbcbe45beb8004a9f1a3f58'}, 'joint': {'status': 'COMPLETE_SELECTED_FROZEN', 'stage': 'joint', 'selected_epoch': 8, 'selected_checkpoint_sha256': '9c640ec9b55c23a7e9778c1a38ad61864856a96b374691c65de379e6bcfc6983'}, 'adapter': {'status': 'EXACT_REUSE', 'checkpoint': '/data/yz/groundingLMM_official/outputs/phase6g3_forensic_adapter_audit/a0/selected.pt', 'sha256': '69cd56bd790b711d7887699868331943d14af41af9db419a032a6305a32eea48', 'selected_epoch': 4}}`
- SEG trigger: `{'n': 1106, 'valid': 1090, 'rate': 0.9855334758758545, 'A0_A1_exact': True}`
- Phase6G.6 reference mean IoU: `0.207724` (not used for selection)

```text
BLOCK11_17_STAGED_R1_NOT_STABLY_BETTER
```
