# Phase 6G.15 — Residual Staged Optimization Audit

## 1. Scientific question
Does base-then-zero-residual training improve the same affine correction class, rather than merely adding capacity or extra updates?

## 2. Function-class equivalence
All final corrections are `support*gamma*W_eff(R2)`. A1/A2 use `W_eff=W_base+W_residual` and can be analytically collapsed to one affine map.

## 3. Arms
- A0: reused G14 direct single, continuous 10 epochs.
- A1: selected epoch5 base frozen with all upstream; zero residual only for epochs6-10.
- A2: base frozen; upstream Rectifier and zero residual co-adapt for epochs6-10.
- C1: A1-matched ownership/restart, but directly updates the existing map.
- C2: A2-matched ownership/restart, but directly updates the existing map.

## 4. Initialization equivalence
All Stage1 arms reuse the exact G14 A1 epoch5 checkpoint. A1/A2 residual weights and biases are zero; stored Stage2 checks have exact zero error.

## 5. Update-budget control
Primary lineage is 5+5 epochs versus A0 10 epochs, with identical global epoch data orders. No extended-budget diagnostic was run. C1/C2 separate restart effects from residual anchoring.

## 6. Trainable ownership
Full parameter names/counts are in each arm `initialization.json`. A1/C1 are frozen-upstream controls; A2/C2 are upstream-adaptive controls.

## 7. DEV results
| Arm | mean IoU | median IoU | mean F1 | global IoU | global F1 |
|---|---:|---:|---:|---:|---:|
| A0 | 0.181284 | 0.088324 | 0.256372 | 0.202926 | 0.337388 |
| A1 | 0.179813 | 0.090295 | 0.255450 | 0.203471 | 0.338140 |
| A2 | 0.184964 | 0.089968 | 0.261463 | 0.205171 | 0.340484 |
| C1 | 0.179316 | 0.091207 | 0.254691 | 0.203371 | 0.338002 |
| C2 | 0.183544 | 0.087630 | 0.259599 | 0.206296 | 0.342033 |

## 8. Paired statistics
- A1 vs A0: delta=-0.001471, CI=[-0.002623951344752645, -0.00030834982179537705], W/T/L=363/284/459, p=0.0003568
- A2 vs A0: delta=+0.003680, CI=[0.0023829611909946214, 0.005029979910195223], W/T/L=512/273/321, p=5.458e-11
- A2 vs A1: delta=+0.005151, CI=[0.0034497895269774734, 0.006874251234457845], W/T/L=532/268/306, p=6.136e-14
- A1 vs C1 residual-isolation control: delta=+0.000498, CI=[-6.162477457252909e-05, 0.0010702181085238778], W/T/L=425/292/389, p=0.1011
- A2 vs C2 residual-isolation control: delta=+0.001420, CI=[-4.439425637352396e-05, 0.0027540978607648436], W/T/L=484/275/347, p=4.657e-05

## 9. Optimization dynamics
Per-epoch loss, DEV IoU, gradient norms, gamma and effective-map norms are saved under each arm. Stage2 uses a fresh identical optimizer/scheduler per arm.

## 10. Residual/base magnitude analysis
```json
{
  "A1": {
    "effective_weight_norm": 5.831964015960693,
    "effective_bias_norm": 0.6633428931236267,
    "residual_weight_norm": 0.5599539279937744,
    "residual_bias_norm": 0.03428111970424652,
    "base_weight_norm": 5.7164812088012695,
    "residual_over_base": 0.09795430302619934,
    "base_residual_cosine": 0.15934205055236816
  },
  "A2": {
    "effective_weight_norm": 5.796289443969727,
    "effective_bias_norm": 0.6615928411483765,
    "residual_weight_norm": 0.5507554411888123,
    "residual_bias_norm": 0.03534388542175293,
    "base_weight_norm": 5.7164812088012695,
    "residual_over_base": 0.09634518623352051,
    "base_residual_cosine": 0.09774592518806458
  }
}
```

## 11. Effective-map comparison
Saved in `results.json`; it contains spectra, ranks, condition numbers and direct-versus-staged matrix similarities.

## 12. R2/Delta dynamics
Saved in `results.json` and per-sample JSONL under `dynamics/`. A1 freezes R2 by construction; A2 permits R2 co-adaptation.

## 13. Historical G11 comparison
Historical G11 main+side IoU 0.185546 is a secondary reference only. Fresh paired conclusions use A0/A1/A2/C1/C2. G14 fresh-vs-G10 absolute drift is recorded in the reproducibility section and does not select a claim.

## 14. Supported interpretation
Decision: **STAGED_GAIN_PRESENT_BUT_RESIDUAL_NOT_ISOLATED**. Restart-matched controls are required before assigning any staged gain specifically to residual anchoring.

## 15. Unsupported interpretation
No result establishes nonlinear capacity, Utility behavior, OOD generalization, or a mandatory two-branch inference architecture. All staged maps remain analytically collapsible.

## 16. Decision
`STAGED_GAIN_PRESENT_BUT_RESIDUAL_NOT_ISOLATED`

`RESIDUAL_STAGED_OPTIMIZATION_EXPLAINS_HISTORICAL_SIDE_GAIN = NO`

COMPLETE STOP. No nonlinear translator was started.
