# Phase 6F.4 — Full-FOV training results

| Arm | Internal-val mean IoU | Internal-val mean F1 | Official1000 mean IoU | Official1000 mean F1 |
|---|---:|---:|---:|---:|
| Full-FOV I2 | 0.204031 | 0.285173 | 0.317837 | 0.437064 |
| Full-FOV staged | 0.203822 | 0.285474 | 0.310626 | 0.428976 |

Both selectors were frozen using internal-validation canonical G0 only. Coverage, aspect-ratio, tile-count strata and paired bootstrap CIs versus current selected new R1 are preserved in each `comparison.json`. No internal test or OOD was accessed.

`COMPLETE_STOP`
