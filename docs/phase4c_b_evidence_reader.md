# Phase 4C-B — Language-Query Evidence Reader Integration

## Outcome

Final gate: **GATE_SPATIAL_REACCESS_EFFECTIVE_FORENSIC_GAIN_NOT_SEPARABLE**. Phase 4C-A Adapter downstream utilization: **False**. The hard stop is active; no joint adaptation, internal test, or official1000 evaluation was started.

## Fresh canonical validation G0

| Arm | Selected epoch | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|---:|
| P1-FROZEN | 0 | 0.148233 | 0.037966 | 0.210697 | 0.146524 | 0.255597 |
| CLIP-READER | 1 | 0.154334 | 0.047992 | 0.220831 | 0.156928 | 0.271284 |
| FORENSIC-READER | 1 | 0.153858 | 0.047444 | 0.220041 | 0.155893 | 0.269736 |

## Paired attribution

- CLIP Reader minus P1: +0.006100, 95% CI [+0.003041, +0.009170].
- Forensic Reader minus P1: +0.005624, 95% CI [+0.002874, +0.008371].
- Forensic Reader minus CLIP Reader: -0.000476, 95% CI [-0.001034, +0.000095].

## Required answers

1. Reader initialization is exactly P1-equivalent: True.
2. Zero-init beta is correct; q_final equals q_seg at initialization and beta receives a nonzero first-step gradient.
3. Two-step gradient audit passed: True.
4. Language output is exact invariant: True; sequence hash `c568f9800f1fc401d364f3386a4fc493dd5eea3565ba53bc5351ae614478a404`.
5. CLIP-READER significantly exceeds P1: True.
6. FORENSIC-READER significantly exceeds P1: True.
7. FORENSIC-READER significantly exceeds CLIP-READER: False.
8. Spatial re-access itself is effective: True.
9. Forensic specialization provides additional gain: False.
10. Reader attention is spatially non-uniform: CLIP=True, forensic=True; qualitative alignment remains diagnostic rather than directly supervised.
11. Phrase-Only Forensic Reader minus P1 IoU: -0.023775.
12. TF-Full Forensic Reader minus P1 IoU: -0.009896; oracle capability is reduced.
13. Final gate: `GATE_SPATIAL_REACCESS_EFFECTIVE_FORENSIC_GAIN_NOT_SEPARABLE`.
14. Phase 4C-A Adapter was downstream-utilized: False.
15. Recommended next step: do not start joint adaptation; revisit interface after report. No next phase was automatically started.

Phase 3B adapted the existing spatial path to generated replay. Phase 4C-B instead freezes the entire original P1 spatial path and learns only an additive query-to-evidence retrieval interface; it is not a repetition of generated replay.
