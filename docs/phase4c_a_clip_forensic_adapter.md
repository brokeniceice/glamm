# Phase 4C-A — CLIP-Anchored Forensic Adapter Learnability

## Outcome

Final gate: **GATE_CLIP_ANCHORED_FORENSIC_ADAPTER_LEARNABLE**. Evidence Reader authorization: **True**. The hard stop is active; internal test and official1000 remained sealed.

## Selected validation results

| Arm | Selected epoch | Mean FG IoU | Median FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |
|---|---:|---:|---:|---:|---:|---:|
| RAW-CLIP-LINEAR | 19 (historical reuse) | 0.143842 | 0.048468 | 0.209971 | 0.178934 | 0.303553 |
| CLIP-PROJ | 4 | 0.138264 | 0.039946 | 0.202635 | 0.170195 | 0.290884 |
| CLIP-FORENSIC-ADAPTER | 4 | 0.171424 | 0.084346 | 0.247279 | 0.207569 | 0.343780 |

## Paired attribution

Adapter minus projection IoU delta is +0.033160, bootstrap 95% CI [+0.027825, +0.038722], with 562/284/260 wins/ties/losses. Adapter minus raw IoU delta is +0.027581, CI [+0.022586, +0.032726].

## Required answers

1. Phase 3C.1 raw feature is P1's frozen CLIP ViT-L/14-336 hidden layer -2 after CLS removal.
2. Its shape is B x 1024 x 24 x 24 (576 patch tokens).
3. CLIP remained hash invariant: ea25ce94579902eb0a94c9638c0277360f2b93cc156eb20f2fc4a481653afc17.
4. CLIP-PROJ vs raw mean IoU: -0.005578.
5. Adapter vs CLIP-PROJ mean IoU: +0.033160, 95% CI [+0.027825, +0.038722].
6. Adapter vs raw mean IoU: +0.027581, 95% CI [+0.022586, +0.032726].
7. Residual-block attribution supported: True.
8. F_forensic non-collapsed: True.
9. Mean F0-to-F_forensic cosine 0.756873; relative L2 1.047637.
10. Qualitative evidence is mixed rather than uniformly better: the full validation pairing favors the adapter (562 wins, 284 ties, 260 losses), and strong wins recover target regions missed by the controls, but fixed-random/loss cases show over-localization and persistent misses. Thus maps are closer on aggregate, not for every image.
11. Final gate: GATE_CLIP_ANCHORED_FORENSIC_ADAPTER_LEARNABLE.
12. Evidence Reader authorized: True.
