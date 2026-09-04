# G1-C 执行协议

本次仅执行授权的 reliability calibration preflight。P1、SAM、CLIP、Phase4C-A 与 ECoLaF 全冻结；仅 TRAIN-FIT 更新 LanguageHead/ForensicHead，TRAIN-CAL 仅拟合两个正 scalar temperature，TRAIN-AUDIT 只正式载入一次。dev validation、internal test、official1000 均封存。

固定 seed `3407`；cache `/data/yz/groundingLMM_official/cache/phase4g1/g1c/frozen_sources`；checkpoint `/data/yz/groundingLMM_official/checkpoints/phase4g1_g1c_reliability_preflight`。

## Source of truth hashes

- `docs/phase4g05/07_split_protocol.md` SHA256 `82baaf6942c3808233b975be452d6ea2e5d03a76aef205733dc04f3ca62764f0`
- `docs/phase4g05/08_reliability_protocol.md` SHA256 `967f42cf961e1c03865441c37cff7e73214715600fa5e277ff182090903197e7`
- `docs/phase4g05/10_pcerf_hardened_architecture.md` SHA256 `ae1f57ae8f68312a0077eaf30176846d1309f697df6e090b696f38eb43c10a6a`
- `docs/phase4g05/11_phase4g1_training_proposal_v2.md` SHA256 `41c1518168ccfa0692c5e022efd3619d86a5e6b97e234e810cfb9594072d60ea`
- `docs/phase4g05/12_phase4g05_final_report.md` SHA256 `e11b9f30427e31a647d2098ab58118e00eb35964fbd2dab4417cb4b96cca9392`
- `outputs/phase4g05/gate_summary.json` SHA256 `41ba45f45e779f7334d3d34f2f6d17e481accbfd97c61181ab67b33b2101e2e5`
- `model/pcerf.py` SHA256 `961568299e24b7027795008b71ff7b3fe286c20fb36026d64dab39a7362f0a5c`
