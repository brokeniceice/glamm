# Phase 4E-0.5 — Execution Readiness

## Checklist

- [x] localization population resolved：8,836 train Fake；Real 不作 empty mask。
- [x] slot-level GT availability resolved：19,173 train refs 均有 id/phrase/polygon。
- [x] union operator frozen：probabilistic OR / soft union。
- [x] K=4 collapse audit defined：active、diversity、overlap、contribution 与 population state。
- [x] student initialization frozen：teacher QG/pyramids/rectification/decoder/head exact copy。
- [x] teacher qualification defined：mask capability + matched/cross/shuffle/zero。
- [x] CLIP/SAM geometry verified：7 synthetic cases PASS。
- [x] rectification block fully specified：original-coordinate cross-attention，gamma=0.01。
- [x] nonzero gradient route verified：Q/K/V/out/projection/gamma 均非零。
- [x] Hungarian cost frozen：region `Dice+BCE`；teacher/student `Dice+logit+(qualified attention)`。
- [x] KD tensor shapes frozen：Q `[B,4,256]`、attention、D2/D4、logit `[B,1,32,32]`。
- [x] KD geometry frozen：teacher/student 同 original-coordinate lattice，continuous logits同32×32。
- [x] loss-gradient audit passed：五项 loss 分别 backward，routing matrix PASS。
- [x] loss scale audit passed：一次 train-only numerical correction 后全部 ratio valid。
- [x] -teacher ablation deconfounded：fresh-init，无 teacher init/KD。
- [x] -forensic ablation deconfounded：no-forensic teacher→no-forensic student。
- [x] K=1 ablation deconfounded：K1 teacher→K1 student。
- [x] FULL-CLIP matched arm defined：同 topology/training，仅 evidence source不同。
- [x] train exposure recalculated：Teacher 44,180；Student 88,360 Fake exposures。
- [x] internal test sealed。
- [x] official1000 sealed。

## Audit evidence

| Audit | Result | Artifact |
|---|---|---|
| population/schema | PASS | `outputs/phase4e05_tf_fdg_hardening/population_audit.json` |
| slot target | PASS | `slot_target_availability.json` |
| geometry | PASS | `geometry_audit.json` |
| feature scale/gamma | PASS，gamma=0.01 | `feature_scale_audit.json` |
| architecture shapes/stop-gradient/slots/params | PASS | `implementation_audit.json` |
| loss gradient routing | PASS | `loss_gradient_routing_audit.json` |
| loss scale | PASS | `loss_scale_audit.json` |
| unit tests | 5 passed | `tests/test_phase4e05_tf_fdg_hardening.py` |

注意：teacher qualification **已定义但尚未执行**，因为必须在未来获授权的 Stage T 完整训练与合法 selector 之后完成。readiness 表示 architecture/protocol/audit 已冻结并可进入单次正式 method study，不表示已授权执行或 teacher 已合格。

```text
TF_FDG_ARCHITECTURE_FROZEN:
YES

PROTOCOL_CAUSALLY_IDENTIFIABLE:
YES

GEOMETRY_VALIDATED:
YES

GRADIENT_ROUTING_VALIDATED:
YES

SLOT_COLLAPSE_RISK_CONTROLLED:
YES

TEACHER_QUALIFICATION_DEFINED:
YES

PHASE_4E1_EXECUTION_READY:
YES — protocol ready; execution still requires explicit user authorization
```

## Stop statement

Phase 4E-0.5 到此停止。未运行 5-epoch teacher、10-epoch student、validation performance comparison、threshold tuning、internal test 或 official1000；不会自动启动 Phase 4E-1。

