#!/usr/bin/env python3
"""Fail-closed reporting for the consumed, interrupted Phase 4G-1Q AUDIT."""

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase4c_b import file_sha256

OUT = ROOT / "outputs/phase4g1q"
DOC = ROOT / "docs/phase4g1q/phase4g1q_report.md"
HASHES = OUT / "hash_manifest.json"
RESULTS = OUT / "results.json"
GATES = OUT / "gate_summary.json"
FIT = OUT / "fit_history.csv"
CKPT = OUT / "csculf_utility_fit_final.pt"
CAL = OUT / "utility_calibration.json"
ERROR = "TypeError: batch_to() missing 1 required positional argument: 'device'"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


manifest = json.loads(HASHES.read_text())
if manifest["audit_access"]["count"] != 1 or manifest["audit_access"]["status"] != "LOADED_IN_MEMORY":
    raise RuntimeError("expected the consumed interrupted AUDIT marker")
calibration = json.loads(CAL.read_text())
history = list(csv.DictReader(FIT.open()))
for row in manifest["source_files"].values():
    row["sha256_after"] = file_sha256(Path(row["path"]))
    row["unchanged"] = row["sha256_before"] == row["sha256_after"]
source_integrity = all(row["unchanged"] for row in manifest["source_files"].values())

results = {
    "schema": "phase4g1q_results_v1",
    "execution_status": "AUDIT_ABORTED_FAIL_CLOSED",
    "failure_class": "AUDIT_EXECUTION_INTEGRITY",
    "failure": {"exception": ERROR, "location": "cross-image corruption call after formal tensors loaded",
                "audit_rerun_performed": False, "reason_not_rerun": "frozen formal read-exactly-once firewall"},
    "populations": {"FIT": {"total": 5258, "valid": 5171, "invalid": 87},
                    "CAL": {"total": 1127, "valid": 1108, "invalid": 19},
                    "AUDIT": {"total": 1126, "valid": 1108, "invalid": 18, "access_count": 1}},
    "fit": {"status": "COMPLETE", "updates": 6470, "epoch_1": history[0], "epoch_10": history[-1],
            "checkpoint_sha256": file_sha256(CKPT)},
    "calibration": calibration,
    "relative_loss": {"status": "NOT_EVALUATED_DUE_TO_AUDIT_ABORT"},
    "four_state": {"status": "NOT_EVALUATED_DUE_TO_AUDIT_ABORT"},
    "cross_image": {"status": "NOT_EVALUATED_DUE_TO_AUDIT_ABORT"},
    "spatial_shuffle": {"status": "NOT_EVALUATED_DUE_TO_AUDIT_ABORT"},
    "qmf": {"status": "NOT_EVALUATED_DUE_TO_AUDIT_ABORT"},
    "utility_intervention_identity": {"status": "NOT_EVALUATED_DUE_TO_AUDIT_ABORT"},
    "utility_permutation": {"status": "NOT_EVALUATED_DUE_TO_AUDIT_ABORT"},
    "vacuous_exact_recovery": {"status": "NOT_EVALUATED_ON_FORMAL_AUDIT", "phase4g1p_frozen_gate": "PASS"},
    "source_integrity": {"checkpoint_files_unchanged": source_integrity, "source_files": manifest["source_files"]},
    "firewall": {"audit_access_count": 1, "audit_second_read": False, "old_g1c_train_audit_accessed": False,
                 "development_validation_accessed": False, "internal_test_accessed": False, "official1000_accessed": False},
}
dump(RESULTS, results)

gates = {
    "schema": "phase4g1q_gate_summary_v1", "UTILITY_FIT_COMPLETE": "YES",
    "CALIBRATION_STATUS": calibration["CALIBRATION_STATUS"],
    "AUDIT_EXECUTION_INTEGRITY": "FAIL",
    "UTILITY_RELATIVE_LOSS_RELATION": "NOT_EVALUATED",
    "CROSS_IMAGE_UTILITY_RESPONSE": "NOT_EVALUATED",
    "SPATIAL_SHUFFLE_UTILITY_RESPONSE": "NOT_EVALUATED",
    "QMF_CONDITIONAL_RELATION": "NOT_EVALUATED",
    "UTILITY_INTERVENTION_IDENTITY": "NOT_EVALUATED",
    "UTILITY_PERMUTATION_SENSITIVITY": "NOT_EVALUATED",
    "VACUOUS_EXACT_RECOVERY": "NOT_EVALUATED_ON_FORMAL_AUDIT",
    "NO_ORACLE_LEAKAGE": "PASS", "INVALID_G0_POLICY": "PASS",
    "SOURCE_HASH_INTEGRITY": "PASS" if source_integrity else "FAIL",
    "UTILITY_AUDIT_ACCESS_COUNT": 1, "UTILITY_AUDIT_SECOND_READ": "NO",
    "DEVELOPMENT_VALIDATION_ACCESSED": "NO", "INTERNAL_TEST_ACCESSED": "NO", "OFFICIAL1000_ACCESSED": "NO",
    "CONDITIONAL_UTILITY_PREFLIGHT": "FAIL", "FORMAL_TRAINING_JUSTIFIED": "NO", "FORMAL_TRAINING_EXECUTED": "NO",
}
dump(GATES, gates)

report = f"""# Phase 4G-1Q CSCU-LF Conditional-Utility Validity Preflight

## 1. Protocol freeze

Phase 4G-1P 冻结的 architecture、target、tau=0.0417200699、split、corruption、permutation、statistics 与 gate threshold 未修改。没有 sweep、checkpoint selection 或 post-hoc threshold/tau tuning。

## 2. Population and firewall

UTILITY-FIT 5,258（valid 5,171 / invalid 87）；UTILITY-CAL 1,127（1,108 / 19）；UTILITY-AUDIT 1,126（1,108 / 18）。formal AUDIT access count=1，tensors 已装入内存后执行中止。按 frozen read-exactly-once rule，没有第二次读取。旧 G1-C TRAIN-AUDIT、development validation、internal test、official1000 均未访问。

## 3. Training configuration

Seed 3407；AdamW lr=1e-4、weight decay=1e-4；batch=8；10 epochs；grad clip=1；无 scheduler/early stopping。只优化 371,803-parameter context/interaction/U branch；source experts frozen。

## 4. FIT results

`UTILITY_FIT_COMPLETE=YES`。6,470/6,470 updates；loss {history[0]['loss']} → {history[-1]['loss']}。逐 epoch loss、U distribution/saturation、五组 gradient norms 与 finite flags见 fit_history.csv。唯一正式 checkpoint 为 epoch 10 final，SHA256 `{file_sha256(CKPT)}`。

## 5. CAL results

`CALIBRATION_STATUS={calibration['CALIBRATION_STATUS']}`；T_U={calibration['T_U']:.8f}；image-balanced soft BCE {calibration['pre']['objective']:.8f} → {calibration['post']['objective']:.8f}。

## 6. Utility relative-loss validity

`UTILITY_RELATIVE_LOSS_RELATION=NOT_EVALUATED`。Formal tensors 已读取，但 statistic 未能在进程中持久化；禁止以第二次 AUDIT read 补算。

## 7. Four-state diagnostic

`NOT_EVALUATED_DUE_TO_AUDIT_ABORT`。

## 8. Cross-image corruption

在进入 cross-image loop 时发生 supervisor 调用错误：`{ERROR}`。该错误属于执行完整性失败，不是 conditional-utility 机制的正面或负面科学证据。

## 9. Spatial-shuffle corruption

`SPATIAL_SHUFFLE_UTILITY_RESPONSE=NOT_EVALUATED`。

## 10. QMF conditional relation

`QMF_CONDITIONAL_RELATION=NOT_EVALUATED`；不作 intrinsic uncertainty 或 weight-quality claim。

## 11. Utility intervention identity

`UTILITY_INTERVENTION_IDENTITY=NOT_EVALUATED`。Phase 4G-1P synthetic frozen gate仍为 PASS，但不能替代本阶段 formal AUDIT gate。

## 12. Utility permutation causal test

`UTILITY_PERMUTATION_SENSITIVITY=NOT_EVALUATED`；identity 未完成，故 fail-closed 不计算 sensitivity。

## 13. Vacuous / invalid-G0 checks

Formal AUDIT real-tensor fallback check未完成；Phase 4G-1P 的 frozen synthetic `VACUOUS_EXACT_P1=PASS` 保持历史事实。`INVALID_G0_POLICY=PASS`，invalid IDs未训练、未被 forensic rescue；`NO_ORACLE_LEAKAGE=PASS`。

## 14. Gate summary

```json
{json.dumps(gates, ensure_ascii=False, indent=2)}
```

## 15. Scientific interpretation

本阶段确认 frozen conditional-utility objective 可完成 FIT，并可用单一正 temperature 改善 CAL objective；但 single-read AUDIT 的预注册机制证据没有完成。失败类别是 **AUDIT_EXECUTION_INTEGRITY**，不能解释为 relative utility learning、cross-image sensitivity、spatial mismatch sensitivity、QMF weighting 或 causal intervention 本身通过/失败。旧 Intrinsic-PCERF failure保持不变，本阶段不声称 CSCU-LF 缓解该 failure，也不声称 localization 改善。

## 16. Final authorization decision

`CONDITIONAL_UTILITY_PREFLIGHT=FAIL`  
`FORMAL_TRAINING_JUSTIFIED=NO`

Phase 4G-1Q 到此 fail-closed STOP。没有自动正式 CSCU-LF localization training、G0/Phrase/TF、AHBFR、Teacher/KD、internal test 或 official1000。
"""
DOC.parent.mkdir(parents=True, exist_ok=True)
DOC.write_text(report, encoding="utf-8")

manifest.update({"status": "AUDIT_ABORTED_FAIL_CLOSED", "audit_failure": {"exception": ERROR, "rerun": False},
                 "results_sha256": file_sha256(RESULTS), "gate_summary_sha256": file_sha256(GATES),
                 "report_sha256": file_sha256(DOC), "fit_checkpoint_sha256_after": file_sha256(CKPT),
                 "calibration_sha256_after": file_sha256(CAL), "outputs_schema_readable": True,
                 "finalized_unix": time.time()})
manifest["audit_access"].update({"status": "ABORTED_FAIL_CLOSED", "second_read": False})
dump(HASHES, manifest)
print(json.dumps(gates, ensure_ascii=False, indent=2))
