#!/usr/bin/env python3
"""Freeze the Phase 3C.0 completion manifest after all read-only gates pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3c0_residual_diagnosis"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    provenance_path = OUT / "provenance.json"
    provenance = load(provenance_path)
    frozen = load(OUT / "frozen_model_hash.json")
    gate = load(OUT / "route_gate.json")
    invariance = load(OUT / "audit/invariance.json")
    regression = OUT / "reports/regression_tests.txt"
    report = OUT / "final_report.md"
    required = [regression, report, ROOT / "docs/phase3c0_residual_localization_diagnosis.md",
                OUT / "human_review/index.html", OUT / "human_review/human_review.csv",
                OUT / "qualitative/index.html", OUT / "paired/paired_conditions.jsonl"]
    missing = [str(path) for path in required if not path.is_file()]
    row_count = sum(1 for line in required[-1].read_text(encoding="utf-8").splitlines() if line)
    tests_passed = "failed" not in regression.read_text(encoding="utf-8").lower()
    gates = {
        "evaluation_complete_1106": provenance.get("status") == "EVALUATION_COMPLETE" and row_count == 1106,
        "checkpoint_hash_exact": frozen["exact_checkpoint_identity"],
        "model_state_hash_exact": frozen["exact_model_state_identity"] and not frozen["model_hash_skipped_preflight_only"],
        "canonical_trace_binary_exact_1106": invariance["trace_vs_canonical_binary_exact_count"] == 1106,
        "repair_token_invariance": (
            invariance["repair_prefix_exact_count"] == invariance["repair_eligible_count"]
            and invariance["repair_seg_suffix_exact_count"] == invariance["repair_eligible_count"]
        ),
        "route_val_fake_only": invariance["route_uses_only_internal_val_fake"],
        "full_regression_passed": tests_passed,
        "required_artifacts_present": not missing,
        "no_training_started": provenance["read_only_no_training"],
    }
    if not all(gates.values()):
        raise RuntimeError(f"Phase3C0 completion gates failed: {gates}; missing={missing}")
    manifest = {
        "status": "COMPLETE", "phase": "Phase 3C.0", "num_val_fake": row_count,
        "completion_gates": gates, "supported_route_gates": gate["supported_gates"],
        "no_training_started": True,
        "artifacts": {str(path.relative_to(ROOT)): sha(path) for path in required},
    }
    (OUT / "completion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    provenance["status"] = "COMPLETE"
    provenance["completion_manifest"] = str((OUT / "completion_manifest.json").resolve())
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
