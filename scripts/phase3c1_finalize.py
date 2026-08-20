#!/usr/bin/env python3
"""Finalize Phase 3C.1 only after route, confirmatory, qualitative, and tests exist."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3c1_spatial_probe"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    required = [
        "provenance.json", "feature_source_audit.json", "geometry_audit.json",
        "frozen_backbone_hashes.json", "probe_training_manifest.json",
        "selected_checkpoints.json", "val_metrics.json", "persistent476_metrics.json",
        "negative_control_statistics.json", "cross_source_statistics.json", "route_gate.json",
        "final_report.md", "reports/regression_tests.txt", "qualitative/index.html",
        "qualitative/selection_manifest.json",
    ]
    missing = [value for value in required if not (OUT / value).is_file()]
    if missing:
        raise RuntimeError(f"missing required artifacts: {missing}")
    route = json.loads((OUT / "route_gate.json").read_text())
    hashes = json.loads((OUT / "frozen_backbone_hashes.json").read_text())
    training = json.loads((OUT / "probe_training_manifest.json").read_text())
    selected = json.loads((OUT / "selected_checkpoints.json").read_text())
    regression = (OUT / "reports/regression_tests.txt").read_text()
    confirmatory = sorted(str(path.relative_to(OUT)) for path in (OUT / "confirmatory").glob("*/*/metrics.json"))
    gates = {
        "all_five_sources_audited": len(hashes) == 5,
        "all_frozen_parameter_hashes_exact": all(value["exact"] for value in hashes.values()),
        "all_five_probes_completed_20_epochs": len(training) == 5 and all(
            value["status"] == "COMPLETE" and value["full_data_passes"] == 20
            for value in training.values()
        ),
        "selected_probes_hash_exact": len(selected) == 5 and all(
            sha(Path(value["path"])) == value["sha256"] for value in selected.values()
        ),
        "route_frozen_on_val_fake_only": route["status"] == "FROZEN_ON_INTERNAL_VAL_FAKE",
        "persistent_membership_fixed_476": len(json.loads(
            (OUT / "audit/persistent476_ids.json").read_text()
        )["sample_ids"]) == 476,
        "confirmatory_labeled_and_after_gate": len(confirmatory) == 2 * len(
            route.get("confirmatory_sources_authorized_after_gate_freeze", [])
        ),
        "full_regression_passed": " failed" not in regression and " error" not in regression,
        "required_artifacts_present": not missing,
    }
    if not all(gates.values()):
        raise RuntimeError(f"completion gates failed: {gates}")
    report = OUT / "final_report.md"
    docs = ROOT / "docs/phase3c1_frozen_spatial_evidence_probe.md"
    docs.write_bytes(report.read_bytes())
    manifest = {
        "status": "COMPLETE", "phase": "Phase 3C.1 — Frozen Spatial Evidence Probe",
        "route_gate": route["selected_gate"], "completion_gates": gates,
        "confirmatory_artifacts": confirmatory,
        "stop_boundary": "no fusion, decoder, architecture modification, P1/B1 tuning, or GRPO started",
        "artifacts": {value: sha(OUT / value) for value in required},
        "docs_report": {str(docs.relative_to(ROOT)): sha(docs)},
    }
    (OUT / "completion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
