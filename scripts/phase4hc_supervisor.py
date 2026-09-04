#!/usr/bin/env python3
"""Wait for both frozen Phase4H-C arms, then evaluate and compare automatically."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase4hc"
CKPT = Path("/data/yz/groundingLMM_official/checkpoints/phase4hc_direct_utility_arms")
RUNNER = ROOT / "scripts/phase4hc_direct_utility_arms.py"
STATE = CKPT / "supervisor_state.json"


def dump(value) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def manifest(arm: str) -> dict | None:
    path = CKPT / arm / "execution_manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def selector(arm: str) -> dict:
    return json.loads((OUT / arm / "selector.json").read_text())


def wait_training() -> None:
    while True:
        values = {arm: manifest(arm) for arm in ("a1", "a2")}
        status = {arm: None if value is None else value.get("status") for arm, value in values.items()}
        dump({"stage": "WAIT_TRAINING", "arm_status": status, "updated_unix": time.time()})
        if all(value == "TRAINING_COMPLETE_SELECTOR_FROZEN" for value in status.values()):
            return
        if any(value == "COMPLETE" for value in status.values()):
            # A previously completed evaluation is acceptable during recovery.
            if all(value in ("TRAINING_COMPLETE_SELECTOR_FROZEN", "COMPLETE") for value in status.values()):
                return
        time.sleep(15)


def evaluate_missing() -> None:
    hashes = {arm: selector(arm)["sample_order_hashes"] for arm in ("a1", "a2")}
    if hashes["a1"] != hashes["a2"]:
        raise RuntimeError("A1/A2 sample-order contract mismatch")
    jobs = []
    for arm, device in (("a1", "cuda:1"), ("a2", "cuda:2")):
        if (OUT / arm / "dev_results.json").exists():
            continue
        log_path = CKPT / arm / "evaluate.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(RUNNER), "evaluate", "--arm", arm, "--device", device],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        )
        jobs.append((arm, process, handle, str(log_path)))
    while jobs:
        active = []
        for arm, process, handle, log_path in jobs:
            code = process.poll()
            if code is None:
                active.append((arm, process, handle, log_path))
            else:
                handle.close()
                if code != 0:
                    raise RuntimeError(f"{arm} selected evaluation failed; see {log_path}")
        jobs = active
        dump({"stage": "EVALUATE_SELECTED", "active_arms": [row[0] for row in jobs], "updated_unix": time.time()})
        if jobs:
            time.sleep(15)


def compare() -> None:
    log_path = CKPT / "comparison.log"
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [sys.executable, str(RUNNER), "compare"], cwd=ROOT,
            stdout=handle, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"comparison failed; see {log_path}")
    required = [OUT / "comparison.json", OUT / "gate_summary.json", ROOT / "docs/phase4hc/report.md"]
    if not all(path.exists() for path in required):
        raise RuntimeError("comparison exited without complete Phase4H-C artifacts")


def main() -> None:
    dump({"stage": "STARTED", "updated_unix": time.time()})
    wait_training()
    evaluate_missing()
    compare()
    gates = json.loads((OUT / "gate_summary.json").read_text())
    dump({"stage": "COMPLETE", "updated_unix": time.time(), "gates": gates,
          "internal_test_accessed": False, "official1000_accessed": False, "stage2_started": False})


if __name__ == "__main__":
    main()
