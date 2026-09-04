#!/usr/bin/env python3
"""Complete Phase4H-D selected evaluations and comparison after both arms train."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase4hd"
CKPT = Path("/data/yz/groundingLMM_official/checkpoints/phase4hd_rectifier_unfreeze_control")
RUNNER = ROOT / "scripts/phase4hd_rectifier_unfreeze_control.py"
STATE = CKPT / "supervisor_state.json"


def dump(value) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def manifest(arm: str):
    path = CKPT / arm / "execution_manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def wait_training() -> None:
    while True:
        status = {arm: None if manifest(arm) is None else manifest(arm).get("status") for arm in ("r0", "r1")}
        dump({"stage": "WAIT_TRAINING", "arm_status": status, "updated_unix": time.time()})
        if all(value in ("TRAINING_COMPLETE_SELECTOR_FROZEN", "COMPLETE") for value in status.values()):
            selectors = [json.loads((OUT / arm / "selector.json").read_text()) for arm in ("r0", "r1")]
            if selectors[0]["sample_order_hashes"] != selectors[1]["sample_order_hashes"]:
                raise RuntimeError("R0/R1 data-order mismatch")
            return
        time.sleep(15)


def evaluate() -> None:
    jobs = []
    for arm, device in (("r0", "cuda:1"), ("r1", "cuda:2")):
        if (OUT / arm / "dev_results.json").exists():
            continue
        handle = (CKPT / arm / "evaluate.log").open("w", encoding="utf-8")
        process = subprocess.Popen([sys.executable, str(RUNNER), "evaluate", "--arm", arm, "--device", device],
                                   cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        jobs.append((arm, process, handle))
    while jobs:
        active = []
        for arm, process, handle in jobs:
            code = process.poll()
            if code is None:
                active.append((arm, process, handle))
            else:
                handle.close()
                if code != 0:
                    raise RuntimeError(f"{arm} selected evaluation failed")
        jobs = active
        dump({"stage": "EVALUATE_SELECTED", "active_arms": [x[0] for x in jobs], "updated_unix": time.time()})
        if jobs:
            time.sleep(15)


def compare() -> None:
    with (CKPT / "comparison.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run([sys.executable, str(RUNNER), "compare"], cwd=ROOT,
                                stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError("Phase4H-D comparison failed")
    required = [OUT / "comparison.json", OUT / "gate_summary.json", ROOT / "docs/phase4hd/report.md"]
    if not all(path.exists() for path in required):
        raise RuntimeError("Phase4H-D final artifacts incomplete")


def main() -> None:
    dump({"stage": "STARTED", "updated_unix": time.time()})
    wait_training(); evaluate(); compare()
    gates = json.loads((OUT / "gate_summary.json").read_text())
    dump({"stage": "COMPLETE", "gates": gates, "updated_unix": time.time(),
          "internal_test_accessed": False, "official1000_accessed": False, "next_stage_started": False})


if __name__ == "__main__":
    main()
