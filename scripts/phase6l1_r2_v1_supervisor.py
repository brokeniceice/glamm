#!/usr/bin/env python3
"""Run the single authorized Phase6L1 arm through internal DEV, then stop."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.phase6l0_r2_preflight import OUT as L0_CACHE, EXPECTED_SHA, CKPT, dump, require
from tools.phase4c_b import file_sha256

RESULTS = ROOT / "outputs/phase6l1_r2_v1"
PY = "/home/yz/miniconda3/envs/glamm_official/bin/python"


def status(stage: str, **details) -> None:
    dump(RESULTS / "supervisor_status.json", {
        "status": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **details,
        "firewall": {"internal_test": False, "official1000": False, "localization_ood": False}})


def run_script(name: str, script: str, *, device: str | None = None) -> None:
    log = RESULTS / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, script] + (["--device", device] if device else [])
    status(name.upper(), command=cmd, log=str(log))
    with log.open("a", buffering=1) as output:
        finished = subprocess.run(cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PYTHONUNBUFFERED": "1",
                                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    require(finished.returncode == 0, f"{name} exited {finished.returncode}; see {log}")


def main(device: str) -> None:
    require(not (RESULTS / "training_status.json").exists() and
            not (RESULTS / "selector.json").exists(), "Phase6L1 artifacts already exist; inspect before relaunch")
    require(file_sha256(CKPT) == EXPECTED_SHA, "C2 checkpoint identity drift")
    for split, n in (("train", 8836), ("val", 1106)):
        manifest = json.loads((L0_CACHE / f"{split}.json").read_text())
        require(manifest["status"] == "COMPLETE" and manifest["n"] == n and
                manifest["c2_sha256"] == EXPECTED_SHA, f"immutable L0 {split} cache incomplete")
    run_script("preflight", "scripts/phase6l1_r2_v1_preflight.py", device=device)
    run_script("train", "scripts/phase6l1_r2_v1_train.py", device=device)
    run_script("finalize", "scripts/phase6l1_r2_v1_finalize.py")
    final = json.loads((RESULTS / "final_internal_dev.json").read_text())
    require(final["status"] == "COMPLETE_STOP_AFTER_INTERNAL_DEV", "Phase6L1 final artifact incomplete")
    status("COMPLETE_STOP_AFTER_INTERNAL_DEV", selected_epoch=final["selected_epoch"],
           final_result=str(RESULTS / "final_internal_dev.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-device", default="cuda:0")
    args = parser.parse_args()
    try:
        main(args.train_device)
    except Exception as error:
        status("FAILED", error=str(error), traceback=traceback.format_exc())
        raise
