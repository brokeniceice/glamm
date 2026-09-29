#!/usr/bin/env python3
"""Gate the authorized R2 run on complete caches and stop after internal DEV."""
from __future__ import annotations

import json
import argparse
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.phase6l0_r2_preflight import OUT, EXPECTED_SHA, CKPT, require, dump
from tools.phase4c_b import file_sha256

RESULTS = OUT.parent
PY = "/home/yz/miniconda3/envs/glamm_official/bin/python"
def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def status(stage: str, **details) -> None:
    dump(RESULTS / "supervisor_status.json", {
        "status": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **details,
        "firewall": {"internal_test": False, "official1000": False, "localization_ood": False}})


def run_script(name: str, script: str, *, device: str | None = None) -> None:
    log_path = RESULTS / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, script] + (["--device", device] if device else [])
    status(name.upper(), command=cmd, log=str(log_path))
    with log_path.open("a", buffering=1) as log:
        result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                                     "TRANSFORMERS_OFFLINE": "1"})
    require(result.returncode == 0, f"{name} exited {result.returncode}; see {log_path}")


def main(pids: dict[str, int], train_device: str) -> None:
    subset = json.loads((OUT / "preflight_subset.json").read_text())
    protocol = json.loads((OUT / "capture_protocol.json").read_text())
    require(subset["status"] == "PASS" and subset["c2_sha256"] == EXPECTED_SHA and
            subset["R2_step0_SAM_exact"] and protocol["capture"] == "A_E_r_prime_q_seg_same_C2_forward" and
            file_sha256(CKPT) == EXPECTED_SHA, "C2 or R2 subset preflight drift")
    for _ in range(24 * 60 * 12):  # at most 12 days; the worker liveness gate fails earlier
        completed = {}
        for split in ("train", "val"):
            path = OUT / f"{split}.json"
            if path.exists():
                summary = json.loads(path.read_text())
                require(summary["status"] == "COMPLETE" and summary["c2_sha256"] == EXPECTED_SHA,
                        f"{split} R2 cache summary invalid")
                completed[split] = summary
            else:
                require(alive(pids[split]), f"{split} R2 cache worker died before completion")
        if len(completed) == 2:
            break
        status("WAITING_FOR_C2_NATIVE_CACHE", train_done="train" in completed,
               dev_done="val" in completed, cache_pids=pids)
        time.sleep(60)
    else:
        raise RuntimeError("R2 cache deadline exceeded")
    require(completed["train"]["n"] == 8836 and completed["val"]["n"] == 1106,
            "canonical TRAIN/DEV population mismatch")
    status("CACHES_COMPLETE", train_valid=completed["train"]["valid_c2_g0"],
           dev_valid=completed["val"]["valid_c2_g0"])
    run_script("train", "scripts/phase6l0_r2_train.py", device=train_device)
    run_script("finalize", "scripts/phase6l0_r2_finalize.py")
    final = json.loads((RESULTS / "final_internal_dev.json").read_text())
    require(final["status"] == "COMPLETE_STOP_AFTER_INTERNAL_DEV", "final R2 result incomplete")
    status("COMPLETE_STOP_AFTER_INTERNAL_DEV", selected_epoch=final["selected_epoch"],
           final_result=str(RESULTS / "final_internal_dev.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pid", type=int, required=True)
    parser.add_argument("--val-pid", type=int, required=True)
    parser.add_argument("--train-device", default="cuda:0")
    args = parser.parse_args()
    try:
        main({"train": args.train_pid, "val": args.val_pid}, args.train_device)
    except Exception as error:
        status("FAILED", error=str(error), traceback=traceback.format_exc())
        raise
