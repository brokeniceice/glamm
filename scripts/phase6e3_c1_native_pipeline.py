#!/usr/bin/env python3
"""Detached fail-closed C1-native staged R1 train -> Official1000 -> OOD chain."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase6e3_c1_native_staged as staged

OUT = staged.OUT
STATUS = OUT / "pipeline_status.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if STATUS.exists():
        raise RuntimeError("pipeline already exists; refusing implicit restart")
    state = {"status": "RUNNING", "started_at_utc": now(), "pid": os.getpid(),
             "gpu_assignments": {"gamma_audit": 1, "rectifier": 1, "utility": 2, "joint": 1,
                                 "official1000": 2, "external_ood": [1, 2]},
             "stages": []}
    staged.dump(STATUS, state)
    tasks = [
        ("GAMMA_AUDIT_C1", 1, ["scripts/phase6e3_c1_gamma_audit.py"]),
        ("RECTIFIER_C1", 1, ["scripts/phase6e3_c1_native_staged.py", "rectifier"]),
        ("UTILITY_C1", 2, ["scripts/phase6e3_c1_native_staged.py", "utility"]),
        ("JOINT_C1", 1, ["scripts/phase6e3_c1_native_staged.py", "joint"]),
        ("OFFICIAL1000", 2, ["scripts/phase6e3_c1_native_official.py"]),
        ("EXTERNAL_OOD_GPU1_GPU2", None, ["scripts/final_eval_c1_native_staged_r1.py", "--mode", "supervisor"]),
    ]
    try:
        for name, gpu, args in tasks:
            state["stage"] = name
            item = {"stage": name, "status": "RUNNING", "started_at_utc": now(), "gpu": gpu}
            state["stages"].append(item)
            staged.dump(STATUS, state)
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "PYTHONDONTWRITEBYTECODE": "1"}
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            else:
                env.pop("CUDA_VISIBLE_DEVICES", None)
            path = OUT / "logs" / f"{name.lower()}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", buffering=1) as log:
                process = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                item["pid"] = process.pid
                staged.dump(STATUS, state)
                code = process.wait()
            item.update(status="COMPLETE" if code == 0 else "FAILED", exit_code=code,
                        ended_at_utc=now())
            staged.dump(STATUS, state)
            if code != 0:
                raise RuntimeError(f"{name} exited with {code}; see {path}")
            if name == "RECTIFIER_C1":
                staged.selected_state("rectifier", "rectifier_state")
            elif name == "UTILITY_C1":
                staged.selected_state("utility", "utility_state")
            elif name == "JOINT_C1":
                staged.selected_state("joint", "utility_state")
            elif name == "OFFICIAL1000":
                result = json.loads((OUT / "official1000_results.json").read_text())
                staged.require(result["status"] == "COMPLETE", "Official1000 incomplete")
            elif name == "EXTERNAL_OOD_GPU1_GPU2":
                result = json.loads((ROOT / "outputs/final_evaluation/c1_native_staged_r1/results.json").read_text())
                staged.require(result["status"] == "COMPLETE", "OOD incomplete")
        state["status"] = "COMPLETE"
    except BaseException as exc:
        state.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        state["updated_at_utc"] = now()
        staged.dump(STATUS, state)


if __name__ == "__main__":
    main()
