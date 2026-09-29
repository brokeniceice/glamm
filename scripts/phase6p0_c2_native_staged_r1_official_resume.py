#!/usr/bin/env python3
"""Resume the selected-only Official1000 stage after the BF16 parity audit."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6p0_c2_native_staged_r1"
STATUS = OUT / "pipeline_status.json"
GPU = int(os.environ.get("PHASE6P0_OFFICIAL_GPU", "0"))
LOG = OUT / "logs" / f"official1000_gpu{GPU}.log"
PY = "/home/yz/miniconda3/envs/glamm_official/bin/python"


def write(state, **extra):
    value = {"status": state, "stage": "OFFICIAL1000",
             "updated_utc": datetime.now(timezone.utc).isoformat(),
             "log": str(LOG.resolve()), **extra}
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, STATUS)


def main():
    previous = json.loads(STATUS.read_text())
    if previous["status"] not in ("FAILED", "PAUSED_FOR_GPU_MOVE") or previous["stage"] != "OFFICIAL1000":
        raise RuntimeError("Official1000 resumable state not found")
    if (OUT / "official1000_results.json").exists():
        raise RuntimeError("Official1000 already finalized")
    completed = sum(1 for _ in (OUT / "official1000_records.jsonl").open())
    write("RUNNING", physical_gpu=GPU, completed_prefix=completed,
          reason="Live C2 output is authoritative; standalone BF16 SAM drift audited")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU), "PYTHONUNBUFFERED": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    with LOG.open("a", buffering=1) as handle:
        code = subprocess.run([PY, "-u", str(ROOT / "scripts/phase6p0_c2_native_staged_r1_official.py")],
                              cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env).returncode
    if code:
        write("FAILED", exit_code=code)
        raise RuntimeError(f"Official1000 retry failed with exit code {code}; see {LOG}")
    result = json.loads((OUT / "official1000_results.json").read_text())
    if result["status"] != "COMPLETE_STOP_AFTER_OFFICIAL1000":
        write("FAILED", error="Result completion marker missing")
        raise RuntimeError("Official1000 result completion marker missing")
    write("COMPLETE_STOP_AFTER_OFFICIAL1000", result=str((OUT / "official1000_results.json").resolve()))


if __name__ == "__main__":
    main()
