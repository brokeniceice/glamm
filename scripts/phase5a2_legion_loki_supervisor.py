#!/usr/bin/env python3
"""Wait for the official1000 matrix, then run the authorized LOKI supplement."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_STATUS = ROOT / "outputs/phase5a2_legion_public_reference/supervisor_status.json"
OUT = ROOT / "outputs/phase5a2_legion_loki_reference"
STATUS = OUT / "supervisor_status.json"
PYTHON = "/home/yz/miniconda3/envs/legion/bin/python"
RUNNER = ROOT / "scripts/phase5a2_legion_shared_evaluate.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(state: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True); state["updated_at_utc"] = now()
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n"); temporary.replace(STATUS)


def main() -> None:
    state = {"schema": "phase5a2_legion_loki_supervisor_v1", "status": "WAITING_OFFICIAL1000_MATRIX", "started_at_utc": now()}; write(state)
    try:
        while True:
            upstream = json.loads(MAIN_STATUS.read_text()) if MAIN_STATUS.is_file() else {}
            if upstream.get("status") == "COMPLETE": break
            if upstream.get("status") == "FAILED_STOP": raise RuntimeError(f"upstream failed: {upstream}")
            time.sleep(30)
        state["status"] = "RUNNING_LOKI"; write(state)
        logs = OUT / "logs"; logs.mkdir(parents=True, exist_ok=True)
        base = [PYTHON, str(RUNNER), "--legion-repo", str(ROOT / "external/LEGION_official"),
                "--model-dir", "/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/legion_LE_f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa",
                "--clip-dir", "/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
                "--manifest", str(ROOT / "datasets/LOKI/legion_localization/manifest.jsonl"), "--output-root", str(OUT),
                "--condition", "original", "--target-kind", "loki_bbox_union", "--device", "cuda:0"]
        processes = []
        for gpu, port, start, end in ((1, 29641, 0, 115), (2, 29642, 115, 229)):
            env = os.environ.copy(); env.update({"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": str(gpu), "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
            log = (logs / f"loki_{start:04d}_{end:04d}.log").open("a", buffering=1)
            command = [*base, "--start", str(start), "--end", str(end)]
            processes.append((subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        codes = [process.wait() for process, _ in processes]
        for _, log in processes: log.close()
        if any(codes): raise RuntimeError(f"LOKI worker return codes: {codes}")
        state["status"] = "FINALIZING"; write(state)
        analysis_python = "/home/yz/miniconda3/envs/glamm_official/bin/python"
        subprocess.run([analysis_python, str(ROOT / "scripts/phase5a2_legion_loki_finalize.py")], cwd=ROOT, check=True)
        subprocess.run([analysis_python, str(ROOT / "scripts/phase5a2_legion_loki_render_report.py")], cwd=ROOT, check=True)
        if subprocess.run(["git", "-C", str(ROOT / "external/LEGION_official"), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip():
            raise RuntimeError("official LEGION checkout is not clean")
        state.update({"status": "COMPLETE", "results": str(OUT / "shared_results.json"), "report": str(ROOT / "docs/phase5a2_legion_loki_localization_diagnostic.md")}); write(state)
    except BaseException as error:
        state.update({"status": "FAILED_STOP", "exception_type": type(error).__name__, "exception": str(error)}); write(state)
        raise


if __name__ == "__main__":
    main()
