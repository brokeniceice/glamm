#!/usr/bin/env python3
"""Resume the authorized Phase 5A-2 condition matrix on two idle GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/yz/miniconda3/envs/legion/bin/python"
RUNNER = ROOT / "scripts/phase5a2_legion_shared_evaluate.py"
FINALIZER = ROOT / "scripts/phase5a2_legion_finalize.py"
RENDERER = ROOT / "scripts/phase5a2_legion_render_report.py"
OUT = ROOT / "outputs/phase5a2_legion_public_reference"
CONDITIONS = ("original", "jpeg70", "jpeg80", "gaussian5", "gaussian10")
BASE_ARGS = [
    "--legion-repo", str(ROOT / "external/LEGION_official"),
    "--model-dir", "/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/legion_LE_f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa",
    "--clip-dir", "/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
    "--manifest", str(ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl"),
    "--output-root", str(OUT), "--device", "cuda:0",
]
STATUS = OUT / "supervisor_status.json"


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(state: dict) -> None:
    state["updated_at_utc"] = stamp()
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def worker_records(condition: str) -> list[dict | None]:
    values: list[dict | None] = []
    for start, end in ((0, 500), (500, 1000)):
        path = OUT / condition / "shards" / f"{start:04d}_{end:04d}.worker.json"
        values.append(json.loads(path.read_text()) if path.is_file() else None)
    return values


def worker_status(condition: str) -> list[str | None]:
    return [record.get("status") if record else None for record in worker_records(condition)]


def worker_pid_alive(record: dict | None) -> bool:
    if not record or record.get("status") != "RUNNING":
        return False
    try:
        pid = int(record["pid"])
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return b"phase5a2_legion_shared_evaluate.py" in command


def run_pair(condition: str, state: dict) -> None:
    records = worker_records(condition)
    if all(record and record.get("status") == "COMPLETE" for record in records):
        return
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen, object, int, int]] = []
    for record, (physical_gpu, port, start, end) in zip(records, ((1, 29541, 0, 500), (2, 29542, 500, 1000))):
        if record and record.get("status") == "COMPLETE":
            continue
        if worker_pid_alive(record):
            continue
        env = os.environ.copy()
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": str(physical_gpu), "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
        command = [PYTHON, str(RUNNER), *BASE_ARGS, "--condition", condition, "--start", str(start), "--end", str(end)]
        log = (logs / f"{condition}_{start:04d}_{end:04d}.log").open("a", buffering=1)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((process, log, start, end))
    try:
        while True:
            status = worker_status(condition)
            state.update({"status": "RUNNING", "condition": condition, "workers": status})
            write(state)
            if all(value == "COMPLETE" for value in status):
                return
            if any(value == "FAILED" for value in status):
                raise RuntimeError(f"worker failed: {condition}: {status}")
            for process, _, start, end in processes:
                code = process.poll()
                if code not in (None, 0):
                    raise RuntimeError(f"worker return code: {condition} [{start}, {end}): {code}")
            time.sleep(30)
    finally:
        for _, log, _, _ in processes:
            log.close()


def main() -> None:
    state = {"schema": "phase5a2_legion_supervisor_v1", "status": "STARTED", "started_at_utc": stamp()}
    write(state)
    try:
        for condition in CONDITIONS:
            run_pair(condition, state)
        state["status"] = "FINALIZING"; write(state)
        subprocess.run(["/home/yz/miniconda3/envs/glamm_official/bin/python", str(FINALIZER)], cwd=ROOT, check=True)
        subprocess.run(["/home/yz/miniconda3/envs/glamm_official/bin/python", str(RENDERER)], cwd=ROOT, check=True)
        if subprocess.run(["git", "-C", str(ROOT / "external/LEGION_official"), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip():
            raise RuntimeError("official LEGION checkout is not clean after evaluation")
        state["status"] = "COMPLETE"; state["results"] = str(OUT / "shared_results.json"); state["report"] = str(ROOT / "docs/phase5a2_legion_public_reference_results.md"); write(state)
    except BaseException as error:
        state.update({"status": "FAILED_STOP", "exception_type": type(error).__name__, "exception": str(error)}); write(state)
        raise


if __name__ == "__main__":
    main()
