#!/usr/bin/env python3
"""Detached Phase 5A-2 recovery: finish LOKI, then finalize both reports."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5a2_legion_loki_reference"
STATUS = OUT / "autoresume_status.json"
LOCK = OUT / "autoresume.lock"
LOGS = OUT / "logs"
PYTHON = "/home/yz/miniconda3/envs/legion/bin/python"
ANALYSIS_PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"
RUNNER = ROOT / "scripts/phase5a2_legion_shared_evaluate.py"
SHARDS = ((1, 29641, 0, 115), (2, 29642, 115, 229))
BASE = [
    PYTHON,
    str(RUNNER),
    "--legion-repo", str(ROOT / "external/LEGION_official"),
    "--model-dir", "/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/legion_LE_f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa",
    "--clip-dir", "/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
    "--manifest", str(ROOT / "datasets/LOKI/legion_localization/manifest.jsonl"),
    "--output-root", str(OUT),
    "--condition", "original",
    "--target-kind", "loki_bbox_union",
    "--device", "cuda:0",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_status(state: dict) -> None:
    state["updated_at_utc"] = now()
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def worker_path(start: int, end: int) -> Path:
    return OUT / "original/shards" / f"{start:04d}_{end:04d}.worker.json"


def prediction_path(start: int, end: int) -> Path:
    return OUT / "original/shards" / f"{start:04d}_{end:04d}.predictions.jsonl"


def record(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def pid_is_runner(value: object) -> bool:
    try:
        command = Path(f"/proc/{int(value)}/cmdline").read_bytes().replace(b"\0", b" ")
    except (OSError, TypeError, ValueError):
        return False
    return b"phase5a2_legion_shared_evaluate.py" in command and b"loki_bbox_union" in command


def completed_lines(start: int, end: int) -> int:
    path = prediction_path(start, end)
    return sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.is_file() else 0


def launch(gpu: int, port: int, start: int, end: int) -> int:
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
    })
    LOGS.mkdir(parents=True, exist_ok=True)
    log = (LOGS / f"loki_{start:04d}_{end:04d}.log").open("a", buffering=1)
    process = subprocess.Popen(
        [*BASE, "--start", str(start), "--end", str(end)],
        cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
    )
    log.close()
    return process.pid


def run_command(command: list[str]) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    state = {
        "schema": "phase5a2_legion_loki_autoresume_v1",
        "status": "STARTED",
        "started_at_utc": now(),
        "supervisor_pid": os.getpid(),
        "supervisor_sid": os.getsid(0),
    }
    atomic_status(state)
    try:
        while True:
            progress = []
            all_complete = True
            for gpu, port, start, end in SHARDS:
                info = record(worker_path(start, end))
                done = completed_lines(start, end)
                complete = info.get("status") == "COMPLETE" and done == end - start
                if not complete:
                    all_complete = False
                    if not pid_is_runner(info.get("pid")):
                        pid = launch(gpu, port, start, end)
                        info = {"status": "RELAUNCHED", "pid": pid}
                progress.append({
                    "start": start, "end": end, "completed": done,
                    "worker_status": info.get("status"), "worker_pid": info.get("pid"),
                })
            state.update({"status": "RUNNING_LOKI", "workers": progress})
            atomic_status(state)
            if all_complete:
                break
            time.sleep(30)

        state["status"] = "FINALIZING_OFFICIAL1000"
        atomic_status(state)
        run_command([ANALYSIS_PYTHON, str(ROOT / "scripts/phase5a2_legion_finalize.py")])
        run_command([ANALYSIS_PYTHON, str(ROOT / "scripts/phase5a2_legion_render_report.py")])

        state["status"] = "FINALIZING_LOKI"
        atomic_status(state)
        run_command([ANALYSIS_PYTHON, str(ROOT / "scripts/phase5a2_legion_loki_finalize.py")])
        run_command([ANALYSIS_PYTHON, str(ROOT / "scripts/phase5a2_legion_loki_render_report.py")])

        dirty = subprocess.run(
            ["git", "-C", str(ROOT / "external/LEGION_official"), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError("official LEGION checkout is not clean after recovery")
        state.update({
            "status": "COMPLETE",
            "official_results": str((ROOT / "outputs/phase5a2_legion_public_reference/shared_results.json").resolve()),
            "official_report": str((ROOT / "docs/phase5a2_legion_public_reference_results.md").resolve()),
            "loki_results": str((OUT / "shared_results.json").resolve()),
            "loki_report": str((ROOT / "docs/phase5a2_legion_loki_localization_diagnostic.md").resolve()),
        })
        atomic_status(state)
    except BaseException as error:
        state.update({
            "status": "FAILED_STOP",
            "exception_type": type(error).__name__,
            "exception": str(error),
        })
        atomic_status(state)
        raise


if __name__ == "__main__":
    main()
