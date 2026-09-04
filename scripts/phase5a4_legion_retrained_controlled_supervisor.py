#!/usr/bin/env python3
"""Persistent two-GPU supervisor for retrained LEGION controlled evaluation."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5a4_legion_retrained_controlled"
STATUS = OUT / "supervisor_status.json"
LOCK = OUT / "supervisor.lock"
LOGS = OUT / "logs"
LEGION_PYTHON = "/home/yz/miniconda3/envs/legion/bin/python"
ANALYSIS_PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"
LOC_RUNNER = ROOT / "scripts/phase5a2_legion_shared_evaluate.py"
CLS_RUNNER = ROOT / "scripts/phase5a4_legion_retrained_cls_evaluate.py"
FINALIZER = ROOT / "scripts/phase5a4_legion_retrained_controlled_finalize.py"
RENDERER = ROOT / "scripts/phase5a4_legion_retrained_controlled_render.py"
LE = ROOT / "checkpoints/phase5a3_legion_retrained/stage1_le"
CLS = ROOT / "checkpoints/phase5a3_legion_retrained/stage2_cls/final_model"
CLIP = "/data/yz/myLISA_storage/checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
MANIFEST = ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl"


def now() -> str: return datetime.now(timezone.utc).isoformat()


def atomic(state: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True); state["updated_at_utc"] = now()
    tmp = STATUS.with_suffix(".json.tmp"); tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n"); tmp.replace(STATUS)


def read(path: Path) -> dict:
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return {}


def lines(path: Path) -> int:
    return sum(1 for row in path.open() if row.strip()) if path.is_file() else 0


def alive(pid: object, token: bytes) -> bool:
    try: command = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ")
    except (OSError, TypeError, ValueError): return False
    return token in command and str(OUT).encode() in command


def environment(gpu: int, port: int) -> dict[str, str]:
    env = os.environ.copy(); env.update({"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": str(gpu), "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    return env


def launch(command: list[str], gpu: int, port: int, log_name: str) -> int:
    LOGS.mkdir(parents=True, exist_ok=True); log = (LOGS / log_name).open("a", buffering=1)
    process = subprocess.Popen(command, cwd=ROOT, env=environment(gpu, port), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    log.close(); return process.pid


def loc_paths(start: int, end: int) -> tuple[Path, Path]:
    base = OUT / "official1000/original/shards"
    return base / f"{start:04d}_{end:04d}.worker.json", base / f"{start:04d}_{end:04d}.predictions.jsonl"


def launch_loc(gpu: int, start: int, end: int) -> int:
    command = [LEGION_PYTHON, str(LOC_RUNNER), "--legion-repo", str(ROOT / "external/LEGION_official"), "--model-dir", str(LE), "--clip-dir", CLIP, "--manifest", str(MANIFEST), "--output-root", str(OUT / "official1000"), "--condition", "original", "--target-kind", "synthscars_union", "--device", "cuda:0", "--start", str(start), "--end", str(end)]
    return launch(command, gpu, 29840 + gpu, f"official1000_{start:04d}_{end:04d}.log")


def ensure_loc(gpu: int, start: int, end: int) -> dict:
    worker_path, prediction_path = loc_paths(start, end); info = read(worker_path); done = lines(prediction_path)
    complete = info.get("status") == "COMPLETE" and done == end - start
    if info.get("status") == "FAILED": raise RuntimeError(f"localization worker failed [{start},{end}): {info.get('exception')}")
    if not complete and not alive(info.get("pid"), b"phase5a2_legion_shared_evaluate.py"):
        info = {"status": "LAUNCHED", "pid": launch_loc(gpu, start, end)}
    return {"gpu": gpu, "start": start, "end": end, "completed": done, "status": "COMPLETE" if complete else info.get("status"), "pid": info.get("pid")}


def ensure_cls() -> dict:
    path = OUT / "classification_worker.json"; info = read(path)
    complete = info.get("status") == "COMPLETE"
    if info.get("status") == "FAILED": raise RuntimeError(f"classification worker failed: {info.get('exception')}")
    if not complete and not alive(info.get("pid"), b"phase5a4_legion_retrained_cls_evaluate.py"):
        command = [LEGION_PYTHON, str(CLS_RUNNER), "--model-dir", str(CLS), "--output-root", str(OUT), "--device", "cuda:0"]
        info = {"status": "LAUNCHED", "pid": launch(command, 2, 29842, "classification.log")}
    progress = {}
    for name, total in (("internal", 2208), ("aigi_test", 1731)):
        progress[name] = {"completed": lines(OUT / "classification" / name / "predictions.jsonl"), "total": total}
    return {"gpu": 2, "status": "COMPLETE" if complete else info.get("status"), "pid": info.get("pid"), "progress": progress}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); lock = LOCK.open("a+")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: return
    started = datetime.now(timezone.utc)
    state = {"schema": "phase5a4_legion_retrained_controlled_supervisor_v1", "status": "STARTED", "started_at_utc": started.isoformat(), "pid": os.getpid(), "estimated_complete_at_utc": (started + timedelta(hours=5)).isoformat(), "schedule": "GPU1 loc[0:500]; GPU2 classifications then loc[500:1000]"}; atomic(state)
    try:
        while True:
            loc0 = ensure_loc(1, 0, 500); cls = ensure_cls()
            loc1 = ensure_loc(2, 500, 1000) if cls["status"] == "COMPLETE" else {"gpu": 2, "start": 500, "end": 1000, "completed": lines(loc_paths(500, 1000)[1]), "status": "WAITING_FOR_CLASSIFICATION"}
            state.update({"status": "RUNNING", "localization": [loc0, loc1], "classification": cls}); atomic(state)
            if loc0["status"] == loc1["status"] == cls["status"] == "COMPLETE": break
            time.sleep(30)
        state["status"] = "FINALIZING"; atomic(state)
        env = os.environ.copy(); env["PYTHONPATH"] = str(ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        subprocess.run([ANALYSIS_PYTHON, str(FINALIZER)], cwd=ROOT, env=env, check=True)
        subprocess.run([ANALYSIS_PYTHON, str(RENDERER)], cwd=ROOT, env=env, check=True)
        if subprocess.run(["git", "-C", str(ROOT / "external/LEGION_official"), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip(): raise RuntimeError("official LEGION repo dirty")
        state.update({"status": "COMPLETE", "completed_at_utc": now(), "results": str((OUT / "results.json").resolve()), "baseline_report": str((ROOT / "docs/p1_reusable_evaluation_baselines.md").resolve())}); atomic(state)
    except BaseException as error:
        state.update({"status": "FAILED_STOP", "exception_type": type(error).__name__, "exception": str(error)}); atomic(state); raise


if __name__ == "__main__": main()
