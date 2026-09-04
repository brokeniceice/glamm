#!/usr/bin/env python3
"""Detached, resumable merge/evaluate/finalize supervisor for Stage-1 epochs."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5a3_stage1_epoch_lfree_validation"
CKPT = ROOT / "checkpoints/phase5a3_legion_retrained"
RUN = CKPT / "stage1_run"
MERGED_ROOT = CKPT / "stage1_epoch_lfree"
BASE = CKPT / "base/GLaMM-GranD-Pretrained"
SAM = ROOT / "checkpoints/phase5b0_fakeshield/models/sam_7790786db131bcdc639f24a915d9f2c331d843ee/checkpoints/sam_vit_h_4b8939.pth"
CLIP = ROOT / "checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
MANIFEST = OUT / "manifests/internal_val_fake.jsonl"
LEGION_PYTHON = Path("/home/yz/miniconda3/envs/legion/bin/python")
ANALYSIS_PYTHON = Path("/home/yz/miniconda3/envs/glamm_official/bin/python")
STATUS = OUT / "supervisor_status.json"
LOCK = OUT / "supervisor.lock"
LOGS = OUT / "logs"
TAGS = {1: "epoch1_global_step561", 2: "epoch2_global_step1122", 3: "epoch3_global_step1683"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(value: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    value["updated_at_utc"] = now()
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def run_logged(command: list[str], log_name: str, env: dict | None = None) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / log_name).open("a", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def merged_dir(epoch: int) -> Path:
    return ROOT / "checkpoints/phase5a3_legion_retrained/stage1_le" if epoch == 3 else MERGED_ROOT / f"epoch{epoch}"


def merge_epoch(epoch: int, state: dict) -> None:
    target = merged_dir(epoch)
    if (target / "pytorch_model.bin.index.json").is_file():
        return
    state.update({"status": "MERGING", "epoch": epoch, "tag": TAGS[epoch]})
    write_status(state)
    fp32_dir = MERGED_ROOT / "merge_inputs" / f"epoch{epoch}"
    fp32 = fp32_dir / "pytorch_model.bin"
    fp32_dir.mkdir(parents=True, exist_ok=True)
    zero = RUN / "deepspeed_checkpoints/zero_to_fp32.py"
    run_logged([str(LEGION_PYTHON), str(zero), str(RUN / "deepspeed_checkpoints"), str(fp32), "--tag", TAGS[epoch]], f"epoch{epoch}_merge_step1.log")
    env = os.environ.copy()
    env.update({"TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(ROOT / "external/LEGION_official")})
    run_logged([
        str(LEGION_PYTHON), str(ROOT / "scripts/phase5a3_merge_stage1.py"),
        "--version", str(BASE), "--weight", str(fp32), "--save_path", str(target),
        "--vision_pretrained", str(SAM), "--vision-tower", str(CLIP),
    ], f"epoch{epoch}_merge_step2.log", env)
    if not (target / "pytorch_model.bin.index.json").is_file():
        raise RuntimeError(f"epoch {epoch} merge incomplete")
    fp32.unlink()
    try:
        fp32_dir.rmdir()
    except OSError:
        pass


def worker_file(epoch: int, start: int, end: int) -> Path:
    return OUT / f"epoch{epoch}/original/shards/{start:04d}_{end:04d}.worker.json"


def records_file(epoch: int, start: int, end: int) -> Path:
    return OUT / f"epoch{epoch}/original/shards/{start:04d}_{end:04d}.predictions.jsonl"


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def lines(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.exists() else 0


def alive(pid) -> bool:
    if not isinstance(pid, int):
        return False
    try:
        text = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        return False
    return "phase5a2_legion_shared_evaluate.py" in text


def launch(epoch: int, gpu: int, start: int, end: int) -> int:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(31000 + epoch * 10 + gpu), "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "PYTHONPATH": str(ROOT / "external/LEGION_official")})
    LOGS.mkdir(parents=True, exist_ok=True)
    log = (LOGS / f"epoch{epoch}_{start:04d}_{end:04d}.log").open("a", buffering=1)
    command = [
        str(LEGION_PYTHON), str(ROOT / "scripts/phase5a2_legion_shared_evaluate.py"),
        "--legion-repo", str(ROOT / "external/LEGION_official"), "--model-dir", str(merged_dir(epoch)),
        "--clip-dir", str(CLIP), "--manifest", str(MANIFEST), "--output-root", str(OUT / f"epoch{epoch}"),
        "--condition", "original", "--target-kind", "internal_val_union", "--start", str(start), "--end", str(end), "--device", "cuda:0",
    ]
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    log.close()
    return process.pid


def evaluate_epoch(epoch: int, state: dict) -> None:
    shards = ((1, 0, 553), (2, 553, 1106))
    while True:
        complete = True
        progress = []
        for gpu, start, end in shards:
            info = load(worker_file(epoch, start, end))
            done = lines(records_file(epoch, start, end))
            ok = info.get("status") == "COMPLETE" and done == end - start
            if not ok:
                complete = False
                if info.get("status") == "FAILED":
                    raise RuntimeError(f"epoch {epoch} shard {start}:{end} failed: {info.get('exception')}")
                if not alive(info.get("pid")):
                    info = {"status": "LAUNCHED", "pid": launch(epoch, gpu, start, end)}
            progress.append({"gpu": gpu, "start": start, "end": end, "completed": done, "status": info.get("status"), "pid": info.get("pid")})
        state.update({"status": "EVALUATING", "epoch": epoch, "workers": progress})
        write_status(state)
        if complete:
            return
        time.sleep(30)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"schema": "phase5a3_stage1_epoch_lfree_supervisor_v1", "status": "PREFLIGHT", "started_at_utc": now(), "pid": os.getpid(), "epochs": [1, 2, 3]}
    write_status(state)
    try:
        run_logged([str(ANALYSIS_PYTHON), str(ROOT / "scripts/phase5a3_epoch_lfree_prepare.py")], "prepare.log")
        for epoch in (1, 2, 3):
            merge_epoch(epoch, state)
            evaluate_epoch(epoch, state)
        state["status"] = "FINALIZING"
        write_status(state)
        run_logged([str(ANALYSIS_PYTHON), str(ROOT / "scripts/phase5a3_epoch_lfree_finalize.py")], "finalize.log")
        state.update({"status": "COMPLETE", "completed_at_utc": now(), "results": str((OUT / "results.json").resolve()), "report": str((ROOT / "docs/phase5a3_stage1_epoch_lfree_validation.md").resolve())})
        write_status(state)
    except BaseException as exc:
        state.update({"status": "FAILED_STOP", "exception_type": type(exc).__name__, "exception": str(exc)})
        write_status(state)
        raise


if __name__ == "__main__":
    main()
