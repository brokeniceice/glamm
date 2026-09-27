#!/usr/bin/env python3
"""Detached GPU1/GPU2 internal-only matched I-JEPA experiment supervisor."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6i_ijepa_r1"
WEIGHT = Path("/data/yz/groundingLMM_official/checkpoints/ijepa/IN1K-vit.h.16-448px-300e.pth.tar")
URL = "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar"
WEIGHT_BYTES = 10_367_908_521


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def run(name, command, gpu=None, arm=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if arm:
        env["PHASE6I_ARM"] = arm
    log = OUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as f:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        code = process.wait()
    if code:
        raise RuntimeError(f"{name} exited {code}; see {log}")


def arm_pipeline(arm):
    gpu = 1 if arm == "A0" else 2
    status_path = OUT / arm / "arm_status.json"
    status = {"status": "RUNNING", "arm": arm, "gpu": gpu,
              "started_at_utc": now(), "stages": [], "official1000_enabled": False,
              "external_ood_enabled": False}
    dump(status_path, status)
    try:
        for stage in ("rectifier", "utility", "joint"):
            entry = {"stage": stage, "status": "RUNNING", "started_at_utc": now()}
            status["stages"].append(entry)
            dump(status_path, status)
            run(f"{arm}_{stage}", [sys.executable, "scripts/phase6i_ijepa_staged.py", stage], gpu, arm)
            selector = json.loads((OUT / arm / stage / "selector.json").read_text())
            if selector["status"] != "COMPLETE":
                raise RuntimeError(f"{arm}/{stage} selector incomplete")
            entry.update(status="COMPLETE", selected_epoch=selector["selected_epoch"],
                         selected_checkpoint_sha256=selector["selected_checkpoint_sha256"],
                         ended_at_utc=now())
            dump(status_path, status)
        status["status"] = "COMPLETE"
    except BaseException as exc:
        status.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status["updated_at_utc"] = now()
        dump(status_path, status)


def orchestrate():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    status_path = OUT / "pipeline_status.json"
    if status_path.exists():
        raise RuntimeError("phase6i pipeline already exists; refusing implicit restart")
    status = {"status": "RUNNING", "started_at_utc": now(),
              "scope": "internal DEV only; no Official1000, external OOD, or internal test",
              "A0_gpu": 1, "A1_gpu": 2, "steps": []}
    dump(status_path, status)
    a0 = a1 = None
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PHASE6I_ARM="A0")
        a0 = subprocess.Popen([sys.executable, __file__, "--arm", "A0"], cwd=ROOT,
                              env=env, start_new_session=True,
                              stdout=(OUT / "logs" / "A0_supervisor.log").open("a"),
                              stderr=subprocess.STDOUT)
        status["A0_pid"] = a0.pid
        status["step"] = "DOWNLOAD_OFFICIAL_IJEPA"
        dump(status_path, status)
        if not WEIGHT.is_file() or WEIGHT.stat().st_size != WEIGHT_BYTES:
            run("download", ["curl", "-L", "--fail", "--silent", "--show-error", "--retry", "4",
                             "--continue-at", "-", "--output", str(WEIGHT), URL])
        if WEIGHT.stat().st_size != WEIGHT_BYTES:
            raise RuntimeError("I-JEPA checkpoint size mismatch")
        for worker in (0, 1):
            status["step"] = f"IJEPA_CACHE_WORKER_{worker}"
            dump(status_path, status)
            run(f"cache_worker_{worker}", [sys.executable, "scripts/phase6i_ijepa_cache.py",
                                           "--worker", str(worker)], gpu=2)
        status["step"] = "A1_STAGED_TRAINING"
        dump(status_path, status)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PHASE6I_ARM="A1")
        a1 = subprocess.Popen([sys.executable, __file__, "--arm", "A1"], cwd=ROOT,
                              env=env, start_new_session=True,
                              stdout=(OUT / "logs" / "A1_supervisor.log").open("a"),
                              stderr=subprocess.STDOUT)
        status["A1_pid"] = a1.pid
        dump(status_path, status)
        a0_code, a1_code = a0.wait(), a1.wait()
        if a0_code or a1_code:
            raise RuntimeError("A0/A1 staged training failed; see arm_status.json and logs")
        for arm, gpu in (("A0", 1), ("A1", 2)):
            status["step"] = f"{arm}_DEV_REPLAY"
            dump(status_path, status)
            run(f"{arm}_dev_replay", [sys.executable, "scripts/phase6i_ijepa_finalize.py", "replay"], gpu, arm)
        status["step"] = "PAIRED_DEV_FINALIZE"
        dump(status_path, status)
        run("paired_dev", [sys.executable, "scripts/phase6i_ijepa_finalize.py", "paired"], arm="A0")
        result = json.loads((OUT / "internal_dev_paired.json").read_text())
        if result["status"] != "COMPLETE_STOP":
            raise RuntimeError("paired internal DEV finalizer incomplete")
        status.update(status="COMPLETE_STOP", gate_passed=result["gate_passed"])
    except BaseException as exc:
        status.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        for process in (a0, a1):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        raise
    finally:
        status["updated_at_utc"] = now()
        dump(status_path, status)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("A0", "A1"))
    args = p.parse_args()
    if args.arm:
        arm_pipeline(args.arm)
    else:
        orchestrate()


if __name__ == "__main__":
    main()
