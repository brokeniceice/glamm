#!/usr/bin/env python3
"""C2 classification on the frozen final-evaluation OOD manifests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/phase6j0_c2/final_evaluation/classification_ood"
CKPT = ROOT / "checkpoints/phase6j0_c2/best/checkpoint/mp_rank_00_model_states.pt"
CFG = ROOT / "configs/phase6j0_c2_preln_cross_attention.yaml"
PRELAUNCH_PID = OUT / "prelaunch_rank1.pid"
os.environ["PHASE6D3_OOD_OUTPUT"] = str(OUT)
os.environ["PHASE6D3_OOD_CONFIG"] = str(CFG)
os.environ["PHASE6D3_OOD_CHECKPOINT"] = str(CKPT)
os.environ["PHASE6D3_OOD_DEVICES"] = "0,2"
os.environ.pop("PHASE6D3_OOD_DATASETS", None)

from scripts import phase6d3_full_ood as routine
from scripts.phase6j0_c2_evaluate import load_c2_model

routine.load_model = load_c2_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("supervisor", "worker", "finalize"), required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    state = __import__("torch").load(CKPT, map_location="cpu", weights_only=False)
    step, epoch = int(state["optimizer_step"]), int(state["epoch"])
    del state
    routine.EXPECTED_STEP, routine.EXPECTED_EPOCH = step, epoch
    if args.mode == "worker":
        routine.worker(args.rank, args.device, args.batch_size)
        return
    if args.mode == "finalize":
        routine.finalize()
        result = json.loads((OUT / "results.json").read_text())
        result.update(schema="phase6j0_c2_full_classification_ood_v1",
                      model="C2-raw", strictly_matched_to_historical_C1=False)
        routine.dump(OUT / "results.json", result)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    routine.dump(OUT / "protocol.json", {
        "schema": "phase6j0_c2_full_ood_protocol_v1", "status": "FROZEN",
        "checkpoint": str(CKPT.resolve()), "checkpoint_sha256": routine.sha(CKPT),
        "selector": "minimum internal-validation total loss", "threshold": 0.5,
        "datasets": {name: {"manifest": str(path.resolve()), "sha256": routine.sha(path),
                            "count": len(routine.rows(path))}
                     for name, path in routine.MANIFESTS.items()},
        "world_size": 2, "gpu_indices": [0, 2], "batch_size_per_gpu": args.batch_size,
        "partition": "manifest_index_mod_2", "selection_or_tuning": False,
    })
    jobs = []
    prelaunch_pid = None
    if PRELAUNCH_PID.exists():
        candidate = int(PRELAUNCH_PID.read_text().strip())
        command_path = Path(f"/proc/{candidate}/cmdline")
        if command_path.exists():
            command = command_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if ("phase6j0_c2_full_ood.py" in command and
                    "--mode worker --rank 1 --device cuda:2" in command):
                prelaunch_pid = candidate
    for rank, gpu in enumerate((0, 2)):
        if rank == 1 and prelaunch_pid is not None:
            continue
        log = (OUT / f"worker_rank{rank}.log").open("a", buffering=1)
        process = subprocess.Popen([
            sys.executable, __file__, "--mode", "worker", "--rank", str(rank),
            "--device", f"cuda:{gpu}", "--batch-size", str(args.batch_size),
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        jobs.append((process, log))
    codes = []
    for process, log in jobs:
        codes.append(process.wait()); log.close()
    if any(codes):
        raise RuntimeError(f"C2 OOD classification workers failed: {codes}")
    if prelaunch_pid is not None:
        while True:
            stat = Path(f"/proc/{prelaunch_pid}/stat")
            if not stat.exists() or stat.read_text().split(") ", 1)[1][0] == "Z":
                break
            time.sleep(20)
        expected = [(name, routine.MANIFESTS[name]) for name in routine.MANIFESTS]
        complete = all(
            (OUT / "shards" / name / "rank1.complete.json").exists() and
            json.loads((OUT / "shards" / name / "rank1.complete.json").read_text()).get("checkpoint_sha256")
            == routine.sha(CKPT) and
            json.loads((OUT / "shards" / name / "rank1.complete.json").read_text()).get("manifest_sha256")
            == routine.sha(manifest)
            for name, manifest in expected
        )
        if not complete:
            with (OUT / "worker_rank1_recovery.log").open("a", buffering=1) as log:
                code = subprocess.run([
                    sys.executable, __file__, "--mode", "worker", "--rank", "1",
                    "--device", "cuda:2", "--batch-size", str(args.batch_size),
                ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}).returncode
            if code:
                raise RuntimeError(f"C2 OOD classification rank1 recovery failed: {code}")
    routine.finalize()
    result = json.loads((OUT / "results.json").read_text())
    result.update(schema="phase6j0_c2_full_classification_ood_v1",
                  model="C2-raw", strictly_matched_to_historical_C1=False)
    routine.dump(OUT / "results.json", result)


if __name__ == "__main__":
    main()
