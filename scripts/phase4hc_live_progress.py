#!/usr/bin/env python3
"""Read-only process/history monitor for the already-running Phase4H-C arms."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase4hc"
CKPT = Path("/data/yz/groundingLMM_official/checkpoints/phase4hc_direct_utility_arms")
LOG = CKPT / "live_progress.jsonl"


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def last_history(arm: str):
    path = OUT / arm / "training_history.csv"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return rows[-1] if rows else None
    except FileNotFoundError:
        return None


def processes():
    result = subprocess.run(
        ["ps", "-eo", "pid,stat,etimes,cmd"], capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return [line.strip() for line in result if "phase4hc_direct_utility_arms.py train --arm" in line]


def gpus():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return [line.strip() for line in result if line.split(",", 1)[0].strip() in ("1", "2")]


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    while True:
        supervisor = read_json(CKPT / "supervisor_state.json")
        row = {
            "unix": time.time(), "processes": processes(), "gpus_index_memoryMiB_utilPercent": gpus(),
            "a1_last_completed": last_history("a1"), "a2_last_completed": last_history("a2"),
            "a1_manifest_status": (read_json(CKPT / "a1/execution_manifest.json") or {}).get("status"),
            "a2_manifest_status": (read_json(CKPT / "a2/execution_manifest.json") or {}).get("status"),
            "supervisor": supervisor,
        }
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if supervisor and supervisor.get("stage") == "COMPLETE":
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
