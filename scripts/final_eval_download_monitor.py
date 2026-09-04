#!/usr/bin/env python3
"""Write user-readable progress logs for detached final-data downloads."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/yz/groundingLMM_official")
LOG_ROOT = ROOT / "outputs/final_eval_datasets/downloads"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

ITEMS = {
    "xaigd": {
        "units": ["finaldata-xaigd-v3.service"],
        "roots": [Path("/data/yz/myLISA_storage/AIGC/X-AIGD/raw/labeled_test-00000-of-00001.parquet")],
        "expected": 3_488_049_189,
    },
    "aigi_holmes": {
        "units": ["finaldata-aigi-holmes-v3.service"],
        "roots": [Path("/data/yz/myLISA_storage/AIGC/AIGI-Holmes/raw/TestSet.zip")],
        "expected": 40_457_899_107,
    },
    "pal4vst": {
        "units": ["finaldata-pal4vst-v2.service"],
        "roots": [Path("/data/yz/myLISA_storage/AIGC/PAL4VST/raw")],
        "expected": None,
    },
    "loki": {
        "units": ["finaldata-loki-archive.service"],
        "roots": [Path("/data/yz/LOKI/loki_media_aggregate/raw/media_data.zip")],
        "expected": 7_607_781_219,
    },
    "genimage": {
        "units": ["finaldata-genimage-test-hf.service"],
        "roots": [Path("/data/yz/myLISA_storage/AIGC/GenImage/raw/huggingface_test/genimage_test.zip")],
        "expected": 25_408_572_820,
    },
}


def unit_state(name: str) -> dict:
    result = subprocess.run(
        ["systemctl", "--user", "show", name,
         "--property=ActiveState,SubState,NRestarts,Result"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    values = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values or {"ActiveState": "not-found"}


def file_bytes(path: Path) -> tuple[int, int]:
    if path.is_file():
        return path.stat().st_size, 1
    if not path.exists():
        return 0, 0
    total = count = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            candidate = Path(root) / name
            try:
                total += candidate.stat().st_size
                count += 1
            except FileNotFoundError:
                pass
    return total, count


def snapshot() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    result = {"updated_at_utc": now, "detached": True, "resume": True,
              "automatic_retry": True, "items": {}}
    for name, spec in ITEMS.items():
        byte_count = file_count = 0
        for root in spec["roots"]:
            size, count = file_bytes(root)
            byte_count += size
            file_count += count
        expected = spec["expected"]
        units = {unit: unit_state(unit) for unit in spec["units"]}
        result["items"][name] = {
            "downloaded_bytes": byte_count,
            "downloaded_gib": round(byte_count / (1024 ** 3), 3),
            "expected_bytes": expected,
            "percent": round(100 * byte_count / expected, 2) if expected else None,
            "file_count_including_partial": file_count,
            "services": units,
        }
    return result


def atomic_status(value: dict) -> None:
    temporary = LOG_ROOT / ".status.json.tmp"
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, LOG_ROOT / "status.json")


def main() -> None:
    previous = {}
    while True:
        value = snapshot()
        atomic_status(value)
        for name, item in value["items"].items():
            compact = {
                "downloaded_bytes": item["downloaded_bytes"],
                "percent": item["percent"],
                "services": item["services"],
            }
            if compact != previous.get(name):
                with (LOG_ROOT / f"{name}.log").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"time": value["updated_at_utc"], **item}, ensure_ascii=False) + "\n")
                previous[name] = compact
        time.sleep(30)


if __name__ == "__main__":
    main()
