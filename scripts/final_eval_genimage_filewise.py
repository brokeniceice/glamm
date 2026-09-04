#!/usr/bin/env python3
"""Resumable file-wise downloader for the official GenImage Google Drive.

Unlike gdown --folder, an inaccessible early part does not block later files.
Only file IDs enumerated from the official pinned Drive folders are used.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import gdown


ROOT = Path("/data/yz/myLISA_storage/AIGC/GenImage/raw/archives")
STATE = Path("/home/yz/groundingLMM_official/outputs/final_eval_datasets/downloads/genimage_filewise_status.json")
LOG = Path("/home/yz/groundingLMM_official/outputs/final_eval_datasets/downloads/genimage_filewise.log")
FOLDERS = {
    "ADM": "1-9o163XaC-7L8Ch9-r7dLxbY_yjN8SVj",
    "BigGAN": "1ajlTuN34gLyJWxRQ6NyUcnkfrS8QEVKt",
    "glide": "1H2_4VPlla4OuKYU2CbMYx-8X0nbGJ2Mc",
    "Midjourney": "1GLZedAqYuBh0kIQCZcbY_RspPJsR7ZyE",
    "sdv14": "12xighYOtu-ryfYEUnNrSeZqrxT8P08Zy",
    "sdv15": "1lG4WheCh3_CM2XNrRwfXVeOiG_1gZ_K2",
    "VQDM": "1-KviCgiBrBpm4e-TTdGk2RaGyV-69SaN",
    "wukong": "1h7685o7i6FNJ1wVtLtbwEWndW25YoaJk",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(value: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    value["updated_at_utc"] = now()
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, STATE)


def log(value: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": now(), **value}, ensure_ascii=False) + "\n")


def inventory() -> list[dict]:
    result = []
    seen_paths = set()
    for generator, folder_id in FOLDERS.items():
        files = gdown.download_folder(id=folder_id, skip_download=True, quiet=True)
        for item in files:
            target = ROOT / generator / item.path
            key = str(target)
            # The official sdv15 folder currently lists one duplicate path.
            if key in seen_paths:
                continue
            seen_paths.add(key)
            result.append({"generator": generator, "id": item.id, "relative_path": item.path, "target": key})
    return result


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    cycle = 0
    while True:
        cycle += 1
        files = inventory()
        missing = [item for item in files if not Path(item["target"]).is_file()]
        state = {
            "schema": "genimage_official_drive_filewise_v1", "status": "RUNNING",
            "cycle": cycle, "official_unique_files": len(files),
            "complete_files": len(files) - len(missing), "missing_files": len(missing),
            "source": "official GenImage Google Drive folders",
            "retry_policy": "skip completed targets; continue after per-file quota errors; retry remaining set every 6 hours",
        }
        write_state(state)
        if not missing:
            (ROOT / ".download_complete").write_text("COMPLETE\n", encoding="utf-8")
            state.update({"status": "COMPLETE", "completed_at_utc": now()})
            write_state(state)
            return
        successes = 0
        for ordinal, item in enumerate(missing, 1):
            target = Path(item["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            state.update({"current": item, "missing_ordinal": ordinal, "cycle_successes": successes})
            write_state(state)
            try:
                output = gdown.download(id=item["id"], output=str(target), resume=True, quiet=False)
                if output is None or not target.is_file():
                    raise RuntimeError("gdown returned without a completed target")
                successes += 1
                log({"event": "COMPLETE", "generator": item["generator"], "path": item["relative_path"], "bytes": target.stat().st_size})
            except BaseException as exc:
                log({"event": "DEFERRED", "generator": item["generator"], "path": item["relative_path"], "file_id": item["id"], "exception_type": type(exc).__name__, "exception": str(exc)})
                continue
        remaining = sum(not Path(item["target"]).is_file() for item in files)
        state.update({"status": "WAITING_RETRY", "cycle_successes": successes, "missing_files": remaining, "next_retry_seconds": 21600})
        state.pop("current", None)
        write_state(state)
        time.sleep(21600)


if __name__ == "__main__":
    main()
