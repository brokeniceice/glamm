#!/usr/bin/env python3
"""Resumable downloader for the exact official GLaMM initialization snapshot."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import requests


DEST = Path("/data/yz/myLISA_storage/checkpoints/phase5a3_legion_retrained/base/GLaMM-GranD-Pretrained")
REV = "a2513f97c9404065cfd5849325e61d5d53456441"
# This transfer mirror redirects to the same immutable Hugging Face objects;
# the two LFS SHA256 values below remain the authority for accepted bytes.
BASE = f"https://hf-mirror.com/MBZUAI/GLaMM-GranD-Pretrained/resolve/{REV}"
FILES = (
    ".gitattributes", "README.md", "added_tokens.json", "config.json", "generation_config.json",
    "pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin",
    "pytorch_model.bin.index.json", "special_tokens_map.json", "tokenizer.model", "tokenizer_config.json",
)
EXPECTED = {
    "pytorch_model-00001-of-00002.bin": "988a0eac1f70372a1e95226c139db44c56fd8de0155a190352e3a2278464fcaf",
    "pytorch_model-00002-of-00002.bin": "557d74ccf62676279ca42a4d635b6641f1ff7208ab73e530625a9dc0f79a692c",
    "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str) -> None:
    destination = DEST / name
    expected = EXPECTED.get(name)
    if destination.is_file() and (expected is None or sha256(destination) == expected):
        print(f"present {name} {destination.stat().st_size}", flush=True)
        return
    partial = DEST / f"{name}.part"
    for attempt in range(1, 101):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            print(f"downloading {name} attempt={attempt} offset={offset}", flush=True)
            with requests.get(f"{BASE}/{name}", headers=headers, stream=True,
                              timeout=(30, 300), allow_redirects=True) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    partial.unlink()
                    offset = 0
                mode = "ab" if offset and response.status_code == 206 else "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            handle.flush()
            os.replace(partial, destination)
            if expected is not None:
                actual = sha256(destination)
                if actual != expected:
                    destination.unlink()
                    raise RuntimeError(f"SHA mismatch for {name}: {actual}")
            print(f"complete {name} {destination.stat().st_size}", flush=True)
            return
        except Exception as exc:
            print(f"retry {name}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(30)
    raise RuntimeError(f"Download retries exhausted for {name}")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(name)
    print(f"phase5a3 base snapshot complete revision={REV}", flush=True)


if __name__ == "__main__":
    main()
