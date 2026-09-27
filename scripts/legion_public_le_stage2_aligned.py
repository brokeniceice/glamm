#!/usr/bin/env python3
"""Train a LEGION classification head from the public LE checkpoint on GPUs 1,2."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/yz/miniconda3/envs/legion/bin/python")
TORCHRUN = PYTHON.parent / "torchrun"
PUBLIC_LE = ROOT / "checkpoints/phase5a1_legion/models/legion_LE_f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa"
CLIP = ROOT / "checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
SAM = ROOT / "checkpoints/phase5b0_fakeshield/models/sam_7790786db131bcdc639f24a915d9f2c331d843ee/checkpoints/sam_vit_h_4b8939.pth"
DATA = ROOT / "outputs/phase5a3_legion_retrained/data/stage2"
OUT = ROOT / "checkpoints/legion_public_le_stage2_aligned"
SCRIPT = ROOT / "scripts/phase5a3_stage2_train.py"
EXPECTED = {
    "train_sha256": "ea46ca7d07c6e585911c757e24f8998c5e67783e4b47866a8feeae8333832f59",
    "val_sha256": "15f668ee771fde740003f53e71f6af7f5e5e3bf18edcdba2774276edd8fe98dc",
    "public_index_sha256": "808b0081a99b019de3bb3c06a83ca016386731e388a50b7707570e9761b39f2f",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(stage: str, command: list[str], env: dict[str, str], status: dict) -> None:
    status.update(stage=stage, updated_at_utc=now())
    save(OUT / "status.json", status)
    log = OUT / "logs" / f"{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"{stage} exited {completed.returncode}; see {log}")


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1,2":
        raise RuntimeError("This run is pinned to physical GPUs 1 and 2")
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Output directory already contains data: {OUT}")
    OUT.mkdir(parents=True)
    source_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/LEGION_official"), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_commit != "d21535dd45f6fea509337a83095966f0b86ac924":
        raise RuntimeError(f"Official source commit drift: {source_commit}")
    for label, path in {
        "train_sha256": DATA / "train.json",
        "val_sha256": DATA / "val.json",
        "public_index_sha256": PUBLIC_LE / "pytorch_model.bin.index.json",
    }.items():
        if sha(path) != EXPECTED[label]:
            raise RuntimeError(f"Frozen input drift: {label}: {path}")
    index = json.loads((PUBLIC_LE / "pytorch_model.bin.index.json").read_text())
    if any(key.startswith("prediction_head.") for key in index["weight_map"]):
        raise RuntimeError("Public LE unexpectedly already contains the classification head")
    if len(json.loads((DATA / "train.json").read_text())) != 17672:
        raise RuntimeError("Training population drift")
    if len(json.loads((DATA / "val.json").read_text())) != 2212:
        raise RuntimeError("Validation population drift")
    for path in (PYTHON, TORCHRUN, PUBLIC_LE, CLIP, SAM, SCRIPT):
        if not path.exists():
            raise FileNotFoundError(path)

    protocol = {
        "schema": "legion_public_le_stage2_aligned_v1",
        "frozen_at_utc": now(),
        "source_commit": source_commit,
        "initialization": str(PUBLIC_LE.resolve()),
        "public_revision": "f8c28b349ccbb0b5d4240c9eb82beaa9a2586ffa",
        "vision_tower": str(CLIP.resolve()),
        "sam_checkpoint": str(SAM.resolve()),
        "train_json": str((DATA / "train.json").resolve()),
        "val_json": str((DATA / "val.json").resolve()),
        "input_hashes": EXPECTED,
        "physical_gpus": [1, 2],
        "model": "official LegionForCls",
        "trainable": "prediction_head only",
        "labels": {"real": 1, "fake": 0},
        "train_n": 17672,
        "val_n": 2212,
        "epochs": 3,
        "per_device_batch": 32,
        "effective_global_batch": 64,
        "gradient_accumulation_steps": 1,
        "lr": 0.001,
        "weight_decay": 0.0,
        "scheduler": "cosine",
        "seed": 3407,
        "selector": "maximum internal-validation accuracy, load_best_model_at_end",
        "held_out_data_used": False,
        "comparison_boundary": "Same-data and effective-batch recipe as legion-retrained; two-GPU DDP is not bitwise identical to its single-GPU run.",
    }
    save(OUT / "protocol.json", protocol)
    status = {"status": "RUNNING", "stage": "prepared", "started_at_utc": now()}
    save(OUT / "status.json", status)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "1,2",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "4",
    })
    base = [
        str(TORCHRUN), "--standalone", "--nproc_per_node=2", str(SCRIPT),
        "--stage1-le", str(PUBLIC_LE), "--vision-pretrained", str(SAM),
        "--vision-tower", str(CLIP),
        "--train-json", str(DATA / "train.json"), "--val-json", str(DATA / "val.json"),
        "--run-dir", str(OUT), "--batch-size", "32", "--epochs", "3",
        "--workers", "4", "--seed", "3407", "--public-legion-le",
    ]
    try:
        run("preflight", base + ["--mode", "preflight"], env, status)
        preflight = json.loads((OUT / "preflight_batch32.json").read_text())
        if (preflight["status"] != "PASS" or not preflight["public_legion_LE_used"]
                or preflight["world_size"] != 2 or preflight["effective_global_batch"] != 64
                or preflight["trainable_parameter_count"] != 2103298):
            raise RuntimeError("Preflight protocol or trainable firewall failed")
        run("training", base + ["--mode", "train"], env, status)
        summary = json.loads((OUT / "training_summary.json").read_text())
        if (summary["status"] != "COMPLETE" or not summary["public_legion_LE_used"]
                or summary["train_count"] != 17672 or summary["validation_count"] != 2212
                or len(summary["validation_by_epoch"]) != 3
                or not Path(summary["final_model"]).is_dir()):
            raise RuntimeError("Training summary failed completeness checks")
        status.update(status="COMPLETE", stage="complete", completed_at_utc=now(),
                      best_metric=summary["best_metric"], best_checkpoint=summary["best_checkpoint"])
        save(OUT / "status.json", status)
    except BaseException as exc:
        status.update(status="FAILED", stage=status["stage"], failed_at_utc=now(),
                      exception=repr(exc), traceback=traceback.format_exc())
        save(OUT / "status.json", status)
        raise


if __name__ == "__main__":
    main()
