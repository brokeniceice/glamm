#!/usr/bin/env python3
"""Detached two-GPU LEGION matched-initialization training pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/yz/miniconda3/envs/legion/bin/python")
DEEPSPEED = Path("/home/yz/miniconda3/envs/legion/bin/deepspeed")
LEGION_BIN = PYTHON.parent
OFFICIAL = ROOT / "external/LEGION_official"
SOURCE_COMMIT = "d21535dd45f6fea509337a83095966f0b86ac924"
BASE = ROOT / "checkpoints/GLaMM-FullScope"
OLD = ROOT / "checkpoints/phase5a3_legion_retrained"
DATA = ROOT / "outputs/phase5a3_legion_retrained/data"
ROOT_OUT = ROOT / "checkpoints/legion_retrained_match"
RUN = ROOT_OUT / "stage1_run_v2"
STAGE1_LE = ROOT_OUT / "stage1_le"
STAGE2 = ROOT_OUT / "stage2_cls"
LOGS = ROOT_OUT / "logs_v2"
TORCH_EXTENSIONS = ROOT_OUT / "torch_extensions"
CLIP = OLD.parent / "phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
SAM = OLD.parent / "phase5b0_fakeshield/models/sam_7790786db131bcdc639f24a915d9f2c331d843ee/checkpoints/sam_vit_h_4b8939.pth"
STATE = ROOT_OUT / "pipeline_state.json"
EPOCHS = 5
VRAM_GATE = 46 * 2**30


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_identity(path: Path):
    files = []
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        files.append({"path": str(item.relative_to(path)), "bytes": item.stat().st_size,
                      "sha256": sha256(item)})
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"root": str(path.resolve()), "files": files,
            "canonical_sha256": hashlib.sha256(canonical).hexdigest()}


def environment():
    return {
        **os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
        "PATH": f"{LEGION_BIN}:{os.environ.get('PATH', '')}",
        "CUDA_HOME": str(LEGION_BIN.parent),
        "LIBRARY_PATH": f"{LEGION_BIN.parent / 'lib'}:{os.environ.get('LIBRARY_PATH', '')}",
        "LD_LIBRARY_PATH": f"{LEGION_BIN.parent / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "TORCH_EXTENSIONS_DIR": str(TORCH_EXTENSIONS), "MAX_JOBS": "8",
    }


def run(command, log_name, allow_failure=False):
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / log_name
    with log_path.open("a") as log:
        log.write(f"\n[{now()}] COMMAND {json.dumps([str(x) for x in command])}\n")
        log.flush()
        process = subprocess.Popen([str(x) for x in command], cwd=ROOT, env=environment(),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code and not allow_failure:
        raise RuntimeError(f"command failed code={code}; see {log_path}")
    return code


def git(*args):
    return subprocess.check_output(["git", "-C", str(OFFICIAL), *args], text=True).strip()


def substantive_git_changes():
    changes = []
    for line in git("status", "--porcelain").splitlines():
        path = line[3:]
        if "/__pycache__/" in path or path.endswith(".pyc"):
            continue
        changes.append(line)
    return changes


def main():
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    state = {"status": "RUNNING", "started_at_utc": now(), "stage": "identity_gate"}
    dump(STATE, state)
    try:
        source_changes = substantive_git_changes()
        if git("rev-parse", "HEAD") != SOURCE_COMMIT or source_changes:
            raise RuntimeError(f"official LEGION source identity/cleanliness gate failed: {source_changes}")
        required = [BASE / "pytorch_model.bin.index.json", SAM, CLIP,
                    DATA / "stage1/train/annotations/train.json", DATA / "stage1/train/annotations/test.json",
                    DATA / "stage2/train.json", DATA / "stage2/val.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(missing)

        common1 = [
            ROOT / "scripts/phase5a3_stage1_train.py", "--base", BASE,
            "--dataset-dir", DATA / "stage1", "--vision-pretrained", SAM,
            "--vision-tower", CLIP, "--run-dir", RUN, "--workers", "2",
            "--epochs", str(EPOCHS), "--preflight-optimizer-steps", "3",
        ]
        selected = None
        preflight_attempts = []
        for micro, accumulation in ((4, 2), (2, 4), (1, 8)):
            state.update(stage="stage1_preflight", candidate={"micro_batch": micro, "grad_accum": accumulation})
            dump(STATE, state)
            command = [DEEPSPEED, "--include", "localhost:1,2", "--master_port", "29641",
                       *common1, "--micro-batch", str(micro), "--grad-accum", str(accumulation),
                       "--mode", "preflight"]
            code = run(command, f"stage1_preflight_m{micro}_g{accumulation}.log", allow_failure=True)
            record_path = RUN / "preflight.json"
            record = json.loads(record_path.read_text()) if code == 0 and record_path.exists() else None
            passed = bool(record and record["status"] == "PASS" and record["peak_memory_bytes"] <= VRAM_GATE)
            preflight_attempts.append({"micro_batch": micro, "grad_accum": accumulation,
                                       "exit_code": code, "record": record, "passed_vram_gate": passed})
            if passed:
                selected = (micro, accumulation, record)
                break
        if selected is None:
            raise RuntimeError("no Stage-1 microbatch candidate passed the three-step 46-GiB gate")
        micro, accumulation, preflight = selected

        train_count = len(json.loads((DATA / "stage1/train/annotations/train.json").read_text()))
        manifest = {
            "schema": "legion_retrained_match_protocol_v1", "frozen_at_utc": now(),
            "status": "FROZEN_BEFORE_FORMAL_TRAINING", "official_source_commit": SOURCE_COMMIT,
            "initialization": {"stage1": str(BASE.resolve()), "stage1_index_sha256": sha256(BASE / "pytorch_model.bin.index.json"),
                               "stage2": str(STAGE1_LE.resolve()), "public_legion_intermediate_used": False},
            "stage1": {"epochs": EPOCHS, "train_annotation_samples": train_count,
                       "sample_exposures": train_count * EPOCHS, "micro_batch": micro,
                       "gradient_accumulation": accumulation, "effective_global_batch": 16,
                       "steps_per_epoch": int(np_ceil(train_count / 16)), "lr": 1e-4,
                       "loss_weights": {"CE": 1.0, "Dice": 0.2, "BCE": 0.4},
                       "selector": "minimum internal-validation native TF total loss; tie earlier epoch",
                       "mandatory_fixed_report_epoch": 3},
            "stage2": {"epochs": EPOCHS, "train_samples": 17672, "sample_exposures": 17672 * EPOCHS,
                       "batch": 64, "steps_per_epoch": 277, "lr": 1e-3, "scheduler": "cosine",
                       "trainable": "prediction_head only", "selector": "validation accuracy; tie lower loss; tie earlier epoch",
                       "mandatory_fixed_report_epoch": 3},
            "preflight_attempts": preflight_attempts,
            "firewall": {"internal_train": True, "internal_validation": True, "internal_test": False,
                         "OOD": False, "official1000": False},
        }
        dump(ROOT_OUT / "execution_manifest.json", manifest)

        state.update(stage="stage1_training", selected_micro_batch=micro, selected_grad_accum=accumulation)
        dump(STATE, state)
        run([DEEPSPEED, "--include", "localhost:1,2", "--master_port", "29642", *common1,
             "--micro-batch", str(micro), "--grad-accum", str(accumulation), "--mode", "train"],
            "stage1_training.log")
        history = json.loads((RUN / "history.json").read_text())
        if len(history) != EPOCHS:
            raise RuntimeError(f"expected {EPOCHS} Stage-1 epochs, got {len(history)}")
        selected_stage1 = min(history, key=lambda row: (row["validation_losses"]["loss"], row["epoch"]))
        manifest["stage1"]["selected_checkpoint"] = selected_stage1["tag"]
        manifest["stage1"]["selected_validation_loss"] = selected_stage1["validation_losses"]["loss"]
        dump(ROOT_OUT / "execution_manifest.json", manifest)

        state["stage"] = "stage1_merge"
        dump(STATE, state)
        ds_root = RUN / "deepspeed_checkpoints"
        fp32 = RUN / "merged_input/pytorch_model.bin"
        fp32.parent.mkdir(parents=True, exist_ok=True)
        run([PYTHON, ds_root / "zero_to_fp32.py", ds_root, fp32, "--tag", selected_stage1["tag"]],
            "stage1_merge_fp32.log")
        run([PYTHON, ROOT / "scripts/phase5a3_merge_stage1.py", "--version", BASE,
             "--weight", fp32, "--save_path", STAGE1_LE, "--vision_pretrained", SAM,
             "--vision-tower", CLIP], "stage1_merge_hf.log")
        dump(ROOT_OUT / "stage1_le_identity.json", directory_identity(STAGE1_LE))

        common2 = [
            PYTHON, "-m", "torch.distributed.run", "--nproc_per_node=2", "--master_port=29643",
            ROOT / "scripts/phase5a3_stage2_train.py", "--stage1-le", STAGE1_LE,
            "--vision-pretrained", SAM, "--vision-tower", CLIP,
            "--train-json", DATA / "stage2/train.json", "--val-json", DATA / "stage2/val.json",
            "--run-dir", STAGE2, "--workers", "4", "--batch-size", "32", "--epochs", str(EPOCHS),
        ]
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
        state["stage"] = "stage2_preflight"
        dump(STATE, state)
        run([*common2, "--mode", "preflight"], "stage2_preflight.log")
        state["stage"] = "stage2_training"
        dump(STATE, state)
        run([*common2, "--mode", "train"], "stage2_training.log")
        summary = json.loads((STAGE2 / "training_summary.json").read_text())
        dump(ROOT_OUT / "stage2_cls_identity.json", directory_identity(Path(summary["final_model"])))

        state.update(status="COMPLETE", stage="complete", completed_at_utc=now(),
                     selected_stage1=selected_stage1["tag"], selected_stage2=summary["best_checkpoint"])
        dump(STATE, state)
    except BaseException as error:
        state.update(status="FAILED", failed_at_utc=now(), exception_type=type(error).__name__, exception=str(error))
        dump(STATE, state)
        raise


def np_ceil(value):
    return int(value) if int(value) == value else int(value) + 1


if __name__ == "__main__":
    main()
