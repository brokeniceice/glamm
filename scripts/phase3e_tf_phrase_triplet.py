#!/usr/bin/env python3
"""Unattended canonical TF-PHRASE triplet diagnostic on Phase 3E validation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/phase3e_joint_language_mask_posttraining.yaml").read_text())
OUT = (ROOT / CFG["experiment"]["output_root"]).resolve()
RUNTIME = OUT / "tf_phrase_triplet"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def launch(name: str, gpu: int, checkpoint: Path, step: int, epoch: int, output: Path):
    command = [
        sys.executable, str(ROOT / "scripts/phase3a_evaluate.py"),
        "--config", str(ROOT / CFG["source"]["model_config"]),
        "--checkpoint", str(checkpoint), "--output-dir", str(output),
        "--manifest-dir", str(OUT / "evaluation/validation_manifest"),
        "--device", "cuda:0", "--modes", "tf_full_context",
        "--expected-step", str(step), "--expected-epoch", str(epoch),
        "--generation-batch-size", "1", "--skip-spatial-save",
        "--tf-user-prompt", "canonical", "--reset",
    ]
    log_path = RUNTIME / f"{name}.log"
    log = log_path.open("a", encoding="utf-8")
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return process, log, log_path, command


def main() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "supervisor.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p1_output = OUT / "evaluation/final/P1_FROZEN"
        p1_summary = p1_output / "summary.json"
        if not p1_summary.is_file():
            raise FileNotFoundError("The canonical Phase 3E P1 TF result required for reuse is missing")
        p1 = json.loads(p1_summary.read_text(encoding="utf-8"))
        if p1.get("checkpoint_file_sha256") != CFG["source"]["checkpoint_sha256"]:
            raise RuntimeError("P1 TF checkpoint provenance mismatch")

        jobs = {
            "SFT_CONT_SELECTED_STEP_0900": {
                "gpu": 0, "step": 900, "epoch": 9,
                "checkpoint": Path(CFG["experiment"]["checkpoint_root"]) / "SFT_CONT/step_0900/checkpoint/mp_rank_00_model_states.pt",
                "output": OUT / "evaluation/final/SFT_CONT",
                "role": "formally_selected_checkpoint",
            },
            "JOINT_TRAINED_BEST_STEP_0300": {
                "gpu": 2, "step": 300, "epoch": 3,
                "checkpoint": Path(CFG["experiment"]["checkpoint_root"]) / "JOINT/step_0300/checkpoint/mp_rank_00_model_states.pt",
                "output": OUT / "evaluation/final/JOINT_TRAINED_STEP_0300",
                "role": "post_selection_diagnostic_best_nonzero_G0_candidate_not_formally_selected",
            },
        }
        state = {
            "status": "RUNNING", "started_utc": now(),
            "population": "internal_validation_fake_only", "num_fake": 1106,
            "tf_user_prompt": "canonical", "target_protocol": "phrase_aligned",
            "mask_logit_threshold": 0.0, "internal_test_used": False,
            "official1000_used": False, "P1_reused_not_rerun": True,
            "P1_output": str(p1_output), "jobs": {},
        }
        processes = {}
        for name, job in jobs.items():
            checkpoint = job["checkpoint"].resolve()
            if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
            process, log, log_path, command = launch(
                name, job["gpu"], checkpoint, job["step"], job["epoch"], job["output"]
            )
            processes[name] = (process, log, log_path)
            state["jobs"][name] = {
                **{k: v for k, v in job.items() if k not in ("checkpoint", "output")},
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "output": str(job["output"]), "command": command, "status": "RUNNING",
            }
        dump(RUNTIME / "state.json", state)

        failed = False
        for name, (process, log, log_path) in processes.items():
            returncode = process.wait(); log.close()
            state["jobs"][name]["returncode"] = returncode
            state["jobs"][name]["log"] = str(log_path)
            state["jobs"][name]["status"] = "COMPLETE" if returncode == 0 else "FAILED"
            failed |= returncode != 0
            dump(RUNTIME / "state.json", state)
        if failed:
            state["status"] = "FAILED"; state["failed_utc"] = now(); dump(RUNTIME / "state.json", state)
            raise RuntimeError("At least one TF-PHRASE diagnostic failed")

        arms = {"P1_FROZEN": p1_output, **{name: job["output"] for name, job in jobs.items()}}
        metrics = {}
        for name, output in arms.items():
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            values = summary["modes"]["tf_full_context"]
            metrics[name] = {
                "checkpoint_role": "frozen_baseline" if name == "P1_FROZEN" else jobs[name]["role"],
                "num_gt_fake": values["num_gt_fake"],
                "per_image_mean": values["per_image_mean"],
                "global_pixel": values["global_pixel"],
                "output": str(output),
            }
        result = {
            "status": "COMPLETE", "completed_utc": now(),
            "population": "internal_validation_fake_only", "tf_user_prompt": "canonical",
            "target_protocol": "phrase_aligned", "threshold_tuned": False,
            "P1_reused_not_rerun": True, "internal_test_used": False,
            "official1000_used": False, "metrics": metrics,
            "interpretation_boundary": "JOINT step 300 is a post-selection trained-checkpoint diagnostic; the formal JOINT selector fell back to P1 step 0.",
        }
        dump(RUNTIME / "triplet_metrics.json", result)
        state["status"] = "COMPLETE"; state["completed_utc"] = result["completed_utc"]
        dump(RUNTIME / "state.json", state)


if __name__ == "__main__":
    main()
