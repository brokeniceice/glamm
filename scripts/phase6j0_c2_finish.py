#!/usr/bin/env python3
"""Quiet, fail-closed continuation: selected C2 checkpoint -> scoped evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6j0_c2"
TRAIN = OUT / "training"
CKPT = ROOT / "checkpoints/phase6j0_c2/best/checkpoint/mp_rank_00_model_states.pt"
PY = "/home/yz/miniconda3/envs/glamm_official/bin/python"
HISTORICAL_C1 = ROOT / "outputs/phase6d3_c1/evaluation"


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alive(pid: int):
    status = Path(f"/proc/{pid}/stat")
    if not status.exists():
        return False
    return status.read_text().split(") ", 1)[1][0] != "Z"


def main():
    pid = int((TRAIN / "train.pid").read_text())
    status_path = OUT / "continuation_status.json"
    write_json(status_path, {"status": "WAITING_FOR_TRAINING", "training_pid": pid,
                             "evaluation_scope": ["internal_test", "classification_ood",
                                                  "official1000_G0"]})
    while alive(pid):
        time.sleep(60)
    try:
        summary = json.loads((TRAIN / "run_summary.json").read_text())
        state = json.loads((TRAIN / "training_state.json").read_text())
        resume_path = TRAIN / "resume_audit.json"
        resume_step = int(json.loads(resume_path.read_text())["optimizer_step"]) if resume_path.exists() else 0
        if summary["mode"] != "train" or summary["optimizer_steps"] != 5000 - resume_step:
            raise RuntimeError("Formal training did not complete the remaining steps to 5000")
        if not summary["all_parameter_groups_synchronized"]:
            raise RuntimeError("Distributed parameter synchronization check failed")
        if state["optimizer_step"] != 5000 or state["epoch"] != 10:
            raise RuntimeError("Last training state is incomplete")
        if not CKPT.is_file():
            raise RuntimeError("Selected best checkpoint is absent")
        checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
        step, epoch = int(checkpoint["optimizer_step"]), int(checkpoint["epoch"])
        del checkpoint
        if step != epoch * 500 or not 1 <= epoch <= 10:
            raise RuntimeError("Selected checkpoint has invalid epoch/step")
        validation_losses = []
        for i in range(1, 11):
            metrics = json.loads((TRAIN / f"validation/epoch_{i:02d}/metrics.json").read_text())
            validation_losses.append(float(metrics["val_total_loss"]))
        if epoch != min(range(1, 11), key=lambda i: validation_losses[i - 1]):
            raise RuntimeError("Best checkpoint does not match validation selector")
        test_summary_path = OUT / "evaluation/test/summary.json"
        prior_result_path = OUT / "results.json"
        prior = json.loads(prior_result_path.read_text()) if prior_result_path.exists() else {}
        test_summary = json.loads(test_summary_path.read_text()) if test_summary_path.exists() else {}
        reuse_internal = (
            prior.get("status") == "COMPLETE" and
            prior.get("selected_epoch") == epoch and prior.get("selected_step") == step and
            prior.get("checkpoint_sha256") == sha256(CKPT) and
            test_summary.get("checkpoint_file_sha256") == prior.get("checkpoint_sha256") and
            test_summary.get("samples") == 2208 and
            all(mode in test_summary.get("modes", {}) for mode in ("detection", "G0", "tf_full_context"))
        )
        write_json(status_path, {"status": "EVALUATING" if not reuse_internal else "REUSING_INTERNAL_TEST",
                                 "best_epoch": epoch, "best_step": step,
                                 "best_val_total_loss": min(validation_losses)})
        if not reuse_internal:
            processes = []
            for split, gpu in (("test", 2),):
                destination = OUT / "evaluation" / split
                destination.mkdir(parents=True, exist_ok=True)
                command = [PY, "scripts/phase6j0_c2_evaluate.py",
                           "--config", "configs/phase6j0_c2_preln_cross_attention.yaml",
                           "--checkpoint", str(CKPT), "--output-dir", str(destination),
                           "--split", split, "--device", f"cuda:{gpu}",
                           "--modes", "detection", "G0", "tf_full_context",
                           "--expected-step", str(step), "--expected-epoch", str(epoch),
                           "--generation-batch-size", "1"]
                log = (OUT / f"evaluate_{split}.log").open("w")
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                processes.append((split, process, log))
            failed = []
            for split, process, log in processes:
                code = process.wait()
                log.close()
                if code:
                    failed.append((split, code))
            if failed:
                raise RuntimeError(f"Internal evaluator failed: {failed}")
        metrics = {}
        for split in ("test",):
            c2 = json.loads((OUT / f"evaluation/{split}/summary.json").read_text())
            c1 = json.loads((HISTORICAL_C1 / f"{split}/summary.json").read_text())
            metrics[split] = {
                "C2": c2["modes"], "historical_C1_raw": c1["modes"],
                "detection_deltas": {
                    key: float(c2["modes"]["detection"]["classification_head"][key])
                         - float(c1["modes"]["detection"]["classification_head"][key])
                    for key in ("accuracy", "roc_auc", "f1", "fake_recall")
                },
            }
        result = {
            "schema": "phase6j0_c2_internal_results_v1", "status": "COMPLETE",
            "strictly_matched_to_historical_C1": False,
            "reason": "C1 was trained on one GPU; C2 was trained on two GPUs",
            "selected_epoch": epoch, "selected_step": step,
            "selected_checkpoint": str(CKPT.resolve()), "checkpoint_sha256": sha256(CKPT),
            "selector": "minimum internal-validation total loss",
            "resume_step": resume_step,
            "training_micro_batch_change_after_resume": (
                yaml.safe_load((TRAIN / "config.yaml").read_text())["training"]["batch_size_per_device"]
                != yaml.safe_load((OUT / "recovery/config_initial_mb10.yaml").read_text())["training"]["batch_size_per_device"]
            ) if resume_step else False,
            "external_ood_evaluated": False, "metrics": metrics,
        }
        write_json(OUT / "results.json", result)
        write_json(status_path, {"status": "SCOPED_EVALUATION", "selected_epoch": epoch,
                                 "selected_step": step, "internal_results": str(OUT / "results.json")})
        with (OUT / "final_evaluation_supervisor.log").open("a", buffering=1) as log:
            code = subprocess.run([PY, "scripts/phase6j0_c2_final_evaluation.py"], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            raise RuntimeError(f"C2 scoped evaluation exited with {code}")
        final = json.loads((OUT / "final_evaluation/results.json").read_text())
        if (final["status"] != "COMPLETE" or
                final["protocol"]["scope"] != ["internal_test", "classification_ood",
                                               "official1000_G0"] or
                final["external_localization_ood"] != "DEFERRED_FOR_C2_PLUS_R1"):
            raise RuntimeError("C2 scoped evaluation did not reach COMPLETE")
        write_json(status_path, {"status": "COMPLETE", "selected_epoch": epoch,
                                 "selected_step": step, "results": str(OUT / "final_evaluation/results.json")})
    except Exception as exc:
        write_json(status_path, {"status": "FAILED", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
