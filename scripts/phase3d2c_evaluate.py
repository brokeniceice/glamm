#!/usr/bin/env python3
"""Run staged matched evaluations for Phase 3D.2-C with a validation hard gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3d2c_spatial_generalization_attribution"
P3D2B = ROOT / "outputs/phase3d2b_matched_spatial_reevaluation"
PYTHON = "/home/yz/miniconda3/envs/glamm_official/bin/python"
STEPS = [0, 50, 100, 250]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def command_for(record: dict, population: str) -> list[str]:
    output = OUT / f"evaluation/{population}/step_{record['candidate_step']:04d}"
    return [
        PYTHON, str(ROOT / "scripts/phase3a_evaluate.py"), "--config", str(ROOT / "configs/phase3a_p1.yaml"),
        "--checkpoint", record["checkpoint"]["path"], "--output-dir", str(output),
        "--manifest-dir", str(OUT / f"populations/{population}"), "--device", "cuda:0",
        "--modes", "tf_full_context", "--expected-step", str(record["stored_optimizer_step"]),
        "--expected-epoch", str(record["logical_epoch"]), "--tf-user-prompt", "canonical",
        "--include-soft-mask-diagnostics", "--reset",
    ]


def evaluate_jobs(jobs: list[tuple[int, str, list[dict]]]) -> None:
    def worker(physical_gpu: int, population: str, records: list[dict]):
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        env["GLAMM_PRESERVE_CUDA_CACHE"] = "1"; env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        for record in records:
            step = int(record["candidate_step"])
            log = OUT / f"logs/{population}_step_{step:04d}.log"; log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as handle:
                subprocess.run(command_for(record, population), cwd=ROOT, env=env,
                               stdout=handle, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(worker, *job) for job in jobs]
        for future in futures: future.result()


def validate_reproduction() -> None:
    records = []
    all_pass = True
    for step in STEPS:
        current = rows(OUT / f"evaluation/validation_fake/step_{step:04d}/tf_full_context/predictions.jsonl")
        frozen = rows(P3D2B / f"evaluation/matched_tf_phrase/step_{step:04d}/tf_full_context/predictions.jsonl")
        ci, fi = ({row["sample_id"]: row for row in values} for values in (current, frozen))
        sample_set_equal = set(ci) == set(fi) and len(ci) == 1106
        iou_max = max(abs(ci[sid]["foreground_iou"] - fi[sid]["foreground_iou"]) for sid in ci) if sample_set_equal else None
        f1_max = max(abs(ci[sid]["foreground_f1"] - fi[sid]["foreground_f1"]) for sid in ci) if sample_set_equal else None
        summary = load(OUT / f"evaluation/validation_fake/step_{step:04d}/summary.json")
        prior = load(P3D2B / f"evaluation/matched_tf_phrase/step_{step:04d}/summary.json")
        mean_iou_diff = abs(summary["modes"]["tf_full_context"]["per_image_mean"]["foreground_iou"] -
                            prior["modes"]["tf_full_context"]["per_image_mean"]["foreground_iou"])
        mean_f1_diff = abs(summary["modes"]["tf_full_context"]["per_image_mean"]["foreground_f1"] -
                           prior["modes"]["tf_full_context"]["per_image_mean"]["foreground_f1"])
        passed = sample_set_equal and iou_max == 0 and f1_max == 0 and mean_iou_diff == 0 and mean_f1_diff == 0
        all_pass &= passed
        records.append({
            "step": step, "sample_set_exact_equal": sample_set_equal,
            "per_sample_foreground_iou_max_abs_diff": iou_max,
            "per_sample_foreground_f1_max_abs_diff": f1_max,
            "mean_foreground_iou_abs_diff": mean_iou_diff,
            "mean_foreground_f1_abs_diff": mean_f1_diff, "status": "PASS" if passed else "FAIL",
        })
    audit = {
        "status": "PASS" if all_pass else "FAIL", "population": "validation_fake", "count": 1106,
        "reference": "Phase 3D.2-B matched per-sample outputs", "records": records,
        "gate": None if all_pass else "GATE_PHASE3D2C_VALIDATION_REPRODUCIBILITY_FAILURE",
    }
    dump(OUT / "statistics/validation_reproducibility_audit.json", audit)
    dump(OUT / "validation_reproducibility_audit.json", audit)
    if not all_pass:
        raise RuntimeError("Phase 3D.2-C validation reproducibility failure; attribution must stop")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validation", "train"), required=True)
    args = parser.parse_args()
    manifest = load(OUT / "checkpoint_attribution_manifest.json")
    by_step = {int(row["candidate_step"]): row for row in manifest["checkpoints"]}
    records = [by_step[step] for step in STEPS]
    if args.stage == "validation":
        evaluate_jobs([(1, "validation_fake", records[::2]), (2, "validation_fake", records[1::2])])
        validate_reproduction()
    else:
        audit = load(OUT / "validation_reproducibility_audit.json")
        if audit["status"] != "PASS":
            raise RuntimeError("validation hard gate not passed")
        evaluate_jobs([(1, "seen_train", records), (2, "train_holdout", records)])
    print(json.dumps({"status": "PASS", "stage": args.stage, "steps": STEPS}, indent=2))


if __name__ == "__main__":
    main()
