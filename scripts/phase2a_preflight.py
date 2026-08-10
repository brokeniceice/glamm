#!/usr/bin/env python3
"""Static and bounded preflight audits around the Phase 2A DeepSpeed runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import deepspeed
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train as glamm_train
from dataset.dataset import custom_collate_fn
from dataset.forensics.distributed import sampler_epoch_indices
from dataset.forensics.unified import (
    CANONICAL_PROMPT_SHA256,
    CANONICAL_PROMPT_TEMPLATE_ID,
    UnifiedForensicsDataset,
)
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import build_train_args


OUTPUT = REPO_ROOT / "outputs/phase2a_unified_baseline"


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2a_unified_baseline_full.yaml")
    parser.add_argument("--action", choices=("sampler", "workers", "compare", "metadata"), required=True)
    parser.add_argument("--worker-candidates", default="2,4,8")
    parser.add_argument("--benchmark-batches", type=int, default=30)
    return parser.parse_args()


def load_config(path):
    path = (REPO_ROOT / path).resolve()
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))


def lightweight_dataset_rows(config):
    path = REPO_ROOT / config["data"]["manifest_dir"] / "train_combined.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return type("Rows", (), {"rows": rows, "__len__": lambda self: len(self.rows)})()


def sampler_audit(config):
    dataset = lightweight_dataset_rows(config)
    seed = int(config["experiment"]["seed"])
    epochs = {epoch: sampler_epoch_indices(dataset, world_size=2, seed=seed, epoch=epoch) for epoch in (0, 1)}
    records = {}
    for epoch, ranks in epochs.items():
        records[f"epoch_{epoch}"] = {}
        for rank, indices in enumerate(ranks):
            steps = []
            for step in range(20):
                selected = indices[step * 10:(step + 1) * 10]
                steps.append({
                    "optimizer_step": step + 1,
                    "samples": [{
                        "sample_id": dataset.rows[i]["sample_id"],
                        "image_path": dataset.rows[i].get("image_path", dataset.rows[i].get("image_relpath")),
                        "gt_label": int(dataset.rows[i]["class_label"]),
                        "content_type": dataset.rows[i].get("content_category"),
                        "source": dataset.rows[i].get("source"),
                    } for i in selected],
                })
            records[f"epoch_{epoch}"][f"rank_{rank}"] = steps
    rank_sets = [set(values) for values in epochs[0]]
    audit = {
        "status": "passed",
        "world_size": 2,
        "seed": seed,
        "rank_overlap_count_full_epoch": len(rank_sets[0] & rank_sets[1]),
        "union_equals_dataset": rank_sets[0] | rank_sets[1] == set(range(len(dataset))),
        "same_seed_reproducible": epochs[0] == sampler_epoch_indices(
            dataset, world_size=2, seed=seed, epoch=0
        ),
        "different_epoch_changes_order": epochs[0] != epochs[1],
        "rank_sample_count": [len(values) for values in epochs[0]],
        "first_20_optimizer_steps": records,
    }
    if not all((audit["rank_overlap_count_full_epoch"] == 0, audit["union_equals_dataset"],
                audit["same_seed_reproducible"], audit["different_epoch_changes_order"])):
        raise AssertionError(audit)
    write_json(OUTPUT / "distributed_sampler_audit.json", audit)
    return audit


def worker_benchmark(config, candidates, batches):
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    args = build_train_args(config)
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(args)
    dataset = UnifiedForensicsDataset(
        REPO_ROOT / config["data"]["manifest_dir"], tokenizer, args.vision_tower, split="train",
        datasets_root=config["data"]["datasets_root"],
        synthscars_root=config["data"]["synthscars_root"], image_size=args.image_size,
    )
    from dataset.forensics.distributed import BalancedDistributedForensicsSampler
    results = []
    reference_ids = None
    for workers in candidates:
        sampler = BalancedDistributedForensicsSampler(
            dataset, num_replicas=2, rank=0, seed=int(config["experiment"]["seed"]), shuffle=True
        )
        kwargs = dict(
            dataset=dataset, sampler=sampler, batch_size=2, num_workers=workers,
            pin_memory=True, drop_last=False,
            collate_fn=lambda items: custom_collate_fn(
                items, tokenizer=tokenizer, inference=False, token_strategy="fixed_cls_query"
            ),
        )
        if workers:
            kwargs.update(persistent_workers=True, prefetch_factor=2)
        loader = torch.utils.data.DataLoader(**kwargs)
        start = time.perf_counter(); ids = []
        for index, batch in enumerate(loader):
            ids.extend(batch["sample_ids"])
            if index + 1 >= batches:
                break
        elapsed = time.perf_counter() - start
        if reference_ids is None:
            reference_ids = ids
        if ids != reference_ids:
            raise AssertionError("Worker count changed deterministic sample order")
        results.append({
            "workers_per_rank": workers, "batches": batches, "samples": len(ids),
            "elapsed_seconds": elapsed, "samples_per_second": len(ids) / elapsed,
            "mean_data_seconds_per_batch": elapsed / batches,
        })
    selected = max(results, key=lambda item: item["samples_per_second"])["workers_per_rank"]
    report = {"status": "passed", "results": results, "selected_workers_per_rank": selected}
    write_json(OUTPUT / "throughput_benchmark_workers.json", report)
    return report


def flatten_step_samples(summary):
    per_step = {}
    for rank in summary["ranks"]:
        for step in rank["sample_audit"]:
            values = per_step.setdefault(step["optimizer_step"], set())
            for micro in step["micro_batches"]:
                values.update(row["sample_id"] for row in micro)
    return per_step


def compare_smokes():
    single = json.loads((OUTPUT / "preflight/single/run_summary.json").read_text())
    dual = json.loads((OUTPUT / "preflight/dual/run_summary.json").read_text())
    single_metrics = [json.loads(line) for line in (OUTPUT / "preflight/single/metrics.jsonl").read_text().splitlines()]
    dual_metrics = [json.loads(line) for line in (OUTPUT / "preflight/dual/metrics.jsonl").read_text().splitlines()]
    single_samples, dual_samples = flatten_step_samples(single), flatten_step_samples(dual)
    common_steps = sorted(set(single_samples) & set(dual_samples))
    sample_equal = all(single_samples[step] == dual_samples[step] for step in common_steps)
    loss_differences = {}
    for key in ("total_loss", "text_loss", "cls_loss", "mask_bce_loss", "mask_dice_loss"):
        differences = [abs(a[key] - b[key]) for a, b in zip(single_metrics, dual_metrics)]
        scales = [max(abs(a[key]), abs(b[key]), 1e-8) for a, b in zip(single_metrics, dual_metrics)]
        loss_differences[key] = {
            "mean_absolute": sum(differences) / len(differences),
            "max_absolute": max(differences),
            "mean_relative": sum(d / s for d, s in zip(differences, scales)) / len(differences),
        }
    parameter_delta = {}
    for group in single["ranks"][0]["parameter_sync"]:
        a = single["ranks"][0]["parameter_sync"][group]["delta_l2"]
        b = dual["ranks"][0]["parameter_sync"][group]["delta_l2"]
        parameter_delta[group] = {"single": a, "dual": b, "relative_difference": abs(a-b)/max(a,b,1e-8)}
    report = {
        "status": "passed" if sample_equal and dual["all_parameter_groups_synchronized"] else "failed",
        "effective_global_batch_single": single["effective_global_batch"],
        "effective_global_batch_dual": dual["effective_global_batch"],
        "sample_sets_equal_each_optimizer_step": sample_equal,
        "loss_differences": loss_differences,
        "parameter_delta": parameter_delta,
        "all_dual_rank_parameters_synchronized": dual["all_parameter_groups_synchronized"],
        "single_samples_per_second": single["samples_per_second"],
        "dual_samples_per_second": dual["samples_per_second"],
        "speedup": dual["samples_per_second"] / single["samples_per_second"],
    }
    if report["status"] != "passed":
        raise AssertionError(report)
    write_json(OUTPUT / "distributed_loss_parity.json", report)
    write_json(OUTPUT / "throughput_benchmark.json", {
        key: report[key] for key in ("single_samples_per_second", "dual_samples_per_second", "speedup")
    })
    return report


def metadata(config_path, config):
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    train_count = sum(1 for _ in (REPO_ROOT / config["data"]["manifest_dir"] / "train_combined.jsonl").open())
    budget = int(config["training"]["total_optimizer_steps"])
    exposure = budget * 20
    gpu_rows = []
    for index in (0, 1):
        prop = torch.cuda.get_device_properties(index)
        gpu_rows.append({"visible_index": index, "name": prop.name, "vram_bytes": prop.total_memory})
    worker_file = OUTPUT / "throughput_benchmark_workers.json"
    selected_workers = json.loads(worker_file.read_text())["selected_workers_per_rank"] if worker_file.is_file() else None
    report = {
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                     text=True, capture_output=True, check=True).stdout.strip(),
        "config": str(config_path.relative_to(REPO_ROOT)), "config_sha256": config_hash,
        "canonical_prompt_id": CANONICAL_PROMPT_TEMPLATE_ID,
        "canonical_prompt_sha256": CANONICAL_PROMPT_SHA256,
        "CUDA_VISIBLE_DEVICES": "0,1", "physical_gpu_ids": [0, 1], "gpus": gpu_rows,
        "world_size": 2, "deepspeed_version": deepspeed.__version__,
        "pytorch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "per_device_batch": int(config["training"]["batch_size_per_device"]),
        "gradient_accumulation_steps": int(config["training"]["gradient_accumulation_steps"]),
        "effective_global_batch": 20, "num_train_samples": train_count,
        "optimizer_step_budget": budget, "samples_per_optimizer_step": 20,
        "planned_total_sample_exposures": exposure,
        "effective_dataset_passes": exposure / train_count,
        "seed": int(config["experiment"]["seed"]), "precision": "bf16",
        "workers_per_rank": selected_workers,
        "checkpoint_root": str((REPO_ROOT / config["checkpoint"]["output_root"]).resolve()),
        "test_or_external_data_used_for_selection": False,
    }
    write_json(OUTPUT / "run_metadata.json", report)
    write_json(OUTPUT / "effective_batch_audit.json", {
        "registered_reference": {"micro_batch": 2, "world_size": 1, "gradient_accumulation": 10},
        "dual_gpu_runtime": {
            "micro_batch": int(config["training"]["batch_size_per_device"]),
            "world_size": 2,
            "gradient_accumulation": int(config["training"]["gradient_accumulation_steps"]),
        },
        "effective_global_batch_both": 20,
        "optimizer_steps": budget, "sample_exposures": exposure,
        "effective_dataset_passes": exposure / train_count,
    })
    return report


def main():
    args = parse_args(); config_path, config = load_config(args.config)
    if args.action == "sampler":
        value = sampler_audit(config)
    elif args.action == "workers":
        value = worker_benchmark(
            config, [int(x) for x in args.worker_candidates.split(",")], args.benchmark_batches
        )
    elif args.action == "compare":
        value = compare_smokes()
    else:
        value = metadata(config_path, config)
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
