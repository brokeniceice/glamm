#!/usr/bin/env python3
"""Incremental C2 adapter around the existing distributed P1/C1 runner."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import deepspeed
import torch
import train as glamm_train
from model.c2_preln_cross_attention import C2GLaMMForCausalLM
from scripts import phase2a_distributed_train as runner


def main():
    cli = runner.parse_args()
    if not cli.rine_conditioned_c1 or not cli.rine_checkpoint:
        raise ValueError("C2 requires --rine-conditioned-c1 and --rine-checkpoint")
    if cli.mode == "smoke" and not cli.save_smoke_checkpoint:
        raise ValueError("C2 smoke must use --save-smoke-checkpoint before validation preflight")
    config = __import__("yaml").safe_load((ROOT / cli.config).read_text())
    if config["experiment"]["name"] != "phase6j0_c2_preln_cross_attention":
        raise ValueError("Refusing to run C2 with another stage's config")

    glamm_train.GLaMMForCausalLM = C2GLaMMForCausalLM
    original_groups = glamm_train.build_optimizer_parameter_groups
    def c2_groups(model, args):
        c2_parameters = [(n, p) for n, p in model.named_parameters()
                         if "c2_cross_attention" in n and p.requires_grad]
        if len(c2_parameters) != 12:
            raise RuntimeError(f"Expected 12 C2 parameter tensors, found {len(c2_parameters)}")
        for _, parameter in c2_parameters:
            parameter.requires_grad_(False)
        try:
            groups = original_groups(model, args)
        finally:
            for _, parameter in c2_parameters:
                parameter.requires_grad_(True)
        groups.append({"name": "c2_cross_attention", "params": [p for _, p in c2_parameters],
                       "lr": float(config["optimizer"]["c2_cross_attention_lr"])})
        return groups
    glamm_train.build_optimizer_parameter_groups = c2_groups

    held = {}
    original_initialize = deepspeed.initialize
    def initialize(*args, **kwargs):
        result = original_initialize(*args, **kwargs)
        held["engine"] = result[0]
        return result
    deepspeed.initialize = initialize

    original_loader = runner.make_loader
    def make_loader(*args, **kwargs):
        result = original_loader(*args, **kwargs)
        if kwargs.get("batch_size") == 2:
            held["validation_loader"] = result
        return result
    runner.make_loader = make_loader

    output_root = ROOT / config["experiment"]["runtime_output_dir"]
    checkpoint_root = ROOT / config["checkpoint"]["output_root"]
    original_validation = runner.validation
    def protected_validation(engine, loader, output_dir, epoch, rank, world_size, device):
        # Runner normally saves last/best only AFTER validation. Preserve a
        # resumable optimizer/scheduler state first, including on validator errors.
        if cli.mode == "train":
            history_path = output_root / "training_state.json"
            previous_best = (json.loads(history_path.read_text()).get("best_val_total_loss", float("inf"))
                             if history_path.exists() else float("inf"))
            runner.checkpoint(engine, checkpoint_root / "pre_validation", best=False,
                              client_state={"optimizer_step": epoch * config["training"]["optimizer_steps_per_epoch"],
                                            "epoch": epoch, "world_size": world_size,
                                            "best_val_total_loss": float(previous_best),
                                            "checkpoint_reason": "before_validation"})
        return original_validation(engine, loader, output_dir, epoch, rank, world_size, device)
    runner.validation = protected_validation

    original_backward = deepspeed.runtime.engine.DeepSpeedEngine.backward
    def backward(engine, *args, **kwargs):
        result = original_backward(engine, *args, **kwargs)
        block = getattr(engine.module, "c2_cross_attention", None)
        if block is not None and engine.global_rank == 0:
            step = int(engine.global_steps) + 1
            if step == 1 or step % 25 == 0:
                output_dir = output_root if cli.mode == "train" else output_root / (cli.output_subdir or "preflight/dual")
                output_dir.mkdir(parents=True, exist_ok=True)
                record = {"optimizer_step": step, **block.last_stats,
                          "gradient_norms": dict(block.last_gradient_norms)}
                with (output_dir / "c2_attention_diagnostics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
        return result
    deepspeed.runtime.engine.DeepSpeedEngine.backward = backward

    runner.main()
    if cli.mode == "smoke":
        if "engine" not in held or "validation_loader" not in held:
            raise RuntimeError("Smoke preflight did not retain engine and validator")
        engine = held["engine"]
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        # A complete distributed validation path on two batches per rank.
        # The formal training validates on the full frozen validation split.
        from itertools import islice
        sample_loader = islice(held["validation_loader"], 2)
        output_dir = output_root / (cli.output_subdir or "preflight/dual")
        metrics = original_validation(engine, sample_loader, output_dir, 0,
                                      rank, world_size, torch.device("cuda", torch.cuda.current_device()))
        if rank == 0:
            (output_dir / "c2_validation_preflight.json").write_text(
                json.dumps({"status": "PASS", "sample_batches_per_rank": 2,
                            "metrics": metrics}, indent=2, allow_nan=False) + "\n")
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
