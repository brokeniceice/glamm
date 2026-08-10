#!/usr/bin/env python3
"""Convert this run's ZeRO-2 AdamW checkpoint from DP world size 2 to 1.

The conversion is deliberately narrow: it requires two source shards, one
world-size-1 template with identical parameter groups, and verifies every
flattened group length before writing a new checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def load(path):
    return torch.load(path, map_location="cpu")


def main():
    args = parse_args()
    source = args.source.resolve() / "checkpoint"
    template_root = args.template.resolve()
    template = template_root / "checkpoint"
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    shards = sorted(source.glob("bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt"))
    if len(shards) != 2:
        raise ValueError(f"Expected exactly two optimizer shards, found {shards}")

    rank0, rank1 = (load(path) for path in shards)
    template_optim_path = template / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
    merged = load(template_optim_path)
    source_opts = [rank0["optimizer_state_dict"], rank1["optimizer_state_dict"]]
    target_opt = merged["optimizer_state_dict"]
    template_opt = load(template_optim_path)["optimizer_state_dict"]

    if any(list(opt["partition_count"]) != [2] * 6 for opt in source_opts):
        raise ValueError("Source is not the expected six-group DP-world-size-2 checkpoint")
    if list(target_opt["partition_count"]) != [1] * 6:
        raise ValueError("Template is not a DP-world-size-1 checkpoint")

    group_report = []
    target_fp32 = []
    target_state = {}
    for group_index, template_tensor in enumerate(template_opt["single_partition_of_fp32_groups"]):
        declared_padding = int(source_opts[1]["group_paddings"][group_index])

        def concatenate(values, target_numel):
            result = torch.cat([values[0], values[1]], dim=0)
            excess = result.numel() - target_numel
            if excess < 0:
                raise ValueError(
                    f"Group {group_index} is short by {-excess} values during merge"
                )
            return (result[:-excess] if excess else result), excess

        fp32, removed_fp32 = concatenate([
            opt["single_partition_of_fp32_groups"][group_index] for opt in source_opts
        ], template_tensor.numel())
        if fp32.numel() != template_tensor.numel():
            raise ValueError(
                f"Group {group_index} fp32 length mismatch: {fp32.numel()} != {template_tensor.numel()}"
            )
        target_fp32.append(fp32)

        states = [opt["base_optimizer_state"]["state"][group_index] for opt in source_opts]
        if float(states[0]["step"]) != float(states[1]["step"]):
            raise ValueError(f"Group {group_index} Adam step differs between ranks")
        exp_avg, removed_exp_avg = concatenate(
            [state["exp_avg"] for state in states], template_tensor.numel()
        )
        exp_avg_sq, removed_exp_avg_sq = concatenate(
            [state["exp_avg_sq"] for state in states], template_tensor.numel()
        )
        if exp_avg.numel() != template_tensor.numel() or exp_avg_sq.numel() != template_tensor.numel():
            raise ValueError(f"Group {group_index} Adam moment length mismatch")
        target_state[group_index] = {
            "step": states[0]["step"].clone(),
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
        }
        group_report.append({
            "group": group_index,
            "numel": fp32.numel(),
            "declared_rank1_padding": declared_padding,
            "removed_fp32_padding": removed_fp32,
            "removed_moment_padding": [removed_exp_avg, removed_exp_avg_sq],
            "adam_step": float(states[0]["step"]),
        })

    target_opt.update({
        "loss_scaler": source_opts[0]["loss_scaler"],
        "dynamic_loss_scale": source_opts[0]["dynamic_loss_scale"],
        "overflow": source_opts[0]["overflow"],
        "clip_grad": source_opts[0]["clip_grad"],
        "single_partition_of_fp32_groups": target_fp32,
        "partition_count": [1] * 6,
        "group_paddings": [0] * 6,
        "param_slice_mappings": template_opt["param_slice_mappings"],
    })
    target_opt["base_optimizer_state"] = {
        "state": target_state,
        "param_groups": source_opts[0]["base_optimizer_state"]["param_groups"],
    }
    merged["ds_config"] = load(template_optim_path)["ds_config"]

    destination_checkpoint = destination / "checkpoint"
    destination_checkpoint.mkdir(parents=True)
    torch.save(
        merged,
        destination_checkpoint / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
    )

    model_state = load(source / "mp_rank_00_model_states.pt")
    template_model_state = load(template / "mp_rank_00_model_states.pt")
    model_state["dp_world_size"] = 1
    model_state["world_size"] = 1
    model_state["ds_config"] = template_model_state["ds_config"]
    torch.save(model_state, destination_checkpoint / "mp_rank_00_model_states.pt")
    shutil.copy2(template_root / "zero_to_fp32.py", destination / "zero_to_fp32.py")
    (destination / "latest").write_text("checkpoint\n", encoding="utf-8")

    report = {
        "status": "converted",
        "source_dp_world_size": 2,
        "destination_dp_world_size": 1,
        "optimizer_step": int(model_state["optimizer_step"]),
        "global_steps": int(model_state["global_steps"]),
        "global_samples": int(model_state["global_samples"]),
        "groups": group_report,
    }
    (destination / "conversion_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
