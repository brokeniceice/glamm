#!/usr/bin/env python3
"""Generate frozen K=8 validation rollouts for a selected Phase 3D.1 policy."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import file_sha256, load_model
from scripts.phase3d0_generate import stable_seed, stochastic_group


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    path = Path(path)
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n"); handle.flush()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("R3", "Q2"), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(1, 2), required=True)
    cli = parser.parse_args(argv)
    expected = 1 if cli.arm == "R3" else 2
    if cli.physical_gpu != expected or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, str(expected)):
        raise RuntimeError("GPU assignment mismatch")
    cfg = yaml.safe_load((ROOT / "configs/phase3d1_policy_optimization.yaml").read_text())
    root = (ROOT / cfg["experiment"]["output_root"]).resolve()
    selector = load(root / "evaluation/selector" / f"{cli.arm}_selector.json")
    checkpoint = Path(selector["selected_checkpoint"]).resolve()
    model_cfg = yaml.safe_load((ROOT / cfg["source"]["model_config"]).read_text())
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    seed = int(cfg["experiment"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, _ = load_model(
        model_cfg, checkpoint, device,
        expected_step=int(selector["optimizer_step"]), expected_epoch=int(selector["logical_epoch"]),
    )
    model.eval(); model.requires_grad_(False)
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(cfg["sampling"]["max_new_tokens"]),
    )
    dataset = UnifiedForensicsDataset(
        ROOT / cfg["data"]["manifest_dir"], tokenizer, model_cfg["model"]["vision_tower"], split="val",
        datasets_root=cfg["data"]["datasets_root"], synthscars_root=cfg["data"]["synthscars_root"],
        image_size=int(model_cfg["model"]["image_size"]), target_protocol="phrase_aligned",
    )
    output = root / "rollouts" / f"{cli.arm}_OPT_selected_K8.jsonl"
    done = {row["sample_id"] for row in rows(output)}
    setting = cfg["sampling"]
    tensor_root = root / "rollouts/tensors" / f"{cli.arm}_OPT"
    started = time.time(); processed = 0
    for index in range(len(dataset)):
        sample = dataset[index]; sample_id = sample["sample_id"]
        if sample_id in done: continue
        group = stochastic_group(
            model, tokenizer, backend, sample, K=8, setting=setting,
            seed=stable_seed(seed, "val", "A", sample_id), text_only=False,
            tensor_root=tensor_root, population="val", setting_name="A",
        )
        append(output, {"sample_id": sample_id, "dataset_index": index, "rollouts": group})
        done.add(sample_id); processed += 1
        if processed % 10 == 0:
            print(json.dumps({"arm": cli.arm, "completed": len(done), "total": len(dataset)}), flush=True)
    dump(root / "rollouts" / f"{cli.arm}_OPT_selected_K8_audit.json", {
        "status": "COMPLETE", "arm": cli.arm, "K": 8, "groups": len(done),
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "sampling": setting, "physical_gpu": cli.physical_gpu,
        "elapsed_seconds": time.time() - started, "reward_tuning": False,
    })


if __name__ == "__main__":
    main()
