#!/usr/bin/env python3
"""Evaluate diagnostic-only Phase 3D.2-C in-memory module swaps B or C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics import _localization_record
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase2a_distributed_train import deterministic_tensor_collection_hash
from scripts.phase2a_final_evaluate import file_sha256, load_model
from scripts.phase3a_evaluate import preserve_spatial_prediction, summarize_localization
from tools.phase3d2 import parameter_group

OUT = ROOT / "outputs/phase3d2c_spatial_generalization_attribution"
P3D2_CONFIG = ROOT / "configs/phase3d2_direct_spatial_path.yaml"
POPULATIONS = ("seen_train", "train_holdout", "validation_fake")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def group_hash(model, group: str) -> dict:
    return deterministic_tensor_collection_hash(
        (name, tensor) for name, tensor in model.state_dict().items() if parameter_group(name) == group
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combination", choices=("B", "C"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = yaml.safe_load(P3D2_CONFIG.read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((ROOT / cfg["source"]["model_config"]).read_text(encoding="utf-8"))
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    device = torch.device(args.device); torch.cuda.set_device(device)
    p1_path = Path(cfg["source"]["checkpoint"])
    step250_path = Path(cfg["experiment"]["checkpoint_root"]) / "step_0250/checkpoint/mp_rank_00_model_states.pt"
    p1_hash_before, step250_hash_before = file_sha256(p1_path), file_sha256(step250_path)
    model, tokenizer, _ = load_model(
        model_cfg, p1_path, device, expected_step=int(cfg["source"]["optimizer_step"]),
        expected_epoch=int(cfg["source"]["epoch"]),
    )
    model.eval(); model.requires_grad_(False)
    swapped_group = "text_hidden_fcs" if args.combination == "B" else "mask_decoder"
    baseline_hashes = {group: group_hash(model, group) for group in ("text_hidden_fcs", "mask_decoder")}
    source = torch.load(step250_path, map_location="cpu")["module"]
    state = model.state_dict(); copied = 0
    with torch.no_grad():
        for name, tensor in source.items():
            if name in state and parameter_group(name) == swapped_group:
                state[name].copy_(tensor.to(device=state[name].device, dtype=state[name].dtype)); copied += 1
    del source
    if copied == 0:
        raise RuntimeError(f"no tensors copied for diagnostic group {swapped_group}")
    composition_hashes = {group: group_hash(model, group) for group in ("text_hidden_fcs", "mask_decoder")}
    if composition_hashes[swapped_group]["sha256"] == baseline_hashes[swapped_group]["sha256"]:
        raise RuntimeError("diagnostic swapped group did not change")
    other_group = "mask_decoder" if swapped_group == "text_hidden_fcs" else "text_hidden_fcs"
    if composition_hashes[other_group]["sha256"] != baseline_hashes[other_group]["sha256"]:
        raise RuntimeError("non-swapped diagnostic group changed")
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(model_cfg["evaluation"]["max_new_tokens"]),
    )
    population_summaries = {}
    for population in POPULATIONS:
        dataset = UnifiedForensicsDataset(
            OUT / f"populations/{population}", tokenizer, model_cfg["model"]["vision_tower"], split="test",
            datasets_root=cfg["data"]["datasets_root"], synthscars_root=cfg["data"]["synthscars_root"],
            image_size=int(model_cfg["model"]["image_size"]), target_protocol="phrase_aligned",
        )
        output_dir = OUT / f"module_swap/{args.combination}/{population}"
        predictions = output_dir / "predictions.jsonl"
        if predictions.exists(): predictions.unlink()
        with torch.no_grad():
            for index in range(len(dataset)):
                sample = dataset[index]
                output = backend.teacher_forced_localization(sample, context="full", user_prompt="canonical")
                record = _localization_record(
                    sample, output, "tf_full_context", uses_gt_authenticity=True,
                    uses_gt_explanation=True, classification_gate=False,
                )
                record = preserve_spatial_prediction(
                    output_dir, "tf_full_context", sample, output, record, save_spatial=False,
                )
                record.update({"module_combination": args.combination, "population": population})
                append(predictions, record)
                if (index + 1) % 25 == 0:
                    print(f"phase3d2c-module-{args.combination} {population} {index + 1}/{len(dataset)}", flush=True)
        values = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line]
        metrics = summarize_localization(values, autoregressive=False)
        dump(output_dir / "metrics.json", metrics)
        population_summaries[population] = metrics
    after_hashes = {group: group_hash(model, group) for group in ("text_hidden_fcs", "mask_decoder")}
    if after_hashes != composition_hashes or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("module-swap model changed during inference")
    if file_sha256(p1_path) != p1_hash_before or file_sha256(step250_path) != step250_hash_before:
        raise RuntimeError("source checkpoint file changed")
    audit = {
        "status": "PASS", "combination": args.combination,
        "composition": "FC_250+Decoder_0" if args.combination == "B" else "FC_0+Decoder_250",
        "swapped_group": swapped_group, "copied_tensor_count": copied,
        "baseline_group_hashes": baseline_hashes, "composition_group_hashes": composition_hashes,
        "after_inference_group_hashes": after_hashes,
        "runtime": {"model_eval": not model.training, "torch_no_grad": True, "gradients_absent": True},
        "checkpoint_files_unchanged": True, "new_checkpoint_saved": False,
        "population_summaries": population_summaries,
    }
    dump(OUT / f"module_swap/{args.combination}/invariance_audit.json", audit)
    print(json.dumps({"status": "PASS", "combination": args.combination}, indent=2))


if __name__ == "__main__":
    main()
