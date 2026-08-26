#!/usr/bin/env python3
"""Freeze Phase 4B-G protocol, provenance, schedule, and validation alias."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.fepn import FEPNv0, ForensicEvidenceProjector, parameter_manifest
from tools.phase4b import balanced_schedule, dump, file_sha256, load_jsonl


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4b_global_fepn_injection.yaml")
    cli = parser.parse_args()
    config_path = (ROOT / cli.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = (ROOT / cfg["experiment"]["output_root"]).resolve(); out.mkdir(parents=True, exist_ok=True)
    p1 = Path(cfg["source"]["checkpoint"]); fepn_path = Path(cfg["fepn"]["checkpoint"])
    checks = {
        "p1_exists": p1.is_file(), "fepn_exists": fepn_path.is_file(),
        "p1_sha_exact": p1.is_file() and file_sha256(p1) == cfg["source"]["checkpoint_sha256"],
        "fepn_sha_exact": fepn_path.is_file() and file_sha256(fepn_path) == cfg["fepn"]["checkpoint_sha256"],
        "canonical_prompt_exact": cfg["prompt"]["user_question"] == "Determine whether this image is authentic and explain the forensic evidence.",
        "legacy_disabled": cfg["prompt"]["legacy_allowed"] is False,
        "four_tokens": int(cfg["projector"]["evidence_tokens"]) == 4,
        "test_sealed": "internal_test" in cfg["prohibited"] and "official1000" in cfg["prohibited"],
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    train_path = ROOT / cfg["data"]["manifest_dir"] / "train_combined.jsonl"
    val_path = ROOT / cfg["data"]["manifest_dir"] / "val_combined.jsonl"
    train_rows, val_rows = load_jsonl(train_path), load_jsonl(val_path)
    if len(train_rows) != int(cfg["data"]["train_count"]) or len(val_rows) != int(cfg["data"]["val_count"]):
        raise RuntimeError("canonical split count drift")
    schedule = balanced_schedule(train_rows, exposures=int(cfg["training"]["total_image_exposures"]),
                                 seed=int(cfg["experiment"]["seed"]))
    dump(out / "training/schedule.json", schedule)
    alias = out / "evaluation/validation_manifest"; alias.mkdir(parents=True, exist_ok=True)
    shutil.copy2(val_path, alias / "test_combined.jsonl")
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    fepn = FEPNv0(image_mean, image_std)
    fepn_state = torch.load(fepn_path, map_location="cpu")
    fepn.load_state_dict(fepn_state["model"], strict=True)
    projector = ForensicEvidenceProjector(**{
        "input_dim": cfg["projector"]["input_dim"], "hidden_dim": cfg["projector"]["hidden_dim"],
        "num_tokens": cfg["projector"]["evidence_tokens"], "llm_dim": cfg["projector"]["llm_dim"]})
    dump(out / "experiment_manifest.json", {"status": "PREPARED_NOT_STARTED", "phase": "Phase 4B-G",
        "arms": cfg["experiment"]["arms"], "authorization": "global_interface_only",
        "canonical_prompt": cfg["prompt"], "internal_test_used": False, "official1000_used": False,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    dump(out / "initial_checkpoint_manifest.json", {"status": "FROZEN", **cfg["source"], **artifact(p1)})
    dump(out / "fepn_checkpoint_manifest.json", {"status": "FROZEN", **cfg["fepn"],
        "artifact": artifact(fepn_path), "state_epoch": fepn_state.get("epoch"),
        "architecture_manifest": parameter_manifest(fepn)})
    dump(out / "fepn_global_feature_spec.json", {"tensor": "FEPNv0.forward()['dense_features'].mean((2,3))",
        "shape": ["batch", 128], "classification_mlp_input": True, "scalar_logit": False,
        "activation_after_gap": None, "normalization_after_gap": None, "stop_gradient": True,
        "preprocess": {"source": "openai/clip-vit-large-patch14-336", "resolution": 336,
                       "image_mean": image_mean, "image_std": image_std}})
    dump(out / "interface_architecture.json", {"sequence": ["576_original_visual_embeddings", "4_continuous_FEPN_embeddings", "raw_text_embeddings"],
        "raw_image_token_net_expansion_without_evidence": 575, "with_evidence": 579,
        "tokenizer_changed": False, "attention": "same_causal_prefix", "evidence_enters_generation_KV_cache": True})
    dump(out / "projector_config.json", {**cfg["projector"], "parameter_count": projector.parameter_count,
        "formula": "Linear(128,256)-GELU-Linear(256,16384)-reshape(B,4,4096)", "sweeps": False})
    dump(out / "training_config.json", {"status": "FROZEN_BEFORE_TRAINING", "training": cfg["training"],
        "optimizer": cfg["optimizer"], "selector": cfg["selector"], "same_schedule_both_arms": True,
        "schedule_sha256": schedule["sha256"], "post_start_changes_allowed": False})
    dump(out / "preparation_audit.json", {"status": "PASS", "checks": checks,
        "train_count": len(train_rows), "val_count": len(val_rows), "schedule": {k: schedule[k] for k in ("exposures", "class_histogram", "sha256")}})
    print(json.dumps({"status": "PASS", "projector_parameters": projector.parameter_count,
                      "schedule_sha256": schedule["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
