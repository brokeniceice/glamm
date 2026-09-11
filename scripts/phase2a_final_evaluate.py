#!/usr/bin/env python3
"""Frozen internal-test evaluation for the completed Phase 2A baseline.

The runner is deliberately resumable at sample granularity.  G0 and Joint
share their one canonical free-generation trajectory; Joint only adds the
preregistered classification-head gate when the stored record is scored.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train as glamm_train
from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics import (
    _localization_record,
    evaluate_detection,
    evaluate_gt_fake_generation_localization,
    evaluate_teacher_forced_full_context,
    summarize_detection_records,
    summarize_localization_records,
)
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from scripts.phase1d_training_policy import configure_args


MODES = ("detection", "G0", "G1", "tf_full_context", "joint")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2a_unified_baseline_full.yaml")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/phase2a_unified_baseline/single/best/checkpoint/mp_rank_00_model_states.pt",
    )
    parser.add_argument("--output-dir", default="outputs/phase2a_unified_baseline/test")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--expected-step", type=int, default=2500)
    parser.add_argument("--expected-epoch", type=int, default=5)
    parser.add_argument("--manifest-dir", default=None)
    parser.add_argument("--synthscars-root", default=None)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args(argv)


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def paths(root: Path, mode: str) -> tuple[Path, Path]:
    directory = root / mode
    return directory / "predictions.jsonl", directory / "metrics.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(config: dict, checkpoint_path: Path, device: torch.device,
               *, expected_step: int, expected_epoch: int):
    args = configure_args(config)
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(args)
    model = glamm_train.initialize_model(args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, args)
    if config.get("forensics", {}).get("conditioning") == "frozen_rine_q2_direct_embedding_token":
        rine_path = Path(config["forensics"]["rine_checkpoint"])
        if not rine_path.is_absolute():
            rine_path = REPO_ROOT / rine_path
        model.enable_rine_conditioning(rine_path.resolve())
    state = torch.load(checkpoint_path, map_location="cpu")
    if int(state.get("optimizer_step", -1)) != expected_step or int(state.get("epoch", -1)) != expected_epoch:
        raise ValueError(
            f"Expected checkpoint at step {expected_step}/epoch {expected_epoch}, got "
            f"{state.get('optimizer_step')}/{state.get('epoch')}"
        )
    missing, unexpected = model.load_state_dict(state["module"], strict=False)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected}")
    model.gradient_checkpointing_disable()
    model.requires_grad_(False)
    model.to(device=device, dtype=torch.bfloat16).eval()
    if hasattr(model, "rine_conditioner"):
        model.rine_conditioner.rine.float()
        model.rine_conditioner.eval()
    return model, tokenizer, {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "optimizer_step": int(state["optimizer_step"]),
        "epoch": int(state["epoch"]),
        # Post-training checkpoints may be selector-driven and therefore do not
        # carry a training-loss best value.  It is metadata only and must not
        # block otherwise compatible frozen evaluation.
        "best_val_total_loss": float(state.get("best_val_total_loss", float("nan"))),
        "missing_frozen_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def summarize(mode: str, records: list[dict]) -> dict:
    if mode == "detection":
        return summarize_detection_records(records)
    return summarize_localization_records(
        records, autoregressive=mode in {"G0", "G1", "joint"}, joint=mode == "joint"
    )


def main(argv=None):
    cli = parse_args(argv)
    config_path = (REPO_ROOT / cli.config).resolve()
    checkpoint_path = (REPO_ROOT / cli.checkpoint).resolve()
    output_root = (REPO_ROOT / cli.output_dir).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    model, tokenizer, checkpoint_meta = load_model(
        config, checkpoint_path, device,
        expected_step=cli.expected_step, expected_epoch=cli.expected_epoch,
    )
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
    )
    manifest_dir = Path(cli.manifest_dir).resolve() if cli.manifest_dir else REPO_ROOT / config["data"]["manifest_dir"]
    synthscars_root = cli.synthscars_root or config["data"]["synthscars_root"]
    dataset = UnifiedForensicsDataset(
        manifest_dir, tokenizer, config["model"]["vision_tower"],
        split="test", datasets_root=config["data"]["datasets_root"],
        synthscars_root=synthscars_root, image_size=int(config["model"]["image_size"]),
    )
    limit = len(dataset) if cli.max_samples is None else min(len(dataset), cli.max_samples)
    requested = set(cli.modes)
    # Joint shares the exact canonical free-generation trajectory with G0.
    if "joint" in requested:
        requested.add("G0")
    for mode in requested:
        prediction_path, _ = paths(output_root, mode)
        if cli.reset and prediction_path.exists():
            prediction_path.unlink()

    completed = {
        mode: {row["sample_id"] for row in read_records(paths(output_root, mode)[0])}
        for mode in requested | ({"joint"} if "joint" in cli.modes else set())
    }
    generation_modes = requested & {"G0", "G1", "joint"}
    if generation_modes:
        fake_indices = [
            index for index in range(limit)
            if int(dataset.rows[index]["class_label"]) == 1
        ]
        batch_size = max(1, int(cli.generation_batch_size))
        for start in range(0, len(fake_indices), batch_size):
            indices = fake_indices[start:start + batch_size]
            samples = [dataset[index] for index in indices]
            g0_samples = [
                sample for sample in samples
                if ("G0" in requested and sample["sample_id"] not in completed["G0"])
                or ("joint" in cli.modes and sample["sample_id"] not in completed["joint"])
            ]
            if g0_samples:
                outputs = backend.generate_localization_batch(
                    g0_samples, provide_gt_fake=False, generation_mode="unified_fake_generate"
                )
                for sample, output in zip(g0_samples, outputs):
                    sample_id = sample["sample_id"]
                    if "G0" in requested and sample_id not in completed["G0"]:
                        g0 = _localization_record(
                            sample, output, "unified_fake_generate", uses_gt_authenticity=False,
                            uses_gt_explanation=False, classification_gate=False,
                        )
                        append_records(paths(output_root, "G0")[0], [g0])
                        completed["G0"].add(sample_id)
                    if "joint" in cli.modes and sample_id not in completed["joint"]:
                        joint_output = copy.copy(output); joint_output["generation_mode"] = "joint"
                        joint = _localization_record(
                            sample, joint_output, "joint", uses_gt_authenticity=False,
                            uses_gt_explanation=False, classification_gate=True,
                        )
                        append_records(paths(output_root, "joint")[0], [joint])
                        completed["joint"].add(sample_id)
            g1_samples = [
                sample for sample in samples
                if "G1" in requested and sample["sample_id"] not in completed["G1"]
            ]
            if g1_samples:
                outputs = backend.generate_localization_batch(
                    g1_samples, provide_gt_fake=True,
                    generation_mode="unified_prompt_gt_fake_prefix",
                )
                for sample, output in zip(g1_samples, outputs):
                    record = _localization_record(
                        sample, output, "gt_fake_generate", uses_gt_authenticity=True,
                        uses_gt_explanation=False, classification_gate=False,
                    )
                    append_records(paths(output_root, "G1")[0], [record])
                    completed["G1"].add(sample["sample_id"])
            if (start // batch_size + 1) % 10 == 0:
                print(
                    f"phase2a-generation progress={min(start + batch_size, len(fake_indices))}/"
                    f"{len(fake_indices)} batch_size={batch_size}", flush=True,
                )
        requested -= {"G0", "G1", "joint"}
    for index in range(limit):
        sample = dataset[index]
        sample_id = sample["sample_id"]
        is_fake = int(sample["cls_label"]) == 1 and bool(sample["seg_valid"])
        if "detection" in requested and sample_id not in completed["detection"]:
            records, _ = evaluate_detection([sample], backend, user_prompt="canonical")
            append_records(paths(output_root, "detection")[0], records)
            completed["detection"].add(sample_id)
        if not is_fake:
            continue
        if "G0" in requested and (
            sample_id not in completed["G0"]
            or ("joint" in cli.modes and sample_id not in completed["joint"])
        ):
            output = backend.generate_localization(
                sample, provide_gt_fake=False, generation_mode="unified_fake_generate"
            )
            if sample_id not in completed["G0"]:
                g0 = _localization_record(
                    sample, output, "unified_fake_generate", uses_gt_authenticity=False,
                    uses_gt_explanation=False, classification_gate=False,
                )
                append_records(paths(output_root, "G0")[0], [g0])
                completed["G0"].add(sample_id)
            if "joint" in cli.modes and sample_id not in completed["joint"]:
                joint_output = copy.copy(output)
                joint_output["generation_mode"] = "joint"
                joint = _localization_record(
                    sample, joint_output, "joint", uses_gt_authenticity=False,
                    uses_gt_explanation=False, classification_gate=True,
                )
                append_records(paths(output_root, "joint")[0], [joint])
                completed["joint"].add(sample_id)
        if "G1" in requested and sample_id not in completed["G1"]:
            records, _ = evaluate_gt_fake_generation_localization([sample], backend)
            append_records(paths(output_root, "G1")[0], records)
            completed["G1"].add(sample_id)
        if "tf_full_context" in requested and sample_id not in completed["tf_full_context"]:
            records, _ = evaluate_teacher_forced_full_context(
                [sample], backend, user_prompt="canonical"
            )
            append_records(paths(output_root, "tf_full_context")[0], records)
            completed["tf_full_context"].add(sample_id)
        if (index + 1) % 25 == 0:
            print(f"phase2a-eval progress={index + 1}/{limit}", flush=True)

    summary = {"checkpoint": checkpoint_meta, "test_samples": limit, "modes": {}}
    final_modes = set(cli.modes) | ({"G0"} if "joint" in cli.modes else set())
    for mode in MODES:
        if mode not in final_modes:
            continue
        prediction_path, metrics_path = paths(output_root, mode)
        records = read_records(prediction_path)
        metrics = summarize(mode, records)
        write_metrics(metrics_path, metrics)
        summary["modes"][mode] = metrics
    summary_suffix = "_".join(cli.modes)
    write_metrics(output_root / f"summary_{summary_suffix}.json", summary)
    if tuple(cli.modes) == MODES:
        write_metrics(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
