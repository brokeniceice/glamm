#!/usr/bin/env python3
"""Frozen C0/P1 classification on LOKI, RAISE, and AIGI-test."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import CLIPImageProcessor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.forensics import summarize_detection_records
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from npr_expert.data import load_aigi_jsonl
from scripts.phase2a_final_evaluate import file_sha256, load_model
from scripts.phase2c_external_evaluate import ExternalClassificationDataset, build_external_rows


DATASETS = ("loki", "raise", "aigi_test")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-root", default="outputs/phase3a1_paired_control/evaluation/external_classification")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def json_scalar(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}")
        return value.item()
    return value


def normalize_prediction_fields(record: dict) -> dict:
    record = dict(record)
    record["cls_pred"] = "fake" if float(record["cls_prob_fake"]) >= .5 else "real"
    record["lm_verdict_pred"] = (
        "fake" if float(record["lm_verdict_prob_fake"]) >= .5 else "real"
    )
    record["cls_lm_agree"] = record["cls_pred"] == record["lm_verdict_pred"]
    return record


def build_rows() -> dict[str, list[dict]]:
    rows = build_external_rows()
    aigi = load_aigi_jsonl(
        ROOT / "datasets/AIGI-Holmes-Dataset/dataset/test.jsonl",
        ROOT / "datasets/AIGI-Holmes-Dataset",
        lambda image: image,
    )
    rows["aigi_test"] = [
        {
            "sample_id": f"aigi_test:{index:06d}",
            "image_path": image_path,
            "class_label": int(label),
            "forensics_domain": "fake" if label else "real",
            "source": source,
            "content_category": None,
        }
        for index, (image_path, label, source) in enumerate(aigi.samples)
    ]
    return rows


def evaluate_dataset(name, source_rows, backend, processor, output_root: Path, batch_size: int):
    destination = output_root / name
    prediction_path = destination / "predictions.jsonl"
    existing = read_jsonl(prediction_path)
    completed = {record["sample_id"] for record in existing}
    dataset = ExternalClassificationDataset(source_rows, processor)
    pending = [index for index, row in enumerate(source_rows) if row["sample_id"] not in completed]
    for start in range(0, len(pending), batch_size):
        indices = pending[start:start + batch_size]
        samples = [dataset[index] for index in indices]
        batch = backend._batch_many(samples, "")
        batch["grounding_enc_images"] = None
        with torch.no_grad():
            output = backend.model.model_forward(**batch)
        records = []
        for index, sample in enumerate(samples):
            prediction = {
                key: json_scalar(value)
                for key, value in backend._prediction_fields(output, index=index).items()
            }
            records.append(normalize_prediction_fields({
                "sample_id": sample["sample_id"],
                "image_path": sample["image_path"],
                "source": sample.get("source"),
                "content_type": sample.get("content_category"),
                "gt_label": "fake" if int(sample["cls_label"]) == 1 else "real",
                "eval_mode": "detection",
                **prediction,
            }))
        append_jsonl(prediction_path, records)
        print(f"external-{name} {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)
    records = read_jsonl(prediction_path)
    expected_ids = [row["sample_id"] for row in source_rows]
    indexed = {record["sample_id"]: record for record in records}
    if len(indexed) != len(expected_ids) or set(indexed) != set(expected_ids):
        raise RuntimeError(f"{name} result scope mismatch: got {len(indexed)}, expected {len(expected_ids)}")
    ordered = [normalize_prediction_fields(indexed[sample_id]) for sample_id in expected_ids]
    write_jsonl(prediction_path, ordered)
    metrics = summarize_detection_records(ordered)
    dump(destination / "metrics.json", metrics)
    return metrics


def main(argv=None):
    cli = parse_args(argv)
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    config_path = (ROOT / cli.config).resolve()
    checkpoint_path = (ROOT / cli.checkpoint).resolve()
    output_root = (ROOT / cli.output_root / cli.model_name).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, checkpoint = load_model(
        config, checkpoint_path, device,
        expected_step=cli.expected_step, expected_epoch=cli.expected_epoch,
    )
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16,
        use_mm_start_end=True, max_new_tokens=1,
    )
    processor = CLIPImageProcessor.from_pretrained(config["model"]["vision_tower"])
    available = build_rows()
    results = {
        name: evaluate_dataset(name, available[name], backend, processor, output_root, cli.batch_size)
        for name in cli.datasets
    }
    summary = {
        "model_name": cli.model_name,
        "checkpoint": checkpoint,
        "checkpoint_file_sha256": file_sha256(checkpoint_path),
        "config": str(config_path),
        "seed": cli.seed,
        "batch_size": cli.batch_size,
        "threshold": 0.5,
        "prompt": "canonical unified forensic question with empty assistant content and fixed [CLS] query",
        "selection_used_external": False,
        "datasets": results,
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
    }
    dump(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
