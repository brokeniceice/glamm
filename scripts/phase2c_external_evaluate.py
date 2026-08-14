#!/usr/bin/env python3
"""One-shot frozen external evaluation after all Phase 2C checkpoints are selected."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.forensics.unified import (
    CANONICAL_PROMPT_SHA256, CANONICAL_PROMPT_TEMPLATE_ID, CANONICAL_UNIFIED_QUESTION,
    UnifiedForensicsDataset,
)
from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from npr_expert.transforms import build_npr_transform
from scripts.phase2a_final_evaluate import load_model
from scripts.phase2c_forensic_fusion import (
    OUTPUT_ROOT, VARIANT_DIRS, binary_metrics, load_config, load_expert, load_fusion,
    score_model, two_class_probability, write_json, write_jsonl,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--datasets", nargs="+", choices=("loki", "raise", "synthscars_official"),
                        default=("loki", "raise", "synthscars_official"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def build_external_rows():
    manifest_root = REPO_ROOT / "outputs/data_audits/unified_forensics_split_v1"
    raise_rows = read_jsonl(manifest_root / "raise_heldout_real_clean.jsonl")
    for row in raise_rows:
        row["image_path"] = str((REPO_ROOT / "datasets" / row["image_relpath"]).resolve())
    synth_rows = read_jsonl(manifest_root / "official_synthscars_test.jsonl")
    loki_raw = json.loads((REPO_ROOT / "datasets/LOKI/true_or_false.json").read_text())
    loki_rows, seen = [], set()
    for item in loki_raw:
        relative = item["image_path"]
        if relative in seen:
            continue
        seen.add(relative)
        answer = str(item.get("answer", "")).strip().lower()
        if answer not in {"yes", "no"}:
            continue
        path = (REPO_ROOT / "datasets/LOKI" / relative).resolve()
        loki_rows.append({
            "sample_id": f"loki:{relative}", "image_path": str(path),
            "class_label": int(answer == "yes"), "forensics_domain": "fake" if answer == "yes" else "real",
            "source": "LOKI", "content_category": None,
        })
    return {"loki": loki_rows, "raise": raise_rows, "synthscars_official": synth_rows}


class ExternalClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["image_path"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None: raise OSError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        global_image = self.processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return {
            "image_path": str(path), "global_enc_image": global_image, "grounding_enc_image": None,
            "bboxes": None, "conversations": ["unused"], "masks": None, "label": None,
            "resize": image.shape[:2], "questions": [CANONICAL_UNIFIED_QUESTION],
            "sampled_classes": [], "cls_label": int(row["class_label"]), "seg_valid": False,
            "sample_id": row["sample_id"], "source": row.get("source"),
            "content_category": row.get("content_category"), "manifest_row": row,
            "prompt_template_id": CANONICAL_PROMPT_TEMPLATE_ID,
            "prompt_sha256": CANONICAL_PROMPT_SHA256,
        }


def extract_dataset(name, rows, backend, expert, transform, device, batch_size, force):
    destination = OUTPUT_ROOT / "external" / name / "features.pt"
    if destination.exists() and not force:
        return torch.load(destination, map_location="cpu")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    dataset = ExternalClassificationDataset(rows, processor)
    captured = []
    hook = backend.model.classification_head.register_forward_hook(
        lambda _module, inputs, _output: captured.append(inputs[0].detach())
    )
    storage = {key: [] for key in ("base_logits", "lm_logits", "h_cls", "npr", "srm")}
    expert_device = next(expert.parameters()).device
    before_all = time.perf_counter()
    try:
        for start in range(0, len(dataset), batch_size):
            samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            batch = backend._batch_many(samples, "")
            batch["grounding_enc_images"] = None
            captured.clear()
            with torch.no_grad(): output = backend.model.model_forward(**batch)
            expert_tensors = []
            for sample in samples:
                with Image.open(sample["image_path"]) as image:
                    expert_tensors.append(transform(image.convert("RGB")))
            tensors = torch.stack(expert_tensors).to(expert_device)
            with torch.no_grad(): branches = expert.extract_branch_features(tensors)
            storage["base_logits"].append(output["cls_logits"].float().cpu())
            storage["lm_logits"].append(output["lm_verdict_logits"].float().cpu())
            storage["h_cls"].append(captured[0].float().cpu().half())
            storage["npr"].append(branches["npr"].cpu().half())
            storage["srm"].append(branches["srm_gated"].cpu().half())
            print(f"external {name}: {min(start + batch_size, len(dataset))}/{len(dataset)}", flush=True)
    finally:
        hook.remove()
    payload = {
        "metadata": {"name": name, "samples": len(rows), "seconds": time.perf_counter() - before_all,
                     "selection_allowed": False, "evaluated_after_all_checkpoints_selected": True},
        "sample_ids": [row["sample_id"] for row in rows],
        "labels": torch.tensor([row["class_label"] for row in rows], dtype=torch.long),
        "sources": [row.get("source") for row in rows],
        "content_categories": [row.get("content_category") for row in rows],
        "image_paths": [row["image_path"] for row in rows],
        **{key: torch.cat(values) for key, values in storage.items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return payload


def external_metrics(name, labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = probabilities >= 0.5
    if name == "raise":
        return {"n": len(labels), "real_accuracy_specificity": float((~predictions).mean()),
                "false_positive_rate": float(predictions.mean()),
                "mean_fake_probability": float(probabilities.mean())}
    if name == "synthscars_official":
        return {"n": len(labels), "fake_recall": float(predictions.mean()),
                "mean_fake_probability": float(probabilities.mean())}
    return binary_metrics(labels, probabilities)


def main(argv=None):
    cli = parse_args(argv)
    # Hard gate: external evaluation is invalid until all three validation selectors exist.
    for dirname in VARIANT_DIRS.values():
        selection = OUTPUT_ROOT / dirname / "train/selection.json"
        best = OUTPUT_ROOT / dirname / "train/best.pt"
        if not selection.exists() or not best.exists():
            raise RuntimeError(f"external evaluation blocked before frozen selection: {dirname}")
    config = load_config("configs/phase2c_npr_srm.yaml")
    phase2a_config = load_config("configs/phase2a_unified_baseline_full.yaml")
    device = torch.device(cli.device)
    model, tokenizer, _ = load_model(
        phase2a_config, REPO_ROOT / config["base_checkpoint"]["path"], device,
        expected_step=2500, expected_epoch=5,
    )
    expert, _ = load_expert(config, device)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=1)
    transform = build_npr_transform(training=False)
    external = {name: rows for name, rows in build_external_rows().items() if name in set(cli.datasets)}
    caches = {name: extract_dataset(name, rows, backend, expert, transform, device,
                                    cli.batch_size, cli.force) for name, rows in external.items()}
    summary = {}
    models = {"B0_phase2a": None}
    for variant, dirname in VARIANT_DIRS.items():
        models[dirname] = load_fusion(OUTPUT_ROOT / dirname, device)[0]
    for model_name, fusion in models.items():
        summary[model_name] = {}
        for name, cache in caches.items():
            if fusion is None:
                probabilities = two_class_probability(cache["base_logits"])
            else:
                _, _, probabilities, _ = score_model(fusion, cache, device)
            metrics = external_metrics(name, cache["labels"].numpy(), probabilities)
            summary[model_name][name] = metrics
            rows = [{"sample_id": sample_id, "gt": int(label), "prob_fake": float(prob),
                     "pred": int(prob >= 0.5)} for sample_id, label, prob in
                    zip(cache["sample_ids"], cache["labels"], probabilities)]
            write_jsonl(OUTPUT_ROOT / "external" / name / model_name / "predictions.jsonl", rows)
            write_json(OUTPUT_ROOT / "external" / name / model_name / "metrics.json", metrics)
    write_json(OUTPUT_ROOT / "external/summary.json", {
        "selection_used_external": False, "models_frozen_before_external": True, "results": summary,
    })
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
