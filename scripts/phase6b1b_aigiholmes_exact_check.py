#!/usr/bin/env python3
"""Online-BF16 AIGI-Holmes parity check for C1-Exact and LEGION-retrained."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import CLIPImageProcessor, CLIPVisionModel


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6b1b_aigiholmes_exact_check"
MANIFEST = ROOT / "datasets/AIGI-Holmes/manifests/eval_manifest.jsonl"
EXACT = ROOT / "outputs/phase6b1a_exact_stage2_control/checkpoints/checkpoint-554"
LEGION = ROOT / "checkpoints/phase5a3_legion_retrained/stage2_cls/checkpoints/checkpoint-554"
CLIP = ROOT / "checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
EXPECTED_MANIFEST_SHA = "380325bc0cbba1e80044d0d2dbb40ff2f788865288bb47402c1d4b7967431982"
EXPECTED_CLIP_HASH = "ea25ce94579902eb0a94c9638c0277360f2b93cc156eb20f2fc4a481653afc17"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(named) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(named):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def load_head(directory: Path) -> tuple[torch.nn.Module, str]:
    index = json.loads((directory / "pytorch_model.bin.index.json").read_text())
    shards = {index["weight_map"][name] for name in index["weight_map"] if name.startswith("prediction_head.")}
    if len(shards) != 1:
        raise RuntimeError(f"prediction head spans unexpected shards: {shards}")
    full = torch.load(directory / next(iter(shards)), map_location="cpu")
    state = {name.removeprefix("prediction_head."): value.clone()
             for name, value in full.items() if name.startswith("prediction_head.")}
    del full
    if set(state) != {"0.weight", "0.bias", "2.weight", "2.bias"}:
        raise RuntimeError("prediction-head key drift")
    head = torch.nn.Sequential(torch.nn.Linear(1024, 2048), torch.nn.ReLU(), torch.nn.Linear(2048, 2))
    head.load_state_dict(state, strict=True)
    return head, tensor_hash(state.items())


def metrics(records: list[dict], prefix: str) -> dict:
    labels = np.asarray([row["gt"] for row in records], dtype=np.int64)
    probability = np.asarray([row[f"{prefix}_prob_fake"] for row in records], dtype=np.float64)
    predicted = probability >= 0.5
    tp = int(((labels == 1) & predicted).sum()); tn = int(((labels == 0) & ~predicted).sum())
    fp = int(((labels == 0) & predicted).sum()); fn = int(((labels == 1) & ~predicted).sum())
    return {
        "n": len(labels), "real": int((labels == 0).sum()), "fake": int((labels == 1).sum()),
        "accuracy": float((labels == predicted).mean()),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "fake_recall": float(recall_score(labels, predicted, zero_division=0)),
        "tnr": float(tn / (tn + fp)), "fpr": float(fp / (tn + fp)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": 0.5,
    }


def write_report(result: dict) -> None:
    a, b, c = result["c1_exact_metrics"], result["legion_retrained_metrics"], result["parity"]
    verdict = result["verdict"]
    report = f"""# Phase 6B.1b — C1-Exact AIGI-Holmes Sanity Check

## Verdict

`{verdict}`

C1-Exact checkpoint-554 与 LEGION-retrained checkpoint-554 在 AIGI-Holmes official TestSet 的同一 99,999 张图上，使用完全相同的顺序、RGB/CLIP preprocessing、逐图 online BF16 CLIP forward、Fake-positive convention 和 threshold=0.5。

| Model | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|
| C1-Exact | {a['accuracy']:.9f} | {a['roc_auc']:.9f} | {a['fake_recall']:.9f} | {a['tnr']:.9f} | {a['fpr']:.9f} | {a['f1']:.9f} |
| LEGION-retrained | {b['accuracy']:.9f} | {b['roc_auc']:.9f} | {b['fake_recall']:.9f} | {b['tnr']:.9f} | {b['fpr']:.9f} | {b['f1']:.9f} |

- prediction disagreements: `{c['prediction_disagreement_count']}`
- maximum absolute probability difference: `{c['probability_max_abs_difference']:.12g}`
- maximum absolute logit difference: `{c['logit_max_abs_difference']:.12g}`
- prediction-head tensors bitwise identical: `{str(c['prediction_head_tensors_bitwise_identical']).lower()}`

## Interpretation

{result['interpretation']}

## Protocol and firewall

- C1-Exact: `{result['checkpoints']['c1_exact']}`
- LEGION-retrained: `{result['checkpoints']['legion_retrained']}`
- manifest SHA256: `{result['manifest_sha256']}`
- CLIP parameter SHA256: `{result['clip_parameter_sha256']}`
- batch size: 1, sample order unchanged
- accessed external datasets: AIGI-Holmes official TestSet only
- no training, threshold tuning, calibration, fusion, or other external dataset access

阶段完成后 STOP。
"""
    path = ROOT / "docs/phase6b1b_aigiholmes_exact_check.md"
    path.write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "status.json"
    state = {"status": "RUNNING", "pid": os.getpid(), "started_at_utc": now(),
             "dataset": "AIGI-Holmes official TestSet", "batch_size": 1}
    atomic_json(status_path, state)
    started = time.monotonic()
    try:
        if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise RuntimeError("AIGI-Holmes manifest drift")
        manifest = rows(MANIFEST)
        if len(manifest) != 99999 or len({row["sample_id"] for row in manifest}) != 99999:
            raise RuntimeError("AIGI-Holmes population drift")
        exact_head, exact_hash = load_head(EXACT)
        legion_head, legion_hash = load_head(LEGION)
        heads_equal = exact_hash == legion_hash and all(
            torch.equal(x, y) for x, y in zip(exact_head.state_dict().values(), legion_head.state_dict().values())
        )

        device = torch.device("cuda:0")
        clip = CLIPVisionModel.from_pretrained(CLIP, local_files_only=True).to(device=device, dtype=torch.bfloat16).eval()
        for parameter in clip.parameters():
            parameter.requires_grad_(False)
        clip_hash = tensor_hash((f"vision_tower.{name}", value) for name, value in clip.state_dict().items())
        if clip_hash != EXPECTED_CLIP_HASH:
            raise RuntimeError(f"CLIP parameter hash drift: {clip_hash}")
        processor = CLIPImageProcessor.from_pretrained(CLIP, local_files_only=True)
        exact_head = exact_head.to(device=device, dtype=torch.bfloat16).eval()
        legion_head = legion_head.to(device=device, dtype=torch.bfloat16).eval()

        prediction_path = OUT / "predictions.jsonl"
        existing = rows(prediction_path) if prediction_path.exists() else []
        done = {row["sample_id"] for row in existing}
        allowed = {row["sample_id"] for row in manifest}
        if not done.issubset(allowed):
            raise RuntimeError("resume output contains out-of-scope samples")
        for index, row in enumerate(manifest):
            if row["sample_id"] in done:
                continue
            bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(row["image_path"])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pixels = processor.preprocess(rgb, return_tensors="pt")["pixel_values"].to(device=device, dtype=torch.bfloat16)
            with torch.inference_mode():
                feature = clip(pixels, output_hidden_states=True).hidden_states[-2][:, 0]
                exact_logits = exact_head(feature).float()[0]
                legion_logits = legion_head(feature).float()[0]
                exact_prob = float(torch.softmax(exact_logits, -1)[0].cpu())
                legion_prob = float(torch.softmax(legion_logits, -1)[0].cpu())
            append_jsonl(prediction_path, [{
                "sample_id": row["sample_id"], "source": row.get("generator/source"),
                "gt": 1 if row["label"] == "Fake" else 0,
                "c1_exact_logits": exact_logits.cpu().tolist(),
                "legion_retrained_logits": legion_logits.cpu().tolist(),
                "c1_exact_prob_fake": exact_prob, "legion_retrained_prob_fake": legion_prob,
            }])
            if (index + 1) % 100 == 0:
                print(json.dumps({"done": index + 1, "total": len(manifest)}), flush=True)

        indexed = {row["sample_id"]: row for row in rows(prediction_path)}
        ordered = [indexed[row["sample_id"]] for row in manifest]
        if len(indexed) != len(manifest):
            raise RuntimeError("incomplete predictions")
        prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
        a = metrics(ordered, "c1_exact"); b = metrics(ordered, "legion_retrained")
        logits_a = np.asarray([row["c1_exact_logits"] for row in ordered]); logits_b = np.asarray([row["legion_retrained_logits"] for row in ordered])
        prob_a = np.asarray([row["c1_exact_prob_fake"] for row in ordered]); prob_b = np.asarray([row["legion_retrained_prob_fake"] for row in ordered])
        disagreement = int(((prob_a >= 0.5) != (prob_b >= 0.5)).sum())
        logit_delta = float(np.max(np.abs(logits_a - logits_b)))
        probability_delta = float(np.max(np.abs(prob_a - prob_b)))
        confirmed = disagreement == 0 and logit_delta <= 1e-6 and probability_delta <= 1e-7
        result = {
            "schema": "phase6b1b_aigiholmes_exact_check_v1", "status": "COMPLETE",
            "verdict": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
            "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha256(MANIFEST),
            "sample_order": "exact manifest order", "n": len(ordered), "threshold": 0.5,
            "online_forward": "per-image RGB + CLIPImageProcessor + BF16 CLIP penultimate CLS",
            "clip_parameter_sha256": clip_hash,
            "checkpoints": {"c1_exact": str(EXACT.resolve()), "legion_retrained": str(LEGION.resolve())},
            "c1_exact_metrics": a, "legion_retrained_metrics": b,
            "parity": {"prediction_disagreement_count": disagreement,
                       "probability_max_abs_difference": probability_delta,
                       "logit_max_abs_difference": logit_delta,
                       "prediction_head_tensors_bitwise_identical": heads_equal,
                       "c1_exact_prediction_head_sha256": exact_hash,
                       "legion_retrained_prediction_head_sha256": legion_hash},
            "interpretation": ("C1-Exact fully reproduces LEGION-retrained on AIGI-Holmes. Phase6B C1-L versus "
                               "LEGION-retrained differences are attributable to the training/numerical recipe, not classifier architecture. "
                               "Phase6B.2 CLIP + LLM fusion is allowed."
                               if confirmed else
                               "Material parity failure remains. Audit inference/checkpoint/preprocessing before any fusion."),
            "firewall": {"other_external_datasets": False, "training": False, "threshold_tuning": False,
                         "calibration": False, "fusion": False},
        }
        atomic_json(OUT / "results.json", result)
        write_report(result)
        state.update(status="COMPLETE", verdict=result["verdict"])
    except BaseException as error:
        state.update(status="FAILED", exception_type=type(error).__name__, exception=str(error))
        raise
    finally:
        state.update(updated_at_utc=now(), elapsed_seconds=time.monotonic() - started)
        atomic_json(status_path, state)


if __name__ == "__main__":
    main()
