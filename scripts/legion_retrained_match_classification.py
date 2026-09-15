#!/usr/bin/env python3
"""Frozen classification evaluation for LEGION-retrained-match."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import final_eval_classification as base

MODEL = ROOT / "checkpoints/legion_retrained_match/stage2_cls/final_model"
IDENTITY = ROOT / "checkpoints/legion_retrained_match/stage2_cls_identity.json"
OUT = ROOT / "outputs/legion_retrained_match_evaluation/classification"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=tuple(base.MANIFESTS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    identity = json.loads(IDENTITY.read_text())
    base.LEGION_CLS = MODEL
    base.LEGION_CLS_IDENTITY = IDENTITY
    base.LEGION_CLS_SHA = identity["canonical_sha256"]
    device = torch.device(args.device); torch.cuda.set_device(device)
    status = OUT / ("worker_" + "_".join(args.datasets) + ".json")
    base.atomic(status, {"status": "RUNNING", "datasets": args.datasets, "pid": os.getpid(),
                         "device": str(device), "batch_size": args.batch_size,
                         "checkpoint": str(MODEL.resolve()), "canonical_sha256": base.LEGION_CLS_SHA})
    try:
        model, _, processor = base.load_legion(device)
        for name in args.datasets:
            scope = base.scope(name); dest = OUT / name; predpath = dest / "predictions.jsonl"
            old = base.rows(predpath) if predpath.exists() else []
            indexed = {row["sample_id"]: row for row in old}
            expected = {row["sample_id"] for row in scope}
            if not set(indexed).issubset(expected): raise RuntimeError(f"{name} resume scope drift")
            pending = [row for row in scope if row["sample_id"] not in indexed]
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]; images = []
                for row in batch:
                    bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
                    if bgr is None: raise OSError(row["image_path"])
                    images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                pixels = processor.preprocess(images, return_tensors="pt")["pixel_values"].to(device=device, dtype=torch.bfloat16)
                with torch.inference_mode():
                    probs = torch.softmax(model(global_enc_images=pixels, inference_cls=True)["logits"].float(), -1)[:, 0].cpu().numpy()
                values = [{**row, "prob_fake": float(prob), "pred": "fake" if prob >= .5 else "real"}
                          for row, prob in zip(batch, probs)]
                base.append(predpath, values); indexed.update((row["sample_id"], row) for row in values)
                done = len(scope) - len(pending) + min(start + len(batch), len(pending))
                if done % 1000 < args.batch_size: print(json.dumps({"dataset": name, "done": done, "total": len(scope)}), flush=True)
            ordered = [indexed[row["sample_id"]] for row in scope]
            tmp = predpath.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered)); os.replace(tmp, predpath)
            result = {"status": "COMPLETE", "model": "legion_retrained_match", "dataset": name,
                      "manifest": str(base.MANIFESTS[name].resolve()), "manifest_sha256": base.sha(base.MANIFESTS[name]),
                      "checkpoint": {"path": str(MODEL.resolve()), "canonical_sha256": base.LEGION_CLS_SHA},
                      "metrics": base.metrics(ordered)}
            base.atomic(dest / "results.json", result)
        base.atomic(status, {"status": "COMPLETE", "datasets": args.datasets, "checkpoint": str(MODEL.resolve())})
    except BaseException as error:
        base.atomic(status, {"status": "FAILED", "datasets": args.datasets,
                             "exception_type": type(error).__name__, "exception": str(error)})
        raise


if __name__ == "__main__": main()
