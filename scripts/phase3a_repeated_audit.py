#!/usr/bin/env python3
"""Quantify fixed-subset BF16 G0 repeatability from two preserved runs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def rows(path):
    return {json.loads(line)["sample_id"]: json.loads(line) for line in path.read_text().splitlines() if line}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    a, b = rows(cli.run_a), rows(cli.run_b)
    if set(a) != set(b):
        raise ValueError("Repeated-run sample IDs differ")
    token_equal, mask_equal, iou_delta = [], [], []
    details = []
    for sample_id in sorted(a):
        aa, bb = a[sample_id], b[sample_id]
        same_tokens = aa["generated_token_ids"] == bb["generated_token_ids"]
        mask_a = torch.load(aa["binary_mask_path"], map_location="cpu")
        mask_b = torch.load(bb["binary_mask_path"], map_location="cpu")
        same_mask = bool(torch.equal(mask_a, mask_b))
        delta = float(bb["foreground_iou"] - aa["foreground_iou"])
        token_equal.append(same_tokens); mask_equal.append(same_mask); iou_delta.append(delta)
        details.append({"sample_id": sample_id, "token_exact": same_tokens,
                        "binary_mask_exact": same_mask, "iou_difference_b_minus_a": delta})
    result = {
        "num_samples": len(details), "fixed_ordering": True, "generation_batch_size": 1,
        "token_exact_agreement_rate": float(np.mean(token_equal)),
        "binary_mask_exact_agreement_rate": float(np.mean(mask_equal)),
        "mean_iou_difference_b_minus_a": float(np.mean(iou_delta)),
        "max_abs_iou_difference": float(np.max(np.abs(iou_delta))),
        "details": details,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
