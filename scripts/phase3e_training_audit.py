#!/usr/bin/env python3
"""Post-training exact schedule, aggregation, checkpoint, and fairness audit for Phase 3E."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    cfg = yaml.safe_load((ROOT / "configs/phase3e_joint_language_mask_posttraining.yaml").read_text())
    root = (ROOT / cfg["experiment"]["output_root"]).resolve(); total = int(cfg["training"]["total_optimizer_steps"])
    interval = int(cfg["training"]["checkpoint_interval"]); gas = int(cfg["training"]["gradient_accumulation_steps"])
    values = {arm: rows(root / f"experiments/{arm}/training_metrics.jsonl") for arm in ("SFT_CONT", "JOINT")}
    schedule = json.loads((root / "training/schedule.json").read_text())
    checks = {
        "both_have_exact_1000_steps": all(len(v) == total and [r["optimizer_step"] for r in v] == list(range(1, total + 1)) for v in values.values()),
        "sample_ids_exact_between_arms": all(a["sample_ids"] == b["sample_ids"] for a, b in zip(values["SFT_CONT"], values["JOINT"])),
        "class_labels_exact_between_arms": all(a["class_labels"] == b["class_labels"] for a, b in zip(values["SFT_CONT"], values["JOINT"])),
        "text_denominators_exact_between_arms": all(a["global_supervision_counts"]["text_tokens"] == b["global_supervision_counts"]["text_tokens"] for a, b in zip(values["SFT_CONT"], values["JOINT"])),
        "lora_lr_exact_between_arms": all(a["learning_rates"]["lora"] == b["learning_rates"]["lora"] for a, b in zip(values["SFT_CONT"], values["JOINT"])),
        "P1_exact_aggregation_declared_every_step": all(r.get("aggregation") == "P1_exact_text_token_mean_and_valid_mask_mean" for v in values.values() for r in v),
        "SFT_has_no_mask_loss_or_spatial_gradient": all(r["mask_loss"] == 0 and r["per_group_gradient_norms"]["text_hidden_fcs"] == 0 and r["per_group_gradient_norms"]["mask_decoder"] == 0 for r in values["SFT_CONT"]),
        "JOINT_has_two_valid_masks_per_window": all(r["global_supervision_counts"]["valid_masks"] == 2 for r in values["JOINT"]),
        "schedule_exact": all(r["sample_ids"] == schedule["sample_ids"][(r["optimizer_step"]-1)*gas:r["optimizer_step"]*gas] for r in values["SFT_CONT"]),
    }
    expected_steps = list(range(interval, total + 1, interval))
    checkpoint_steps = {}
    for arm in values:
        metadata = rows(root / f"experiments/{arm}/checkpoint_metadata.jsonl"); checkpoint_steps[arm] = [int(x["optimizer_step"]) for x in metadata]
        checks[f"{arm}_checkpoint_steps_exact"] = checkpoint_steps[arm] == expected_steps and all(Path(x["checkpoint"]).is_file() for x in metadata)
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "optimizer_steps_per_arm": {arm: len(v) for arm, v in values.items()},
              "image_exposures_per_arm": total * gas, "checkpoint_steps": checkpoint_steps,
              "invalid_engineering_run_excluded": str(root / "invalid_runs/equal_sample_normalization_20260822"),
              "internal_test_used": False, "official1000_used": False}
    dump(root / "audits/training_fairness_runtime_audit.json", result)
    if result["status"] != "PASS": raise RuntimeError(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
