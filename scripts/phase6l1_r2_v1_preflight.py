#!/usr/bin/env python3
"""Read-only Phase6L1 source audit and fixed-sample C2 grouping diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.forensics_eval import GLaMMForensicsBackend
from model.c2_r2_localizer_v1 import C2R2LocalizerV1
from model.llava import conversation as conversation_lib
from scripts.phase3c2_p3 import dataset_for
from scripts.phase6j0_c2_evaluate import load_c2_model
from scripts.phase6l0_r2_preflight import CKPT, EXPECTED_SHA, OUT as L0_CACHE, dump, require
from scripts.phase6l1_r2_v1_train import RESULTS, audit_cache_provenance, audit_initial_sampling
from tools.phase4c_b import file_sha256


def _tensor(value):
    return value.detach().cpu().clone() if isinstance(value, torch.Tensor) else None


def _difference(left, right):
    if left is None or right is None:
        return {"both_none": left is None and right is None}
    require(left.shape == right.shape, "batch grouping changed tensor shape")
    return {"exact": bool(torch.equal(left, right)),
            "max_abs": float((left.float() - right.float()).abs().max())}


def batch_grouping_audit(device: torch.device) -> dict:
    """Record, never select on, B=1/2/4 effects for one fixed C2 sample."""
    cfg = yaml.safe_load((ROOT / "configs/phase6j0_c2_preln_cross_attention.yaml").read_text())
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, _ = load_c2_model(cfg, CKPT, device, expected_step=3500, expected_epoch=7)
    model.eval().requires_grad_(False)
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=int(cfg["evaluation"]["max_new_tokens"]))
    dataset = dataset_for(tokenizer, cfg, "train")
    fake_rows = [i for i, row in enumerate(dataset.rows) if int(row["class_label"]) == 1]
    first = torch.load(Path(json.loads((L0_CACHE / "train.json").read_text())["shards"][0]["path"]),
                       map_location="cpu", weights_only=False)
    ordinals = first["valid_c2_g0"].nonzero().flatten()[:4].tolist()
    require(len(ordinals) == 4, "batch audit requires four valid fixed TRAIN samples")
    samples = [dataset[fake_rows[i]] for i in ordinals]
    require([str(s["sample_id"]) for s in samples] == [first["sample_ids"][i] for i in ordinals],
            "batch audit canonical sample IDs drift")
    attention = model.c2_cross_attention
    fused = []
    hook = attention.register_forward_hook(lambda _m, _i, output: fused.append(output.detach().cpu().clone()))
    attention.capture_spatial_intermediates = True
    captures = {}
    try:
        for size in (1, 2, 4):
            fused.clear()
            attention.last_spatial_intermediates = None
            with torch.no_grad():
                results = backend.generate_localization_batch(samples[:size], provide_gt_fake=False,
                                                               generation_mode="unified_fake_generate")
            cap = attention.last_spatial_intermediates
            require(cap is not None and fused and fused[-1].shape == (size, 1, 4096) and
                    cap["A"].shape == (size, 8, 24, 24) and
                    cap["E"].shape == (size, 512, 24, 24),
                    f"batch audit C2 capture missing at B={size}")
            result = results[0]
            captures[size] = {"A": _tensor(cap["A"][0]), "E": _tensor(cap["E"][0]),
                              "r_prime": _tensor(fused[-1][0, 0]),
                              "q_seg": _tensor(result["projected_seg_embeddings"]),
                              "mask": _tensor(result["pred_mask"]),
                              "generated_token_ids": result["generated_token_ids"]}
    finally:
        hook.remove()
        attention.capture_spatial_intermediates = False
    comparison = {}
    for size in (2, 4):
        comparison[f"B{size}_minus_B1"] = {
            key: (_difference(captures[1][key], captures[size][key]) if key != "generated_token_ids"
                  else {"exact": captures[1][key] == captures[size][key]})
            for key in captures[1]}
    return {"status": "RECORDED_NOT_A_SELECTOR", "sample_id": samples[0]["sample_id"],
            "group_sizes": [1, 2, 4], "comparison": comparison,
            "formal_cache_generation_batch": {"train": 4, "dev": 1},
            "no_cache_rebuilt": True, "c2_checkpoint_sha256": EXPECTED_SHA}


def main(device: torch.device) -> None:
    torch.cuda.set_device(device)
    require(file_sha256(CKPT) == EXPECTED_SHA, "C2 checkpoint SHA drift")
    provenance = audit_cache_provenance()
    torch.manual_seed(3407)
    init = audit_initial_sampling(C2R2LocalizerV1())
    prior = ROOT / "outputs/phase6l0_r2/preflight_gates.json"
    require(json.loads(prior.read_text())["status"] == "PASS", "Phase6L0 frozen preflight missing")
    grouping = batch_grouping_audit(device)
    dump(RESULTS / "preflight/batch_grouping_audit.json", grouping)
    dump(RESULTS / "preflight/prelaunch.json", {
        "status": "PASS", "cache_provenance_sha256": file_sha256(RESULTS / "preflight/cache_provenance.json"),
        "initial_sampling_sha256": file_sha256(RESULTS / "preflight/initial_sampling_pattern.json"),
        "batch_grouping_sha256": file_sha256(RESULTS / "preflight/batch_grouping_audit.json"),
        "sampling_points": len(init["points"]), "train_n": provenance["splits"]["train"]["n"],
        "dev_n": provenance["splits"]["val"]["n"],
        "formal_training_still_requires_full_DEV_step0_gate": True})
    print(json.dumps({"status": "PASS", "stage": "PRELAUNCH", "train_n": 8836,
                      "dev_n": 1106}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(torch.device(args.device))
