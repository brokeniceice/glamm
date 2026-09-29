#!/usr/bin/env python3
"""Selected-only C2-native staged R1 Official1000 canonical G0 evaluation."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.forensics_eval import GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from model.pcerf import sam_lowres_to_original_normalized
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase6p0_c2_native_staged_r1 as staged
from scripts.phase3c1_cache import clip_grid
from scripts.phase3c2_p3 import dataset_for
from scripts.phase6j0_c2_evaluate import load_c2_model
from scripts.phase6l0_r2_finalize import global_bootstrap
from scripts.phase6l0_r2_preflight import CKPT as C2_CKPT, dump, require
from tools.phase3c1 import geometry_for, paired_statistics
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import clip_coordinates, sam_coordinates, summarize, tensor_state_sha256
from tools.phase4f import load_evidence_source, load_rectifier

OUT = staged.OUT
PREDICTIONS = OUT / "official1000_records.jsonl"
BASE = ROOT / "outputs/phase6j0_c2/final_evaluation/official1000/G0/predictions.jsonl"
DOC = ROOT / "docs/phase6p0_c2_native_staged_r1.md"
CFG_PATH = ROOT / "configs/phase6j0_c2_preln_cross_attention.yaml"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def selected_state(device):
    internal = json.loads((OUT / "summary.json").read_text())
    require(internal["status"] == "COMPLETE_STOP_AFTER_INTERNAL_DEV",
            "official evaluation requires completed internal DEV finalizer")
    selector = json.loads((OUT / "joint/selector.json").read_text())
    require(selector["status"] == "COMPLETE" and not selector["official1000_used_for_selection"] and
            selector["primary"] == "canonical internal DEV Mean FG IoU only",
            "joint DEV selector not frozen")
    checkpoint = Path(selector["selected_checkpoint"])
    require(file_sha256(checkpoint) == selector["selected_checkpoint_sha256"] ==
            internal["selected_joint_checkpoint_sha256"],
            "selected R1 checkpoint hash drift")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(state["stage"] == "joint" and state["epoch"] == selector["selected_epoch"] and
            state["c2_sha256"] == staged.C2_SHA,
            "selected R1 checkpoint provenance drift")
    gates = json.loads((OUT / "preflight_gates.json").read_text())
    require(gates["status"] == "PASS" and gates["C2_G0_full_DEV_exact"] and
            gates["R1_utility_S64_original_geometry_exact"],
            "geometry-corrected C2-native R1 preflight missing")
    utility, _ = hc.load_utility("a2", device)
    rectifier = load_rectifier(staged.CFG, float(gates["gamma"]), device)
    utility.load_state_dict(state["utility_state"], strict=True)
    rectifier.load_state_dict(state["rectifier_state"], strict=True)
    utility.eval().requires_grad_(False)
    rectifier.eval().requires_grad_(False)
    sam, _ = staged.native_sam(device)
    source = load_evidence_source(staged.CFG, "forensic_rect", device)
    require(tensor_state_sha256(source.state_dict()) == gates["adapter_state_sha256"] and
            tensor_state_sha256(sam.state_dict()) == gates["C2_native_sam_state_sha256"] and
            tensor_state_sha256(staged.q.source_state(utility)) ==
            gates["source_heads_state_sha256"],
            "frozen source/SAM state drift before Official1000")
    return selector, utility, rectifier, sam, source


def baseline_count(row):
    return {key: int(row[key]) for key in ("tp", "fp", "fn", "tn")}


def evaluate(device):
    selector, utility, rectifier, sam, source = selected_state(device)
    require(file_sha256(C2_CKPT) == staged.C2_SHA, "C2 checkpoint drift")
    config = yaml.safe_load(CFG_PATH.read_text())
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, _ = load_c2_model(config, C2_CKPT, device,
                                        expected_step=3500, expected_epoch=7)
    model.eval().requires_grad_(False)
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16,
        use_mm_start_end=True, max_new_tokens=int(config["evaluation"]["max_new_tokens"]))
    dataset = dataset_for(tokenizer, config, "official1000")
    indices = [i for i, item in enumerate(dataset.rows) if int(item["class_label"]) == 1]
    reference = rows(BASE)
    require(len(indices) == len(reference) == 1000 and
            [str(dataset.rows[i]["sample_id"]) for i in indices] ==
            [item["sample_id"] for item in reference],
            "Official1000 canonical Fake ID/order drift")
    existing = rows(PREDICTIONS) if PREDICTIONS.exists() else []
    require(len(existing) <= 1000 and
            [item["sample_id"] for item in existing] ==
            [item["sample_id"] for item in reference[:len(existing)]] and
            all(item["selected_checkpoint_sha256"] ==
                selector["selected_checkpoint_sha256"] and
                "native_c2_tp" in item for item in existing),
            "Official1000 resume prefix/checkpoint drift")
    vision = model.get_model().get_vision_tower()
    for ordinal in range(len(existing), 1000):
        sample = dataset[indices[ordinal]]
        sid = str(sample["sample_id"])
        target = torch.as_tensor(sample["masks"]).bool().any(0)
        base = reference[ordinal]
        require(sid == base["sample_id"] and int(target.sum()) ==
                int(base["gt_foreground_pixels"]),
                f"Official1000 GT/ID mismatch: {sid}")
        output = backend.generate_localization_batch(
            [sample], provide_gt_fake=False,
            generation_mode="unified_fake_generate")[0]
        require(output["generated_token_ids"] == base["generated_token_ids"],
                f"Official1000 C2 generated trajectory drift: {sid}")
        q_values = output["projected_seg_embeddings"]
        seg_count = 0 if q_values is None else int(q_values.shape[0])
        if seg_count == 0:
            require(output["pred_mask"] is None, f"unexpected C2 mask without [SEG]: {sid}")
            record = {"sample_id": sid, "foreground_iou": 0.0, "foreground_f1": 0.0,
                      "tp": 0, "fp": 0, "fn": int(target.sum()),
                      "tn": int(target.numel() - target.sum())}
            native_base = dict(record)
            live_base = dict(record)
        else:
            h, w = target.shape
            sam_geo = geometry_for("sam", (h, w))
            clip_geo = geometry_for("clip", (h, w))
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw_s64 = model.get_grounding_encoder_embs(
                    sample["grounding_enc_image"][None].to(device, dtype=torch.bfloat16))
                tokens, _ = vision(
                    sample["global_enc_image"][None].to(device, dtype=torch.bfloat16))
                forensic = source(clip_grid(tokens), return_features=True)
            f24 = forensic["F_forensic"].to(torch.bfloat16)
            zf24 = forensic["logits"].to(torch.bfloat16)
            s64_original = sam_lowres_to_original_normalized(
                raw_s64, sam_geo, output_hw=(64, 64)).to(torch.bfloat16)
            sam_coords = sam_coordinates(sam_geo, grid=64)[None].to(device)
            clip_coords = clip_coordinates(clip_geo, grid=24)[None].to(device)
            token_valid = torch.ones(1, 576, dtype=torch.bool, device=device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                corrected = rectifier(raw_s64, f24, sam_coords, clip_coords,
                                      token_valid)
            support = corrected["support"].reshape(1, 1, 64, 64)
            adapted_logits, raw_logits = [], []
            with torch.no_grad():
                for q_one in q_values:
                    query = q_one[None].to(device, dtype=torch.bfloat16)
                    with torch.autocast(device_type="cuda", enabled=False):
                        raw_low = sam(query, raw_s64.to(torch.bfloat16))
                    z_original = sam_lowres_to_original_normalized(
                        raw_low, sam_geo, output_hw=(256, 256)).to(torch.bfloat16)
                    batch = {"S64": s64_original, "q_seg": query, "z_L": z_original,
                             "F24": f24, "z_F24": zf24,
                             "clip_geometries": [clip_geo]}
                    utility_out = hc.utility_forward(utility, batch)
                    gate = hc.gate_to_sam_grid(utility_out["U"], sam_coords) * support.float()
                    embedding = hc.gated_embedding(
                        raw_s64, corrected["image_embeddings"], gate)
                    with torch.autocast(device_type="cuda", enabled=False):
                        adapted_low = sam(query, embedding.to(torch.bfloat16))
                    raw_logits.append(inverse_sam_logits(raw_low, sam_geo))
                    adapted_logits.append(inverse_sam_logits(adapted_low, sam_geo))
            record = metric_record(sid, torch.stack(adapted_logits).amax(0), target)
            native_base = metric_record(sid, torch.stack(raw_logits).amax(0), target)
            for item in (record, native_base):
                item["tn"] = int(target.numel()) - item["tp"] - item["fp"] - item["fn"]
            require(output["pred_mask"] is not None, f"missing live C2 mask: {sid}")
            live_base = metric_record(sid, output["pred_mask"].amax(0), target)
            live_base["tn"] = int(target.numel()) - live_base["tp"] - live_base["fp"] - live_base["fn"]
        # The original C2 forward is the authoritative published baseline.
        # The standalone frozen SAM runtime uses BF16 decoder weights and can
        # differ by a few threshold-border pixels from C2's FP32 path.
        require(baseline_count(live_base) == baseline_count(base),
                f"Official1000 live C2-G0 mask parity drift: {sid}; "
                f"reference={baseline_count(base)}, live={baseline_count(live_base)}")
        record.update({"seg_count": seg_count, "valid_q_seg": seg_count > 0,
                       "selected_checkpoint_sha256": selector["selected_checkpoint_sha256"],
                       "c2_checkpoint_sha256": staged.C2_SHA,
                       "generated_token_ids_sha256": hashlib.sha256(
                           json.dumps(output["generated_token_ids"]).encode()).hexdigest(),
                       "native_c2_tp": native_base["tp"],
                       "native_c2_fp": native_base["fp"],
                       "native_c2_fn": native_base["fn"],
                       "native_c2_tn": native_base["tn"],
                       "native_c2_foreground_iou": native_base["foreground_iou"],
                       "native_c2_foreground_f1": native_base["foreground_f1"]})
        append(PREDICTIONS, record)
        if (ordinal + 1) % 20 == 0:
            print(json.dumps({"stage": "OFFICIAL1000", "done": ordinal + 1,
                              "total": 1000}), flush=True)
    return selector, reference


def finalize(selector, reference):
    selected = rows(PREDICTIONS)
    require(len(selected) == len(reference) == 1000 and
            [item["sample_id"] for item in selected] ==
            [item["sample_id"] for item in reference] and
            all(item["selected_checkpoint_sha256"] ==
                selector["selected_checkpoint_sha256"] for item in selected),
            "Official1000 selected-only full-N drift")
    paired = {metric: paired_statistics(
        [item[metric] for item in selected], [item[metric] for item in reference],
        repeats=2000, seed=3407)
        for metric in ("foreground_iou", "foreground_f1")}
    global_paired = global_bootstrap(selected, reference)
    metrics = summarize(selected)
    c2 = summarize(reference)
    native_reference = [{"sample_id": row["sample_id"],
                         "tp": row["native_c2_tp"], "fp": row["native_c2_fp"],
                         "fn": row["native_c2_fn"], "tn": row["native_c2_tn"],
                         "foreground_iou": row["native_c2_foreground_iou"],
                         "foreground_f1": row["native_c2_foreground_f1"]}
                        for row in selected]
    native_metrics = summarize(native_reference)
    native_drift = {
        "different_images": sum(baseline_count(a) != baseline_count(b)
                                for a, b in zip(native_reference, reference)),
        "sum_abs_tp_difference": sum(abs(a["tp"] - b["tp"])
                                     for a, b in zip(native_reference, reference)),
        "sum_abs_fp_difference": sum(abs(a["fp"] - b["fp"])
                                     for a, b in zip(native_reference, reference)),
        "max_abs_tp_difference": max(abs(a["tp"] - b["tp"])
                                     for a, b in zip(native_reference, reference)),
        "max_abs_fp_difference": max(abs(a["fp"] - b["fp"])
                                     for a, b in zip(native_reference, reference)),
    }
    result = {"status": "COMPLETE_STOP_AFTER_OFFICIAL1000", "population": 1000,
              "protocol": "canonical G0 Fake-only, generated C2 query, full reference-mask union, threshold logit >0",
              "selected_joint_epoch": selector["selected_epoch"],
              "selected_checkpoint_sha256": selector["selected_checkpoint_sha256"],
              "c2_checkpoint_sha256": staged.C2_SHA,
              "records_sha256": file_sha256(PREDICTIONS),
              "c2_g0_records_sha256": file_sha256(BASE),
              "metrics": {"C2_G0": c2, "C2_native_BF16_runtime": native_metrics,
                          "C2_native_staged_R1": metrics},
              "paired_R1_minus_C2_G0": paired, "global_paired": global_paired,
              "paired_R1_minus_native_BF16": {metric: paired_statistics(
                  [item[metric] for item in selected],
                  [item[metric] for item in native_reference], repeats=2000, seed=3407)
                  for metric in ("foreground_iou", "foreground_f1")},
              "live_generated_C2_mask_exact": True,
              "standalone_BF16_runtime_drift": native_drift,
              "selection": "joint checkpoint frozen by internal DEV before Official1000",
              "firewall": {"internal_test": False, "localization_OOD": False,
                           "classification_OOD": False}}
    dump(OUT / "official1000_results.json", result)
    document = DOC.read_text()
    require("Status: **COMPLETE STOP AFTER INTERNAL DEV**." in document and
            "Official1000, internal test and OOD were not accessed." in document,
            "internal DEV report changed before Official1000 finalization")
    document = document.replace(
        "Status: **COMPLETE STOP AFTER INTERNAL DEV**.",
        "Status: **COMPLETE STOP AFTER OFFICIAL1000**.")
    document = document.replace(
        "Official1000, internal test and OOD were not accessed.",
        "Official1000 was accessed only after the internal DEV selector was frozen. "
        "C2-native R1 did not use internal test or localization OOD.")
    f = lambda value: f"{float(value):.6f}"
    p = paired["foreground_iou"]
    g = global_paired["iou"]
    document += ("\n## Selected joint R1 on Official1000\n\n"
        "| Model | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| C2-G0 | {f(c2['mean_foreground_iou'])} | {f(c2['mean_foreground_f1'])} | "
        f"{f(c2['global_foreground_iou'])} | {f(c2['global_foreground_f1'])} |\n"
        f"| C2-native staged R1 | {f(metrics['mean_foreground_iou'])} | "
        f"{f(metrics['mean_foreground_f1'])} | {f(metrics['global_foreground_iou'])} | "
        f"{f(metrics['global_foreground_f1'])} |\n\n"
        f"Paired Mean FG IoU difference R1−C2-G0: {f(p['mean_difference'])}, "
        f"95% bootstrap CI [{f(p['bootstrap_95_ci'][0])}, {f(p['bootstrap_95_ci'][1])}], "
        f"W/T/L {p['wins']}/{p['ties']}/{p['losses']}, Wilcoxon p={p['wilcoxon_pvalue']:.3g}. "
        f"Global FG IoU difference: {f(g['difference'])}, paired bootstrap CI "
        f"[{f(g['bootstrap_95_ci'][0])}, {f(g['bootstrap_95_ci'][1])}].\n\n"
        "C2 generation tokens, IDs, GT foreground counts and live C2 forward "
        "mask confusion counts matched the frozen C2-G0 Official1000 record per image. "
        f"The standalone BF16 SAM runtime differed from live C2 on {native_drift['different_images']} "
        "images; its unadapted counts and paired result are recorded separately. "
        "Official1000 was not used to retrain or reselect the joint checkpoint. "
        "Historical access to this test set limits independence of this comparison.\n")
    DOC.write_text(document)
    internal = json.loads((OUT / "summary.json").read_text())
    internal.update({"status": "COMPLETE_STOP_AFTER_OFFICIAL1000",
                     "official1000_result": str(OUT / "official1000_results.json"),
                     "official1000_result_sha256": file_sha256(OUT / "official1000_results.json"),
                     "report_sha256": file_sha256(DOC)})
    dump(OUT / "summary.json", internal)
    print(json.dumps({"status": "COMPLETE_STOP_AFTER_OFFICIAL1000",
                      "mean_fg_iou": metrics["mean_foreground_iou"]}), flush=True)


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    selector, reference = evaluate(device)
    finalize(selector, reference)


if __name__ == "__main__":
    main()
