#!/usr/bin/env python3
"""Frozen old-R1 transfer audit on C1, Official1000 canonical G0 only."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import CANONICAL_UNIFIED_QUESTION
from eval.forensics_eval import GLaMMForensicsBackend
from model.GLaMM import extract_seg_predictor_hidden
from model.llava import conversation as conversation_lib
from model.pcerf import sam_lowres_to_original_normalized
from scripts import phase4ha_utility_gated_rectification as ha
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts import p1_r1_reusable_matrix as matrix
from scripts.phase2a_final_evaluate import load_model
from scripts.phase3c1_cache import clip_grid
from tools.phase3c1 import geometry_for
from tools.phase3f_aogd import capture_representations, core_model
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import clip_coordinates, compare, sam_coordinates, summarize
from tools.phase4f import load_evidence_source, load_sam_runtime

SEED = 3407
OUT = ROOT / "outputs/phase6e1_c1_old_r1_transfer"
RESULT = OUT / "phase6e1_c1_old_r1_transfer.json"
DOC = ROOT / "docs/phase6e1_c1_old_r1_transfer.md"
C1_CFG_PATH = ROOT / "configs/phase6d3_c1_rine_conditioned_p1.yaml"
C1_CKPT = ROOT / "checkpoints/phase6d3_c1/best/checkpoint/mp_rank_00_model_states.pt"
P1_CFG_PATH = ROOT / "configs/phase3a_p1.yaml"
P1_CKPT = matrix.P1
R1_CKPT = matrix.R1
HIST_P1 = ROOT / "outputs/phase3a_phrase_grounding/evaluation/official1000/G0/predictions.jsonl"
HIST_JOB = ROOT / "outputs/p1_r1_reusable_matrix/jobs/official_g0.json"
SHARED_MANIFEST = ROOT / "outputs/phase2b_legion_parity/split_audit/official1000_manifest/test_combined.jsonl"


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def rows(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f if x.strip()]


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n"); f.flush()


def freeze_protocol():
    historical = json.loads(HIST_JOB.read_text())
    manifest_ids = [x["sample_id"] for x in rows(SHARED_MANIFEST)]
    p1_ids = [x["sample_id"] for x in historical["P1"]["records"]]
    r1_ids = [x["sample_id"] for x in historical["R1"]["records"]]
    generated_ids = [x["sample_id"] for x in rows(HIST_P1)]
    checks = {
        "n_1000": len(manifest_ids) == len(p1_ids) == len(r1_ids) == len(generated_ids) == 1000,
        "ordered_ids_exact": manifest_ids == p1_ids == r1_ids == generated_ids,
        "p1_checkpoint_exact": file_sha256(P1_CKPT) == matrix.P1_SHA,
        "r1_checkpoint_exact": file_sha256(R1_CKPT) == matrix.R1_SHA,
        "historical_population": historical.get("population") == "official1000",
        "historical_condition": historical.get("condition") == "original",
        "historical_mode": historical.get("mode") == "g0",
    }
    if not all(checks.values()):
        raise RuntimeError(f"historical reuse provenance failed: {checks}")
    cfg_p1 = yaml.safe_load(P1_CFG_PATH.read_text())
    cfg_c1 = yaml.safe_load(C1_CFG_PATH.read_text())
    prompt_checks = {
        "canonical_prompt_sha_equal": cfg_p1["forensics"]["canonical_prompt_sha256"] == cfg_c1["forensics"]["canonical_prompt_sha256"],
        "g0_prompt_equal": cfg_p1["evaluation"]["g0_prompt"] == cfg_c1["evaluation"]["g0_prompt"] == "canonical_unified",
        "max_new_tokens_equal": cfg_p1["evaluation"]["max_new_tokens"] == cfg_c1["evaluation"]["max_new_tokens"] == 400,
        "mask_threshold_equal": cfg_p1["evaluation"]["mask_logit_threshold"] == cfg_c1["evaluation"]["mask_logit_threshold"] == 0.0,
    }
    if not all(prompt_checks.values()):
        raise RuntimeError(f"prompt/evaluator drift: {prompt_checks}")
    protocol = {
        "schema": "phase6e1_protocol_v1", "status": "FROZEN_BEFORE_C1_INFERENCE",
        "created_at_utc": now(), "seed": SEED,
        "scope": "SynthScars Official1000 Fake-only, original RGB, canonical G0",
        "models": {
            "P1": {"path": str(P1_CKPT.resolve()), "sha256": file_sha256(P1_CKPT), "historical_reuse": True},
            "old_R1": {"path": str(R1_CKPT.resolve()), "sha256": file_sha256(R1_CKPT), "selected_epoch": 9, "historical_reuse": True},
            "C1": {"path": str(C1_CKPT.resolve()), "sha256": file_sha256(C1_CKPT), "epoch": 5, "step": 2500},
        },
        "manifest": {"path": str(SHARED_MANIFEST.resolve()), "sha256": file_sha256(SHARED_MANIFEST), "count": 1000},
        "historical_reuse_audit": {"checks": checks, "prompt_checks": prompt_checks,
            "historical_job": str(HIST_JOB.resolve()), "historical_job_sha256": file_sha256(HIST_JOB),
            "historical_p1_generation": str(HIST_P1.resolve()), "historical_p1_generation_sha256": file_sha256(HIST_P1)},
        "protocol": {"gt_authenticity": False, "gt_phrase": False, "gt_explanation": False,
            "generation": "greedy", "seg_trigger": "generated [SEG]", "invalid": "no [SEG] => empty prediction and score 0",
            "mask_threshold": 0.0, "training": False, "parameter_updates": False},
        "preregistered_decisions": {
            "transfer_yes": "C1+oldR1 minus C1 paired-bootstrap lower bounds >0 for mean IoU and mean F1, and both global metrics non-decreasing",
            "transfer_no": "neither mean metric has positive-CI evidence and neither global metric improves; otherwise MIXED",
            "base_changed": "C1 minus P1 paired-bootstrap CI excludes zero for mean IoU or mean F1",
            "seg_shift": "joint-valid mean cosine <0.95 or relative mean L2 norm difference >0.10",
            "retrain_needed": "transfer result is not YES",
        },
    }
    dump(OUT / "protocol.json", protocol)
    return historical, protocol


def union_metric(sid, pred_mask, target):
    if pred_mask is None:
        return {"sample_id": sid, "foreground_iou": 0.0, "foreground_f1": 0.0,
                "tp": 0, "fp": 0, "fn": int(target.sum()), "valid_q_seg": False}
    logits = torch.as_tensor(pred_mask).detach().float().cpu()
    if logits.ndim == 2: logits = logits[None]
    row = metric_record(sid, logits.amax(0), target)
    row["valid_q_seg"] = bool(logits.shape[0])
    return row


def c1_run(device):
    cfg = yaml.safe_load(C1_CFG_PATH.read_text())
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, meta = load_model(cfg, C1_CKPT, device, expected_step=2500, expected_epoch=5)
    model.eval().requires_grad_(False)
    if not hasattr(model, "rine_conditioner"):
        raise RuntimeError("C1 missing frozen RINE conditioner")
    model.rine_conditioner.eval().requires_grad_(False)
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=400)
    full, indices = matrix.fake_dataset(tokenizer, "official1000")
    source = load_evidence_source(matrix.P4F_CFG, "forensic_rect", device).eval().requires_grad_(False)
    utility, rectifier, sam = matrix.load_r1(device)
    freeze_audit = {
        "C1": {"training": bool(model.training), "parameter_count": sum(p.numel() for p in model.parameters()),
               "requires_grad_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "RINE_conditioner": {"training": bool(model.rine_conditioner.training),
               "parameter_count": sum(p.numel() for p in model.rine_conditioner.parameters()),
               "requires_grad_parameter_count": sum(p.numel() for p in model.rine_conditioner.parameters() if p.requires_grad)},
        "old_R1_utility": {"training": bool(utility.training), "requires_grad_parameter_count": sum(p.numel() for p in utility.parameters() if p.requires_grad)},
        "old_R1_rectifier": {"training": bool(rectifier.training), "requires_grad_parameter_count": sum(p.numel() for p in rectifier.parameters() if p.requires_grad)},
        "old_R1_forensic_source": {"training": bool(source.training), "requires_grad_parameter_count": sum(p.numel() for p in source.parameters() if p.requires_grad)},
        "SAM_runtime": {"training": bool(sam.training), "requires_grad_parameter_count": sum(p.numel() for p in sam.parameters() if p.requires_grad)},
        "optimizer_created": False,
    }
    if any(x.get("training") or x.get("requires_grad_parameter_count") for x in freeze_audit.values() if isinstance(x, dict)):
        raise RuntimeError(f"freeze audit failed: {freeze_audit}")
    dump(OUT / "freeze_audit.json", freeze_audit)
    core = core_model(model); vision_tower = model.get_model().get_vision_tower()
    output_path = OUT / "c1_records.jsonl"
    existing = rows(output_path) if output_path.exists() else []
    expected_prefix = [full.rows[i]["sample_id"] for i in indices[:len(existing)]]
    if [x["sample_id"] for x in existing] != expected_prefix:
        raise RuntimeError("C1 resume prefix drift")
    start = len(existing)
    for ordinal, full_index in enumerate(indices[start:], start=start):
        sample = full[full_index]; sid = sample["sample_id"]
        target = torch.as_tensor(sample["masks"]).bool().any(0)
        captured_raw, captured_vision = [], []
        original = core._extract_projected_seg_predictor_hidden
        def wrapped(hidden_all, input_ids, *args, **kwargs):
            last = core._get_last_hidden_state(hidden_all)
            raw, _ = extract_seg_predictor_hidden(last, input_ids, core.seg_token_idx)
            captured_raw.append([x.detach().to(torch.bfloat16).cpu() for x in raw])
            return original(hidden_all, input_ids, *args, **kwargs)
        def vision_hook(_module, _inputs, value):
            value = value[0] if isinstance(value, (tuple, list)) else value
            captured_vision.append(value.detach().to(torch.bfloat16).cpu())
        core._extract_projected_seg_predictor_hidden = wrapped
        hook = vision_tower.register_forward_hook(vision_hook)
        try:
            output = backend.generate_localization_batch([sample], provide_gt_fake=False,
                generation_mode="unified_fake_generate")[0]
        finally:
            hook.remove(); core._extract_projected_seg_predictor_hidden = original
        q_values = output.get("projected_seg_embeddings")
        if q_values is None: q_values = torch.empty((0, 256), device=device)
        q_values = q_values.detach()
        raw_values = captured_raw[-1][0] if captured_raw and captured_raw[-1] else torch.empty((0, 4096), dtype=torch.bfloat16)
        if q_values.shape[0] != raw_values.shape[0]:
            raise RuntimeError(f"SEG raw/projected mismatch {sid}: {q_values.shape} {raw_values.shape}")
        hidden_path = OUT / "seg_hidden" / f"{hashlib.sha256(sid.encode()).hexdigest()[:24]}.pt"
        hidden_path.parent.mkdir(parents=True, exist_ok=True); torch.save(raw_values, hidden_path)
        c1_record = union_metric(sid, output.get("pred_mask"), target)
        r1_record = {"sample_id": sid, "foreground_iou": 0.0, "foreground_f1": 0.0,
                     "tp": 0, "fp": 0, "fn": int(target.sum()), "valid_q_seg": False}
        if q_values.shape[0]:
            if not captured_vision:
                raise RuntimeError(f"missing GLaMM visual tokens for {sid}")
            raw = model.get_grounding_encoder_embs(sample["grounding_enc_image"][None].to(device=device, dtype=torch.bfloat16))
            raw_clip = clip_grid(captured_vision[-1]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16): forensic = source(raw_clip, return_features=True)
            h, w = target.shape; sg, cg = geometry_for("sam", (h, w)), geometry_for("clip", (h, w))
            s64 = sam_lowres_to_original_normalized(raw, sg, output_hw=(64, 64)).to(torch.bfloat16)
            r1_original_logits = []
            for q in q_values:
                qone = q.reshape(1, -1).to(device=device, dtype=torch.bfloat16)
                with torch.autocast(device_type=device.type, enabled=False): low_p1 = sam(qone, raw.to(torch.bfloat16))
                zl = sam_lowres_to_original_normalized(low_p1, sg, output_hw=(256, 256)).to(torch.bfloat16)
                f24 = forensic["F_forensic"].to(torch.bfloat16); zf = forensic["logits"].to(torch.bfloat16)
                batch = {"S64": s64, "q_seg": qone, "z_L": zl, "F24": f24, "z_F24": zf,
                    "clip_geometries": [cg], "valid_g0": torch.ones(1, dtype=torch.bool, device=device),
                    "forensic_present": torch.ones(1, dtype=torch.bool, device=device),
                    "forensic_vacuous": torch.zeros(1, dtype=torch.bool, device=device),
                    "forensic_off": torch.zeros(1, dtype=torch.bool, device=device)}
                uout = hc.utility_forward(utility, batch)
                sc = sam_coordinates(sg, grid=64)[None].to(device); cc = clip_coordinates(cg, grid=24)[None].to(device)
                valid = torch.ones(1, 576, dtype=torch.bool, device=device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16): p4f = rectifier(raw, f24, sc, cc, valid)
                gate = ha.gate_to_sam_grid(uout["U"], sc) * p4f["support"].reshape(1, 1, 64, 64).float()
                adapted = ha.gated_embedding(raw, p4f["image_embeddings"], gate)
                with torch.autocast(device_type=device.type, enabled=False): low = sam(qone, adapted.to(torch.bfloat16))
                r1_original_logits.append(inverse_sam_logits(low, sg))
            union = torch.stack(r1_original_logits).amax(0)
            r1_record = metric_record(sid, union, target); r1_record["valid_q_seg"] = True
        append(output_path, {"sample_id": sid, "generated_token_ids": output["generated_token_ids"],
            "seg_triggered": bool(output["seg_triggered"]), "seg_count": int(q_values.shape[0]),
            "c1": c1_record, "c1_old_r1": r1_record, "c1_seg_hidden_path": str(hidden_path.resolve())})
        if (ordinal + 1) % 10 == 0:
            print(json.dumps({"stage": "C1_G0_AND_OLD_R1", "done": ordinal + 1, "total": len(indices)}), flush=True)
    del sam, rectifier, utility, source, backend, model
    gc.collect(); torch.cuda.empty_cache()
    return meta


def p1_hidden_diagnostic(device):
    cfg = yaml.safe_load(P1_CFG_PATH.read_text())
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, _ = load_model(cfg, P1_CKPT, device, expected_step=3500, expected_epoch=7)
    model.eval().requires_grad_(False)
    p1_audit = {"training": bool(model.training), "parameter_count": sum(p.numel() for p in model.parameters()),
                "requires_grad_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "optimizer_created": False}
    if p1_audit["training"] or p1_audit["requires_grad_parameter_count"]:
        raise RuntimeError(f"P1 freeze audit failed: {p1_audit}")
    dump(OUT / "p1_freeze_audit.json", p1_audit)
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=400)
    full, indices = matrix.fake_dataset(tokenizer, "official1000")
    hist = {x["sample_id"]: x for x in rows(HIST_P1)}
    c1 = {x["sample_id"]: x for x in rows(OUT / "c1_records.jsonl")}
    path = OUT / "seg_hidden_diagnostic.jsonl"
    existing = rows(path) if path.exists() else []
    valid_indices = [i for i in indices if hist[full.rows[i]["sample_id"]].get("seg_triggered") and hist[full.rows[i]["sample_id"]].get("has_pred_mask")]
    expected = [full.rows[i]["sample_id"] for i in valid_indices[:len(existing)]]
    if [x["sample_id"] for x in existing] != expected: raise RuntimeError("P1 hidden resume prefix drift")
    core = core_model(model)
    for ordinal, full_index in enumerate(valid_indices[len(existing):], start=len(existing)):
        sample = full[full_index]; sid = sample["sample_id"]; old = hist[sid]
        batch = backend._batch(sample, "", question=CANONICAL_UNIFIED_QUESTION)
        ids = old["prompt_token_ids"] + old["generated_token_ids"]
        token_ids = torch.tensor(ids, dtype=torch.long, device=device)[None]
        batch["input_ids"] = token_ids; batch["attention_masks"] = torch.ones_like(token_ids, dtype=torch.bool)
        with torch.inference_mode(): raw, _projected, _rp, _pp = capture_representations(core, batch)
        c1_hidden = torch.load(c1[sid]["c1_seg_hidden_path"], map_location="cpu", weights_only=True)
        if c1_hidden.shape[0] != 1:
            append(path, {"sample_id": sid, "p1_valid": True, "c1_valid": False, "c1_seg_count": int(c1_hidden.shape[0])})
        else:
            a, b = raw.detach().float().cpu(), c1_hidden[0].float()
            append(path, {"sample_id": sid, "p1_valid": True, "c1_valid": True,
                "p1_l2_norm": float(a.norm()), "c1_l2_norm": float(b.norm()),
                "cosine_similarity": float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())})
        if (ordinal + 1) % 25 == 0:
            print(json.dumps({"stage": "P1_C1_SEG_HIDDEN", "done": ordinal + 1, "total": len(valid_indices)}), flush=True)


def finalize(historical, protocol):
    c1_rows = rows(OUT / "c1_records.jsonl")
    if len(c1_rows) != 1000: raise RuntimeError(f"C1 incomplete: {len(c1_rows)}")
    p1 = historical["P1"]["records"]; r1 = historical["R1"]["records"]
    c1 = [x["c1"] for x in c1_rows]; c1r1 = [x["c1_old_r1"] for x in c1_rows]
    arms = {"P1": summarize(p1), "P1+old_R1": summarize(r1), "C1": summarize(c1), "C1+old_R1": summarize(c1r1)}
    p1_trigger = sum(bool(x.get("seg_triggered") and x.get("has_pred_mask")) for x in rows(HIST_P1)) / 1000
    c1_trigger = sum(bool(x["seg_triggered"] and x["seg_count"] > 0) for x in c1_rows) / 1000
    arms["P1"]["seg_trigger_rate"] = p1_trigger; arms["P1+old_R1"]["seg_trigger_rate"] = p1_trigger
    arms["C1"]["seg_trigger_rate"] = c1_trigger; arms["C1+old_R1"]["seg_trigger_rate"] = c1_trigger
    stats = {"C1+old_R1_vs_C1": compare(c1r1, c1, seed=SEED), "P1+old_R1_vs_P1": compare(r1, p1, seed=SEED),
             "C1_vs_P1": compare(c1, p1, seed=SEED)}
    hidden_all = rows(OUT / "seg_hidden_diagnostic.jsonl")
    hidden = [x for x in hidden_all if x.get("p1_valid") and x.get("c1_valid")]
    cos = np.array([x["cosine_similarity"] for x in hidden]); pn = np.array([x["p1_l2_norm"] for x in hidden]); cn = np.array([x["c1_l2_norm"] for x in hidden])
    diag = {"p1_valid_seg_samples": len(hidden_all), "joint_valid_seg_samples": len(hidden),
            "p1_l2_norm_mean": float(pn.mean()), "p1_l2_norm_std": float(pn.std()),
            "c1_l2_norm_mean": float(cn.mean()), "c1_l2_norm_std": float(cn.std()),
            "cosine_similarity_mean": float(cos.mean()), "cosine_similarity_median": float(np.median(cos)),
            "relative_mean_l2_norm_difference": float(abs(cn.mean() - pn.mean()) / pn.mean())}
    t = stats["C1+old_R1_vs_C1"]
    pos_iou = t["foreground_iou"]["bootstrap_95_ci"][0] > 0
    pos_f1 = t["foreground_f1"]["bootstrap_95_ci"][0] > 0
    global_ok = arms["C1+old_R1"]["global_foreground_iou"] >= arms["C1"]["global_foreground_iou"] and arms["C1+old_R1"]["global_foreground_f1"] >= arms["C1"]["global_foreground_f1"]
    if pos_iou and pos_f1 and global_ok: transfer = "YES"
    elif not pos_iou and not pos_f1 and not (arms["C1+old_R1"]["global_foreground_iou"] > arms["C1"]["global_foreground_iou"] or arms["C1+old_R1"]["global_foreground_f1"] > arms["C1"]["global_foreground_f1"]): transfer = "NO"
    else: transfer = "MIXED"
    base = stats["C1_vs_P1"]
    changed = any(not (x["bootstrap_95_ci"][0] <= 0 <= x["bootstrap_95_ci"][1]) for x in (base["foreground_iou"], base["foreground_f1"]))
    shift = diag["cosine_similarity_mean"] < .95 or diag["relative_mean_l2_norm_difference"] > .10
    decisions = {"OLD_R1_TRANSFERS_TO_C1": transfer, "C1_CHANGES_BASE_LOCALIZATION": "YES" if changed else "NO",
                 "SEG_QUERY_SHIFT_OBSERVED": "YES" if shift else "NO", "RETRAIN_R1_ON_C1_NEEDED": "NO" if transfer == "YES" else "YES"}
    result = {"schema": "phase6e1_c1_old_r1_transfer_v1", "status": "COMPLETE", "completed_at_utc": now(),
              "protocol": protocol, "arms": arms, "paired_bootstrap": stats, "seg_hidden_diagnostic": diag,
              "decisions": decisions, "raw_records": {"c1": str((OUT/'c1_records.jsonl').resolve()), "hidden": str((OUT/'seg_hidden_diagnostic.jsonl').resolve())}}
    dump(RESULT, result)
    lines = ["# Phase 6E.1 — C1 + old R1 frozen transfer", "", "Official1000 canonical G0 only; all parameters frozen.", "",
        "| Arm | Mean FG IoU | Global FG IoU | Mean FG F1 | Global FG F1 | SEG trigger rate |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in arms.items(): lines.append(f"| {name} | {m['mean_foreground_iou']:.6f} | {m['global_foreground_iou']:.6f} | {m['mean_foreground_f1']:.6f} | {m['global_foreground_f1']:.6f} | {m['seg_trigger_rate']:.6f} |")
    lines += ["", "## Decisions", ""] + [f"- `{k} = {v}`" for k, v in decisions.items()]
    DOC.write_text("\n".join(lines) + "\n")


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:2"); torch.cuda.set_device(device)
    historical, protocol = freeze_protocol()
    dump(OUT / "worker_status.json", {"status": "RUNNING", "pid": os.getpid(), "stage": "C1_G0_AND_OLD_R1", "started_at_utc": now()})
    try:
        c1_run(device)
        dump(OUT / "worker_status.json", {"status": "RUNNING", "pid": os.getpid(), "stage": "P1_C1_SEG_HIDDEN", "updated_at_utc": now()})
        p1_hidden_diagnostic(device)
        finalize(historical, protocol)
        dump(OUT / "worker_status.json", {"status": "COMPLETE", "pid": os.getpid(), "stage": "STOP", "updated_at_utc": now()})
    except BaseException as error:
        dump(OUT / "worker_status.json", {"status": "FAILED", "pid": os.getpid(), "exception_type": type(error).__name__, "exception": str(error), "updated_at_utc": now()})
        raise


if __name__ == "__main__": main()
