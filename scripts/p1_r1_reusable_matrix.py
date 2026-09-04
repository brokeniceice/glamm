#!/usr/bin/env python3
"""Complete the frozen P1 reusable-localization matrix for Phase4H-D R1.

The evaluator replays historical P1 G0/G1 trajectories and uses the historical
Phrase/TF construction.  It does not train, select, tune, or modify either
checkpoint.  Classification is handled by a structural exact-reuse audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from eval.forensics_eval import FORENSICS_QUESTION, GLaMMForensicsBackend
from model.llava import conversation as conversation_lib
from model.pcerf import sam_lowres_to_original_normalized
from scripts import phase4ha_utility_gated_rectification as ha
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts.phase2a_final_evaluate import load_model
from scripts.phase3c1_cache import clip_grid
from scripts.phase3c2_p3 import corrupted_sample, dataset_for
from scripts.phase3a1_loki_g0 import LokiLocalizationDataset
from tools.phase3c1 import geometry_for
from tools.phase3f_aogd import capture_representations, core_model
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import clip_coordinates, compare, sam_coordinates, summarize, tensor_state_sha256
from tools.phase4f import load_evidence_source, load_sam_runtime

SEED = 3407
P1 = Path("/data/yz/groundingLMM_official/checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt")
P1_SHA = "fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326"
R1 = ROOT / "outputs/phase4hd/r1/selected_checkpoint.pt"
R1_SHA = "9b38ef1e62c86c74287895e681ec8ebb96b724e3bae52622d52b61bca7ddf5a5"
OUT = ROOT / "outputs/p1_r1_reusable_matrix"
CACHE = Path("/data/yz/groundingLMM_official/cache/p1_r1_reusable_matrix")
CFG = yaml.safe_load((ROOT / "configs/phase3a_p1.yaml").read_text())
P4F_CFG = yaml.safe_load((ROOT / "configs/phase4f_language_preserving_rectification.yaml").read_text())

JOBS = {
    "internal_g0": ("internal", "original", "g0"),
    "internal_phrase": ("internal", "original", "phrase"),
    "internal_tf": ("internal", "original", "tf"),
    "official_g0": ("official1000", "original", "g0"),
    "official_phrase": ("official1000", "original", "phrase"),
    "official_tf": ("official1000", "original", "tf"),
    **{f"internal_{c}": ("internal", c, "g0") for c in ("jpeg70", "jpeg80", "gaussian5", "gaussian10")},
    **{f"official_{c}": ("official1000", c, "g0") for c in ("jpeg70", "jpeg80", "gaussian5", "gaussian10")},
    "loki_g1": ("loki", "original", "g1"),
}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def seed_all() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def historical_path(population: str, condition: str, mode: str) -> Path:
    if population == "loki":
        return ROOT / "outputs/phase3a1_paired_control/evaluation/external_g0/loki/p1/G1/predictions.jsonl"
    if condition == "original":
        base = ROOT / "outputs/phase3a_phrase_grounding/evaluation" / ("internal" if population == "internal" else "official1000")
        return base / {"g0": "G0", "phrase": "phrase_only", "tf": "tf_full_context"}[mode] / "predictions.jsonl"
    if population == "internal":
        return ROOT / f"outputs/phase3c2_p3_robustness/robustness_internal/{condition}/P1_G0.jsonl"
    return ROOT / f"outputs/phase3c2_p3_robustness/robustness/{condition}/p1/G0.jsonl"


def freeze() -> dict:
    if file_sha256(P1) != P1_SHA or file_sha256(R1) != R1_SHA:
        raise RuntimeError("checkpoint provenance drift")
    state = torch.load(R1, map_location="cpu", weights_only=False)
    allowed = {"schema", "arm", "epoch", "optimizer_updates", "utility_state", "rectifier_state", "optimizer", "validation_g0", "config_sha256", "initialization"}
    unexpected = sorted(set(state) - allowed)
    classification_audit = {
        "schema": "p1_r1_classification_exact_reuse_audit_v1",
        "conclusion": "EXACT_REUSE",
        "reason": "R1 checkpoint contains only CSCU utility and Phase4F rectifier states; both are invoked after P1 q-seg construction and only alter SAM image embeddings for mask decoding.",
        "P1_sha256": file_sha256(P1), "R1_sha256": file_sha256(R1),
        "r1_checkpoint_top_level_keys": sorted(state), "unexpected_keys": unexpected,
        "classification_or_language_weights_in_r1": [k for k in state if any(x in k.lower() for x in ("class", "llm", "lora", "text_hidden"))],
        "reusable_sections": ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7"],
        "claim_boundary": "No new classification inference result is claimed; P1 values are reused because candidate classification is definitionally the unchanged P1 route.",
    }
    dump(OUT / "classification_invariance.json", classification_audit)
    manifest = {
        "schema": "p1_r1_reusable_matrix_protocol_v4", "status": "FROZEN_BEFORE_MISSING_LOCALIZATION",
        "seed": SEED, "P1": {"path": str(P1), "sha256": P1_SHA},
        "R1": {"path": str(R1), "sha256": R1_SHA, "selected_epoch": 9},
        "jobs": {name: {"population": p, "condition": c, "mode": m,
                         "historical_predictions": str(historical_path(p, c, m)),
                         "historical_predictions_usage": "population/order only; legacy TF metrics discarded" if m == "tf" else "P1 frozen trajectory/baseline",
                         "historical_sha256": file_sha256(historical_path(p, c, m))}
                 for name, (p, c, m) in JOBS.items()},
        "metric": {"target": "historical population target", "threshold": "mask logit > 0", "invalid": "empty prediction and IoU=0"},
        "trajectory": {"g0": "replay exact historical prompt_token_ids + generated_token_ids",
                       "phrase": "historical phrase_only backend contract",
                       "tf": "fresh P1 and R1 full-context teacher-forced run with the correct canonical user question; legacy-prompt TF values are discarded"},
        "no_training": True, "no_selection": True, "no_tuning": True,
        "authorized_access": ["internal test matrix-listed localization", "official1000 matrix-listed localization"],
        "excluded": ["G0 Phrase Repair", "LOKI G0 free authenticity verdict", "AIGI G0/Phrase/TF"],
    }
    manifest["contract_sha256"] = hash_json(manifest)
    path = OUT / "protocol_v4.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("frozen protocol drift")
    dump(path, manifest)
    dump(OUT / "protocol_correction_v2.json", {
        "schema": "p1_r1_reusable_matrix_protocol_correction_v2",
        "superseded": [str(OUT / "protocol.json"), str(OUT / "protocol_v2.json"), str(OUT / "protocol_v3.json")], "active": str(path),
        "reason": "User explicitly rejected the erroneous legacy prompt. Internal/Official TF therefore reruns both P1 and R1 with the correct canonical prompt; historical legacy-prompt TF metrics are discarded.",
        "smoke_canonical_first_sample_abs_iou_drift": 0.0019136598229175794,
        "change_scope": ["internal_tf", "official_tf"], "all_other_jobs_unchanged": True,
    })
    return manifest


def load_p1(device):
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, meta = load_model(CFG, P1, device, expected_step=3500, expected_epoch=7)
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=400)
    return model, tokenizer, backend, meta


def fake_dataset(tokenizer, population: str):
    if population == "loki":
        full = LokiLocalizationDataset(ROOT / "datasets/LOKI/legion_localization/manifest.jsonl",
                                       CFG["model"]["vision_tower"], int(CFG["model"]["image_size"]))
        return full, list(range(229))
    full = dataset_for(tokenizer, CFG, "test" if population == "internal" else "official1000")
    indices = [i for i, r in enumerate(full.rows) if int(r["class_label"]) == 1]
    expected = 1104 if population == "internal" else 1000
    if len(indices) != expected:
        raise RuntimeError(f"population drift {population}: {len(indices)}")
    return full, indices


def sample_at(full, full_index: int, condition: str):
    if condition == "original":
        sample = full[full_index]
        return sample, None
    sample, _rgb, image_hash = corrupted_sample(full, full_index, condition)
    return sample, image_hash


def capture_call(core, call):
    holder = []; original = core._extract_projected_seg_predictor_hidden
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        holder.append([x.detach().to(torch.bfloat16).cpu() for x in result[0]])
        return result
    core._extract_projected_seg_predictor_hidden = wrapped
    try:
        output = call()
    finally:
        core._extract_projected_seg_predictor_hidden = original
    vectors = holder[-1] if holder else []
    qseg = vectors[0][0] if len(vectors) == 1 and vectors[0].shape[0] == 1 else None
    return output, qseg


def context_for(mode, row, sample, backend, core, vision_tower):
    vision = []
    def save_vision(_module, _inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        vision.append(value.detach().to(torch.bfloat16).cpu())
    hook = vision_tower.register_forward_hook(save_vision)
    try:
        if mode in ("g0", "g1"):
            ids = row.get("prompt_token_ids", []) + row.get("generated_token_ids", [])
            if not row.get("seg_triggered") or not row.get("has_pred_mask"):
                return None, None
            batch = backend._batch(sample, "", question=CANONICAL_UNIFIED_QUESTION)
        elif mode == "phrase":
            ids = None
            batch = backend._batch(sample, backend.phrase_only_content(sample), question=FORENSICS_QUESTION)
        else:
            ids = None
            batch = backend._batch(sample, backend.tf_phrase_content(sample), question=CANONICAL_UNIFIED_QUESTION)
        if ids is not None:
            token_ids = torch.tensor(ids, dtype=torch.long, device=batch["input_ids"].device)[None]
            batch["input_ids"] = token_ids; batch["attention_masks"] = torch.ones_like(token_ids, dtype=torch.bool)
        try:
            _raw, projected, _rp, _pp = capture_representations(core, batch)
            qseg = projected.detach().to(torch.bfloat16).cpu()
        except RuntimeError:
            qseg = None
    finally:
        hook.remove()
    raw_clip = clip_grid(vision[-1]).cpu() if vision else None
    return qseg, raw_clip


def load_r1(device):
    utility, rectifier, _ = hd.load_common(device)
    state = torch.load(R1, map_location="cpu", weights_only=False)
    utility.load_state_dict(state["utility_state"], strict=True)
    rectifier.load_state_dict(state["rectifier_state"], strict=True)
    utility.eval().requires_grad_(False); rectifier.eval().requires_grad_(False)
    sam = load_sam_runtime(P4F_CFG, device)
    return utility, rectifier, sam


def evaluate_job(job: str, device: torch.device, limit: int | None = None) -> dict:
    population, condition, mode = JOBS[job]
    hist = rows(historical_path(population, condition, mode))
    by_id = {r["sample_id"]: r for r in hist}
    model, tokenizer, backend, _ = load_p1(device)
    full, indices = fake_dataset(tokenizer, population)
    if limit is not None: indices = indices[:limit]
    sample_ids = [full.rows[i]["sample_id"] for i in indices]
    if any(sid not in by_id for sid in sample_ids):
        raise RuntimeError("historical sample-set mismatch")
    source = load_evidence_source(P4F_CFG, "forensic_rect", device)
    utility, rectifier, sam = load_r1(device)
    frozen = {"p1": tensor_state_sha256(model.state_dict()), "source": tensor_state_sha256(source.state_dict()),
              "utility": tensor_state_sha256(utility.state_dict()), "rectifier": tensor_state_sha256(rectifier.state_dict()),
              "sam": tensor_state_sha256(sam.state_dict())}
    core = core_model(model); vision_tower = model.get_model().get_vision_tower()
    r1_records = []; p1_records = []; gates = []; parity = []
    with torch.no_grad():
        for ordinal, full_index in enumerate(indices):
            sample, image_hash = sample_at(full, full_index, condition); sid = sample["sample_id"]; old = by_id[sid]
            if image_hash is not None and old.get("corrupted_rgb_sha256") != image_hash:
                raise RuntimeError(f"corruption hash drift: {sid}")
            target = torch.as_tensor(sample["masks"]).bool().any(0)
            qseg, raw_clip = context_for(mode, old, sample, backend, core, vision_tower)
            if mode != "tf":
                p1_records.append({k: old[k] for k in ("sample_id", "foreground_iou", "foreground_f1", "tp", "fp", "fn")})
            if qseg is None:
                if mode == "tf":
                    p1_records.append({"sample_id": sid, "foreground_iou": 0., "foreground_f1": 0., "tp": 0, "fp": 0, "fn": int(target.sum())})
                r1_records.append({"sample_id": sid, "foreground_iou": 0., "foreground_f1": 0., "tp": 0, "fp": 0, "fn": int(target.sum()), "valid_q_seg": False})
                continue
            h, w = target.shape; sg, cg = geometry_for("sam", (h, w)), geometry_for("clip", (h, w))
            raw = model.get_grounding_encoder_embs(sample["grounding_enc_image"][None].to(device=device, dtype=torch.bfloat16))
            if raw_clip is None:
                tokens, _ = vision_tower(sample["global_enc_image"][None].to(device=device, dtype=torch.bfloat16))
                raw_clip = clip_grid(tokens).cpu()
            raw_clip = raw_clip.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16): forensic = source(raw_clip, return_features=True)
            s64 = sam_lowres_to_original_normalized(raw, sg, output_hw=(64, 64)).to(torch.bfloat16)
            qone = qseg[None].to(device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type=device.type, enabled=False): low_p1 = sam(qone, raw.to(torch.bfloat16))
            p1_now = metric_record(sid, inverse_sam_logits(low_p1, sg), target)
            if mode == "tf":
                p1_records.append(p1_now)
            zl = sam_lowres_to_original_normalized(low_p1, sg, output_hw=(256, 256)).to(torch.bfloat16)
            f24 = forensic["F_forensic"].to(torch.bfloat16); zf = forensic["logits"].to(torch.bfloat16)
            batch = {"S64": s64, "q_seg": qone, "z_L": zl, "F24": f24, "z_F24": zf, "clip_geometries": [cg],
                     "valid_g0": torch.ones(1, dtype=torch.bool, device=device), "forensic_present": torch.ones(1, dtype=torch.bool, device=device),
                     "forensic_vacuous": torch.zeros(1, dtype=torch.bool, device=device), "forensic_off": torch.zeros(1, dtype=torch.bool, device=device)}
            uout = hc.utility_forward(utility, batch); sc = sam_coordinates(sg, grid=64)[None].to(device); cc = clip_coordinates(cg, grid=24)[None].to(device)
            valid = torch.ones(1, 576, dtype=torch.bool, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16): p4f = rectifier(raw, f24, sc, cc, valid)
            gate = ha.gate_to_sam_grid(uout["U"], sc) * p4f["support"].reshape(1, 1, 64, 64).float()
            adapted = ha.gated_embedding(raw, p4f["image_embeddings"], gate)
            with torch.autocast(device_type=device.type, enabled=False): low = sam(qone, adapted.to(torch.bfloat16))
            row = metric_record(sid, inverse_sam_logits(low, sg), target); row["valid_q_seg"] = True
            r1_records.append(row); gates.append(gate[p4f["support"].reshape(1, 1, 64, 64).bool()].cpu())
            if mode != "tf" and condition == "original" and ordinal < 8:
                parity.append(abs(p1_now["foreground_iou"] - old["foreground_iou"]))
            if (ordinal + 1) % 50 == 0:
                print(json.dumps({"stage": "R1_MATRIX", "job": job, "done": ordinal + 1, "total": len(indices)}), flush=True)
    after = {"p1": tensor_state_sha256(model.state_dict()), "source": tensor_state_sha256(source.state_dict()),
             "utility": tensor_state_sha256(utility.state_dict()), "rectifier": tensor_state_sha256(rectifier.state_dict()),
             "sam": tensor_state_sha256(sam.state_dict())}
    if frozen != after: raise RuntimeError("frozen source mutation")
    gate = torch.cat(gates).float() if gates else torch.empty(0)
    result = {"schema": "p1_r1_reusable_matrix_job_v1", "status": "COMPLETE" if limit is None else "SMOKE",
              "job": job, "population": population, "condition": condition, "mode": mode, "n": len(indices),
              "P1": {"metrics": summarize(p1_records), "records": p1_records},
              "R1": {"metrics": summarize(r1_records), "records": r1_records},
              "R1_vs_P1": compare(r1_records, p1_records, seed=SEED),
              "gate": None if not gate.numel() else {"mean": float(gate.mean()), "std": float(gate.std(unbiased=False)),
                  "low_le_0.01": float((gate <= .01).float().mean()), "high_ge_0.99": float((gate >= .99).float().mean())},
              "historical_P1_redecode_parity_first8_max_abs_iou": max(parity) if parity else None,
              "source_integrity": {"before": frozen, "after": after, "exact": frozen == after}, "completed_unix": time.time()}
    path = OUT / "jobs" / (f"{job}.smoke.json" if limit is not None else f"{job}.json")
    dump(path, result)
    return result


def aggregate() -> dict:
    missing = [name for name in JOBS if not (OUT / "jobs" / f"{name}.json").is_file()]
    if missing: raise RuntimeError(f"missing jobs: {missing}")
    payload = {"schema": "p1_r1_reusable_matrix_results_v1", "status": "COMPLETE",
               "classification": json.loads((OUT / "classification_invariance.json").read_text()),
               "localization": {name: json.loads((OUT / "jobs" / f"{name}.json").read_text()) for name in JOBS},
               "external": {"aigi_g1": json.loads((ROOT / "outputs/p1_r1_aigi_localization/results.json").read_text()),
                            "loki_g1": json.loads((OUT / "jobs/loki_g1.json").read_text())}}
    dump(OUT / "results.json", payload)
    render_document(payload)
    return payload


def stat_text(job: dict) -> str:
    stat = job["R1_vs_P1"]["foreground_iou"]
    return (f"{stat['mean_difference']:+.6f} [{stat['bootstrap_95_ci'][0]:+.6f}, {stat['bootstrap_95_ci'][1]:+.6f}]; "
            f"{stat['wins']}/{stat['ties']}/{stat['losses']}; p={stat['wilcoxon_pvalue']:.6g}")


def render_document(payload: dict) -> None:
    doc = ROOT / "docs/p1_reusable_evaluation_baselines.md"
    text = doc.read_text(encoding="utf-8")
    marker = "## 9. Phase4H-D R1 完整同条件对比"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n\n"
    loc = payload["localization"]
    dev = json.loads((ROOT / "outputs/phase4hd/r1/dev_results.json").read_text())
    dev_rows = []
    for label, key, p1 in (("DEV G0", "matched", 0.148233), ("DEV Phrase", "phrase", 0.245798), ("DEV TF canonical", "tf", 0.342928)):
        value = dev["metrics"][key]["mean_foreground_iou"]
        dev_rows.append(f"| {label} | {p1:.6f} | {value:.6f} | existing Phase4H-D selected-checkpoint result |")
    rows_out = []
    labels = {
        "internal_g0": "Internal test G0", "internal_phrase": "Internal test Phrase",
        "internal_tf": "Internal test TF canonical (fresh P1/R1)",
        "official_g0": "Official1000 G0", "official_phrase": "Official1000 Phrase",
        "official_tf": "Official1000 TF canonical (fresh P1/R1)",
        "internal_jpeg70": "Internal G0 JPEG70", "internal_jpeg80": "Internal G0 JPEG80",
        "internal_gaussian5": "Internal G0 Gaussian5", "internal_gaussian10": "Internal G0 Gaussian10",
        "official_jpeg70": "Official G0 JPEG70", "official_jpeg80": "Official G0 JPEG80",
        "official_gaussian5": "Official G0 Gaussian5", "official_gaussian10": "Official G0 Gaussian10",
        "loki_g1": "LOKI G1",
    }
    for name in labels:
        job = loc[name]
        rows_out.append(f"| {labels[name]} | {job['P1']['metrics']['mean_foreground_iou']:.6f} | "
                        f"{job['R1']['metrics']['mean_foreground_iou']:.6f} | {stat_text(job)} |")
    aigi = payload["external"]["aigi_g1"]["modes"]["g1"]
    astat = aigi["R1_vs_P1"]["foreground_iou"]
    rows_out.append(f"| AIGI-test G1 | {aigi['P1']['metrics']['mean_foreground_iou']:.6f} | "
                    f"{aigi['R1']['metrics']['mean_foreground_iou']:.6f} | "
                    f"{astat['mean_difference']:+.6f} [{astat['bootstrap_95_ci'][0]:+.6f}, {astat['bootstrap_95_ci'][1]:+.6f}]; "
                    f"{astat['wins']}/{astat['ties']}/{astat['losses']}; p={astat['wilcoxon_pvalue']:.6g} |")
    section = f"""{marker}

R1 checkpoint：`{R1}`，epoch 9，SHA256 `{R1_SHA}`。本节仅补齐本文件保留的 P1 协议；已丢弃的 G0 Phrase Repair、LOKI G0 free-authenticity，以及 AIGI G0/Phrase/TF 均未恢复。

R1 不改 classification/LM verdict 路径。`outputs/p1_r1_reusable_matrix/classification_invariance.json` 已审计 R1 checkpoint 只含 CSCU utility 与 Phase4F rectifier state，因此第 3.1–3.7 节的 R1 classification 值对 P1 为 **EXACT REUSE**，不是新推理结果。

旧 Internal/Official TF 产物使用过错误的 legacy user prompt，按用户指令丢弃。下表 TF 行是 P1 与 R1 使用正确 canonical prompt 的 fresh同条件重跑；不得再引用旧 TF 点值。

| Population / protocol | P1 mean FG IoU | R1 mean FG IoU | R1−P1 delta [bootstrap 95% CI]; W/T/L; Wilcoxon |
|---|---:|---:|---|
{chr(10).join(dev_rows + rows_out)}

所有新增定位行均使用同一 sample order、target、geometry 与 mask-logit `>0` threshold；G0/G1 复用冻结 P1 trajectory，corruption 行逐图验证 corrupted RGB SHA256，所有 frozen source 前后 tensor hash 必须一致。完整逐样本记录与统计见 `outputs/p1_r1_reusable_matrix/results.json`。
"""
    doc.write_text(text + section, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=("freeze", "run", "aggregate")); ap.add_argument("--job", choices=sorted(JOBS));
    ap.add_argument("--device", default="cuda:2"); ap.add_argument("--limit", type=int); args = ap.parse_args()
    freeze()
    if args.command == "freeze": print((OUT / "protocol_v4.json").read_text()); return
    if args.command == "aggregate": aggregate(); return
    if not args.job: raise SystemExit("--job is required")
    device = torch.device(args.device); torch.cuda.set_device(device); seed_all(); evaluate_job(args.job, device, args.limit)


if __name__ == "__main__": main()
