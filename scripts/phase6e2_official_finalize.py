#!/usr/bin/env python3
"""Final selected-new-R1 Official1000 evaluation and Phase6E.2 report."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from model.llava import conversation as conversation_lib
from model.pcerf import sam_lowres_to_original_normalized
from scripts import phase4ha_utility_gated_rectification as ha
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts import p1_r1_reusable_matrix as matrix
from scripts.phase2a_final_evaluate import load_model
from scripts.phase3c1_cache import clip_grid
from scripts.phase6e1_c1_old_r1_transfer import C1_CFG_PATH, C1_CKPT, OUT as E1_OUT
from tools.phase3c1 import geometry_for
from tools.phase3f_aogd import core_model
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import clip_coordinates, compare, sam_coordinates, summarize
from tools.phase4f import load_evidence_source, load_sam_runtime

SEED = 3407
ARM = os.environ.get("PHASE6E2_ARM", "main")
if ARM not in ("main", "i1", "i2"): raise RuntimeError(f"unsupported Phase6E.2 arm: {ARM}")
BASE_OUT = ROOT / "outputs/phase6e2_c1_specific_r1"
OUT = BASE_OUT if ARM == "main" else BASE_OUT / ARM
SELECTED = OUT / "selected_checkpoint.pt"
RESULT = OUT / ({"main": "phase6e2_c1_specific_r1.json", "i1": "phase6e2_i1_random_utility.json", "i2": "phase6e2_i2_random_init.json"}[ARM])
DOC = ({"main": ROOT / "docs/phase6e2_c1_specific_r1.md", "i1": ROOT / "docs/phase6e2_i1_random_utility.md",
        "i2": ROOT / "docs/phase6e2_i2_random_init.md"}[ARM])
E1_RESULT = E1_OUT / "phase6e1_c1_old_r1_transfer.json"
RECORDS = OUT / "official1000_new_r1_records.jsonl"
C1_SHA = "85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff"


def rows(path):
    with Path(path).open() as f: return [json.loads(x) for x in f if x.strip()]


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n"); os.replace(tmp, path)


def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f: f.write(json.dumps(value, ensure_ascii=False) + "\n"); f.flush()


def evaluate(device):
    selector = json.loads((OUT / "selector.json").read_text())
    if selector.get("official1000_used") or selector.get("internal_test_used"):
        raise RuntimeError("selector firewall failed")
    if file_sha256(C1_CKPT) != C1_SHA: raise RuntimeError("C1 checkpoint drift")
    state = torch.load(SELECTED, map_location="cpu", weights_only=False)
    if state.get("c1_sha256") != C1_SHA: raise RuntimeError("new R1 checkpoint provenance drift")
    utility, rectifier, _ = hd.load_common(device)
    utility.load_state_dict(state["utility_state"], strict=True); rectifier.load_state_dict(state["rectifier_state"], strict=True)
    utility.eval().requires_grad_(False); rectifier.eval().requires_grad_(False)
    sam = load_sam_runtime(hd.CFG, device); source = load_evidence_source(hd.CFG, "forensic_rect", device)
    cfg = yaml.safe_load(C1_CFG_PATH.read_text()); conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, _ = load_model(cfg, C1_CKPT, device, expected_step=2500, expected_epoch=5)
    model.eval().requires_grad_(False); model.rine_conditioner.eval().requires_grad_(False)
    core = core_model(model); vision = model.get_model().get_vision_tower()
    full, indices = matrix.fake_dataset(tokenizer, "official1000")
    c1_rows = rows(E1_OUT / "c1_records.jsonl"); c1_by_id = {x["sample_id"]: x for x in c1_rows}
    existing = rows(RECORDS) if RECORDS.exists() else []
    expected = [full.rows[i]["sample_id"] for i in indices[:len(existing)]]
    if [x["sample_id"] for x in existing] != expected: raise RuntimeError("Official1000 resume-prefix drift")
    for ordinal, index in enumerate(indices[len(existing):], start=len(existing)):
        sample = full[index]; sid = sample["sample_id"]; target = torch.as_tensor(sample["masks"]).bool().any(0)
        old = c1_by_id[sid]; raw_hidden = torch.load(old["c1_seg_hidden_path"], map_location="cpu", weights_only=True)
        if raw_hidden.shape[0] != int(old["seg_count"]): raise RuntimeError(f"hidden count drift {sid}")
        if raw_hidden.shape[0] == 0:
            record = {"sample_id": sid, "foreground_iou": 0., "foreground_f1": 0., "tp": 0, "fp": 0,
                      "fn": int(target.sum()), "valid_q_seg": False}
        else:
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                q_values = core.model.text_hidden_fcs[0](raw_hidden.to(device=device, dtype=torch.bfloat16))
                raw = model.get_grounding_encoder_embs(sample["grounding_enc_image"][None].to(device=device, dtype=torch.bfloat16))
                tokens, _ = vision(sample["global_enc_image"][None].to(device=device, dtype=torch.bfloat16))
                forensic = source(clip_grid(tokens), return_features=True)
            h, w = target.shape; sg, cg = geometry_for("sam", (h, w)), geometry_for("clip", (h, w))
            s64 = sam_lowres_to_original_normalized(raw, sg, output_hw=(64, 64)).to(torch.bfloat16)
            f24, zf = forensic["F_forensic"].to(torch.bfloat16), forensic["logits"].to(torch.bfloat16)
            sc = sam_coordinates(sg, grid=64)[None].to(device); cc = clip_coordinates(cg, grid=24)[None].to(device)
            valid = torch.ones(1, 576, dtype=torch.bool, device=device); logits = []
            with torch.no_grad():
                for qone in q_values:
                    qone = qone[None].to(torch.bfloat16)
                    with torch.autocast(device_type=device.type, enabled=False): low0 = sam(qone, raw.to(torch.bfloat16))
                    zl = sam_lowres_to_original_normalized(low0, sg, output_hw=(256, 256)).to(torch.bfloat16)
                    batch = {"S64": s64, "q_seg": qone, "z_L": zl, "F24": f24, "z_F24": zf,
                        "clip_geometries": [cg], "valid_g0": torch.ones(1, dtype=torch.bool, device=device),
                        "forensic_present": torch.ones(1, dtype=torch.bool, device=device),
                        "forensic_vacuous": torch.zeros(1, dtype=torch.bool, device=device),
                        "forensic_off": torch.zeros(1, dtype=torch.bool, device=device)}
                    u = hc.utility_forward(utility, batch)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16): p4f = rectifier(raw, f24, sc, cc, valid)
                    gate = ha.gate_to_sam_grid(u["U"], sc) * p4f["support"].reshape(1, 1, 64, 64).float()
                    adapted = ha.gated_embedding(raw, p4f["image_embeddings"], gate)
                    with torch.autocast(device_type=device.type, enabled=False): low = sam(qone, adapted.to(torch.bfloat16))
                    logits.append(inverse_sam_logits(low, sg))
            record = metric_record(sid, torch.stack(logits).amax(0), target); record["valid_q_seg"] = True
        append(RECORDS, record)
        if (ordinal + 1) % 20 == 0: print(json.dumps({"stage": "OFFICIAL1000", "done": ordinal + 1, "total": 1000}), flush=True)


def finalize():
    new = rows(RECORDS)
    if len(new) != 1000: raise RuntimeError(f"Official1000 incomplete: {len(new)}")
    e1 = json.loads(E1_RESULT.read_text()); c1_rows = rows(E1_OUT / "c1_records.jsonl")
    c1 = [x["c1"] for x in c1_rows]; old = [x["c1_old_r1"] for x in c1_rows]
    arm_name = {"main": "C1+new_R1", "i1": "C1+I1_random_utility_R1", "i2": "C1+I2_random_R1"}[ARM]
    arms = dict(e1["arms"]); arms[arm_name] = summarize(new)
    arms[arm_name]["seg_trigger_rate"] = arms["C1"]["seg_trigger_rate"]
    stats = {f"{arm_name}_vs_C1+old_R1": compare(new, old, seed=SEED),
             f"{arm_name}_vs_C1": compare(new, c1, seed=SEED)}
    iou = stats[f"{arm_name}_vs_C1+old_R1"]["foreground_iou"]
    f1 = stats[f"{arm_name}_vs_C1+old_R1"]["foreground_f1"]
    global_ok = (arms[arm_name]["global_foreground_iou"] >= arms["C1+old_R1"]["global_foreground_iou"] and
                 arms[arm_name]["global_foreground_f1"] >= arms["C1+old_R1"]["global_foreground_f1"])
    if iou["bootstrap_95_ci"][0] > 0 and f1["bootstrap_95_ci"][0] > 0 and global_ok: verdict = "YES"
    elif iou["bootstrap_95_ci"][1] < 0 and f1["bootstrap_95_ci"][1] < 0: verdict = "NO"
    else: verdict = "MIXED"
    delta = arms[arm_name]["mean_foreground_iou"] - arms["C1+old_R1"]["mean_foreground_iou"]
    decisions = ({"C1_SPECIFIC_R1_BEATS_OLD_R1": verdict, "C1_SPECIFIC_ADAPTATION_GAIN": delta,
                  "FINAL_R1_CANDIDATE": "NEW_R1" if verdict == "YES" else "OLD_R1"} if ARM == "main" else
                 ({"I1_RANDOM_UTILITY_R1_BEATS_OLD_R1": verdict, "I1_RANDOM_UTILITY_ADAPTATION_GAIN": delta}
                  if ARM == "i1" else {"I2_RANDOM_R1_BEATS_OLD_R1": verdict, "I2_RANDOM_ADAPTATION_GAIN": delta}))
    result = {"schema": "phase6e2_c1_specific_r1_v1", "status": "COMPLETE", "arm": ARM, "arms": arms,
              "paired_statistics": stats, "decisions": decisions,
              "provenance": {"c1_checkpoint": str(C1_CKPT.resolve()), "c1_sha256": C1_SHA,
                "new_r1_selected": str(SELECTED.resolve()), "new_r1_sha256": file_sha256(SELECTED),
                "selector": json.loads((OUT / "selector.json").read_text()),
                "initialization": json.loads((OUT / "initialization_provenance.json").read_text()),
                "phase6e1_reuse": str(E1_RESULT.resolve()), "phase6e1_sha256": file_sha256(E1_RESULT),
                "official_records": str(RECORDS.resolve()), "official_records_sha256": file_sha256(RECORDS)}}
    dump(RESULT, result)
    table = ["| Arm | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 | SEG trigger |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in arms.items(): table.append(f"| {name} | {m['mean_foreground_iou']:.6f} | {m['mean_foreground_f1']:.6f} | {m['global_foreground_iou']:.6f} | {m['global_foreground_f1']:.6f} | {m['seg_trigger_rate']:.6f} |")
    title = {"main": "C1-specific new R1", "i1": "I1 random utility + Phase4F epoch9 rectifier",
             "i2": "I2 random utility + random rectifier"}[ARM]
    text = f"# Phase 6E.2 — {title}\n\n" + "\n".join(table) + "\n\n## Decisions\n\n" + "\n".join(f"- `{k} = {v}`" for k, v in decisions.items()) + "\n\nOfficial1000仅在internal-validation selector冻结后访问；完成后STOP。\n"
    DOC.write_text(text)
    dump(OUT / "worker_status.json", {"status": "COMPLETE", "stage": "STOP", "result": str(RESULT.resolve())})


def main():
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    dump(OUT / "worker_status.json", {"status": "RUNNING", "stage": "OFFICIAL1000_SELECTED_ONLY", "pid": os.getpid()})
    evaluate(device); finalize()


if __name__ == "__main__": main()
