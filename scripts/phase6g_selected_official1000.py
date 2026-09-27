#!/usr/bin/env python3
"""One frozen Official1000 pass for each preselected Phase6G complete-R1 candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.clip_forensic_adapter import CLIPSpatialArm
from model.cross_layer_attention_fusion import CrossLayerPatchAttention
from model.llava import conversation as conversation_lib
from model.pcerf import sam_lowres_to_original_normalized
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts import p1_r1_reusable_matrix as matrix
from scripts.phase2a_final_evaluate import load_model
from scripts.phase6e1_c1_old_r1_transfer import C1_CFG_PATH, C1_CKPT, OUT as E1_OUT
from tools.phase3c1 import geometry_for
from tools.phase3f_aogd import core_model
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import clip_coordinates, compare, sam_coordinates, summarize
from tools.phase4f import load_sam_runtime

OUT = ROOT / "outputs/phase6g_selected_official1000"
BASELINE = ROOT / "outputs/phase6e2_c1_specific_r1/official1000_new_r1_records.jsonl"
C1_SHA = "85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff"
FUSION = ROOT / "outputs/phase6g2_multilevel_attention/phase6g2a/selected.pt"
ARMS = {
    "g1": {
        "name": "Phase6G.1 block17 staged R1",
        "root": ROOT / "outputs/phase6g1_block17_single_layer_replacement/r1/joint",
        "adapter": ROOT / "outputs/phase6g1_block17_single_layer_replacement/adapter/selected.pt",
        "schema": "phase6g1_block17_staged_checkpoint_v1",
        "dev_mean": 0.20562396993788443,
        "dev_global": 0.2234764836052761,
        "reason": "complete R1; DEV mean and global IoU both exceed the same-condition baseline",
    },
    "g7": {
        "name": "Phase6G.7 block11+17 staged R1",
        "root": ROOT / "outputs/phase6g7_block11_17_staged_r1/joint",
        "adapter": ROOT / "outputs/phase6g3_forensic_adapter_audit/a0/selected.pt",
        "schema": "phase6g7_stage_checkpoint_v1",
        "dev_mean": 0.2102029836103252,
        "dev_global": 0.21185677717830342,
        "reason": "largest complete-R1 DEV mean IoU among unevaluated alternatives; global IoU decreased",
    },
}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def freeze_selection() -> dict:
    """Record both choices before either Official1000 candidate is opened."""
    selection = {
        "schema": "phase6g_official1000_selection_v1",
        "status": "FROZEN_BEFORE_CANDIDATE_INFERENCE",
        "selector_data": "internal DEV canonical G0 only",
        "historical_6e2_dev_mean": 0.20248798113178615,
        "historical_6e2_dev_global": 0.22258106040231374,
        "candidate_order": ["g7", "g1"],
        "candidates": {},
        "excluded": {
            "g6": "complete R1 but lower DEV mean than G.7 and global IoU also lower than baseline",
            "g13": "positive paired effect is versus its own no-Utility arm, not a direct complete-R1 baseline",
            "g19_g20": "lower absolute DEV final-mask mean than complete G.1/G.7 candidates",
            "g14_g15": "Rectifier-stage target, not a selected complete-R1 replacement",
            "g18": "architecture audit only; no trained checkpoint",
        },
        "official_use": "one fixed pass per candidate; no checkpoint or threshold reselection",
    }
    for arm in selection["candidate_order"]:
        spec = ARMS[arm]
        root = spec["root"]
        selector = json.loads((root / "selector.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        if selector["status"] != "COMPLETE" or summary["status"] != "COMPLETE_SELECTED_FROZEN":
            raise RuntimeError(f"{arm} selection incomplete")
        if selector.get("official1000_used") or selector.get("test_used") or selector.get("ood_used"):
            raise RuntimeError(f"{arm} selector firewall failed")
        curve = selector["candidates"]
        if len(curve) != 10:
            raise RuntimeError(f"{arm} incomplete DEV trajectory")
        best = max(curve, key=lambda r: (r["dev_g0_mean_iou"], -r["epoch"]))
        ckpt = root / "selected_checkpoint.pt"
        actual_hash = file_sha256(ckpt)
        if (best["epoch"] != selector["selected_epoch"] != summary["selected_epoch"] or
                actual_hash != selector["selected_checkpoint_sha256"] or
                actual_hash != summary["selected_checkpoint_sha256"] or
                abs(best["dev_g0_mean_iou"] - spec["dev_mean"]) > 1e-12):
            raise RuntimeError(f"{arm} DEV selector or selected checkpoint drift")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if (state["schema"] != spec["schema"] or state["epoch"] != best["epoch"] or
                state.get("stage") != "joint" or "utility_state" not in state or
                "rectifier_state" not in state):
            raise RuntimeError(f"{arm} checkpoint payload drift")
        adapter_hash = file_sha256(spec["adapter"])
        protocol = json.loads((root / "protocol.json").read_text())
        if arm == "g1":
            if protocol["adapter_sha256"] != adapter_hash:
                raise RuntimeError("G.1 adapter hash drift")
        else:
            if (protocol["adapter"]["sha256"] != adapter_hash or
                    protocol["fusion"]["sha256"] != file_sha256(FUSION)):
                raise RuntimeError("G.7 source hash drift")
        selection["candidates"][arm] = {
            "name": spec["name"], "checkpoint": str(ckpt.resolve()),
            "checkpoint_sha256": actual_hash, "selected_epoch": best["epoch"],
            "adapter_sha256": adapter_hash,
            "fusion_sha256": file_sha256(FUSION) if arm == "g7" else None,
            "dev_mean_iou": spec["dev_mean"], "dev_global_iou": spec["dev_global"],
            "reason": spec["reason"],
        }
    path = OUT / "selection.json"
    if path.exists():
        if json.loads(path.read_text()) != selection:
            raise RuntimeError("frozen two-candidate selection drift")
    else:
        dump(path, selection)
    return selection


def preflight(arm: str) -> tuple[dict, list[dict]]:
    selection = freeze_selection()
    if file_sha256(C1_CKPT) != C1_SHA:
        raise RuntimeError("C1 checkpoint hash drift")
    c1_cfg = yaml.safe_load(C1_CFG_PATH.read_text())
    clip_cfg = yaml.safe_load((ROOT / "configs/phase6g0_multilevel_dense_clip.yaml").read_text())
    if c1_cfg["model"]["vision_tower"] != clip_cfg["clip"]["checkpoint"]:
        raise RuntimeError("C1 CLIP tower and Phase6G evidence source differ")
    baseline = rows(BASELINE)
    c1_records = rows(E1_OUT / "c1_records.jsonl")
    sample_ids = [r["sample_id"] for r in baseline]
    if (len(sample_ids) != 1000 or len(set(sample_ids)) != 1000 or
            sample_ids != [r["sample_id"] for r in c1_records]):
        raise RuntimeError("Official1000 baseline/C1 sample identity drift")
    root = OUT / arm
    records = root / "predictions.jsonl"
    protocol = {
        "schema": "phase6g_selected_official1000_protocol_v1",
        "status": "FROZEN_BEFORE_INFERENCE", "arm": arm,
        "checkpoint_sha256": selection["candidates"][arm]["checkpoint_sha256"],
        "adapter_sha256": selection["candidates"][arm]["adapter_sha256"],
        "fusion_sha256": selection["candidates"][arm]["fusion_sha256"],
        "c1_sha256": C1_SHA, "clip_source": c1_cfg["model"]["vision_tower"],
        "population": 1000, "sample_ids_sha256": ids_sha(sample_ids),
        "baseline_records_sha256": file_sha256(BASELINE),
        "mode": "canonical G0", "target": "all-reference pixel mask union",
        "threshold_logit": 0.0,
        "no_seg": "zero-IoU record", "multi_seg": "pixelwise maximum of all mask logits",
        "baseline": "historical Phase6E.2 C1+new R1 on identical Official1000 order",
        "selection": "frozen on internal DEV before either Official1000 run",
    }
    if (root / "protocol.json").exists():
        if json.loads((root / "protocol.json").read_text()) != protocol:
            raise RuntimeError(f"{arm} Official1000 protocol drift")
    else:
        dump(root / "protocol.json", protocol)
    if records.exists():
        old = rows(records)
        if len(old) > 1000 or [r["sample_id"] for r in old] != sample_ids[:len(old)]:
            raise RuntimeError(f"{arm} resume prefix drift")
    return selection["candidates"][arm], baseline


def load_candidate(arm: str, device: torch.device):
    spec = ARMS[arm]
    state = torch.load(spec["root"] / "selected_checkpoint.pt", map_location="cpu", weights_only=False)
    utility, rectifier, _ = hd.load_common(device)
    utility.load_state_dict(state["utility_state"], strict=True)
    rectifier.load_state_dict(state["rectifier_state"], strict=True)
    utility.eval().requires_grad_(False)
    rectifier.eval().requires_grad_(False)
    adapter = CLIPSpatialArm(blocks=3)
    adapter.load_state_dict(torch.load(spec["adapter"], map_location="cpu", weights_only=False)["model"], strict=True)
    adapter = adapter.to(device).eval().requires_grad_(False)
    fusion = None
    if arm == "g7":
        fusion = CrossLayerPatchAttention(1024, 8, .01)
        fusion.load_state_dict(torch.load(FUSION, map_location="cpu", weights_only=False)["fusion"], strict=True)
        fusion = fusion.to(device).eval().requires_grad_(False)
    return utility, rectifier, adapter, fusion


def evidence(arm: str, vision, image, adapter, fusion, device):
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        pixel = image[None].to(device=device, dtype=vision.dtype)
        hidden = vision.vision_tower(pixel, output_hidden_states=True).hidden_states
        if len(hidden) != 25:
            raise RuntimeError("CLIP hidden-state count drift")
        def grid(index):
            patches = hidden[index][:, 1:]
            if patches.shape != (1, 576, 1024):
                raise RuntimeError(f"CLIP patch shape drift at hidden[{index}]")
            return patches.transpose(1, 2).reshape(1, 1024, 24, 24).to(torch.bfloat16).float()
        f17 = grid(18)
        source = f17 if arm == "g1" else fusion(grid(12), f17)
        value = adapter(source, return_features=True)
    return value["F_forensic"].detach(), value["logits"].detach()


def evaluate(arm: str, baseline: list[dict]):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    utility, rectifier, adapter, fusion = load_candidate(arm, device)
    sam = load_sam_runtime(hd.CFG, device)
    cfg = yaml.safe_load(C1_CFG_PATH.read_text())
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, _ = load_model(cfg, C1_CKPT, device, expected_step=2500, expected_epoch=5)
    model.eval().requires_grad_(False)
    model.rine_conditioner.eval().requires_grad_(False)
    core = core_model(model)
    vision = model.get_model().get_vision_tower()
    full, indices = matrix.fake_dataset(tokenizer, "official1000")
    expected_ids = [r["sample_id"] for r in baseline]
    if [full.rows[i]["sample_id"] for i in indices] != expected_ids:
        raise RuntimeError("Official1000 manifest/order differs from historical C1 baseline")
    c1_by_id = {r["sample_id"]: r for r in rows(E1_OUT / "c1_records.jsonl")}
    records = OUT / arm / "predictions.jsonl"
    existing = rows(records) if records.exists() else []
    if [r["sample_id"] for r in existing] != expected_ids[:len(existing)]:
        raise RuntimeError("Official1000 resume prefix drift")
    valid = torch.ones(1, 576, dtype=torch.bool, device=device)
    for ordinal, index in enumerate(indices[len(existing):], start=len(existing)):
        sample = full[index]
        sid = sample["sample_id"]
        target = torch.as_tensor(sample["masks"]).bool().any(0)
        old = c1_by_id[sid]
        raw_hidden = torch.load(old["c1_seg_hidden_path"], map_location="cpu", weights_only=True)
        if raw_hidden.shape[0] != int(old["seg_count"]):
            raise RuntimeError(f"C1 [SEG] hidden count drift: {sid}")
        if len(raw_hidden) == 0:
            record = {"sample_id": sid, "foreground_iou": 0., "foreground_f1": 0.,
                      "tp": 0, "fp": 0, "fn": int(target.sum()), "valid_q_seg": False}
        else:
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                q_values = core.model.text_hidden_fcs[0](raw_hidden.to(device=device, dtype=torch.bfloat16))
                raw = model.get_grounding_encoder_embs(sample["grounding_enc_image"][None].to(
                    device=device, dtype=torch.bfloat16))
            f24, zf24 = evidence(arm, vision, sample["global_enc_image"], adapter, fusion, device)
            h, w = target.shape
            sg, cg = geometry_for("sam", (h, w)), geometry_for("clip", (h, w))
            s64 = sam_lowres_to_original_normalized(raw, sg, output_hw=(64, 64)).to(torch.bfloat16)
            sc = sam_coordinates(sg, grid=64)[None].to(device)
            cc = clip_coordinates(cg, grid=24)[None].to(device)
            logits = []
            with torch.no_grad():
                for qone in q_values:
                    qone = qone[None].to(torch.bfloat16)
                    with torch.autocast(device_type=device.type, enabled=False):
                        low0 = sam(qone, raw.to(torch.bfloat16))
                    zl = sam_lowres_to_original_normalized(low0, sg, output_hw=(256, 256)).to(torch.bfloat16)
                    batch = {"S64": s64, "q_seg": qone, "z_L": zl,
                             "F24": f24, "z_F24": zf24, "clip_geometries": [cg],
                             "valid_g0": torch.ones(1, dtype=torch.bool, device=device),
                             "forensic_present": torch.ones(1, dtype=torch.bool, device=device),
                             "forensic_vacuous": torch.zeros(1, dtype=torch.bool, device=device),
                             "forensic_off": torch.zeros(1, dtype=torch.bool, device=device)}
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        rv = rectifier(s64, f24, sc, cc, valid)
                        if arm == "g1":
                            u = hc.utility_forward(utility, batch)
                    if arm == "g7":
                        u = hc.utility_forward(utility, batch)
                    gate = hc.gate_to_sam_grid(u["U"], sc) * rv["support"].reshape(1, 1, 64, 64)
                    embedding = hc.gated_embedding(s64, rv["image_embeddings"], gate)
                    with torch.autocast(device_type=device.type, enabled=False):
                        low = sam(qone, embedding.to(torch.bfloat16))
                    logits.append(inverse_sam_logits(low, sg))
            record = metric_record(sid, torch.stack(logits).amax(0), target)
            record["valid_q_seg"] = True
        with records.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        if (ordinal + 1) % 20 == 0:
            print(json.dumps({"stage": "OFFICIAL1000", "arm": arm,
                              "done": ordinal + 1, "total": 1000}), flush=True)


def finalize(arm: str, candidate: dict, baseline: list[dict]):
    records = OUT / arm / "predictions.jsonl"
    current = rows(records)
    if len(current) != 1000 or [r["sample_id"] for r in current] != [r["sample_id"] for r in baseline]:
        raise RuntimeError("Official1000 final population/order drift")
    metrics, original = summarize(current), summarize(baseline)
    for source, rs in ((metrics, current), (original, baseline)):
        source["seg_trigger_rate"] = sum(bool(r["valid_q_seg"]) for r in rs) / len(rs)
    paired = compare(current, baseline, seed=3407)
    result = {"schema": "phase6g_selected_official1000_result_v1", "status": "COMPLETE",
              "arm": arm, "candidate": candidate, "A1": metrics,
              "historical_6e2": original, "paired_A1_minus_historical": paired,
              "sample_ids_sha256": ids_sha([r["sample_id"] for r in current]),
              "records_sha256": file_sha256(records), "baseline_sha256": file_sha256(BASELINE)}
    dump(OUT / arm / "results.json", result)
    dump(OUT / arm / "status.json", {"status": "COMPLETE", "records": 1000,
                                     "checkpoint_sha256": candidate["checkpoint_sha256"]})
    print(json.dumps({"stage": "COMPLETE", "arm": arm,
                      "mean_foreground_iou": metrics["mean_foreground_iou"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "run", "finalize"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    if args.stage == "preflight":
        freeze_selection()
        for arm in ARMS:
            candidate, baseline = preflight(arm)
            print(json.dumps({"stage": "PREFLIGHT_PASS", "arm": arm,
                              "selected_epoch": candidate["selected_epoch"],
                              "population": len(baseline)}), flush=True)
        return
    if args.arm is None:
        parser.error("--arm required for run/finalize")
    try:
        candidate, baseline = preflight(args.arm)
        if args.stage == "run":
            dump(OUT / args.arm / "status.json", {"status": "RUNNING", "pid": os.getpid(),
                                                   "checkpoint_sha256": candidate["checkpoint_sha256"]})
            evaluate(args.arm, baseline)
        finalize(args.arm, candidate, baseline)
    except Exception as exc:
        dump(OUT / args.arm / "status.json", {"status": "FAILED", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
