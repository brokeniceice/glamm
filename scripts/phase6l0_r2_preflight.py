#!/usr/bin/env python3
"""C2-native R2 source, capture, runtime, and step-zero identity audit.

This script does not train or touch sealed evaluation populations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.forensics_eval import GLaMMForensicsBackend
from model.c2_r2_localizer import C2R2Localizer
from model.llava import conversation as conversation_lib
from model.sam_forensic_rectifier import FrozenP1SAMPath
from scripts.phase3c2_p3 import dataset_for
from scripts.phase3c1_cache import clip_grid, module_hash
from scripts.phase6j0_c2_evaluate import load_c2_model
from tools.phase3c1 import geometry_for
from tools.phase4c_b import file_sha256
from tools.phase4e1 import tensor_state_sha256
from tools.phase3f_aogd import core_model

CKPT = ROOT / "checkpoints/phase6j0_c2/best/checkpoint/mp_rank_00_model_states.pt"
PROTOCOL = ROOT / "outputs/phase6j0_c2/final_evaluation/protocol.json"
P1_RUNTIME = Path("/data/yz/groundingLMM_official/cache/phase4f_language_preserving_rectification/p1_sam_runtime.pt")
OUT = ROOT / "outputs/phase6l0_r2/cache"
EXPECTED_SHA = "4a67e6a87c453d554fa5bd6cf93329ae54c1c2853f0925dc7a63397eef27e8ce"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def c2_native_sam_state() -> tuple[dict, dict]:
    """Use base-frozen prompt state plus the trained C2 decoder, never P1 decoder."""
    require(file_sha256(CKPT) == EXPECTED_SHA, "C2 checkpoint SHA mismatch")
    protocol = json.loads(PROTOCOL.read_text())
    require(protocol["status"] == "FROZEN" and protocol["selected_checkpoint_sha256"] == EXPECTED_SHA and
            (protocol["selected_epoch"], protocol["selected_step"]) == (7, 3500), "C2 protocol mismatch")
    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)["module"]
    base = torch.load(P1_RUNTIME, map_location="cpu", weights_only=False)
    prefix = "base_model.model.model.grounding_encoder."
    trained_decoder = {key[len(prefix + "mask_decoder."):]: value for key, value in checkpoint.items()
                       if key.startswith(prefix + "mask_decoder.")}
    prompt_key = prefix + "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"
    require(prompt_key in checkpoint and len(trained_decoder) == len(base["mask_decoder"]),
            "C2 SAM checkpoint coverage mismatch")
    require(torch.equal(checkpoint[prompt_key], base["prompt_encoder"]["pe_layer.positional_encoding_gaussian_matrix"]),
            "C2/P1 base-frozen prompt position matrix mismatch")
    require(set(trained_decoder) == set(base["mask_decoder"]), "C2 decoder key mismatch")
    unequal = sum(not torch.equal(trained_decoder[k], base["mask_decoder"][k]) for k in trained_decoder)
    require(unequal > 0, "unexpected P1/C2 decoder identity; inspect provenance")
    return {"prompt_encoder": base["prompt_encoder"], "mask_decoder": trained_decoder}, {
        "c2_sha256": EXPECTED_SHA, "p1_prompt_source_sha256": file_sha256(P1_RUNTIME),
        "p1_c2_decoder_unequal_keys": unequal, "decoder_keys": len(trained_decoder)}


def load_native_runtime(state: dict, device: torch.device) -> FrozenP1SAMPath:
    sam = FrozenP1SAMPath()
    sam.prompt_encoder.load_state_dict(state["prompt_encoder"], strict=True)
    sam.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
    return sam.to(device=device, dtype=torch.bfloat16).freeze()


def main(device: torch.device) -> None:
    torch.cuda.set_device(device)
    state, provenance = c2_native_sam_state()
    runtime = load_native_runtime(state, device)
    sam_before = tensor_state_sha256(runtime.state_dict())
    cfg = yaml.safe_load((ROOT / "configs/phase6j0_c2_preln_cross_attention.yaml").read_text())
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, meta = load_c2_model(cfg, CKPT, device, expected_step=3500, expected_epoch=7)
    model.eval().requires_grad_(False)
    core = core_model(model)
    c2_sam = core.model.grounding_encoder
    sam_complete = json.loads((ROOT / "outputs/phase3c1_spatial_probe/cache/sam/train/complete.json").read_text())
    clip_complete = json.loads((ROOT / "outputs/phase3c1_spatial_probe/cache/clip/train/complete.json").read_text())
    live_sam_source_hash = module_hash(c2_sam.image_encoder)
    live_clip_source_hash = module_hash(model.get_model().get_vision_tower())
    require(live_sam_source_hash == sam_complete["source_parameter_hash_before"] and
            live_clip_source_hash == clip_complete["source_parameter_hash_before"],
            "C2/P1 frozen spatial source parameter hash mismatch")
    backend = GLaMMForensicsBackend(model, tokenizer, device=device, dtype=torch.bfloat16,
                                    use_mm_start_end=True, max_new_tokens=int(cfg["evaluation"]["max_new_tokens"]))
    dataset = dataset_for(tokenizer, cfg, "train")
    first_fake = next(i for i, row in enumerate(dataset.rows) if int(row["class_label"]) == 1)
    sample = dataset[first_fake]
    sid = str(sample["sample_id"])
    c2_cache = torch.load("/data/yz/groundingLMM_official/cache/phase6k0_c2_after_v2/train/shard_000000.pt",
                          map_location="cpu", weights_only=False)
    spatial_sam = torch.load(ROOT / "outputs/phase3c1_spatial_probe/cache/sam/train/shard_000000_000032.pt",
                             map_location="cpu", weights_only=False)
    spatial_clip = torch.load(ROOT / "outputs/phase3c1_spatial_probe/cache/clip/train/shard_000000_000032.pt",
                              map_location="cpu", weights_only=False)
    require(sid == c2_cache["sample_ids"][0] == spatial_sam["records"][0]["sample_id"] ==
            spatial_clip["records"][0]["sample_id"], "first canonical sample ID mismatch")
    require(bool(c2_cache["valid"][0]), "first canonical C2 sample is invalid")

    with torch.no_grad():
        live_s64 = model.get_grounding_encoder_embs(sample["grounding_enc_image"][None].to(device=device, dtype=torch.bfloat16))
        tokens, _ = model.get_model().get_vision_tower()(sample["global_enc_image"][None].to(device=device, dtype=torch.bfloat16))
        live_clip = clip_grid(tokens)
    s64_error = float((live_s64.float().cpu() - spatial_sam["features"][0].float()).abs().max())
    clip_error = float((live_clip.float().cpu() - spatial_clip["features"][0].float()).abs().max())
    require(s64_error == 0 and clip_error == 0, f"C2 versus P1 spatial source drift: SAM={s64_error}, CLIP={clip_error}")

    captured = {"fused": [], "low": []}
    logits = []
    original_cls = core._extract_cls_logits
    def capture_cls(*args, **kwargs):
        value = original_cls(*args, **kwargs)
        logits.append(value[0].detach().cpu().clone())
        return value
    core._extract_cls_logits = capture_cls
    handles = [model.c2_cross_attention.register_forward_hook(
        lambda _m, _i, output: captured["fused"].append(output.detach().cpu().clone())),
        c2_sam.mask_decoder.register_forward_hook(
            lambda _m, _inputs, output: captured["low"].append(output[0].detach().cpu().clone()))]
    attn = model.c2_cross_attention
    try:
        attn.capture_spatial_intermediates = False
        old = backend.generate_localization_batch([sample], provide_gt_fake=False,
                                                   generation_mode="unified_fake_generate")[0]
        old_fused = captured["fused"][-1]
        old_low = captured["low"][-1]
        require(len(logits) == 1, "C2 classification logit capture missing")
        old_logits = logits[-1]
        captured = {"fused": [], "low": []}
        logits = []
        attn.capture_spatial_intermediates = True
        new = backend.generate_localization_batch([sample], provide_gt_fake=False,
                                                   generation_mode="unified_fake_generate")[0]
        cap = attn.last_spatial_intermediates
        require(len(logits) == 1 and torch.equal(old_logits, logits[-1]),
                "C2 raw classification logits changed by capture")
        require(old["generated_token_ids"] == new["generated_token_ids"] and
                torch.equal(old["projected_seg_embeddings"], new["projected_seg_embeddings"]) and
                torch.equal(old["pred_mask"], new["pred_mask"]) and
                torch.equal(old_fused, captured["fused"][-1]) and torch.equal(old_low, captured["low"][-1]),
                "capture changed C2 output")
        # Phase6K generated this shard in a different batch. It is useful as
        # provenance, but R2 must capture A/E/q/r_prime in *one* fresh forward.
        prior_differences = {
            "A_max_abs": float((cap["A"][0] - c2_cache["A"][0]).abs().max()),
            "E_max_abs": float((cap["E"][0].float() - c2_cache["E"][0].float()).abs().max()),
            "q_seg_max_abs": float((new["projected_seg_embeddings"][0].float().cpu() - c2_cache["q_seg"][0].float()).abs().max()),
        }
        a_error = float((cap["A"].flatten(2).sum(-1) - 1).abs().max())
        e_context = cap["E"].float().reshape(1, 8, 64, 576).sum(-1)
        context_error = float((e_context - cap["head_context"].float()).abs().max())
        context_relative = context_error / max(float(cap["head_context"].float().abs().max()), 1e-12)
        require(a_error < 1e-5 and context_relative <= .01, "C2 A/E context reconstruction failed")
        q = new["projected_seg_embeddings"].to(device=device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            native_low = runtime(q, live_s64)
        require(torch.equal(native_low.cpu(), captured["low"][-1]), "C2-native lightweight SAM output differs from direct C2")

        clip_geometry = spatial_clip["records"][0]["geometry"]
        require(clip_geometry == geometry_for("clip", clip_geometry["original_hw"]), "CLIP geometry mismatch")
        torch.manual_seed(3407)
        r2 = C2R2Localizer().to(device).eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = r2(raw_clip_grid=live_clip, attention_map=cap["A"].to(device),
                        evidence_map=cap["E"].to(device), r_prime=captured["fused"][-1].to(device),
                        q_seg=q, s64=live_s64, z_l=native_low, clip_geometry=[clip_geometry])
        require(torch.count_nonzero(output["deltaS"]) == 0 and torch.equal(output["S_adapt"], live_s64),
                "R2 step-zero S64 identity failed")
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            adapted_low = runtime(q, output["S_adapt"].to(torch.bfloat16))
        require(torch.equal(adapted_low, native_low), "R2 step-zero C2-G0 SAM logits differ")
        require(sam_before == tensor_state_sha256(runtime.state_dict()), "frozen SAM state changed")
        c2_hash = tensor_state_sha256(model.c2_cross_attention.state_dict())
        output_path = OUT / "c2_sam_runtime.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp = output_path.with_suffix(".pt.tmp")
        torch.save({"schema": "phase6l0_c2_sam_runtime_v1", "state": state,
                    "c2_sha256": EXPECTED_SHA, "sam_state_sha256": sam_before}, temp)
        os.replace(temp, output_path)
        dump(OUT / "preflight_subset.json", {
            "status": "PASS", "sample_id": sid, "c2_sha256": EXPECTED_SHA,
            "runtime_sha256": file_sha256(output_path), "sam_state_sha256": sam_before,
            "c2_attention_state_sha256": c2_hash, "c2_checkpoint_meta": meta,
            "p1_c2_decoder_unequal_keys": provenance["p1_c2_decoder_unequal_keys"],
            "C2_P1_S64_max_abs": s64_error, "C2_P1_raw_CLIP_max_abs": clip_error,
            "C2_SAM_image_encoder_hash": live_sam_source_hash,
            "C2_CLIP_vision_tower_hash": live_clip_source_hash,
            "A_max_mass_error": a_error, "E_context_max_abs_error": context_error,
            "E_context_relative_error": context_relative, "capture_exact": True,
            "classification_logits_exact": True,
            "prior_Phase6K_batch_cache_max_abs": prior_differences,
            "C2_native_SAM_low_exact": True, "R2_step0_SAM_exact": True,
            "q_seg_shape": list(q.shape), "r_prime_shape": list(captured["fused"][-1].shape),
            "S64_shape": list(live_s64.shape), "z_L_shape": list(native_low.shape),
            "support_fraction": float(output["support64"].float().mean())})
        print(json.dumps({"status": "PASS", "sample_id": sid, "output": str(OUT / "preflight_subset.json")}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
        attn.capture_spatial_intermediates = False
        core._extract_cls_logits = original_cls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(torch.device(args.device))
