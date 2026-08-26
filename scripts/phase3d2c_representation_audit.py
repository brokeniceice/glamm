#!/usr/bin/env python3
"""Read-only step-0/step-250 representation-change audit for Phase 3D.2-C."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import UnifiedForensicsDataset
from eval.forensics_eval import GLaMMForensicsBackend, UNIFIED_FORENSICS_QUESTION
from eval.inference_trace import unwrap_glamm
from model.GLaMM import extract_seg_predictor_hidden
from model.llava import conversation as conversation_lib
from scripts.phase2a_final_evaluate import file_sha256, load_model
from scripts.phase3d2b_consistency_audit import decode_masks
from tools.phase3d2 import parameter_group

OUT = ROOT / "outputs/phase3d2c_spatial_generalization_attribution"
P3D2 = ROOT / "outputs/phase3d2_direct_spatial_path"
CONFIG = ROOT / "configs/phase3d2_direct_spatial_path.yaml"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compare(left: torch.Tensor, right: torch.Tensor) -> dict:
    a, b = left.float().reshape(-1), right.float().reshape(-1)
    delta = b - a
    return {
        "shape": list(left.shape), "exact_equal": bool(torch.equal(left, right)),
        "max_abs_diff": float(delta.abs().max()), "mean_abs_diff": float(delta.abs().mean()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        "l2_distance": float(delta.norm()), "step0_norm": float(a.norm()),
        "step250_norm": float(b.norm()), "norm_delta": float(b.norm() - a.norm()),
    }


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((ROOT / cfg["source"]["model_config"]).read_text(encoding="utf-8"))
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    p1_path = Path(cfg["source"]["checkpoint"])
    step250_path = Path(cfg["experiment"]["checkpoint_root"]) / "step_0250/checkpoint/mp_rank_00_model_states.pt"
    checkpoint_hashes_before = {"step0": file_sha256(p1_path), "step250": file_sha256(step250_path)}
    model, tokenizer, _ = load_model(
        model_cfg, p1_path, device, expected_step=int(cfg["source"]["optimizer_step"]),
        expected_epoch=int(cfg["source"]["epoch"]),
    )
    model.eval(); model.requires_grad_(False)
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16, use_mm_start_end=True,
        max_new_tokens=int(model_cfg["evaluation"]["max_new_tokens"]),
    )
    dataset = UnifiedForensicsDataset(
        OUT / "populations/seen_train", tokenizer, model_cfg["model"]["vision_tower"], split="test",
        datasets_root=cfg["data"]["datasets_root"], synthscars_root=cfg["data"]["synthscars_root"],
        image_size=int(model_cfg["model"]["image_size"]), target_protocol="phrase_aligned",
    )
    fixed_ids = json.loads((P3D2 / "training/schedule.json").read_text(encoding="utf-8"))["sample_ids"][:8]
    index = {row["sample_id"]: i for i, row in enumerate(dataset.rows)}
    core = unwrap_glamm(model)

    def capture() -> dict:
        result = {}
        with torch.no_grad():
            for sample_id in fixed_ids:
                sample = dataset[index[sample_id]]
                batch = backend._batch(sample, backend.tf_phrase_content(sample), question=UNIFIED_FORENSICS_QUESTION)
                _, hidden_all = core._inference_path(
                    batch["input_ids"], batch["global_enc_images"], batch["attention_masks"],
                    batch["offset"], batch["bboxes"],
                )
                last = core._get_last_hidden_state(hidden_all)
                raw, _ = extract_seg_predictor_hidden(last, batch["input_ids"], core.seg_token_idx)
                projected, _ = core._extract_projected_seg_predictor_hidden(hidden_all, batch["input_ids"], batch["offset"])
                image_embeddings = core.get_grounding_encoder_embs(batch["grounding_enc_images"])
                native, post = decode_masks(core, projected, image_embeddings, batch["resize_list"], batch["label_list"])
                result[sample_id] = {
                    "seg_hidden": raw[0].detach().float().cpu(),
                    "projected_256d": projected[0].detach().float().cpu(),
                    "native_mask_logits": native[0].detach().float().cpu(),
                    "postprocessed_mask_logits": post[0].detach().float().cpu(),
                }
        return result

    step0 = capture()
    source = torch.load(step250_path, map_location="cpu")["module"]
    state = model.state_dict(); copied = {"text_hidden_fcs": 0, "mask_decoder": 0}
    with torch.no_grad():
        for name, tensor in source.items():
            group = parameter_group(name)
            if name in state and group in copied:
                state[name].copy_(tensor.to(device=state[name].device, dtype=state[name].dtype)); copied[group] += 1
    del source
    if any(count == 0 for count in copied.values()):
        raise RuntimeError(f"step250 spatial modules not copied: {copied}")
    step250 = capture()
    records = []
    for sample_id in fixed_ids:
        records.append({
            "sample_id": sample_id,
            **{key: compare(step0[sample_id][key], step250[sample_id][key]) for key in step0[sample_id]},
        })
    if any(not row["seg_hidden"]["exact_equal"] for row in records):
        raise RuntimeError("frozen [SEG] hidden changed between step0 and step250")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("gradient materialized during representation audit")
    checkpoint_hashes_after = {"step0": file_sha256(p1_path), "step250": file_sha256(step250_path)}
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise RuntimeError("checkpoint file changed during representation audit")
    result = {
        "status": "PASS", "sample_count": len(records), "sample_ids": fixed_ids,
        "copied_step250_tensor_counts": copied, "records": records,
        "summary": {
            key: {
                "exact_equal_count": sum(row[key]["exact_equal"] for row in records),
                "mean_cosine_similarity": sum(row[key]["cosine_similarity"] for row in records) / len(records),
                "mean_l2_distance": sum(row[key]["l2_distance"] for row in records) / len(records),
                "max_of_max_abs_diff": max(row[key]["max_abs_diff"] for row in records),
            } for key in ("seg_hidden", "projected_256d", "native_mask_logits", "postprocessed_mask_logits")
        },
        "runtime": {"model_eval": not model.training, "torch_no_grad": True, "gradients_absent": True},
        "checkpoint_files_unchanged": True, "new_probe_trained": False,
    }
    dump(OUT / "representations/representation_change_audit.json", result)
    dump(OUT / "representation_change_audit.json", result)
    print(json.dumps({"status": "PASS", "sample_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
