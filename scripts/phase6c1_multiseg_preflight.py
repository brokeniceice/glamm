#!/usr/bin/env python3
"""Phase 6C.1 TRAIN-only native MultiSEG restoration preflight.

No optimizer, training loop, validation/test/OOD data, or checkpoint write is
permitted here. The only backward calls are deterministic tensor-level parity
checks for the unchanged native GLaMM loss/indexing path.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as glamm_train
from dataset.dataset import custom_collate_fn
from dataset.forensics.unified import TARGET_PROTOCOL_NATIVE_MULTISEG, UnifiedForensicsDataset
from model import pcerf
from model.GLaMM import calculate_dice_loss, compute_sigmoid_cross_entropy, extract_seg_predictor_hidden
from model.llava import conversation as conversation_lib
from scripts import phase4g1q_conditional_utility as r1q
from scripts import phase4hd_rectifier_unfreeze_control as r1hd
from scripts.phase1d_training_policy import configure_args
from tools.phase4e1 import tensor_state_sha256

OUT = ROOT / "outputs/phase6c1_multiseg_preflight"
DOC = ROOT / "docs/phase6c1_multiseg_preflight.md"
CFG = yaml.safe_load((ROOT / "configs/phase3a_p1.yaml").read_text())
P1 = Path("/data/yz/groundingLMM_official/checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt")
R1 = ROOT / "outputs/phase4hd/r1/selected_checkpoint.pt"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def tokenizer():
    args = configure_args(CFG)
    return glamm_train.setup_tokenizer_and_special_tokens(args)


def dataset_checks(tok) -> dict:
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    dataset = UnifiedForensicsDataset(
        ROOT / CFG["data"]["manifest_dir"], tok, CFG["model"]["vision_tower"], split="train",
        datasets_root=ROOT / CFG["data"]["datasets_root"],
        synthscars_root=ROOT / CFG["data"]["synthscars_root"],
        image_size=CFG["model"]["image_size"], target_protocol=TARGET_PROTOCOL_NATIVE_MULTISEG,
    )
    fake_indices = [i for i, row in enumerate(dataset.rows) if row["forensics_domain"] == "fake"]
    chosen = {}
    for wanted in (1, 2, 3):
        chosen[wanted] = next(i for i in fake_indices if len(dataset.rows[i].get("refs") or []) == wanted)
    samples = [dataset[chosen[k]] for k in (1, 2, 3)]
    batch = custom_collate_fn(samples, tokenizer=tok, inference=False, token_strategy="fixed_cls_query")
    seg_id = tok("[SEG]", add_special_tokens=False).input_ids[0]
    observed = [int(row.eq(seg_id).sum()) for row in batch["input_ids"]]
    masks = [int(value.shape[0]) for value in batch["masks_list"]]
    refs = batch["multiseg_ref_indices"]
    if observed != [1, 2, 3] or masks != observed or refs != [[0], [0, 1], [0, 1, 2]]:
        raise RuntimeError(f"TRAIN adapter K/order mismatch: SEG={observed} masks={masks} refs={refs}")
    k1 = samples[0]
    old_union = dataset._fake_union_mask(dataset.rows[chosen[1]], *k1["masks"].shape[-2:])
    if not torch.equal(old_union, k1["masks"]):
        raise RuntimeError("K=1 native mask differs from historical union mask")

    synthetic = {"sample_id": "atomic", "forensics_domain": "fake", "explanation": "evidence"}
    pairs = [{"ref_index": i, "phrase": " ".join([f"phrase{i}"] * 24), "mask": torch.ones(2, 2)} for i in range(5)]
    original_max = tok.model_max_length
    try:
        _, conv2 = dataset._native_multiseg_conversation(synthetic, pairs[:2])
        exact_two_budget = dataset._native_multiseg_token_length(conv2[0])
        tok.model_max_length = exact_two_budget + 575
        selected, dropped, _, atomic_conversations = dataset._truncate_native_pairs(synthetic, pairs, [])
    finally:
        tok.model_max_length = original_max
    if [x["ref_index"] for x in selected] != [0, 1]:
        raise RuntimeError("atomic truncation did not retain a complete ordered prefix")
    if [x["ref_index"] for x in dropped] != [4, 3, 2]:
        raise RuntimeError("atomic truncation accounting/order mismatch")
    if atomic_conversations[0].count("[SEG]") != len(selected):
        raise RuntimeError("atomic truncation SEG mismatch")

    # Exercise the empty-mask policy on TRAIN only and retain exact ref accounting.
    actual_drops = []
    fully_dropped = []
    for index in fake_indices:
        row = dataset.rows[index]
        image = cv2.imread(str(dataset._resolve_image_path(row)), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(row["sample_id"])
        pairs_now, dropped_now = dataset.ordered_phrase_mask_pairs(row, *image.shape[:2])
        actual_drops.extend({"sample_id": row["sample_id"], **item} for item in dropped_now)
        if dropped_now and not pairs_now:
            fully_dropped.append(row["sample_id"])
    reasons = Counter(item["reason"] for item in actual_drops)
    if reasons != {"empty_mask": 1} or fully_dropped:
        raise RuntimeError(f"unexpected TRAIN invalid-pair accounting: {reasons}, full={fully_dropped}")
    write(OUT / "dropped_pairs.json", {
        "scope": "frozen internal TRAIN Fake only", "records": actual_drops,
        "fully_dropped_images": fully_dropped,
    })
    return {
        "status": "PASS", "selected_sample_ids": [sample["sample_id"] for sample in samples],
        "k_values": observed, "mask_counts": masks, "ref_indices": refs,
        "k1_union_mask_bit_exact": True,
        "atomic_truncation": {
            "input_k": 5, "retained_ref_indices": [0, 1], "dropped": dropped,
            "phrase_seg_mask_aligned": True,
        },
        "train_invalid_pair_accounting": {"records": actual_drops, "fully_dropped_images": []},
        "batch_offsets": batch["offset"].tolist(),
    }


def p1_core_checks() -> dict:
    seg, image = 99, -200
    ids = torch.tensor([[1, image, 7, seg, 8, 0, 0], [1, image, 4, seg, 5, seg, 6]])
    hidden = torch.randn(2, ids.shape[1] + 575, 16, generator=torch.Generator().manual_seed(3407), requires_grad=True)
    grouped, positions = extract_seg_predictor_hidden(hidden, ids, seg)
    if [len(x) for x in grouped] != [1, 2]:
        raise RuntimeError("native GLaMM did not extract every SEG predictor state")
    grouped[0].sum().backward(retain_graph=True); gradient = hidden.grad.clone(); hidden.grad.zero_()
    hidden[0, positions[0]].sum().backward()
    if not torch.equal(hidden.grad, gradient):
        raise RuntimeError("K=1 SEG-state backward parity failed")
    pred_old = torch.randn(1, 16, 16, generator=torch.Generator().manual_seed(3408), requires_grad=True)
    pred_new = pred_old.detach().clone().requires_grad_(True)
    target = torch.randint(0, 2, pred_old.shape, generator=torch.Generator().manual_seed(3409)).float()
    def loss(value):
        return 2 * compute_sigmoid_cross_entropy(value, target, 1) + .5 * calculate_dice_loss(value, target, 1)
    old, new = loss(pred_old), loss(pred_new); old.backward(); new.backward()
    if not torch.equal(old, new) or not torch.equal(pred_old.grad, pred_new.grad):
        raise RuntimeError("K=1 native mask loss backward parity failed")
    return {
        "status": "PASS", "seg_counts": [1, 2], "predictor_positions": [x.tolist() for x in positions],
        "k1_seg_state_backward_bit_exact": True, "k1_mask_loss_backward_bit_exact": True,
        "loss_reduction": "GLaMM native global per-mask mean (unchanged)",
    }


def r1_checks(device: torch.device) -> dict:
    before_file = sha256(R1)
    utility, rectifier, _ = r1hd.load_common(device)
    utility.eval().requires_grad_(False); rectifier.eval().requires_grad_(False)
    before_utility = tensor_state_sha256(utility.state_dict())
    before_rectifier = tensor_state_sha256(rectifier.state_dict())
    generator = torch.Generator(device=device).manual_seed(3410)
    b, counts = 3, [1, 2, 3]
    s64 = torch.randn(b, 256, 64, 64, device=device, dtype=torch.bfloat16, generator=generator)
    evidence = torch.randn(b, 256, 24, 24, device=device, dtype=torch.bfloat16, generator=generator)
    sam_coords = pcerf.normalized_cell_centers(64, 64, device=device).reshape(1, -1, 2).expand(b, -1, -1)
    clip_coords = pcerf.normalized_cell_centers(24, 24, device=device).reshape(1, -1, 2).expand(b, -1, -1)
    valid = torch.ones(b, 576, dtype=torch.bool, device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        rectified = rectifier(s64, evidence, sam_coords, clip_coords, valid)["image_embeddings"]
    grouped_q = [torch.randn(k, 256, device=device, dtype=torch.bfloat16, generator=generator) for k in counts]
    grouped_z = [torch.randn(k, 1, 256, 256, device=device, dtype=torch.bfloat16, generator=generator) for k in counts]
    geometry = [{"resized_hw": [336, 336], "crop_box_yxyx": [0, 0, 336, 336]} for _ in range(b)]
    image_batch = {
        "S64": rectified, "F24": evidence,
        "z_F24": torch.randn(b, 1, 24, 24, device=device, dtype=torch.bfloat16, generator=generator),
        "clip_geometries": geometry,
    }
    batch, offsets = pcerf.multiseg_r1_batch(grouped_q, grouped_z, image_batch)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        output = r1q.utility_forward(utility, batch)
    t = sum(counts)
    if output["U"].shape[0] != t or offsets.tolist() != [0, 1, 3, 6]:
        raise RuntimeError("R1 T=sum(K) output/offset mismatch")
    regrouped = pcerf.multiseg_regroup(output["U"], offsets)
    if [x.shape[0] for x in regrouped] != counts:
        raise RuntimeError("R1 regroup mismatch")
    if before_utility != tensor_state_sha256(utility.state_dict()) or before_rectifier != tensor_state_sha256(rectifier.state_dict()):
        raise RuntimeError("R1 weights mutated during preflight")
    if before_file != sha256(R1):
        raise RuntimeError("R1 checkpoint file changed during preflight")
    return {
        "status": "PASS", "K": counts, "T": t, "offsets": offsets.tolist(),
        "regrouped_K": [x.shape[0] for x in regrouped],
        "shared_utility_parameter_count": sum(x.numel() for x in utility.parameters()),
        "shared_rectifier_parameter_count": sum(x.numel() for x in rectifier.parameters()),
        "parameter_copies_created": 0, "utility_state_sha256_before_after_equal": True,
        "rectifier_state_sha256_before_after_equal": True, "checkpoint_sha256_before_after_equal": True,
    }


def render(result: dict) -> None:
    data, p1, r1 = result["data_path"], result["p1_path"], result["r1_path"]
    text = f"""# Phase 6C.1 — Native MultiSEG Restoration Preflight

## Scope

TRAIN-only implementation/preflight. No formal training, optimizer step, validation/test/OOD access, checkpoint selection, classifier change, or loss-weighting change occurred. Phase 6A is the architecture-audit source boundary.

## Restored contract

`refs[i].phrase ↔ refs[i].polygon mask ↔ <p>phrase_i</p>[SEG]` is retained in annotation order. Invalid masks remove only their own pair and record `ref_index + reason`. Token overflow removes complete suffix pairs atomically. GLaMM's native global per-mask mean remains unchanged.

## Gates

| Gate | Result |
|---|---:|
| TRAIN adapter K | {data['k_values']} |
| phrase / SEG / mask counts | PASS |
| ordered ref indices | {data['ref_indices']} |
| K=1 union mask identity | PASS |
| atomic truncation retained refs | {data['atomic_truncation']['retained_ref_indices']} |
| actual empty-mask drops | {len(data['train_invalid_pair_accounting']['records'])} pair, 0 images |
| P1 K=1 SEG-state backward parity | PASS |
| P1 K=1 mask-loss backward parity | PASS |
| R1 K / T / offsets | {r1['K']} / {r1['T']} / {r1['offsets']} |
| R1 parameter copies | 0 |
| R1 weights unchanged | PASS |
| loss | native global per-mask mean |

`P1_MULTI_READY = {result['P1_MULTI_READY']}`

`R1_MULTI_READY = {result['R1_MULTI_READY']}`

## Boundary

This result authorizes no formal training. Per-image-balanced loss was not implemented or tested in this phase.
"""
    DOC.write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status = {"schema": "phase6c1_multiseg_preflight_v1", "status": "RUNNING", "started_at_utc": now()}
    write(OUT / "status.json", status)
    p1_before, r1_before = sha256(P1), sha256(R1)
    try:
        unit = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_phase6c1_multiseg.py",
             "tests/test_phase3a_phrase_grounding.py", "tests/test_unified_forensics_pipeline.py",
             "tests/test_forensics_evaluation_protocol.py"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        tok = tokenizer()
        data = dataset_checks(tok)
        p1 = p1_core_checks()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        r1 = r1_checks(device)
        checkpoint_integrity = {
            "P1": {"path": str(P1), "sha256": p1_before, "unchanged": sha256(P1) == p1_before},
            "R1": {"path": str(R1), "sha256": r1_before, "unchanged": sha256(R1) == r1_before},
        }
        result = {
            "schema": "phase6c1_multiseg_preflight_results_v1", "status": "COMPLETE",
            "scope": "frozen internal TRAIN only", "phase6a_source": "docs/phase6a_architecture_audit.md",
            "unit_tests": {"status": "PASS", "stdout": unit.stdout.strip()},
            "data_path": data, "p1_path": p1, "r1_path": r1,
            "loss": {"reduction": "global_per_mask_mean", "changed": False, "per_image_balanced": False},
            "checkpoint_integrity": checkpoint_integrity,
            "firewall": {"formal_training": False, "optimizer_steps": 0, "validation": False,
                         "internal_test": False, "official1000": False, "OOD": False,
                         "classifier_modified": False, "checkpoint_selected": False},
            "P1_MULTI_READY": "YES", "R1_MULTI_READY": "YES", "completed_at_utc": now(),
        }
        if not all(item["unchanged"] for item in checkpoint_integrity.values()):
            raise RuntimeError("checkpoint integrity failure")
        write(OUT / "results.json", result); render(result)
        status.update(status="COMPLETE", results=str(OUT / "results.json"), report=str(DOC))
    except BaseException as error:
        status.update(status="FAILED", exception_type=type(error).__name__, exception=str(error))
        raise
    finally:
        status["updated_at_utc"] = now(); write(OUT / "status.json", status)


if __name__ == "__main__":
    main()
