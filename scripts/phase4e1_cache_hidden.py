#!/usr/bin/env python3
"""Build frozen raw-4096 TF/G0 hidden caches for Phase 4E-1."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CFG = yaml.safe_load((ROOT / "configs/phase4e1_tf_fdg_full_method.yaml").read_text())
OUT = ROOT / CFG["experiment"]["output_root"]
CACHE = Path("/data/yz/groundingLMM_official/cache/phase4e1_tf_fdg_raw_hidden")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def id_manifest(name: str) -> list[str]:
    return json.loads((OUT / "manifests" / f"{name}.json").read_text())["sample_ids"]


def save_shard(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if json.loads((OUT / "preflight/approved_protocol_amendment.json").read_text())["status"] != "PASS":
        raise RuntimeError("approved amendment manifest PASS required")

    # Reuse the already-audited canonical P1 loader/backend/dataset contract.
    from scripts.phase4c_b_cache import setup
    from scripts.phase3f_train import teacher_batch
    from tools.phase3f_aogd import capture_representations, core_model, replay_batch
    from eval.forensics_eval import UNIFIED_FORENSICS_QUESTION
    from tools.phase4c_b import rows

    bcfg = yaml.safe_load((ROOT / "configs/phase4c_b_evidence_reader.yaml").read_text())
    device = torch.device(args.device)
    model, backend, dataset, _ = setup(bcfg, args.split, device)
    model.eval(); model.requires_grad_(False); core = core_model(model)
    fake = [(index, row) for index, row in enumerate(dataset.rows) if int(row["class_label"]) == 1]
    expected = 8836 if args.split == "train" else 1106
    if len(fake) != expected:
        raise RuntimeError(f"{args.split} Fake count mismatch")

    if args.split == "train":
        valid_ids = id_manifest("train_valid_g0_ids")
        invalid_ids = id_manifest("train_invalid_g0_ids")
        replay = {row["sample_id"]: row for row in rows(ROOT / CFG["data"]["replay_cache"])}
        teacher_source = torch.load(
            "/data/yz/groundingLMM_official/checkpoints/phase3f_autonomous_oracle_grounding_distillation/teacher_cache/projected_256d.pt",
            map_location="cpu", weights_only=False,
        )["vectors"]
        auto_source = replay
    else:
        valid_ids = id_manifest("val_valid_g0_ids")
        invalid_ids = id_manifest("val_invalid_g0_ids")
        teacher_source = torch.load(
            "/data/yz/groundingLMM_official/checkpoints/phase3f_autonomous_oracle_grounding_distillation/validation_teacher_cache/p1_oracle_representations.pt",
            map_location="cpu", weights_only=False,
        )["vectors"]
        prediction_path = ROOT / "outputs/phase3f_autonomous_oracle_grounding_distillation/evaluation/selector/P1_FROZEN/G0/predictions.jsonl"
        auto_source = {row["sample_id"]: row for row in rows(prediction_path)}
        phrase_source = {}
        phrase_root = ROOT / CFG["data"]["validation_context_cache"] / "phrase_only"
        for path in sorted(phrase_root.glob("shard_*.pt")):
            for row in torch.load(path, map_location="cpu", weights_only=False)["records"]:
                phrase_source[str(row["sample_id"])] = row
        if set(phrase_source) != {str(row["sample_id"]) for _, row in fake}:
            raise RuntimeError("phrase-only context population mismatch")

    ordered_ids = [str(row["sample_id"]) for _, row in fake]
    if set(ordered_ids) != set(valid_ids) | set(invalid_ids):
        raise RuntimeError("hidden cache population differs from frozen ID manifests")
    destination = CACHE / args.split
    complete = destination / "complete.json"
    if complete.exists() and json.loads(complete.read_text()).get("status") == "COMPLETE":
        print(complete.read_text()); return

    started = time.time(); shard_size = 64
    index_by_id = {str(row["sample_id"]): index for index, row in fake}
    valid_set = set(valid_ids)
    files = []
    for start in range(0, len(ordered_ids), shard_size):
        ids = ordered_ids[start:start + shard_size]
        path = destination / f"shard_{start:06d}_{start + len(ids):06d}.pt"
        if path.exists():
            files.append(path); continue
        tf_values, g0_values, g0_positions, phrase_values = [], [], [], []
        with torch.no_grad():
            for local, sid in enumerate(ids):
                sample = dataset[index_by_id[sid]]
                if sid in teacher_source:
                    tf_hidden = teacher_source[sid]["raw_4096d"].reshape(-1)
                else:
                    tf_hidden, _, _, _ = capture_representations(core, teacher_batch(backend, sample))
                tf_values.append(tf_hidden.detach().to(torch.bfloat16).cpu())
                if sid in valid_set:
                    source = auto_source[sid]
                    replay_row = {"full_input_token_ids": source["full_input_token_ids"]} if args.split == "train" else {
                        "full_input_token_ids": source["prompt_token_ids"] + source["generated_token_ids"]
                    }
                    g0_hidden, _, _, _ = capture_representations(
                        core, replay_batch(backend, sample, replay_row, UNIFIED_FORENSICS_QUESTION)
                    )
                    g0_values.append(g0_hidden.detach().to(torch.bfloat16).cpu())
                    g0_positions.append(local)
                if args.split == "val":
                    phrase_hidden, _, _, _ = capture_representations(
                        core, replay_batch(
                            backend, sample,
                            {"full_input_token_ids": phrase_source[sid]["generated_token_ids"]},
                            UNIFIED_FORENSICS_QUESTION,
                        ),
                    )
                    phrase_values.append(phrase_hidden.detach().to(torch.bfloat16).cpu())
        payload = {
            "schema": "phase4e1_raw_hidden_v1", "split": args.split,
            "sample_ids": ids, "tf_hidden": torch.stack(tf_values),
            "g0_positions": torch.tensor(g0_positions, dtype=torch.int16),
            "g0_hidden": torch.stack(g0_values) if g0_values else torch.empty(0, 4096, dtype=torch.bfloat16),
            "phrase_hidden": torch.stack(phrase_values) if phrase_values else None,
            "hidden_extraction_rule": CFG["p1"]["hidden_extraction"],
            "p1_checkpoint_sha256": CFG["p1"]["checkpoint_sha256"],
        }
        save_shard(path, payload); files.append(path)
        done = start + len(ids)
        print(json.dumps({"split": args.split, "done": done, "total": len(ordered_ids),
                          "seconds_per_sample": (time.time() - started) / done}), flush=True)
    valid_seen = 0; ids_seen = []
    file_rows = []
    for path in files:
        value = torch.load(path, map_location="cpu", weights_only=False)
        ids_seen.extend(value["sample_ids"]); valid_seen += int(value["g0_hidden"].shape[0])
        file_rows.append({"path": str(path), "sha256": file_hash(path)})
    if ids_seen != ordered_ids or valid_seen != len(valid_ids):
        raise RuntimeError("completed hidden cache count/order mismatch")
    manifest = {
        "status": "COMPLETE", "split": args.split, "total": len(ordered_ids),
        "valid_g0": valid_seen, "invalid_g0": len(invalid_ids), "tf_count": len(ordered_ids),
        "phrase_only_count": len(ordered_ids) if args.split == "val" else 0,
        "dtype": "torch.bfloat16", "dimension": 4096, "files": file_rows,
        "ordered_sample_ids_sha256": hashlib.sha256("".join(f"{sid}\n" for sid in ordered_ids).encode()).hexdigest(),
        "hidden_extraction_rule": CFG["p1"]["hidden_extraction"],
        "canonical_prompt_sha256": CFG["p1"]["prompt_sha256"],
        "p1_checkpoint_sha256": CFG["p1"]["checkpoint_sha256"],
        "surrogate_hidden_count": 0, "elapsed_seconds": time.time() - started,
        "internal_test_accessed": False, "official1000_accessed": False,
    }
    dump(complete, manifest)
    dump(OUT / "cache" / f"{args.split}_raw_hidden_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
