#!/usr/bin/env python3
"""Phase 6D.1 frozen authenticity-evidence audit on internal validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPVisionModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.forensics.unified import (  # noqa: E402
    CANONICAL_PROMPT_SHA256,
    CANONICAL_UNIFIED_QUESTION,
    UnifiedForensicsDataset,
)
from eval.forensics_eval import GLaMMForensicsBackend  # noqa: E402
from model.rine_on_c1 import RINEOnHFCLIP  # noqa: E402
from model.llava import conversation as conversation_lib  # noqa: E402
from scripts.phase2a_final_evaluate import load_model  # noqa: E402

OUT = ROOT / "outputs/phase6d1_authenticity_audit"
MANIFEST_ROOT = ROOT / "outputs/data_audits/unified_forensics_split_v1"
UNIFIED_VAL = MANIFEST_ROOT / "val_combined.jsonl"
STAGE2_VAL = ROOT / "outputs/phase5a3_legion_retrained/data/stage2/val.json"
P1 = ROOT / "checkpoints/phase3a_phrase_grounding/p1/best/checkpoint/mp_rank_00_model_states.pt"
RINE = ROOT / "outputs/phase6b6_rine_training/selected_checkpoint.pt"
RINE_PRED = ROOT / "outputs/phase6b6_rine_training/validation_predictions_epoch1.pt"
OLD_B_CACHE = ROOT / "outputs/phase6b2_fusion/features/val/features.pt"
CLIP = ROOT / "checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
EXPECTED_N = 2212
EXPECTED_P1_SHA = "fa4856f8c173c300050c3ae2149354e63a52bbced762d83fe518e9666136b326"
EXPECTED_RINE_SHA = "5286b05c82416e3a11d067b1f449b566c39360ffe7133499d09aecd0fcb3b562"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unified_rows() -> list[dict]:
    rows = jsonl(UNIFIED_VAL)
    if len(rows) != EXPECTED_N or len({row["sample_id"] for row in rows}) != EXPECTED_N:
        raise RuntimeError("Internal-validation manifest count/uniqueness drift")
    if any(int(row["class_label"]) != (row["forensics_domain"] == "fake") for row in rows):
        raise RuntimeError("Internal-validation label/domain drift")
    return rows


def prepare() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite an existing audit: {OUT}")
    rows = unified_rows()
    stage2 = json.loads(STAGE2_VAL.read_text())
    stage2_ids = [row["sample_id"] for row in stage2]
    ids = [row["sample_id"] for row in rows]
    if len(stage2_ids) != EXPECTED_N or set(stage2_ids) != set(ids):
        raise RuntimeError("RINE and LLM validation populations are not the same ID set")
    if sha256(P1) != EXPECTED_P1_SHA or sha256(RINE) != EXPECTED_RINE_SHA:
        raise RuntimeError("Frozen checkpoint provenance drift")
    protocol = {
        "schema": "phase6d1_authenticity_evidence_protocol_v1",
        "status": "FROZEN_BEFORE_EXTRACTION",
        "population": {
            "name": "internal_validation", "n": EXPECTED_N,
            "manifest": str(UNIFIED_VAL.resolve()), "manifest_sha256": sha256(UNIFIED_VAL),
            "sample_order_sha256": canonical_hash(ids),
            "rine_stage2_manifest": str(STAGE2_VAL.resolve()),
            "same_sample_id_set": True,
        },
        "evidence": {
            "A": "RINE-on-C1 raw scalar BCE logit; positive means Fake",
            "B": "P1/R1 shared frozen classification-head margin z_Fake-z_Real at fixed [CLS] query",
            "C": "P1 frozen LM-head next-token margin z_[FAKE]-z_[REAL] at the same fixed [CLS] query",
        },
        "generative_verdict": {
            "prompt": CANONICAL_UNIFIED_QUESTION,
            "prompt_sha256": CANONICAL_PROMPT_SHA256,
            "assistant_input": "fixed [CLS] query only",
            "position": "LM logits at expanded [CLS] position, predicting the immediately following token",
            "true_next_token_logits": True,
            "real_fake_each_single_special_token_required": True,
            "ground_truth_verdict_in_input": False,
            "teacher_forced_authenticity_token": False,
        },
        "score_contract": {"positive": "Fake", "decision": "score > 0", "calibration": False},
        "decision_rules_frozen_before_results": {
            "GEN_TOKEN_DISTINCT_FROM_CLS": (
                "YES iff B/C raw margins are not allclose(atol=1e-6,rtol=1e-6) and "
                "(prediction disagreement > 0 or abs(Pearson correlation) < 0.999999)"
            ),
            "GEN_TOKEN_COMPLEMENTS_RINE": "YES iff at least one A-wrong sample is correct under C",
            "GEN_TOKEN_BETTER_COMPLEMENT_THAN_CLS": (
                "YES iff, within the identical A-wrong subset, C correct count is strictly greater than B correct count"
            ),
            "ALLOW_COLLABORATIVE_DECODING_EXPERIMENT": "YES iff the preceding three gates are all YES",
        },
        "checkpoints": {
            "P1": {"path": str(P1.resolve()), "sha256": sha256(P1), "epoch": 7, "step": 3500},
            "RINE": {"path": str(RINE.resolve()), "sha256": sha256(RINE), "selected_epoch": 1},
        },
        "firewall": {"training": False, "fusion": False, "threshold_tuning": False,
                     "calibration": False, "internal_test": False, "OOD": False},
    }
    atomic_json(OUT / "protocol.json", protocol)
    atomic_json(OUT / "status.json", {"status": "PREPARED"})


def load_p1(device: torch.device):
    cfg = yaml.safe_load((ROOT / "configs/phase3a_p1.yaml").read_text())
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    model, tokenizer, meta = load_model(cfg, P1, device, expected_step=3500, expected_epoch=7)
    model.requires_grad_(False).eval()
    backend = GLaMMForensicsBackend(
        model, tokenizer, device=device, dtype=torch.bfloat16,
        use_mm_start_end=True, max_new_tokens=1,
    )
    dataset = UnifiedForensicsDataset(
        MANIFEST_ROOT, tokenizer, cfg["model"]["vision_tower"], split="val",
        datasets_root=ROOT / cfg["data"]["datasets_root"],
        synthscars_root=ROOT / cfg["data"]["synthscars_root"], image_size=1024,
    )
    return cfg, model, tokenizer, backend, dataset, meta


def extract_llm(device_name: str, batch_size: int) -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    _, model, tokenizer, backend, dataset, meta = load_p1(device)
    if len(dataset) != EXPECTED_N:
        raise RuntimeError("LLM dataset population drift")
    real_ids = tokenizer("[REAL]", add_special_tokens=False).input_ids
    fake_ids = tokenizer("[FAKE]", add_special_tokens=False).input_ids
    if real_ids != [model.real_token_idx] or fake_ids != [model.fake_token_idx]:
        raise RuntimeError(f"Verdict tokenization is not singleton: real={real_ids}, fake={fake_ids}")

    sample_ids, labels, b_logits, c_logits = [], [], [], []
    smoke = None
    atomic_json(OUT / "status_llm.json", {"status": "RUNNING", "pid": os.getpid(), "done": 0})
    with torch.inference_mode():
        for start in range(0, EXPECTED_N, batch_size):
            samples = [dataset[index] for index in range(start, min(start + batch_size, EXPECTED_N))]
            batch = backend._batch_many(samples, "", question=CANONICAL_UNIFIED_QUESTION)
            batch["grounding_enc_images"] = None
            # Labels remain metadata only: inference=True bypasses language targets and cls_labels is
            # explicitly removed, so neither authenticity label can affect this forward.
            batch["labels"] = None
            batch["cls_labels"] = None
            batch["inference"] = True
            output = model.model_forward(**batch)
            if not output["cls_valid_mask"].all() or not output["lm_verdict_valid_mask"].all():
                raise RuntimeError(f"Missing fixed [CLS] query in batch beginning {start}")
            if smoke is None:
                nonpad = batch["attention_masks"].sum(1).long() - 1
                last_ids = batch["input_ids"][torch.arange(len(samples), device=device), nonpad]
                cls_counts = batch["input_ids"].eq(model.cls_token_idx).sum(1)
                if not cls_counts.eq(1).all() or not last_ids.eq(model.cls_token_idx).all():
                    raise RuntimeError("Canonical prompt does not terminate in exactly one fixed [CLS] query")
                smoke = {
                    "status": "PASS", "batch_size": len(samples),
                    "real_token_ids": real_ids, "fake_token_ids": fake_ids,
                    "cls_token_id": int(model.cls_token_idx),
                    "cls_count_per_row": cls_counts.cpu().tolist(),
                    "last_nonpad_is_cls": True,
                    "lm_logit_source": "model output logits at expanded CLS position",
                    "predicts_immediately_following_token": True,
                    "labels_argument": None, "cls_labels_argument": None,
                    "teacher_forced_authenticity_token": False,
                }
                atomic_json(OUT / "next_token_position_audit.json", smoke)
            b_logits.append(output["cls_logits"].float().cpu())
            c_logits.append(output["lm_verdict_logits"].float().cpu())
            sample_ids.extend(sample["sample_id"] for sample in samples)
            labels.extend(int(sample["cls_label"]) for sample in samples)
            done = min(start + batch_size, EXPECTED_N)
            if done % 128 < batch_size or done == EXPECTED_N:
                print(json.dumps({"arm": "LLM_B_C", "done": done, "total": EXPECTED_N}), flush=True)
                atomic_json(OUT / "status_llm.json", {"status": "RUNNING", "pid": os.getpid(), "done": done})

    if sample_ids != [row["sample_id"] for row in dataset.rows]:
        raise RuntimeError("LLM sample order drift")
    payload = {
        "schema": "phase6d1_llm_raw_logits_v1", "sample_ids": sample_ids,
        "labels_fake_positive": torch.tensor(labels, dtype=torch.int64),
        "cls_logits_real_fake": torch.cat(b_logits),
        "gen_token_logits_real_fake": torch.cat(c_logits),
        "checkpoint": meta, "prompt_sha256": CANONICAL_PROMPT_SHA256,
    }
    path = OUT / "llm_raw_logits.pt"
    torch.save(payload, path)
    old = torch.load(OLD_B_CACHE, map_location="cpu", weights_only=False)
    if old["sample_ids"] != sample_ids or not torch.equal(old["labels"], payload["labels_fake_positive"]):
        raise RuntimeError("Historical B cache population drift")
    delta = (old["llm_logits"].float() - payload["cls_logits_real_fake"]).abs()
    atomic_json(OUT / "b_cache_reproduction_audit.json", {
        "status": "PASS" if float(delta.max()) <= 1e-3 else "MISMATCH",
        "historical_cache": str(OLD_B_CACHE.resolve()),
        "max_abs_logit_difference": float(delta.max()), "mean_abs_logit_difference": float(delta.mean()),
        "same_sample_order": True, "same_labels": True,
    })
    atomic_json(OUT / "status_llm.json", {"status": "COMPLETE", "n": EXPECTED_N,
                                           "artifact": str(path.resolve()), "sha256": sha256(path)})


class RineImages(Dataset):
    def __init__(self, rows: list[dict], processor: CLIPImageProcessor):
        self.rows, self.processor = rows, processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = Path(str(row.get("image_path") or ""))
        if not path.is_file():
            root = Path("/data/yz/myLISA_storage/AIGC/SynthScars") if row["forensics_domain"] == "fake" else ROOT / "datasets"
            path = root / row["image_relpath"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return index, pixels


def extract_rine(device_name: str, batch_size: int, workers: int) -> None:
    rows = unified_rows()
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    processor = CLIPImageProcessor.from_pretrained(CLIP, local_files_only=True)
    vision = CLIPVisionModel.from_pretrained(CLIP, local_files_only=True).to(
        device=device, dtype=torch.bfloat16
    ).eval().requires_grad_(False)
    model = RINEOnHFCLIP(vision).to(device).eval()
    checkpoint = torch.load(RINE, map_location="cpu", weights_only=False)
    state = checkpoint.get("rine_state_dict") or checkpoint.get("state_dict")
    if state is None:
        raise RuntimeError("Selected RINE checkpoint has no head state")
    model.rine.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    dataset = RineImages(rows, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    indices, logits = [], []
    atomic_json(OUT / "status_rine.json", {"status": "RUNNING", "pid": os.getpid(), "done": 0})
    with torch.inference_mode():
        for index, pixels in loader:
            pixels = pixels.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            raw, _, _ = model(pixels)
            indices.extend(index.tolist())
            logits.append(raw[:, 0].float().cpu())
            done = len(indices)
            if done % 256 < len(index) or done == EXPECTED_N:
                print(json.dumps({"arm": "RINE_A", "done": done, "total": EXPECTED_N}), flush=True)
                atomic_json(OUT / "status_rine.json", {"status": "RUNNING", "pid": os.getpid(), "done": done})
    if indices != list(range(EXPECTED_N)):
        raise RuntimeError("RINE sample order drift")
    payload = {
        "schema": "phase6d1_rine_raw_logits_v1",
        "sample_ids": [row["sample_id"] for row in rows],
        "labels_fake_positive": torch.tensor([int(row["class_label"]) for row in rows]),
        "raw_binary_logits": torch.cat(logits),
        "checkpoint_sha256": sha256(RINE),
    }
    path = OUT / "rine_raw_logits.pt"
    torch.save(payload, path)
    historical = torch.load(RINE_PRED, map_location="cpu", weights_only=False)
    by_id = {sample_id: float(prob) for sample_id, prob in zip(historical["sample_ids"], historical["prob_fake"])}
    reference = torch.tensor([by_id[sample_id] for sample_id in payload["sample_ids"]])
    reproduced = torch.sigmoid(payload["raw_binary_logits"])
    delta = (reference - reproduced).abs()
    atomic_json(OUT / "rine_reproduction_audit.json", {
        "status": "PASS" if float(delta.max()) <= 1e-3 else "MISMATCH",
        "historical_predictions": str(RINE_PRED.resolve()),
        "same_sample_id_set": set(historical["sample_ids"]) == set(payload["sample_ids"]),
        "max_abs_probability_difference": float(delta.max()),
        "mean_abs_probability_difference": float(delta.mean()),
    })
    atomic_json(OUT / "status_rine.json", {"status": "COMPLETE", "n": EXPECTED_N,
                                            "artifact": str(path.resolve()), "sha256": sha256(path)})


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    pred = scores > 0
    tp = int(((labels == 1) & pred).sum())
    tn = int(((labels == 0) & ~pred).sum())
    fp = int(((labels == 0) & pred).sum())
    fn = int(((labels == 1) & ~pred).sum())
    return {
        "n": int(len(labels)), "accuracy": float((pred == labels).mean()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "fake_recall": tp / (tp + fn), "tnr": tn / (tn + fp), "fpr": fp / (tn + fp),
        "f1": float(f1_score(labels, pred)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": 0.0,
    }


def compare(first: str, second: str, labels: np.ndarray, scores: dict[str, np.ndarray]) -> dict:
    p, q = scores[first] > 0, scores[second] > 0
    first_correct, second_correct = p == labels, q == labels
    out = {}
    for name, take in (("all", np.ones(len(labels), dtype=bool)),
                       ("Real", labels == 0), ("Fake", labels == 1)):
        n = int(take.sum())
        out[name] = {
            "n": n,
            "both_correct": int((take & first_correct & second_correct).sum()),
            "first_only_correct": int((take & first_correct & ~second_correct).sum()),
            "second_only_correct": int((take & ~first_correct & second_correct).sum()),
            "both_wrong": int((take & ~first_correct & ~second_correct).sum()),
            "disagreement_count": int((take & (p != q)).sum()),
            "disagreement_rate": float((take & (p != q)).sum() / n),
        }
    pearson = pearsonr(scores[first], scores[second])
    spearman = spearmanr(scores[first], scores[second])
    out["score_correlation"] = {
        "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
    }
    return out


def finalize() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    a = torch.load(OUT / "rine_raw_logits.pt", map_location="cpu", weights_only=False)
    bc = torch.load(OUT / "llm_raw_logits.pt", map_location="cpu", weights_only=False)
    historical_b = torch.load(OLD_B_CACHE, map_location="cpu", weights_only=False)
    if a["sample_ids"] != bc["sample_ids"] or not torch.equal(
        a["labels_fake_positive"], bc["labels_fake_positive"]
    ):
        raise RuntimeError("A/B/C sample or label alignment failure")
    if historical_b["sample_ids"] != a["sample_ids"] or not torch.equal(
        historical_b["labels"], a["labels_fake_positive"]
    ):
        raise RuntimeError("Frozen historical B cache sample or label alignment failure")
    labels = a["labels_fake_positive"].numpy()
    score = {
        "A_RINE": a["raw_binary_logits"].double().numpy(),
        # B is the already-frozen Phase 6B.2 evidence used by the prior B+RINE
        # study.  The fresh same-forward B is retained only as a reproduction
        # diagnostic because it did not reproduce that cache numerically.
        "B_LLM_CLS": (historical_b["llm_logits"][:, 1] - historical_b["llm_logits"][:, 0]).double().numpy(),
        "C_GEN_TOKEN": (bc["gen_token_logits_real_fake"][:, 1] - bc["gen_token_logits_real_fake"][:, 0]).double().numpy(),
    }
    if any(not np.isfinite(value).all() for value in score.values()):
        raise RuntimeError("Non-finite raw score")
    pairs = {
        "A_vs_B": compare("A_RINE", "B_LLM_CLS", labels, score),
        "A_vs_C": compare("A_RINE", "C_GEN_TOKEN", labels, score),
        "B_vs_C": compare("B_LLM_CLS", "C_GEN_TOKEN", labels, score),
    }
    a_wrong = (score["A_RINE"] > 0) != labels
    b_rescue = int((a_wrong & ((score["B_LLM_CLS"] > 0) == labels)).sum())
    c_rescue = int((a_wrong & ((score["C_GEN_TOKEN"] > 0) == labels)).sum())
    bc_allclose = bool(np.allclose(score["B_LLM_CLS"], score["C_GEN_TOKEN"], atol=1e-6, rtol=1e-6))
    bc_corr = abs(pairs["B_vs_C"]["score_correlation"]["pearson_r"])
    distinct = (not bc_allclose) and (
        pairs["B_vs_C"]["all"]["disagreement_count"] > 0 or bc_corr < 0.999999
    )
    complements = c_rescue > 0
    better = c_rescue > b_rescue
    gates = {
        "GEN_TOKEN_DISTINCT_FROM_CLS": "YES" if distinct else "NO",
        "GEN_TOKEN_COMPLEMENTS_RINE": "YES" if complements else "NO",
        "GEN_TOKEN_BETTER_COMPLEMENT_THAN_CLS": "YES" if better else "NO",
        "ALLOW_COLLABORATIVE_DECODING_EXPERIMENT": "YES" if distinct and complements and better else "NO",
    }
    records = []
    for i, sample_id in enumerate(a["sample_ids"]):
        records.append({
            "sample_id": sample_id, "label": int(labels[i]),
            "A_rine_logit": float(score["A_RINE"][i]),
            "B_cls_logits_real_fake": [float(x) for x in historical_b["llm_logits"][i]],
            "B_cls_margin": float(score["B_LLM_CLS"][i]),
            "C_token_logits_real_fake": [float(x) for x in bc["gen_token_logits_real_fake"][i]],
            "C_token_margin": float(score["C_GEN_TOKEN"][i]),
        })
    records_path = OUT / "matched_raw_logits.jsonl"
    records_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    results = {
        "schema": "phase6d1_authenticity_evidence_results_v1", "status": "COMPLETE",
        "protocol_sha256": sha256(OUT / "protocol.json"),
        "population": {"name": "internal_validation", "n": EXPECTED_N,
                       "real": int((labels == 0).sum()), "fake": int((labels == 1).sum()),
                       "sample_order_sha256": canonical_hash(a["sample_ids"])},
        "metrics": {name: metrics(labels, value) for name, value in score.items()},
        "comparisons": pairs,
        "rine_error_rescue": {
            "rine_wrong_n": int(a_wrong.sum()),
            "B_rescued_n": b_rescue, "B_rescue_rate": b_rescue / int(a_wrong.sum()),
            "C_rescued_n": c_rescue, "C_rescue_rate": c_rescue / int(a_wrong.sum()),
            "C_minus_B_rescued_n": c_rescue - b_rescue,
        },
        "B_C_distinctness": {"raw_margins_allclose": bc_allclose,
                             "max_abs_margin_difference": float(np.max(np.abs(score["B_LLM_CLS"] - score["C_GEN_TOKEN"])))},
        "B_provenance": {
            "primary": str(OLD_B_CACHE.resolve()),
            "reason": "frozen existing B logits used by the prior B+RINE comparison",
            "fresh_same_forward_B_used_as_primary": False,
            "fresh_B_reproduction_audit": str((OUT / "b_cache_reproduction_audit.json").resolve()),
            "reproduction_status": json.loads((OUT / "b_cache_reproduction_audit.json").read_text())["status"],
        },
        "gates": gates,
        "artifacts": {"raw_tensor_A": str((OUT / "rine_raw_logits.pt").resolve()),
                      "raw_tensor_BC": str((OUT / "llm_raw_logits.pt").resolve()),
                      "matched_jsonl": str(records_path.resolve())},
        "firewall": protocol["firewall"],
    }
    atomic_json(OUT / "results.json", results)
    write_report(results, protocol)
    atomic_json(OUT / "status.json", {"status": "COMPLETE", "gates": gates})


def write_report(results: dict, protocol: dict) -> None:
    lines = [
        "# Phase 6D.1 — LLM Authenticity Evidence Audit", "",
        "本审计仅使用 internal validation，A/B/C 全部冻结；未训练、融合、拟合权重、调阈值或校准。", "",
        "## Evidence contract", "",
        "- A: RINE-on-C1 原始 binary logit。",
        "- B: 固定 `[CLS]` hidden 经过现有 classification head 后的 `z_Fake-z_Real`。",
        "- C: canonical prompt 末尾固定 `[CLS]` 位置的 LM-head true next-token logits，取 `z_[FAKE]-z_[REAL]`。",
        "- `[REAL]` 与 `[FAKE]` 均经运行时检查为单一 special token。输入不含 GT verdict，未 teacher-force authenticity token。",
        "- 三者统一以 score > 0 判 Fake；保存原始 logits，未 calibration。", "",
        "B 使用 Phase 6B.2 已冻结、用于此前 B+RINE 比较的原始 logits。本次同前向重放 B 未数值复现历史 cache（详见 `b_cache_reproduction_audit.json`），因此没有用重放值替换历史 B；C 仍来自本次经位置审计的冻结 next-token forward。", "",
        "## Metrics", "",
        "| Evidence | Accuracy | ROC-AUC | Fake recall | TNR | FPR | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in results["metrics"].items():
        lines.append(f"| {name} | {value['accuracy']:.6f} | {value['roc_auc']:.6f} | {value['fake_recall']:.6f} | {value['tnr']:.6f} | {value['fpr']:.6f} | {value['f1']:.6f} |")
    for pair_name, pair in results["comparisons"].items():
        lines += ["", f"## {pair_name}", "",
                  "| Scope | N | Both correct | First only | Second only | Both wrong | Disagreement |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for scope in ("all", "Real", "Fake"):
            value = pair[scope]
            lines.append(f"| {scope} | {value['n']} | {value['both_correct']} | {value['first_only_correct']} | {value['second_only_correct']} | {value['both_wrong']} | {value['disagreement_rate']:.6f} |")
        corr = pair["score_correlation"]
        lines.append(f"\nScore correlation: Pearson `{corr['pearson_r']:.6f}`, Spearman `{corr['spearman_rho']:.6f}`.")
    rescue = results["rine_error_rescue"]
    lines += ["", "## RINE error rescue", "",
              f"RINE errors: `{rescue['rine_wrong_n']}`; B rescues `{rescue['B_rescued_n']}` (`{rescue['B_rescue_rate']:.6f}`); C rescues `{rescue['C_rescued_n']}` (`{rescue['C_rescue_rate']:.6f}`); C-B = `{rescue['C_minus_B_rescued_n']:+d}`.",
              "", "## Gates", ""]
    for key, value in results["gates"].items():
        lines.append(f"`{key} = {value}`")
    lines += ["", "阶段完成并 STOP；没有运行 collaborative decoding。"]
    (ROOT / "docs/phase6d1_llm_authenticity_evidence_audit.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "extract-llm", "extract-rine", "finalize"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    random.seed(3407); np.random.seed(3407); torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    if args.command == "prepare": prepare()
    elif args.command == "extract-llm": extract_llm(args.device, args.batch_size)
    elif args.command == "extract-rine": extract_rine(args.device, args.batch_size, args.workers)
    else: finalize()


if __name__ == "__main__":
    main()
