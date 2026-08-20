#!/usr/bin/env python3
"""Build deterministic Phase 3C.1 qualitative panels without manual selection."""

from __future__ import annotations

import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase3c1 import SOURCES, inverse_logits

OUT = ROOT / "outputs/phase3c1_spatial_probe"


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def order(sample_id):
    return hashlib.sha256(f"3407:{sample_id}:qualitative".encode()).hexdigest()


def save_mask(value, path):
    tensor = torch.as_tensor(value).bool()
    while tensor.ndim > 2:
        tensor = tensor.any(dim=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor.numpy().astype(np.uint8) * 255).save(path)


def main():
    route = json.loads((OUT / "route_gate.json").read_text())
    phase3c0_rows = rows(ROOT / "outputs/phase3c0_residual_diagnosis/paired/paired_conditions.jsonl")
    phase3c0 = {row["sample_id"]: row for row in phase3c0_rows}
    persistent = set(json.loads((OUT / "audit/persistent476_ids.json").read_text())["sample_ids"])
    metrics, logits, geometries = {}, {}, {}
    for source in SOURCES:
        source_rows = rows(OUT / "validation" / source / "normal_predictions.jsonl")
        metrics[source] = {row["sample_id"]: row for row in source_rows}
        payload = torch.load(OUT / "validation" / source / "selected_low_res_logits.pt", map_location="cpu")
        logits[source] = {sid: value for sid, value in zip(payload["sample_ids"], payload["low_res_logits"])}
        geometries[source] = {row["sample_id"]: row["geometry"] for row in source_rows}
    controls = {
        source: {row["sample_id"]: row for row in rows(OUT / "controls" / source / "spatial_shuffle.jsonl")}
        for source in ("npr", "srm", "focal")
    }
    forensic = max(
        ("npr", "srm", "focal"),
        key=lambda source: json.loads((OUT / "validation" / source / "metrics.json").read_text())[
            "persistent476"
        ]["mean_foreground_iou"],
    )
    ids = [row["sample_id"] for row in phase3c0_rows]
    definitions = {
        "persistent_forensic_success": lambda sid: sid in persistent and metrics[forensic][sid]["foreground_iou"] > .50,
        "persistent_sam_success": lambda sid: sid in persistent and metrics["sam"][sid]["foreground_iou"] > .50,
        "persistent_forensic_and_sam_success": lambda sid: sid in persistent and metrics[forensic][sid]["foreground_iou"] > .50 and metrics["sam"][sid]["foreground_iou"] > .50,
        "persistent_all_probes_fail": lambda sid: sid in persistent and max(metrics[s][sid]["foreground_iou"] for s in SOURCES) <= .30,
        "forensic_normal_success_spatial_shuffle_failure": lambda sid: metrics[forensic][sid]["foreground_iou"] > .50 and controls[forensic][sid]["foreground_iou"] <= .30,
        "negative_G0_good_controls": lambda sid: phase3c0[sid]["A"]["foreground_iou"] > .30,
    }
    definition_text = {
        "persistent_forensic_success": "fixed persistent476 and best forensic probe IoU > 0.50",
        "persistent_sam_success": "fixed persistent476 and SAM probe IoU > 0.50",
        "persistent_forensic_and_sam_success": "fixed persistent476 and both best forensic/SAM IoU > 0.50",
        "persistent_all_probes_fail": "fixed persistent476 and every probe IoU <= 0.30",
        "forensic_normal_success_spatial_shuffle_failure": "best forensic normal IoU > 0.50 and its spatial-shuffle IoU <= 0.30",
        "negative_G0_good_controls": "Phase3C.0 stored canonical A IoU > 0.30",
    }
    selected, used = {}, set()
    for name, predicate in definitions.items():
        eligible = sorted((sid for sid in ids if predicate(sid)), key=order)
        fresh = [sid for sid in eligible if sid not in used][:8]
        if len(fresh) < min(8, len(eligible)):
            fresh += [sid for sid in eligible if sid not in fresh][:8 - len(fresh)]
        selected[name] = fresh; used.update(fresh)
    manifest = {
        "seed": 3407, "forensic_display_source": forensic,
        "selection_rule": "predicate eligibility then SHA256(3407:sample_id:qualitative), no manual choice",
        "category_definitions": definition_text,
        "selections": selected,
    }
    qualitative = OUT / "qualitative"
    qualitative.mkdir(parents=True, exist_ok=True)
    (qualitative / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    blocks = []
    for category, sample_ids in selected.items():
        cards = []
        for sid in sample_ids:
            row = phase3c0[sid]
            stem = hashlib.sha256(sid.encode()).hexdigest()[:24]
            sample_dir = qualitative / "assets" / stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            image = Image.open(row["image_path"]).convert("RGB")
            image.save(sample_dir / "image.jpg", quality=90)
            save_mask(torch.load(row["A"]["gt_mask_path"], map_location="cpu"), sample_dir / "gt.png")
            for label, condition in (("p1_g0", "A"), ("p1_phrase_only", "C"), ("p1_tf_phrase", "D")):
                save_mask(torch.load(row[condition]["binary_mask_path"], map_location="cpu"), sample_dir / f"{label}.png")
            for source in SOURCES:
                original = inverse_logits(logits[source][sid], geometries[source][sid]).gt(0)
                save_mask(original, sample_dir / f"{source}.png")
            columns = [
                ("image", "image.jpg", None), ("GT", "gt.png", None),
                ("P1 G0", "p1_g0.png", row["A"]["foreground_iou"]),
                ("P1 Phrase-Only", "p1_phrase_only.png", row["C"]["foreground_iou"]),
                ("P1 TF-PHRASE", "p1_tf_phrase.png", row["D"]["foreground_iou"]),
            ] + [(source.upper(), f"{source}.png", metrics[source][sid]["foreground_iou"]) for source in SOURCES]
            cells = "".join(
                f'<div><b>{label}</b><img src="assets/{stem}/{filename}"><span>{"" if score is None else f"IoU {score:.4f}"}</span></div>'
                for label, filename, score in columns
            )
            cards.append(f'<article><h3>{html.escape(sid)}</h3><div class="grid">{cells}</div></article>')
        blocks.append(f'<section><h2>{html.escape(category)}</h2>{"".join(cards)}</section>')
    page = """<!doctype html><meta charset="utf-8"><title>Phase 3C.1 qualitative</title>
<style>body{font-family:sans-serif;margin:20px;background:#fafafa}article{background:white;padding:12px;margin:16px 0;border:1px solid #ddd}.grid{display:grid;grid-template-columns:repeat(5,minmax(140px,1fr));gap:8px}.grid div{display:flex;flex-direction:column;gap:4px}.grid img{width:100%;height:180px;object-fit:contain;background:#111}.grid span{font-variant-numeric:tabular-nums;color:#444}h3{font-size:14px}</style>""" + "".join(blocks)
    (qualitative / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "samples": sum(map(len, selected.values())),
                      "forensic": forensic}, ensure_ascii=False))


if __name__ == "__main__":
    main()
