#!/usr/bin/env python3
"""Frozen I-JEPA predictive-discrepancy diagnostic on internal validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.phase6i_ijepa_cache import REPO, WEIGHT, image_tensor
from tools.phase4c_b import file_sha256
from tools.phase3c1 import geometry_for

sys.path.insert(0, str(REPO))
from src.models.vision_transformer import vit_huge, vit_predictor

OUT = ROOT / "outputs/phase6i_jepa_discrepancy"
CLIP = ROOT / "outputs/phase3c1_spatial_probe/cache/clip/val"
VAL = ROOT / "outputs/data_audits/unified_forensics_split_v1/val_combined.jsonl"
PRIOR = ROOT / "outputs/phase6i_ijepa_r1/A0/internal_dev_records.jsonl"
GRID = 28
N_PATCH = GRID * GRID
SEED = 3407
MIN_AREA = math.ceil(.15 * N_PATCH)
MAX_AREA = math.floor(.20 * N_PATCH)
MAX_BLOCKS = 4
SAFETY = 2


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def hash_json(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def hash_lines(items) -> str:
    return hashlib.sha256("\n".join(map(str, items)).encode()).hexdigest()


def bbox_indices(box):
    y0, x0, h, w = box
    return [y * GRID + x for y in range(y0, y0 + h) for x in range(x0, x0 + w)]


def descriptor(box):
    y0, x0, h, w = box
    cy = (y0 + h / 2) / GRID - .5
    cx = (x0 + w / 2) / GRID - .5
    return {"area_fraction": h * w / N_PATCH, "aspect_ratio": w / h,
            "center_distance": math.hypot(cx, cy),
            "center_distance_bucket": min(3, int(math.hypot(cx, cy) * 8))}


def all_boxes(h, w):
    for y in range(GRID - h + 1):
        for x in range(GRID - w + 1):
            yield (y, x, h, w)


def legal_shapes():
    return [(h, w) for h in range(1, GRID + 1) for w in range(1, GRID + 1)
            if MIN_AREA <= h * w <= MAX_AREA and .75 <= w / h <= 1.5]


SHAPES = legal_shapes()


def dilate(grid, radius=SAFETY):
    a = np.pad(grid.astype(bool), radius)
    out = np.zeros_like(grid, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= a[dy:dy + GRID, dx:dx + GRID]
    return out


def components(grid):
    seen = np.zeros_like(grid, dtype=bool)
    result = []
    for y, x in zip(*np.where(grid)):
        if seen[y, x]:
            continue
        todo = [(int(y), int(x))]
        seen[y, x] = True
        comp = []
        while todo:
            yy, xx = todo.pop()
            comp.append((yy, xx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    y2, x2 = yy + dy, xx + dx
                    if 0 <= y2 < GRID and 0 <= x2 < GRID and grid[y2, x2] and not seen[y2, x2]:
                        seen[y2, x2] = True
                        todo.append((y2, x2))
        result.append(comp)
    return sorted(result, key=lambda c: (-len(c), min(c)))


def mask_grid(mask: torch.Tensor, geo: dict):
    rh, rw = map(int, geo["resized_hw"])
    top, left, bottom, right = map(int, geo["crop_box_yxyx"])
    assert tuple(mask.shape) == tuple(geo["original_hw"])
    pil = Image.fromarray(mask.to(torch.uint8).numpy() * 255, "L")
    pil = pil.resize((rw, rh), Image.Resampling.NEAREST)
    full_gt = int((np.asarray(pil)>0).sum())
    cropped = pil.crop((left, top, right, bottom))
    crop_gt = int((np.asarray(cropped)>0).sum())
    pil = cropped.resize((448, 448), Image.Resampling.NEAREST)
    pixels = np.asarray(pil, dtype=np.uint8).reshape(GRID, 16, GRID, 16)
    counts = (pixels > 0).sum(axis=(1, 3))
    return counts > 0, counts, crop_gt / full_gt if full_gt else 0.0


def artifact_boxes(pos, counts):
    total = int(counts.sum())
    if total == 0:
        return [], "no_gt_in_crop", 0
    comps = components(pos)
    covered = np.zeros_like(pos, dtype=bool)
    selected = []
    for comp in comps:
        if len(selected) >= MAX_BLOCKS or int((covered * counts).sum()) / total >= .5:
            break
        ys, xs = zip(*comp)
        ymin, ymax, xmin, xmax = min(ys), max(ys), min(xs), max(xs)
        candidates = []
        for h, w in SHAPES:
            if ymax - ymin + 1 > h or xmax - xmin + 1 > w:
                continue
            for y in range(max(0, ymax - h + 1), min(ymin, GRID - h) + 1):
                for x in range(max(0, xmax - w + 1), min(xmin, GRID - w) + 1):
                    box = (y, x, h, w)
                    if any(y < oy+oh and oy < y+h and x < ox+ow and ox < x+w
                           for oy,ox,oh,ow in selected):
                        continue
                    gained = int((counts[y:y+h, x:x+w] * ~covered[y:y+h, x:x+w]).sum())
                    candidates.append((-gained, h * w, abs((y + h/2) - (ymin + ymax + 1)/2)
                                       + abs((x + w/2) - (xmin + xmax + 1)/2), y, x, box))
        if not candidates:
            continue
        box = min(candidates)[-1]
        selected.append(box)
        y, x, h, w = box
        covered[y:y+h, x:x+w] = True
    recall = int((covered * counts).sum()) / total
    if recall < .5:
        return selected, "gt_coverage_below_0.5", recall
    return selected, None, recall


def normal_box(shape, forbidden, anchor):
    h, w = shape
    target = descriptor(anchor)["center_distance"]
    choices = []
    for box in all_boxes(h, w):
        y, x, _, _ = box
        if forbidden[y:y+h, x:x+w].any():
            continue
        d = descriptor(box)["center_distance"]
        ay, ax, _, _ = anchor
        choices.append((abs(d - target), abs(y-ay) + abs(x-ax), y, x, box))
    return min(choices)[-1] if choices else None


def fake_regions(pos, counts, crop_gt_fraction):
    aa, reason, recall = artifact_boxes(pos, counts)
    if reason:
        return [], [], reason, recall
    if recall * crop_gt_fraction < .5:
        return aa, [], "full_image_gt_coverage_below_0.5", recall
    forbidden = dilate(pos)
    bb = []
    for box in aa:
        control = normal_box(box[2:], forbidden, box)
        if control is None:
            return aa, bb, "no_safe_matched_normal", recall
        bb.append(control)
    return aa, bb, None, recall


def region_record(group, sample_id, box, ordinal, counts=None, total_gt=0):
    y, x, h, w = box
    row = {"group": group, "sample_id": sample_id, "block_ordinal": ordinal,
           "target_bbox_patch": [y, x, h, w], "target_patch_count": h*w, **descriptor(box)}
    if counts is not None:
        contained = int(counts[y:y+h, x:x+w].sum())
        row["gt_coverage_inside_target"] = contained / total_gt if total_gt else None
        row["gt_positive_patch_fraction_inside_target"] = float((counts[y:y+h, x:x+w]>0).mean())
        row["gt_positive_patch_count_inside_target"] = int((counts[y:y+h, x:x+w]>0).sum())
    return row


def prepare():
    if (OUT / "protocol_frozen.json").exists():
        raise RuntimeError("protocol already frozen; refuse silent regeneration")
    if not WEIGHT.is_file() or not REPO.is_dir():
        raise RuntimeError("official I-JEPA checkpoint/source missing")
    manifest = list(jsonl(VAL))
    fake = [r for r in manifest if int(r["class_label"]) == 1]
    real = [r for r in manifest if int(r["class_label"]) == 0]
    if len(fake) != 1106 or len(real) != 1106:
        raise RuntimeError("internal validation population drift")
    fake_ids = [str(r["sample_id"]) for r in fake]
    prior_ids = [str(r["sample_id"]) for r in jsonl(PRIOR)]
    if fake_ids != prior_ids:
        raise RuntimeError("canonical Phase6I internal DEV ID/order drift")
    source_rows, masks, shards = [], [], []
    for path in sorted(CLIP.glob("shard_*.pt")):
        d = torch.load(path, map_location="cpu", weights_only=False)
        shards.append({"path": str(path.resolve()), "n": len(d["records"]), "sha256": file_sha256(path)})
        source_rows.extend(d["records"])
        masks.extend(d["original_masks"])
    if [str(r["sample_id"]) for r in source_rows] != fake_ids:
        raise RuntimeError("CLIP spatial source ID/order drift")
    fake_regions_rows, real_regions_rows, sample_rows = [], [], []
    reasons = Counter()
    for i, (frow, rrow, source, mask) in enumerate(zip(fake, real, source_rows, masks)):
        if str(frow["sample_id"]) != str(source["sample_id"]):
            raise RuntimeError("fake/source mismatch")
        pos, counts, crop_gt_fraction = mask_grid(mask, source["geometry"])
        aa, bb, reason, coverage = fake_regions(pos, counts, crop_gt_fraction)
        rid = str(rrow["sample_id"])
        image_path = (ROOT / "datasets" / str(rrow["image_relpath"])).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as im:
            hw = [im.height, im.width]
        # Geometry is the same deterministic ResizeShortest336 -> CenterCrop336 rule.
        rgeo = geometry_for("clip", hw)
        cc = []
        if reason is None:
            for j, (a, b) in enumerate(zip(aa, bb)):
                c = normal_box(a[2:], np.zeros((GRID, GRID), dtype=bool), a)
                if c is None:
                    reason = "real_block_unavailable"
                    break
                cc.append(c)
        reasons[reason or "valid"] += 1
        sample_rows.append({"ordinal": i, "sample_id": fake_ids[i], "real_sample_id": rid,
                            "class_label": 1, "real_class_label": 0,
                            "image_path": source["image_path"], "geometry": source["geometry"],
                            "original_hw": list(mask.shape), "gt_mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                            "gt_area_fraction": float(mask.float().mean()),
                            "crop_coverage": 336*336/(int(source["geometry"]["resized_hw"][0])*int(source["geometry"]["resized_hw"][1])),
                            "real_image_path": str(image_path), "real_geometry": rgeo,
                            "gt_positive_patch_count": int(pos.sum()), "gt_coverage_by_targets": coverage * crop_gt_fraction,
                            "gt_fraction_inside_crop": crop_gt_fraction,
                            "valid": reason is None, "invalid_reason": reason})
        if reason is None:
            for j, (a, b, c) in enumerate(zip(aa, bb, cc)):
                fake_regions_rows.extend((region_record("A", fake_ids[i], a, j, counts, int(counts.sum())),
                                          region_record("B", fake_ids[i], b, j, counts, int(counts.sum()))))
                real_regions_rows.append(region_record("C", rid, c, j))
    protocol = {"schema": "phase6i_jd_protocol_v1", "status": "FROZEN_BEFORE_DEV_DISCREPANCY",
                "question": "frozen contextual prediction discrepancy at artifact versus matched normal blocks",
                "checkpoint": str(WEIGHT), "checkpoint_sha256": file_sha256(WEIGHT),
                "source_commit": "52c1ae95d05f743e000e8f10a1f3a79b10cff048",
                "variant": "I-JEPA ViT-H/16 IN1K epoch299", "input_resolution": 448, "patch_size": 16,
                "patch_grid": [28, 28], "embedding_dim": 1280, "predictor_depth": 12,
                "predictor_embed_dim": 384,
                "preprocessing": "CLIP ResizeShortest336 CenterCrop336 geometry; bicubic RGB to 448; ImageNet mean/std; no TTA",
                "target_block": {"area_patch_min": MIN_AREA, "area_patch_max": MAX_AREA,
                                 "aspect_ratio": [.75, 1.5], "max_blocks": MAX_BLOCKS,
                                 "minimum_gt_patch_recall": .5},
                "control": {"same_shape": True, "dilation_safety_patches": SAFETY,
                            "selection": "minimum center distance difference, then location distance, then row/col"},
                "metrics": {"primary": "cosine_distance", "secondary": ["normalized_l2", "raw_l2"],
                            "target_reference": "full-image target encoder then per-token layer_norm as original I-JEPA training"},
                "statistics": {"unit": "image", "bootstrap_repeats": 10000, "permutation_repeats": 10000,
                               "seed": SEED, "ci": .95},
                "go": {"mean_delta_cosine_min": .02, "paired_bootstrap_ci_lower_gt_zero": True,
                       "win_rate_min_exclusive": .55, "delta_A_B_gt_delta_B_C": True},
                "scope": {"main": "1106 canonical internal DEV Fake", "secondary": "1106 same-validation Real",
                          "official1000": False, "internal_test": False, "external_ood": False},
                "source_manifest": str(VAL), "source_manifest_sha256": file_sha256(VAL),
                "prior_phase6i_records_sha256": file_sha256(PRIOR), "clip_shards": shards,
                "implementation_sha256": file_sha256(Path(__file__))}
    sample_manifest = {"sample_count": len(fake_ids), "sample_ids_sha256": hash_lines(sorted(fake_ids)),
                       "sample_order_sha256": hash_lines(fake_ids),
                       "gt_mask_sha256": hash_lines([r["gt_mask_sha256"] for r in sample_rows]),
                       "real_fake_label_sha256": hash_lines([1]*len(fake_ids)),
                       "original_hw_sha256": hash_json([r["original_hw"] for r in sample_rows]),
                       "real_control_count": len(real), "real_sample_ids_sha256": hash_lines(sorted(r["sample_id"] for r in real)),
                       "real_sample_order_sha256": hash_lines(r["sample_id"] for r in real),
                       "real_fake_labels": {"canonical_dev": "1106 Fake", "real_secondary": "1106 Real"},
                       "source_manifest_sha256": file_sha256(VAL), "invalid_reasons": dict(reasons)}
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "sample_records.jsonl", sample_rows)
    write_jsonl(OUT / "region_manifest.jsonl", fake_regions_rows + real_regions_rows)
    dump(OUT / "model_manifest.json", {k: protocol[k] for k in ("checkpoint", "checkpoint_sha256", "source_commit", "variant", "input_resolution", "patch_size", "patch_grid", "embedding_dim", "predictor_depth", "predictor_embed_dim", "preprocessing")})
    dump(OUT / "sample_manifest.json", sample_manifest)
    dump(OUT / "protocol_frozen.json", protocol)
    dump(OUT / "pipeline_status.json", {"status": "PREPARED", "valid_fake": reasons["valid"],
                                        "invalid_reasons": dict(reasons), "scope": protocol["scope"]})
    print(json.dumps({"status": "PREPARED", "reasons": dict(reasons),
                      "A_blocks": len(fake_regions_rows)//2, "C_blocks": len(real_regions_rows)}), flush=True)


def load_models(device):
    enc = vit_huge(img_size=[448], patch_size=16).eval()
    target = vit_huge(img_size=[448], patch_size=16).eval()
    pred = vit_predictor(num_patches=N_PATCH, embed_dim=1280, predictor_embed_dim=384,
                         depth=12, num_heads=16).eval()
    ckpt = torch.load(WEIGHT, map_location="cpu", weights_only=False)
    for model, key in ((enc, "encoder"), (target, "target_encoder"), (pred, "predictor")):
        state = ckpt[key]
        if state and all(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False)
    del ckpt
    return enc.to(device), target.to(device), pred.to(device)


def predict_one(x, target_features, box, enc, pred):
    ids = bbox_indices(box)
    wanted = set(ids)
    context = [i for i in range(N_PATCH) if i not in wanted]
    assert len(ids) + len(context) == N_PATCH and not wanted.intersection(context)
    mi = torch.tensor(ids, device=x.device, dtype=torch.long)[None]
    mc = torch.tensor(context, device=x.device, dtype=torch.long)[None]
    z = enc(x, mc)
    phat = pred(z, mc, mi).float()[0]
    truth = target_features[0].index_select(0, mi[0]).float()
    cosine = 1 - F.cosine_similarity(phat, truth, dim=-1)
    nl2 = (F.normalize(phat, dim=-1) - F.normalize(truth, dim=-1)).norm(dim=-1)
    raw = (phat - truth).norm(dim=-1)
    values = {"cosine_distance": float(cosine.mean()),
              "normalized_l2": float(nl2.mean()), "raw_l2": float(raw.mean())}
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError("nonfinite discrepancy")
    return values, mc, mi


def run(group):
    assert group in ("fake", "real")
    protocol = json.loads((OUT / "protocol_frozen.json").read_text())
    if file_sha256(WEIGHT) != protocol["checkpoint_sha256"]:
        raise RuntimeError("checkpoint drift")
    samples = list(jsonl(OUT / "sample_records.jsonl"))
    regions = list(jsonl(OUT / "region_manifest.jsonl"))
    by_id = {}
    for r in regions:
        by_id.setdefault(r["sample_id"], []).append(r)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    enc, target, pred = load_models(device)
    rows = []
    audit = None
    dest = OUT / f"discrepancy_{group}.jsonl"
    if dest.exists():
        raise RuntimeError(f"already exists: {dest}")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, sample in enumerate(samples):
            sid = sample["sample_id"] if group == "fake" else sample["real_sample_id"]
            group_regions = by_id.get(sid, [])
            if not sample["valid"]:
                rows.append({"sample_id": sid, "group": group, "valid": False,
                             "invalid_reason": sample["invalid_reason"]})
                continue
            image_row = {"sample_id": sid, "image_path": sample["image_path"] if group == "fake" else sample["real_image_path"],
                         "geometry": sample["geometry"] if group == "fake" else sample["real_geometry"]}
            x = image_tensor(image_row)[None].to(device)
            truth = F.layer_norm(target(x), (1280,))
            for region in group_regions:
                if group == "fake" and region["group"] not in ("A", "B"):
                    continue
                if group == "real" and region["group"] != "C":
                    continue
                box = region["target_bbox_patch"]
                metrics, mc, mi = predict_one(x, truth, box, enc, pred)
                rows.append({"sample_id": sid, "group": region["group"], "valid": True,
                             "block_ordinal": region["block_ordinal"], "metrics": metrics})
                if audit is None:
                    altered = x.clone()
                    y0, x0, h, w = box
                    altered[:, :, y0*16:(y0+h)*16, x0*16:(x0+w)*16] = 0
                    before = pred(enc(x, mc), mc, mi)
                    after = pred(enc(altered, mc), mc, mi)
                    diff = float((before.float() - after.float()).abs().max())
                    audit = {"sample_id": sid, "group": region["group"],
                             "target_context_indices_disjoint": True,
                             "context_encoder_mask_applied_before_transformer": True,
                             "patch_embed_nonoverlapping_16x16": True,
                             "target_pixel_perturbation_predictor_max_abs_difference": diff,
                             "TARGET_VISIBLE_TO_CONTEXT": "NO" if diff == 0 else "AUDIT_FAIL",
                             "TARGET_FEATURE_USED_BY_PREDICTOR_INPUT": "NO"}
                    if diff != 0:
                        raise RuntimeError(f"target leakage audit failed: {diff}")
            if (i+1) % 50 == 0:
                print(json.dumps({"group": group, "processed": i+1, "total": len(samples),
                                  "region_rows": len(rows)}), flush=True)
    write_jsonl(dest, rows)
    dump(OUT / f"leakage_audit_{group}.json", audit)
    dump(OUT / f"worker_{group}.json", {"status": "COMPLETE", "sample_count": len(samples),
                                      "rows": len(rows), "records_sha256": file_sha256(dest),
                                      "checkpoint_sha256": protocol["checkpoint_sha256"],
                                      "leakage_audit": audit})
    print(json.dumps({"status": "COMPLETE", "group": group, "rows": len(rows)}), flush=True)


def ci_bootstrap(values, paired=True):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return None
    rng = np.random.default_rng(SEED)
    means = np.empty(10000)
    for start in range(0, 10000, 500):
        n = min(500, 10000-start)
        means[start:start+n] = a[rng.integers(0, len(a), size=(n, len(a)))].mean(1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def summarize_delta(a, b, paired):
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if not len(aa) or not len(bb):
        return {"n_a": len(aa), "n_b": len(bb), "valid": False}
    if paired:
        if len(aa) != len(bb):
            raise RuntimeError("paired length drift")
        dd = aa-bb
        rng = np.random.default_rng(SEED + 1)
        perm = np.empty(10000)
        for start in range(0, 10000, 500):
            n = min(500, 10000-start)
            signs = rng.choice([-1, 1], size=(n, len(dd)))
            perm[start:start+n] = (signs*dd).mean(1)
        return {"n": len(dd), "mean_a": float(aa.mean()), "mean_b": float(bb.mean()),
                "mean_delta": float(dd.mean()), "median_delta": float(np.median(dd)),
                "bootstrap_95_ci": ci_bootstrap(dd), "wins": int((dd>0).sum()),
                "ties": int((dd==0).sum()), "losses": int((dd<0).sum()),
                "win_rate": float((dd>0).mean()), "paired_effect_d": float(dd.mean()/dd.std(ddof=1)) if len(dd)>1 and dd.std(ddof=1)>0 else None,
                "paired_permutation_p_two_sided": float((1+(abs(perm)>=abs(dd.mean())).sum())/10001)}
    rng = np.random.default_rng(SEED)
    diff = np.empty(10000)
    for start in range(0, 10000, 500):
        n = min(500, 10000-start)
        diff[start:start+n] = (aa[rng.integers(len(aa), size=(n,len(aa)))].mean(1)
                              -bb[rng.integers(len(bb), size=(n,len(bb)))].mean(1))
    return {"n_a": len(aa), "n_b": len(bb), "mean_a": float(aa.mean()), "mean_b": float(bb.mean()),
            "mean_delta": float(aa.mean()-bb.mean()),
            "bootstrap_95_ci": [float(x) for x in np.quantile(diff,[.025,.975])]}


def finalize():
    protocol = json.loads((OUT / "protocol_frozen.json").read_text())
    sample = json.loads((OUT / "sample_manifest.json").read_text())
    samples = list(jsonl(OUT / "sample_records.jsonl"))
    regions = list(jsonl(OUT / "region_manifest.jsonl"))
    for group in ("fake", "real"):
        worker = json.loads((OUT / f"worker_{group}.json").read_text())
        if worker["status"] != "COMPLETE" or worker["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
            raise RuntimeError(f"{group} worker invalid")
        if file_sha256(OUT / f"discrepancy_{group}.jsonl") != worker["records_sha256"]:
            raise RuntimeError(f"{group} records hash drift")
    rows = list(jsonl(OUT / "discrepancy_fake.jsonl")) + list(jsonl(OUT / "discrepancy_real.jsonl"))
    rkey = {(r["sample_id"],r["group"],r.get("block_ordinal")): r for r in rows if r["valid"]}
    expected = {(r["sample_id"],r["group"],r["block_ordinal"]) for r in regions}
    if set(rkey) != expected or len(rkey) != len(expected):
        raise RuntimeError("region/result mismatch")
    observed_fake_ids = [r["sample_id"] for r in jsonl(OUT / "discrepancy_fake.jsonl") if not r["valid"]]
    expected_invalid = [s["sample_id"] for s in samples if not s["valid"]]
    if observed_fake_ids != expected_invalid:
        raise RuntimeError("invalid fake accounting drift")
    records = []
    metrics = ("cosine_distance", "normalized_l2", "raw_l2")
    for s in samples:
        row = {"sample_id": s["sample_id"], "real_sample_id": s["real_sample_id"],
               "valid": s["valid"], "invalid_reason": s["invalid_reason"],
               "gt_area_fraction": s["gt_area_fraction"], "crop_coverage": s["crop_coverage"],
               "gt_coverage_by_targets": s["gt_coverage_by_targets"]}
        if s["valid"]:
            for g, sid in (("A",s["sample_id"]),("B",s["sample_id"]),("C",s["real_sample_id"])):
                blocks = [r for r in regions if r["group"]==g and r["sample_id"]==sid]
                if not blocks:
                    raise RuntimeError("valid image without blocks")
                row[g] = {m: float(np.mean([rkey[(sid,g,b["block_ordinal"])]["metrics"][m] for b in blocks]))
                          for m in metrics}
                row[g]["n_blocks"] = len(blocks)
        records.append(row)
    if len(records) != sample["sample_count"] or [r["sample_id"] for r in records] != [r["sample_id"] for r in samples]:
        raise RuntimeError("image-level accounting drift")
    write_jsonl(OUT / "discrepancy_per_region.jsonl", rows)
    write_jsonl(OUT / "discrepancy_per_image.jsonl", records)
    valid = [r for r in records if r["valid"]]
    stats = {}
    for metric in metrics:
        stats[metric] = {"A_minus_B_paired": summarize_delta([r["A"][metric] for r in valid], [r["B"][metric] for r in valid], True),
                         "A_minus_C_unpaired": summarize_delta([r["A"][metric] for r in valid], [r["C"][metric] for r in valid], False),
                         "B_minus_C_unpaired": summarize_delta([r["B"][metric] for r in valid], [r["C"][metric] for r in valid], False)}
    subgroup = {}
    bins = {"area_le_1pct": lambda r: r["gt_area_fraction"]<=.01,
            "area_1_to_5pct": lambda r: .01<r["gt_area_fraction"]<=.05,
            "area_5_to_20pct": lambda r: .05<r["gt_area_fraction"]<=.20,
            "area_gt_20pct": lambda r: r["gt_area_fraction"]>.20,
            "coverage_0p5_to_0p7": lambda r: .5<=r["gt_coverage_by_targets"]<.7,
            "coverage_0p7_to_0p9": lambda r: .7<=r["gt_coverage_by_targets"]<.9,
            "coverage_ge_0p9": lambda r: r["gt_coverage_by_targets"]>=.9,
            "crop_lt_0p8": lambda r: r["crop_coverage"]<.8,
            "crop_ge_0p8": lambda r: r["crop_coverage"]>=.8}
    for name, condition in bins.items():
        all_n = sum(condition(r) for r in records)
        vv = [r for r in valid if condition(r)]
        subgroup[name] = {"n_all": all_n, "n_valid": len(vv),
                          "A_minus_B_cosine": summarize_delta([r["A"]["cosine_distance"] for r in vv],
                                                                 [r["B"]["cosine_distance"] for r in vv], True)}
    primary = stats["cosine_distance"]
    ab, bc = primary["A_minus_B_paired"], primary["B_minus_C_unpaired"]
    gates = {"mean_delta_cosine_ge_0p02": ab["mean_delta"]>=.02,
             "paired_ci_lower_gt_zero": ab["bootstrap_95_ci"][0]>0,
             "win_rate_gt_0p55": ab["win_rate"]>.55,
             "artifact_increment_gt_synthetic_global": ab["mean_delta"]>bc["mean_delta"]}
    go = all(gates.values())
    if go:
        conclusion = "SUPPORTED"
    elif ab["mean_delta"]>0:
        conclusion = "WEAK / INCONCLUSIVE"
    else:
        conclusion = "NOT SUPPORTED"
    dump(OUT / "paired_statistics.json", {"status": "COMPLETE_STOP", "n_all_fake": len(samples),
                                         "n_valid_paired": len(valid), "n_invalid": len(samples)-len(valid),
                                         "primary": "cosine_distance", "metrics": stats, "go_gates": gates,
                                         "go": go, "conclusion": conclusion})
    dump(OUT / "subgroup_statistics.json", subgroup)
    audits = [json.loads((OUT / f"leakage_audit_{g}.json").read_text()) for g in ("fake", "real")]
    dump(OUT / "leakage_audit.json", {"TARGET_VISIBLE_TO_CONTEXT": "NO", "TARGET_FEATURE_USED_BY_PREDICTOR_INPUT": "NO",
                                     "dynamic_checks": audits})
    dump(OUT / "pipeline_status.json", {"status": "COMPLETE_STOP", "go": go, "conclusion": conclusion,
                                        "valid_paired": len(valid), "invalid": len(samples)-len(valid),
                                        "official1000_accessed": False, "internal_test_accessed": False,
                                        "external_ood_accessed": False})
    doc = ROOT / "docs/phase6i_jepa_predictive_discrepancy.md"
    def fmt(x): return f"{x:+.6f}"
    content = ["# Phase6I-JD：冻结 I-JEPA 预测误差诊断", "",
               f"状态：**COMPLETE STOP**；结论：**{conclusion}**；下一阶段门槛：**{'GO' if go else 'STOP'}**。", "",
               "## 人群与方法", "",
               f"主分析为 canonical internal DEV 的 1106 张 Fake，{len(valid)} 张形成合法 A/B 配对，{len(samples)-len(valid)} 张显式 invalid。",
               "同一 internal validation 清单的 1106 张 Real 用作次级 C 对照，不改变主 DEV 人群和顺序。",
               "沿用 Phase6I 官方 ViT-H/16@448 checkpoint。encoder、target encoder、predictor 全冻结；target latent 来自完整图像，predictor 仅接收排除 target patch 后的 context tokens。",
               f"目标块面积 {MIN_AREA}–{MAX_AREA}/784 patches，长宽比 0.75–1.5；正常块与 artifact 块宽高完全相同，GT 膨胀 {SAFETY} patches 后避让。",
               "primary 为 cosine distance；normalized L2 和 raw L2 为预定次级指标。", "",
               "## 主结果：Artifact A 与同图正常 B", "",
               "| 指标 | Mean A | Mean B | A−B | 配对 bootstrap 95% CI | 胜/平/负 |", "|---|---:|---:|---:|---|---:|",]
    for metric in metrics:
        z = stats[metric]["A_minus_B_paired"]
        content.append(f"| {metric} | {z['mean_a']:.6f} | {z['mean_b']:.6f} | {fmt(z['mean_delta'])} | [{fmt(z['bootstrap_95_ci'][0])}, {fmt(z['bootstrap_95_ci'][1])}] | {z['wins']}/{z['ties']}/{z['losses']} |")
    content.extend(["", "## 次级对照与稳健性", "",
                    "| cosine distance 比较 | Mean difference | bootstrap 95% CI |", "|---|---:|---|",])
    for label, key in (("A−C（非配对）", "A_minus_C_unpaired"),("B−C（非配对）", "B_minus_C_unpaired")):
        z = primary[key]
        content.append(f"| {label} | {fmt(z['mean_delta'])} | [{fmt(z['bootstrap_95_ci'][0])}, {fmt(z['bootstrap_95_ci'][1])}] |")
    content.extend(["", "| GT 面积分组 | 全部 / 有效 | A−B cosine | 95% CI |", "|---|---:|---:|---|",])
    for name in ("area_le_1pct","area_1_to_5pct","area_5_to_20pct","area_gt_20pct"):
        z = subgroup[name]; v = z["A_minus_B_cosine"]
        effect = fmt(v["mean_delta"]) if "mean_delta" in v else "-"
        interval = f"[{fmt(v['bootstrap_95_ci'][0])}, {fmt(v['bootstrap_95_ci'][1])}]" if "bootstrap_95_ci" in v else "-"
        content.append(f"| {name} | {z['n_all']} / {z['n_valid']} | {effect} | {interval} |")
    content.extend(["", f"配对标准化效应 d={ab['paired_effect_d']:.4f}；双侧配对置换 p={ab['paired_permutation_p_two_sided']:.6f}。",
                    f"预注册 GO 四项检查：`{json.dumps(gates, ensure_ascii=False)}`。", "",
                    "模型没有训练或更新 R1；此实验不能推出 JEPA 改善 R1。未访问 Official1000、internal test 或外部 OOD。",
                    "完整人群哈希、区域、逐区域/逐图数值和 leakage audit 位于 `outputs/phase6i_jepa_discrepancy/`。", ""])
    doc.write_text("\n".join(content))
    print(json.dumps({"status": "COMPLETE_STOP", "n_valid": len(valid), "go": go,
                      "conclusion": conclusion, "A_minus_B": ab, "B_minus_C": bc}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare","fake","real","finalize"))
    args = parser.parse_args()
    {"prepare": prepare, "fake": lambda: run("fake"), "real": lambda: run("real"),
     "finalize": finalize}[args.mode]()


if __name__ == "__main__":
    main()
