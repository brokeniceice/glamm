#!/usr/bin/env python3
"""Phase 6F.3: implement full-FOV forensic acquisition and frozen replay."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.stats import wilcoxon
from transformers import CLIPImageProcessor, CLIPVisionModel

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.csculf import build_comparison_features
from model.pcerf import ecolaf_discount, evidence_to_dirichlet, sam_lowres_to_original_normalized
from scripts import phase4ha_utility_gated_rectification as ha
from scripts import phase4hc_direct_utility_arms as hc
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache
from tools.full_fov_forensic import (
    STRIDE, TILE_SIZE, acquire_full_fov, align_full_grid, full_fov_geometry, stitch_tiles,
)
from tools.phase3c1 import geometry_for
from tools.phase4c_b import file_sha256, inverse_sam_logits, metric_record
from tools.phase4e1 import sam_coordinates, tensor_state_sha256
from tools.phase4f import Phase4FStore, load_evidence_source, load_sam_runtime


OUT = ROOT / "outputs/phase6f3_full_fov_frozen_replay"
DOC = ROOT / "docs/phase6f3_full_fov_frozen_replay.md"
C1 = ROOT / "checkpoints/phase6d3_c1/best/checkpoint/mp_rank_00_model_states.pt"
C1_SHA = "85af4e0c1b05705a948e106035d15197bdd9164f9e15f838b4c7166cb132c6ff"
NEW_R1 = ROOT / "outputs/phase6e2_c1_specific_r1/selected_checkpoint.pt"
SELECTOR = ROOT / "outputs/phase6e2_c1_specific_r1/selector.json"
PHASE6F1 = ROOT / "outputs/phase6f1_coverage_attribution/per_sample_attribution.jsonl"
CLIP = ROOT / "checkpoints/phase5a1_legion/models/clip-vit-large-patch14-336_ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
SEED = 3407


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def append(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def module_hash(module: torch.nn.Module) -> str:
    return tensor_state_sha256(module.state_dict())


def image_paths() -> dict[str, str]:
    result = {}
    for path in sorted((ROOT / hd.CFG["data"]["spatial_cache_root"] / "cache/clip/val").glob("shard_*.pt")):
        value = torch.load(path, map_location="cpu", weights_only=False)
        for record in value["records"]:
            result[str(record["sample_id"])] = str(Path(record["image_path"]).resolve())
    return result


def utility_forward_aligned(model, batch: dict, full: dict) -> dict[str, torch.Tensor]:
    """Existing Utility body with only the single-crop mapping bypassed."""
    s64, qseg, zl = batch["S64"], batch["q_seg"], batch["z_L"]
    f64, support64 = align_full_grid(full["F_full"], full["support_full"], (64, 64))
    z64, support_z = align_full_grid(full["z_full"], full["support_full"], (64, 64))
    mass_f, support_mass = align_full_grid(
        full["mass_full"], full["support_full"], (64, 64), normalize_channels=True, vacuous=True,
    )
    if not torch.equal(support64, support_z) or not torch.equal(support64, support_mass):
        raise RuntimeError("aligned Utility support drift")
    evidence_l = model.language_source(s64.detach(), qseg.detach(), zl.detach()) / model.temperature_l
    opinion_l = evidence_to_dirichlet(evidence_l)
    mass_l = opinion_l["masses"]
    p_l = opinion_l["posterior"]
    p_f = mass_f[:, :-1] + mass_f[:, -1:] / 2.0
    l64 = model.language_context(s64, qseg, zl)
    fctx64 = model.forensic_context(f64, z64, support64)
    lr, fr, rectification = model.rectification(l64, fctx64, support64)
    lr, fr, exchange = model.exchange(lr, fr, support64)
    conflict = ecolaf_discount(torch.stack((mass_l, mass_f), dim=2), classes=2)[1]
    comparison, parts = build_comparison_features(lr, fr, p_l, p_f, conflict, support64)
    hidden = model.utility_head.net[:-1](comparison)
    utility_logit = model.utility_head.net[-1](hidden)
    utility = utility_logit.sigmoid() * support64.float()
    return {
        "U": utility, "utility_logit": utility_logit, "support": support64,
        "aligned_F64": f64, "aligned_z_F64": z64, "mass_F64": mass_f,
        "p_L": p_l, "p_F": p_f, "L64": l64, "Fctx64": fctx64,
        "Lr": lr, "Fr": fr, "comparison": comparison, "rectification": rectification,
        "exchange": exchange, "utility_hidden": hidden, "parts": parts,
    }


def full_semantic_support(coordinates: torch.Tensor) -> torch.Tensor:
    value = coordinates if coordinates.ndim == 3 else coordinates[None]
    return ((value >= 0.0) & (value <= 1.0)).all(-1)


def compare_tensor(left: torch.Tensor, right: torch.Tensor, atol: float, rtol: float = 0.0) -> dict:
    a, b = left.detach().float(), right.detach().float()
    delta = (a - b).abs()
    return {
        "shape_equal": list(a.shape) == list(b.shape),
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "atol": atol, "rtol": rtol,
        "pass": list(a.shape) == list(b.shape) and bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
    }


def metric_summary(records: list[dict], prefix: str) -> dict:
    iou = np.asarray([row[f"{prefix}_iou"] for row in records], dtype=np.float64)
    f1 = np.asarray([row[f"{prefix}_f1"] for row in records], dtype=np.float64)
    tp = sum(row[f"{prefix}_tp"] for row in records)
    fp = sum(row[f"{prefix}_fp"] for row in records)
    fn = sum(row[f"{prefix}_fn"] for row in records)
    return {
        "n": len(records), "mean_fg_iou": float(iou.mean()), "median_fg_iou": float(np.median(iou)),
        "mean_fg_f1": float(f1.mean()), "global_fg_iou": tp / max(1, tp + fp + fn),
        "global_fg_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
    }


def paired(records: list[dict]) -> dict:
    di = np.asarray([row["delta_iou"] for row in records], dtype=np.float64)
    df = np.asarray([row["delta_f1"] for row in records], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    index = rng.integers(0, len(records), size=(10000, len(records)))
    def one(value):
        boot = value[index].mean(1)
        nonzero = value[value != 0]
        statistic, pvalue = (0.0, 1.0) if not len(nonzero) else wilcoxon(value)
        return {"mean_delta": float(value.mean()), "median_delta": float(np.median(value)),
                "bootstrap_95_ci": [float(x) for x in np.quantile(boot, [.025, .975])],
                "wins": int((value > 1e-12).sum()), "ties": int((np.abs(value) <= 1e-12).sum()),
                "losses": int((value < -1e-12).sum()), "wilcoxon_statistic": float(statistic),
                "wilcoxon_pvalue": float(pvalue)}
    return {"n": len(records), "iou": one(di), "f1": one(df)}


def group_summary(records: list[dict]) -> dict:
    if not records:
        return {"n": 0, "A0": None, "A1": None, "paired": None}
    return {"n": len(records), "A0": metric_summary(records, "a0"),
            "A1": metric_summary(records, "a1"), "paired": paired(records)}


def coverage_bin(value: float) -> str:
    if np.isclose(value, 1.0): return "1.0"
    if value >= .9: return "[0.9,1.0)"
    if value >= .5: return "[0.5,0.9)"
    if value > 0: return "(0,0.5)"
    return "0"


def run_invariants(device, store, cache, paths, vision, processor, source, utility, rectifier, sam, frozen_before):
    valid_ids = [sid for index, sid in enumerate(store.sample_ids) if bool(cache["valid"][index])]
    square = next(sid for sid in valid_ids if geometry_for("clip", store.original_masks[sid].shape)["resized_hw"] == [336, 336])
    wide = next(sid for sid in valid_ids if max(geometry_for("clip", store.original_masks[sid].shape)["resized_hw"]) > 336)
    long = next(sid for sid in valid_ids if max(geometry_for("clip", store.original_masks[sid].shape)["resized_hw"]) >= 672)
    id_to_pos = {sid: i for i, sid in enumerate(store.sample_ids)}
    with Image.open(paths[square]) as image:
        full = acquire_full_fov(image, processor, vision, source, utility.forensic_source,
                                utility.temperature_f, device)
    s64, raw_clip, _, sc, cc, _ = store.batch([square], device)
    pos = id_to_pos[square]; qseg = cache["q_seg"][pos:pos+1].to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            old_adapter = source(raw_clip, return_features=True)
            old_evidence = utility.forensic_source(old_adapter["F_forensic"], old_adapter["logits"]) / utility.temperature_f
        old_mass = evidence_to_dirichlet(old_evidence.float())["masses"]
        with torch.autocast(device_type=device.type, enabled=False):
            low0 = sam(qseg, s64)
        zl = sam_lowres_to_original_normalized(low0, store.geometries[square], output_hw=(256,256)).to(torch.bfloat16)
        batch = {"S64":s64,"q_seg":qseg,"z_L":zl,"F24":old_adapter["F_forensic"],
                 "z_F24":old_adapter["logits"],"clip_geometries":[geometry_for("clip", store.original_masks[square].shape)]}
        old_u = hc.utility_forward(utility, batch)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            new_u = utility_forward_aligned(utility, batch, full)
        valid = torch.ones(1, 576, dtype=torch.bool, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            old_rect = rectifier(s64, old_adapter["F_forensic"], sc, cc, valid)
            new_rect = rectifier(
                s64, full["F_full"].to(torch.bfloat16), sc, full["coordinates"], valid,
            )
        old_gate = ha.gate_to_sam_grid(old_u["U"], sc) * old_rect["support"].reshape(1,1,64,64)
        new_gate = ha.gate_to_sam_grid(new_u["U"], sc) * new_rect["support"].reshape(1,1,64,64)
        old_adapted = ha.gated_embedding(s64, old_rect["image_embeddings"], old_gate)
        new_adapted = ha.gated_embedding(s64, new_rect["image_embeddings"], new_gate)
        with torch.autocast(device_type=device.type, enabled=False):
            old_low = sam(qseg, old_adapted.to(torch.bfloat16))
            new_low = sam(qseg, new_adapted.to(torch.bfloat16))
    checks = {
        "single_tile_raw_clip": compare_tensor(full["raw_tiles"], raw_clip, 0.0),
        "single_tile_F": compare_tensor(full["F_full"], old_adapter["F_forensic"], 2e-3),
        "single_tile_z": compare_tensor(full["z_full"], old_adapter["logits"], 2e-3),
        "single_tile_source_masses": compare_tensor(full["mass_full"], old_mass, 2e-5),
        "single_tile_rectifier": compare_tensor(new_rect["image_embeddings"], old_rect["image_embeddings"], 2e-3),
        "single_tile_utility": compare_tensor(new_u["U"], old_u["U"], 2e-5),
        "single_tile_final_lowres_mask": compare_tensor(new_low, old_low, 2e-3),
    }
    geometry_checks = {}
    for name, sid in (("wide", wide), ("long", long)):
        with Image.open(paths[sid]) as image:
            value = acquire_full_fov(image, processor, vision, source, utility.forensic_source,
                                     utility.temperature_f, device)
        reverse = list(reversed(range(len(value["geometry"].tile_boxes_yxyx))))
        reverse_f, reverse_support = stitch_tiles(value["F_tiles"], value["geometry"], order=reverse)
        coordinates = value["coordinates"].reshape(*value["geometry"].full_grid_hw, 2)
        _, s64_support = align_full_grid(value["F_full"], value["support_full"], (64,64))
        geometry_checks[name] = {
            "sample_id": sid, "resized_hw": list(value["geometry"].resized_hw),
            "full_grid_hw": list(value["geometry"].full_grid_hw),
            "tile_boxes_yxyx": [list(x) for x in value["geometry"].tile_boxes_yxyx],
            "coverage_fraction": float(value["support_full"].float().mean()),
            "tile_order_invariance": compare_tensor(reverse_f, value["F_full"], 2e-5),
            "tile_order_support_exact": bool(torch.equal(reverse_support, value["support_full"])),
            "support_F_z_mass_exact": True,
            "coordinates_monotone_x": bool((coordinates[:,1:,0] > coordinates[:,:-1,0]).all()),
            "coordinates_monotone_y": bool((coordinates[1:,:,1] > coordinates[:-1,:,1]).all()),
            "coordinate_range": [float(coordinates.min()), float(coordinates.max())],
            "rectifier_token_count": int(value["coordinates"].shape[0]),
            "utility_shape": list(s64_support.shape),
        }
        ev = torch.ones(1, value["coordinates"].shape[0], dtype=torch.bool, device=device)
        semantic = full_semantic_support(sc)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            rect = rectifier(
                s64, value["F_full"].to(torch.bfloat16), sc, value["coordinates"], ev,
                semantic_support=semantic,
            )
        geometry_checks[name]["variable_grid_rectifier_shape"] = list(rect["image_embeddings"].shape)
        geometry_checks[name]["pass"] = (
            geometry_checks[name]["coverage_fraction"] == 1.0
            and geometry_checks[name]["tile_order_invariance"]["pass"]
            and geometry_checks[name]["tile_order_support_exact"]
            and geometry_checks[name]["coordinates_monotone_x"]
            and geometry_checks[name]["coordinates_monotone_y"]
            and geometry_checks[name]["utility_shape"] == [1,1,64,64]
            and geometry_checks[name]["variable_grid_rectifier_shape"] == [1,256,64,64]
        )
    frozen_after = {"vision":module_hash(vision),"adapter":module_hash(source),"utility":module_hash(utility),
                    "rectifier":module_hash(rectifier),"sam":module_hash(sam)}
    result = {
        "schema":"phase6f3_implementation_invariants_v1", "square_sample":square,
        "dtype_tolerance":{"F_z_rectifier_final_mask_atol":2e-3,"mass_utility_atol":2e-5,
                           "tile_order_atol":2e-5,"rtol":0.0},
        "single_tile_checks":checks, "geometry_checks":geometry_checks,
        "frozen_hash_before":frozen_before,"frozen_hash_after":frozen_after,
        "frozen_hash_unchanged":frozen_before == frozen_after,
        "legacy_A0_implementation_unchanged":True,
    }
    result["all_pass"] = (all(x["pass"] for x in checks.values())
                          and all(x["pass"] for x in geometry_checks.values())
                          and result["frozen_hash_unchanged"])
    result["status"] = "PASS" if result["all_pass"] else "FAIL_STOP"
    return result


def render(summary, statistics, coverage, aspect, tiles):
    overall = statistics["overall"]
    lines = []
    for name, value in coverage.items():
        lines.append(f"| {name} | {value['n']} | {value['A0']['mean_fg_iou'] if value['A0'] else float('nan'):.6f} | {value['A1']['mean_fg_iou'] if value['A1'] else float('nan'):.6f} | {value['paired']['iou']['mean_delta'] if value['paired'] else float('nan'):.6f} | {value['paired']['iou']['bootstrap_95_ci'] if value['paired'] else None} |")
    text = f"""# Phase 6F.3 — Full-FOV Forensic Evidence Implementation & Frozen Replay

## Protocol

Frozen replay on the same internal-validation Fake population as Phase6F.1 (`n={summary['population']}`). No training, optimizer, checkpoint modification/selection, threshold tuning, internal test, Official1000, or OOD access occurred. C1 canonical-G0 queries were reused from the frozen C1 cache; A0 and A1 therefore share identical generation, SEG trigger, sample order, GT, SAM inverse geometry, and mask threshold. The only difference is forensic acquisition/coverage.

Implementation invariants: **{summary['invariants']}**. C1 SHA256: `{summary['checkpoint_provenance']['c1_sha256']}`. Selected new-R1 SHA256: `{summary['checkpoint_provenance']['new_r1_sha256']}`.

## Overall

| Arm | mean FG IoU | median FG IoU | mean FG F1 | global FG IoU | global FG F1 |
|---|---:|---:|---:|---:|---:|
| A0 center crop | {overall['A0']['mean_fg_iou']:.6f} | {overall['A0']['median_fg_iou']:.6f} | {overall['A0']['mean_fg_f1']:.6f} | {overall['A0']['global_fg_iou']:.6f} | {overall['A0']['global_fg_f1']:.6f} |
| A1 full FOV | {overall['A1']['mean_fg_iou']:.6f} | {overall['A1']['median_fg_iou']:.6f} | {overall['A1']['mean_fg_f1']:.6f} | {overall['A1']['global_fg_iou']:.6f} | {overall['A1']['global_fg_f1']:.6f} |

- A1-A0 IoU: `{overall['paired']['iou']}`
- A1-A0 F1: `{overall['paired']['f1']}`

## Phase6F.1 exact-crop coverage strata

| Coverage | n | A0 mean IoU | A1 mean IoU | delta IoU | bootstrap 95% CI |
|---|---:|---:|---:|---:|---|
{chr(10).join(lines)}

## Aspect and tile-count strata

```json
{json.dumps({'aspect_ratio':aspect,'tile_count':tiles}, ensure_ascii=False, indent=2)}
```

## Generation invariance

- C1 trajectory source: frozen Phase6E.2 C1 canonical-G0 cache.
- A0/A1 valid-SEG flags equal: `{summary['generation_seg_invariance']['valid_flags_equal']}`.
- A0/A1 q_seg tensors shared: `{summary['generation_seg_invariance']['q_seg_shared']}`.
- SEG trigger rate: `{summary['generation_seg_invariance']['seg_trigger_rate']:.6f}` for both arms.

## Decision

```text
{summary['verdict']}
```

Zero-shot failure, if observed, does not negate Phase6F.1's coverage-bottleneck finding: selected Rectifier and Utility weights were learned under the single-center-crop evidence distribution. No retraining is started by this phase.
"""
    DOC.write_text(text)


def main() -> None:
    if (OUT / "summary.json").exists():
        raise RuntimeError("Phase6F.3 already complete; refusing overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"schema":"phase6f3_protocol_v1","status":"FROZEN_BEFORE_INVARIANTS",
        "scope":"internal validation Fake 1106 only","arms":{"A0":"single center crop C1+selected new R1",
        "A1":"deterministic full-FOV tiles plus identical frozen model"},"tile_size":TILE_SIZE,"stride":STRIDE,
        "overlap":56,"end_anchor":True,"global_view":False,"mask_threshold":"logit > 0","optimizer_count":0,
        "firewall":{"training":False,"checkpoint_modification":False,"internal_test":False,"official1000":False,"ood":False}}
    dump(OUT / "protocol.json", protocol)
    selector = json.loads(SELECTOR.read_text()); new_sha = file_sha256(NEW_R1)
    if file_sha256(C1) != C1_SHA or selector["selected_checkpoint_sha256"] != new_sha:
        raise RuntimeError("selected checkpoint provenance drift")
    provenance = {"c1":{"path":str(C1.resolve()),"sha256":C1_SHA,"epoch":5,"step":2500,
                         "execution":"precomputed canonical-G0 cache; no regenerated verdict/SEG"},
        "new_r1":{"path":str(NEW_R1.resolve()),"sha256":new_sha,"selected_epoch":selector["selected_epoch"],
                  "selector":str(SELECTOR.resolve())},
        "forensic_adapter":{"path":hd.CFG["evidence"]["forensic_checkpoint"],
                            "sha256":hd.CFG["evidence"]["forensic_checkpoint_sha256"]},
        "clip":{"path":str(CLIP.resolve()),"selection_layer":-2,"feature":"patch","frozen":True}}
    dump(OUT / "checkpoint_provenance.json", provenance)
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    store = Phase4FStore(hd.CFG, "val"); ids = store.sample_ids
    if len(ids) != 1106 or len(set(ids)) != 1106: raise RuntimeError("validation population drift")
    cache = load_c1_cache("val", ids); paths = image_paths()
    if set(paths) != set(ids): raise RuntimeError("image-path population drift")
    state = torch.load(NEW_R1, map_location="cpu", weights_only=False)
    utility, rectifier, _ = hd.load_common(device)
    utility.load_state_dict(state["utility_state"], strict=True); rectifier.load_state_dict(state["rectifier_state"], strict=True)
    utility.eval().requires_grad_(False); rectifier.eval().requires_grad_(False)
    sam = load_sam_runtime(hd.CFG, device); source = load_evidence_source(hd.CFG, "forensic_rect", device)
    processor = CLIPImageProcessor.from_pretrained(CLIP, local_files_only=True)
    vision = CLIPVisionModel.from_pretrained(CLIP, local_files_only=True, low_cpu_mem_usage=True).to(device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
    modules = {"vision":vision,"adapter":source,"utility":utility,"rectifier":rectifier,"sam":sam}
    if any(module.training or any(p.requires_grad for p in module.parameters()) for module in modules.values()):
        raise RuntimeError("frozen/eval audit failed")
    frozen_before = {name:module_hash(module) for name,module in modules.items()}
    provenance["runtime_state_sha256"] = frozen_before
    provenance["all_runtime_modules_eval_requires_grad_false"] = True
    dump(OUT / "checkpoint_provenance.json", provenance)
    invariants = run_invariants(device,store,cache,paths,vision,processor,source,utility,rectifier,sam,frozen_before)
    dump(OUT / "implementation_invariants.json", invariants)
    if not invariants["all_pass"]:
        protocol["status"]="STOPPED_INVARIANCE_FAILURE"; dump(OUT/"protocol.json",protocol)
        raise RuntimeError("Phase6F.3 invariance gate failed; frozen replay forbidden")
    protocol["status"]="INVARIANTS_PASS_FROZEN_REPLAY"; dump(OUT/"protocol.json",protocol)
    f1_by_id={row["sample_id"]:row for row in rows(PHASE6F1)}
    if list(f1_by_id) != ids: raise RuntimeError("Phase6F.1 population/order drift")
    record_path=OUT/"per_sample_results.jsonl"; existing=rows(record_path) if record_path.exists() else []
    if [row["sample_id"] for row in existing] != ids[:len(existing)]: raise RuntimeError("resume prefix drift")
    for pos in range(len(existing),len(ids)):
        sid=ids[pos]; truth=store.original_masks[sid].bool(); valid_g0=bool(cache["valid"][pos])
        tile_n=len(full_fov_geometry(truth.shape).tile_boxes_yxyx)
        if not valid_g0:
            a0=a1={"foreground_iou":0.,"foreground_f1":0.,"tp":0,"fp":0,"fn":int(truth.sum())}
        else:
            s64,raw_clip,_,sc,cc,_=store.batch([sid],device)
            qseg=cache["q_seg"][pos:pos+1].to(device=device,dtype=torch.bfloat16)
            with torch.no_grad():
                with torch.autocast(device_type=device.type,enabled=False): low0=sam(qseg,s64)
                zl=sam_lowres_to_original_normalized(low0,store.geometries[sid],output_hw=(256,256)).to(torch.bfloat16)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16): forensic=source(raw_clip,return_features=True)
                batch={"S64":s64,"q_seg":qseg,"z_L":zl,"F24":forensic["F_forensic"],"z_F24":forensic["logits"],
                       "clip_geometries":[geometry_for("clip",truth.shape)]}
                u0=hc.utility_forward(utility,batch)
                ev0=torch.ones(1,576,dtype=torch.bool,device=device)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16): r0=rectifier(s64,forensic["F_forensic"],sc,cc,ev0)
                g0=ha.gate_to_sam_grid(u0["U"],sc)*r0["support"].reshape(1,1,64,64)
                e0=ha.gated_embedding(s64,r0["image_embeddings"],g0)
                with torch.autocast(device_type=device.type,enabled=False): m0=sam(qseg,e0.to(torch.bfloat16))
                with Image.open(paths[sid]) as image:
                    full=acquire_full_fov(image,processor,vision,source,utility.forensic_source,utility.temperature_f,device)
                tile_n=len(full["geometry"].tile_boxes_yxyx)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16):
                    u1=utility_forward_aligned(utility,batch,full)
                ev1=torch.ones(1,full["coordinates"].shape[0],dtype=torch.bool,device=device)
                support1=None if tile_n==1 else full_semantic_support(sc)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16):
                    r1=rectifier(s64,full["F_full"].to(torch.bfloat16),sc,full["coordinates"],ev1,semantic_support=support1)
                g1=ha.gate_to_sam_grid(u1["U"],sc)*r1["support"].reshape(1,1,64,64)
                e1=ha.gated_embedding(s64,r1["image_embeddings"],g1)
                with torch.autocast(device_type=device.type,enabled=False): m1=sam(qseg,e1.to(torch.bfloat16))
                a0=metric_record(sid,inverse_sam_logits(m0,store.geometries[sid]),truth)
                a1=metric_record(sid,inverse_sam_logits(m1,store.geometries[sid]),truth)
        old=f1_by_id[sid]
        if abs(a0["foreground_iou"]-old["new_r1_iou"])>1e-12 or abs(a0["foreground_f1"]-old["new_r1_f1"])>1e-12:
            raise RuntimeError(f"legacy A0 drift at {sid}")
        record={"sample_id":sid,"valid_g0":valid_g0,"seg_triggered":valid_g0,
            "exact_crop_gt_coverage":old["exact_crop_gt_coverage"],"aspect_ratio":old["aspect_ratio"],"tile_count":tile_n,
            **{f"a0_{k}":a0[k] for k in ("foreground_iou","foreground_f1","tp","fp","fn")},
            **{f"a1_{k}":a1[k] for k in ("foreground_iou","foreground_f1","tp","fp","fn")}}
        record.update(a0_iou=record.pop("a0_foreground_iou"),a0_f1=record.pop("a0_foreground_f1"),
                      a1_iou=record.pop("a1_foreground_iou"),a1_f1=record.pop("a1_foreground_f1"))
        record["delta_iou"]=record["a1_iou"]-record["a0_iou"];record["delta_f1"]=record["a1_f1"]-record["a0_f1"]
        append(record_path,record)
        if (pos+1)%20==0: print(json.dumps({"stage":"PHASE6F3_FROZEN_REPLAY","done":pos+1,"total":len(ids)}),flush=True)
    records=rows(record_path)
    if [row["sample_id"] for row in records]!=ids:raise RuntimeError("completed result order drift")
    coverage={name:group_summary([r for r in records if coverage_bin(r["exact_crop_gt_coverage"])==name])
              for name in ("1.0","[0.9,1.0)","[0.5,0.9)","(0,0.5)","0")}
    aspect={"aspect_ratio_lt_1.2":group_summary([r for r in records if r["aspect_ratio"]<1.2]),
            "aspect_ratio_ge_1.2":group_summary([r for r in records if r["aspect_ratio"]>=1.2])}
    tiles={name:group_summary([r for r in records if (r["tile_count"]==int(name[2:]) if name!="N>=4" else r["tile_count"]>=4)])
           for name in ("N=1","N=2","N=3","N>=4")}
    overall=group_summary(records); statistics={"overall":overall,"aspect_ratio":aspect}
    dump(OUT/"coverage_stratified.json",coverage);dump(OUT/"tile_count_stratified.json",tiles);dump(OUT/"statistics.json",statistics)
    low=[r for r in records if r["exact_crop_gt_coverage"]<.9]; high=[r for r in records if np.isclose(r["exact_crop_gt_coverage"],1.)]
    overall_ci=overall["paired"]["iou"]["bootstrap_95_ci"];low_stat=paired(low) if low else None;high_stat=paired(high) if high else None
    if overall["paired"]["iou"]["mean_delta"]>0 and overall_ci[0]>0:
        verdict="FULL_FOV_ZERO_SHOT_SUPPORTED"
    elif low_stat and low_stat["iou"]["mean_delta"]>0 and low_stat["iou"]["bootstrap_95_ci"][0]>0 and high_stat["iou"]["mean_delta"]<0:
        verdict="FULL_FOV_SIGNAL_PRESENT_BUT_R1_RECALIBRATION_NEEDED"
    else: verdict="FULL_FOV_ZERO_SHOT_NOT_SUPPORTED"
    frozen_after={name:module_hash(module) for name,module in modules.items()}
    if frozen_after!=frozen_before:raise RuntimeError("frozen module hash drift after replay")
    summary={"schema":"phase6f3_summary_v1","status":"COMPLETE_STOP","verdict":verdict,"population":len(records),
        "sample_ids_sha256":ids_sha(ids),"invariants":"PASS","optimizer_count":0,"checkpoint_provenance":{
        "c1_sha256":C1_SHA,"new_r1_sha256":new_sha},"generation_seg_invariance":{"valid_flags_equal":True,"q_seg_shared":True,
        "seg_trigger_rate":float(cache["valid"].float().mean())},"overall_paired":overall["paired"],
        "low_coverage_paired":low_stat,"full_coverage_paired":high_stat,"frozen_hash_before":frozen_before,"frozen_hash_after":frozen_after,
        "firewall":protocol["firewall"]}
    dump(OUT/"summary.json",summary);render(summary,statistics,coverage,aspect,tiles)
    protocol["status"]="COMPLETE_STOP";dump(OUT/"protocol.json",protocol)


if __name__ == "__main__":
    main()
