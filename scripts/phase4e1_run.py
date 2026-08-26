#!/usr/bin/env python3
"""Formal Stage T/Stage S runner and validation evaluator for Phase 4E-1."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.clip_forensic_adapter import CLIPSpatialArm
from model.tf_fdg import (
    AssignmentWeights, TFFDGStudent, clone_frozen_teacher,
    hungarian_teacher_assignment, kd_losses, region_aware_slot_loss,
    slot_collapse_diagnostics,
)
from tools.phase4e1 import (
    FrozenStore, append, compare, deterministic_order, dump, file_sha256,
    metric, rows, summarize, tensor_state_sha256,
)

CFG = yaml.safe_load((ROOT / "configs/phase4e1_tf_fdg_full_method.yaml").read_text())
OUT = ROOT / CFG["experiment"]["output_root"]
CKPT = Path(CFG["experiment"]["checkpoint_root"])


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "stage_t", "qualify", "stage_s", "evaluate"), required=True)
    parser.add_argument("--arm", choices=("full", "no_teacher", "no_forensic", "k1", "full_clip", "no_kd", "no_rectification", "mask_logit_only"), required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_all(seed: int = 3407) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def topology(arm: str) -> dict:
    return {
        "slots": 1 if arm == "k1" else 4,
        "use_forensic": arm != "no_forensic",
        "use_rectification": arm != "no_rectification" and arm != "no_forensic",
        "evidence": "clip" if arm == "full_clip" else "forensic",
        "teacher_parent": "full" if arm in {"no_kd", "mask_logit_only"} else arm,
    }


def common_model(arm: str, device: torch.device) -> TFFDGStudent:
    spec = topology(arm); seed_all(3407)
    model = TFFDGStudent(
        slots=spec["slots"], gamma_init=float(CFG["architecture"]["rectification_gamma"]),
        use_forensic=spec["use_forensic"], use_rectification=spec["use_rectification"],
    )
    return model.to(device)


def evidence_model(kind: str, device: torch.device) -> CLIPSpatialArm:
    blocks = 0 if kind == "clip" else 3
    key = "clip_checkpoint" if kind == "clip" else "forensic_checkpoint"
    expected_key = key + "_sha256"
    path = Path(CFG["evidence"][key])
    if file_sha256(path) != CFG["evidence"][expected_key]:
        raise RuntimeError("4C-A evidence checkpoint hash mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if int(value["epoch"]) != 4:
        raise RuntimeError("4C-A selected epoch mismatch")
    model = CLIPSpatialArm(blocks=blocks); model.load_state_dict(value["model"])
    return model.to(device).eval().requires_grad_(False)


def forensic_feature(source: CLIPSpatialArm, raw_clip: torch.Tensor, kind: str) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        value = source(raw_clip, return_features=True)
    return (value["F0"] if kind == "clip" else value["F_forensic"]).detach()


def mask_loss(output: dict, targets: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    terms = []
    for index, target in enumerate(targets):
        value = region_aware_slot_loss(output["slot_logits"][index], target.to(device=device, dtype=torch.float32))
        terms.append(value["slot_loss"] + value["union_loss"])
    return torch.stack(terms).mean()


def model_forward(model, hidden, sam, evidence, sam_coord, clip_coord):
    return model(hidden, sam, evidence, sam_coord, clip_coord, torch.ones(evidence.shape[0], 576, dtype=torch.bool, device=evidence.device))


def optimizer_t(model):
    return torch.optim.AdamW(model.parameters(), lr=float(CFG["stage_t"]["learning_rate"]), weight_decay=float(CFG["stage_t"]["weight_decay"]), betas=(0.9, 0.999))


def scheduler_t(optimizer):
    total = int(CFG["stage_t"]["optimizer_steps"]); warm = round(total * float(CFG["stage_t"]["warmup_fraction"])); floor = float(CFG["stage_t"]["minimum_learning_rate"]) / float(CFG["stage_t"]["learning_rate"])
    def scale(step):
        if step < warm: return float(step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total - warm))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def optimizer_s(model):
    low, high = [], []
    for name, parameter in model.named_parameters():
        if name.startswith(("semantic_pyramid", "forensic_pyramid", "rectification")):
            low.append(parameter)
        else:
            high.append(parameter)
    return torch.optim.AdamW([
        {"name": "qg_decoder_head", "params": high, "lr": float(CFG["stage_s"]["qg_decoder_head_lr"])},
        {"name": "pyramid_rectification", "params": low, "lr": float(CFG["stage_s"]["pyramid_rectification_lr"])},
    ], weight_decay=float(CFG["stage_s"]["weight_decay"]), betas=(0.9, 0.999))


def scheduler_s(optimizer, total_updates: int):
    warm = round(total_updates * float(CFG["stage_s"]["warmup_fraction"])); base = [group["lr"] for group in optimizer.param_groups]
    def scale(step):
        if step < warm: return float(step + 1) / max(1, warm)
        progress = min(1.0, (step - warm) / max(1, total_updates - warm))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale), base


def save_checkpoint(path: Path, model, optimizer, scheduler, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"schema": "phase4e1_tf_fdg_checkpoint_v1", "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "scheduler": scheduler.state_dict() if scheduler else None, **metadata}, temporary)
    temporary.replace(path)


def load_selected_teacher(arm: str, device: torch.device):
    parent = topology(arm)["teacher_parent"]
    selector = json.loads((OUT / "selectors" / f"{parent}_teacher.json").read_text())
    model = common_model(parent, device)
    state = torch.load(selector["selected_checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"]); model.eval().requires_grad_(False)
    return model, selector


def evaluate_model(model, source, store: FrozenStore, mode: str, device: torch.device, *, condition: str = "matched"):
    model.eval(); records_out, collapse_rows = [], []
    ids = list(store.sample_ids); cross = {sid: ids[(index + 1) % len(ids)] for index, sid in enumerate(ids)}
    permutation = torch.randperm(576, generator=torch.Generator().manual_seed(3407)).to(device)
    with torch.no_grad():
        for position, sid in enumerate(ids, 1):
            if mode == "g0" and sid not in store.g0_hidden:
                original = store.original_masks[sid]
                records_out.append({"sample_id": sid, "foreground_iou": 0.0, "foreground_f1": 0.0,
                                    "background_iou": float((~original).sum() / original.numel()), "fg_bg_miou": 0.0,
                                    "tp": 0, "fp": 0, "fn": int(original.sum()), "tn": int((~original).sum()),
                                    "valid_g0": False})
                continue
            spatial_sid = cross[sid] if condition == "cross_image" else sid
            sam, raw_clip, sam_coord, clip_coord = store.spatial_batch([sid], device)
            if spatial_sid != sid:
                _, raw_clip, _, _ = store.spatial_batch([spatial_sid], device)
            evidence = forensic_feature(source, raw_clip, topology("full_clip" if source.forensic_blocks.__len__() == 0 else "full")["evidence"])
            if condition == "spatial_shuffle": evidence = evidence.flatten(2)[:, :, permutation].reshape_as(evidence)
            elif condition == "zero": evidence = torch.zeros_like(evidence)
            hidden = store.hidden_batch([sid], mode, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model_forward(model, hidden, sam, evidence, sam_coord, clip_coord)
            row = metric(sid, output["union_logit"][0, 0].float().cpu(), store.original_masks[sid], store.geometries[sid])
            row["valid_g0"] = True; records_out.append(row)
            if model.slots > 1:
                diag = slot_collapse_diagnostics(output["slot_logits"], output["queries"], output["forensic_attention"][-1])
                collapse_rows.append({"sample_id": sid, **{key: value.detach().float().cpu().tolist() for key, value in diag.items()}})
            if position % 100 == 0: print(json.dumps({"eval": mode, "condition": condition, "done": position, "total": len(ids)}), flush=True)
    return summarize(records_out), records_out, collapse_rows


def preflight(arm: str, device: torch.device):
    model = common_model(arm, device); source = evidence_model(topology(arm)["evidence"], device)
    store = FrozenStore(CFG, "train")
    ids = store.sample_ids[:8]; sam, raw, sc, cc = store.spatial_batch(ids, device); evidence = forensic_feature(source, raw, topology(arm)["evidence"])
    torch.cuda.reset_peak_memory_stats(device)
    hidden = store.hidden_batch(ids, "tf", device); model.train(); model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16): output = model_forward(model, hidden, sam, evidence, sc, cc)
    loss = mask_loss(output, [store.targets[sid] for sid in ids], device); loss.backward()
    gradients = {name: float(parameter.grad.float().norm()) for name, parameter in model.named_parameters() if parameter.grad is not None}
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    forward_flops = None
    try:
        with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_flops=True) as profile:
            with torch.autocast("cuda", dtype=torch.bfloat16): model_forward(model, hidden, sam, evidence, sc, cc)
        forward_flops = int(sum(event.flops for event in profile.key_averages() if event.flops))
    except Exception:
        forward_flops = None
    result = {"status": "PASS" if torch.isfinite(loss) and gradients else "FAIL", "arm": arm,
              "loss": float(loss.detach()), "gradient_tensor_count": len(gradients),
              "all_gradients_finite": all(math.isfinite(x) for x in gradients.values()),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "profile_batch_size": len(ids), "forward_flops_profiled": forward_flops,
              "peak_allocated_bytes_forward_backward": peak_memory, "formal_optimizer_updates": 0,
              "internal_test_accessed": False, "official1000_accessed": False}
    dump(OUT / "preflight" / f"{arm}_integrated_preflight.json", result)
    if result["status"] != "PASS": raise RuntimeError("integrated preflight failed")
    print(json.dumps(result, indent=2))


def stage_t(arm: str, device: torch.device):
    if arm == "no_teacher": raise RuntimeError("no_teacher has no Stage T")
    store = FrozenStore(CFG, "train"); val = FrozenStore(CFG, "val", keep_original=True)
    model = common_model(arm, device); source = evidence_model(topology(arm)["evidence"], device)
    optimizer = optimizer_t(model); scheduler = scheduler_t(optimizer); root = CKPT / arm / "teacher"
    history_path = OUT / "training" / arm / "teacher_history.json"
    history = []
    if not (root / "epoch_0.pt").exists():
        save_checkpoint(root / "epoch_0.pt", model, optimizer, scheduler, {"arm": arm, "stage": "T", "epoch": 0, "global_step": 0})
    else:
        existing = sorted(root.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1])); state = torch.load(existing[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        if history_path.exists(): history = json.loads(history_path.read_text())
    start_epoch = int(state["epoch"]) if 'state' in locals() else 0; global_step = int(state["global_step"]) if 'state' in locals() else 0
    if not any(row["epoch"] == 0 for row in history):
        metrics, records_out, _ = evaluate_model(model, source, val, "tf", device)
        history.append({"epoch": 0, "global_step": 0, "validation": metrics, "train": None})
        dump(history_path, history); dump(OUT / "evaluation" / arm / "teacher_epoch_0_metrics.json", metrics)
        (OUT / "evaluation" / arm).mkdir(parents=True, exist_ok=True)
        (OUT / "evaluation" / arm / "teacher_epoch_0_predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records_out))
    for epoch in range(start_epoch + 1, 6):
        began = time.time(); model.train(); order = deterministic_order(store.sample_ids, 3407, epoch); sums = 0.0; seen = 0
        for begin in range(0, len(order), 8):
            ids = order[begin:begin+8]; sam, raw, sc, cc = store.spatial_batch(ids, device); evidence = forensic_feature(source, raw, topology(arm)["evidence"]); hidden = store.hidden_batch(ids, "tf", device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): output = model_forward(model, hidden, sam, evidence, sc, cc)
            loss = mask_loss(output, [store.targets[sid] for sid in ids], device)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite Stage T loss")
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(CFG["stage_t"]["gradient_clip_norm"]))
            if not torch.isfinite(norm): raise RuntimeError("nonfinite Stage T gradient")
            optimizer.step(); scheduler.step(); global_step += 1; sums += float(loss.detach()) * len(ids); seen += len(ids)
            if global_step <= 5 or global_step % 50 == 0: print(json.dumps({"arm": arm, "stage": "T", "epoch": epoch, "step": global_step, "loss": float(loss.detach()), "lr": optimizer.param_groups[0]["lr"]}), flush=True)
        if seen != 8836 or global_step != epoch * 1105: raise RuntimeError("Stage T exposure/step mismatch")
        metrics, records_out, _ = evaluate_model(model, source, val, "tf", device)
        row = {"epoch": epoch, "global_step": global_step, "train": {"mean_loss": sums/seen, "exposures": seen}, "validation": metrics, "seconds": time.time()-began}
        history.append(row); dump(history_path, history); dump(OUT / "evaluation" / arm / f"teacher_epoch_{epoch}_metrics.json", metrics)
        (OUT / "evaluation" / arm / f"teacher_epoch_{epoch}_predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records_out))
        save_checkpoint(root / f"epoch_{epoch}.pt", model, optimizer, scheduler, {"arm": arm, "stage": "T", "epoch": epoch, "global_step": global_step, "validation": metrics})
    best = max(history, key=lambda row: (row["validation"]["mean_foreground_iou"], -row["epoch"]))
    selected = root / f"epoch_{best['epoch']}.pt"
    selector = {"status": "COMPLETE", "arm": arm, "primary": "validation Fake canonical TF mean FG IoU", "candidates": history, "selected_epoch": best["epoch"], "selected_checkpoint": str(selected), "selected_metrics": best["validation"], "internal_test_used": False, "official1000_used": False}
    dump(OUT / "selectors" / f"{arm}_teacher.json", selector); print(json.dumps(selector, indent=2))


def qualify(arm: str, device: torch.device):
    teacher, selector = load_selected_teacher(arm, device); val = FrozenStore(CFG, "val", keep_original=True); source = evidence_model(topology(arm)["evidence"], device)
    conditions = {}
    for condition in ("matched", "cross_image", "spatial_shuffle", "zero"):
        metrics, records_out, _ = evaluate_model(teacher, source, val, "tf", device, condition=condition)
        conditions[condition] = {"metrics": metrics, "records": records_out}
        root = OUT / "qualification" / arm; root.mkdir(parents=True, exist_ok=True)
        (root / f"{condition}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records_out))
    baseline = []
    for path in sorted((ROOT / "outputs/phase4c_b_evidence_reader/cache/validation/tf_full_context").glob("shard_*.pt")):
        baseline.extend(torch.load(path, map_location="cpu", weights_only=False)["records"])
    baseline = [{"sample_id": r["sample_id"], "foreground_iou": r["foreground_iou"], "foreground_f1": r["foreground_f1"]} for r in baseline]
    matched = conditions["matched"]["records"]
    delta = conditions["matched"]["metrics"]["mean_foreground_iou"] - float(np.mean([r["foreground_iou"] for r in baseline]))
    capability = "ADEQUATE" if delta >= -0.010 else ("WEAK" if delta >= -0.030 else "FAILED")
    image_stats = compare(matched, conditions["cross_image"]["records"])["foreground_iou"]
    spatial_stats = compare(matched, conditions["spatial_shuffle"]["records"])["foreground_iou"]
    state = lambda stats: "TRUE" if stats["bootstrap_95_ci"][0] > 0 else ("FALSE" if stats["bootstrap_95_ci"][1] <= 0 else "INCONCLUSIVE")
    result = {"status": "COMPLETE", "arm": arm, "selected_teacher": selector, "conditions": {key:value["metrics"] for key,value in conditions.items()},
              "teacher_vs_p1_tf_mean_iou_delta": delta, "TEACHER_MASK_CAPABILITY": capability,
              "TEACHER_IMAGE_SPECIFIC_USE": state(image_stats), "TEACHER_SPATIAL_SPECIFIC_USE": state(spatial_stats),
              "matched_vs_cross": image_stats, "matched_vs_spatial_shuffle": spatial_stats,
              "kd_enable": {"relation": capability != "FAILED", "attention": capability != "FAILED" and state(spatial_stats) == "TRUE", "feature": capability != "FAILED", "logit": capability == "ADEQUATE"},
              "teacher_reselected": False, "internal_test_accessed": False, "official1000_accessed": False}
    dump(OUT / "qualification" / arm / "qualification.json", result); print(json.dumps(result, indent=2))


def student_loss(student_output, teacher_output, targets, device, kd_enable, mode):
    losses = {"mask": mask_loss(student_output, targets, device)}
    if teacher_output is not None and mode != "no_kd":
        values = {key: [] for key in ("relation", "attention", "feature", "logit")}
        for index in range(student_output["slot_logits"].shape[0]):
            s = {key: ([item[index:index+1] for item in value] if isinstance(value, list) else value[index:index+1]) for key, value in student_output.items()}
            t = {key: ([item[index:index+1] for item in value] if isinstance(value, list) else value[index:index+1]) for key, value in teacher_output.items()}
            weight = AssignmentWeights(attention=float(CFG["loss"]["assignment_attention"]) if kd_enable["attention"] else 0.0)
            permutation = hungarian_teacher_assignment(s["slot_logits"][0], t["slot_logits"][0], s["forensic_attention"][-1][0].mean(0), t["forensic_attention"][-1][0].mean(0), weight)
            current = kd_losses(s, t, permutation, float(CFG["loss"]["temperature"]))
            for key in values: values[key].append(current[key])
        losses.update({key: torch.stack(value).mean() for key, value in values.items()})
    total = losses["mask"]
    if teacher_output is not None and mode != "no_kd":
        allowed = {"relation", "attention", "feature", "logit"} if mode != "mask_logit_only" else {"logit"}
        for key in allowed:
            if kd_enable[key]: total = total + float(CFG["loss"][key]) * losses[key]
    return total, losses


def stage_s(arm: str, device: torch.device):
    store = FrozenStore(CFG, "train"); val = FrozenStore(CFG, "val", keep_original=True); spec = topology(arm); source = evidence_model(spec["evidence"], device)
    if arm == "no_teacher":
        model = common_model(arm, device); teacher = None; kd_enable = {key: False for key in ("relation","attention","feature","logit")}; copy_hash = None
    else:
        teacher, _ = load_selected_teacher(arm, device); qualification = json.loads((OUT / "qualification" / spec["teacher_parent"] / "qualification.json").read_text())
        if qualification["TEACHER_MASK_CAPABILITY"] == "FAILED": raise RuntimeError("teacher FAILED; Stage S prohibited")
        model = common_model(arm, device); model.load_state_dict(teacher.state_dict(), strict=True); kd_enable = qualification["kd_enable"]
        copy_hash = tensor_state_sha256(model.state_dict())
        if copy_hash != tensor_state_sha256(teacher.state_dict()): raise RuntimeError("teacher/student copy hash mismatch")
    optimizer = optimizer_s(model); total_updates = 11050; scheduler, _ = scheduler_s(optimizer, total_updates); root = CKPT / arm / "student"; history_path = OUT / "training" / arm / "student_history.json"; history=[]
    if not (root / "epoch_0.pt").exists(): save_checkpoint(root / "epoch_0.pt", model, optimizer, scheduler, {"arm": arm, "stage": "S", "epoch": 0, "optimizer_updates": 0, "copy_hash": copy_hash})
    else:
        existing=sorted(root.glob("epoch_*.pt"),key=lambda p:int(p.stem.split("_")[-1])); state=torch.load(existing[-1],map_location="cpu",weights_only=False); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        if history_path.exists(): history=json.loads(history_path.read_text())
    start_epoch=int(state["epoch"]) if 'state' in locals() else 0; updates=int(state["optimizer_updates"]) if 'state' in locals() else 0
    if not any(row["epoch"]==0 for row in history):
        metrics,records_out,_=evaluate_model(model,source,val,"g0",device); history.append({"epoch":0,"optimizer_updates":0,"validation":metrics,"train":None}); dump(history_path,history)
        root_eval=OUT/"evaluation"/arm;root_eval.mkdir(parents=True,exist_ok=True);(root_eval/"student_epoch_0_predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records_out));dump(root_eval/"student_epoch_0_metrics.json",metrics)
    valid_set=set(store.g0_hidden)
    for epoch in range(start_epoch+1,11):
        began=time.time();model.train();order=deterministic_order(store.sample_ids,3407,epoch); traversal=eligible=invalid=empty=0;sums={key:0.0 for key in ("total","mask","relation","attention","feature","logit")};batch_log=OUT/"training"/arm/f"student_epoch_{epoch}_batches.jsonl"
        for begin in range(0,len(order),8):
            all_ids=order[begin:begin+8];ids=[sid for sid in all_ids if sid in valid_set];n_total=len(all_ids);n_valid=len(ids);traversal+=n_total;eligible+=n_valid;invalid+=n_total-n_valid
            record={"epoch":epoch,"dataloader_step":begin//8+1,"n_total":n_total,"n_valid_g0":n_valid,"n_invalid_g0":n_total-n_valid}
            if not ids:
                empty+=1;record["status"]="EMPTY_VALID_G0_BATCH";append(batch_log,record);continue
            sam,raw,sc,cc=store.spatial_batch(ids,device);evidence=forensic_feature(source,raw,spec["evidence"]);hidden=store.hidden_batch(ids,"g0",device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16): teacher_output=None if teacher is None else model_forward(teacher,store.hidden_batch(ids,"tf",device),sam,evidence,sc,cc)
            with torch.autocast("cuda",dtype=torch.bfloat16): student_output=model_forward(model,hidden,sam,evidence,sc,cc)
            total,losses=student_loss(student_output,teacher_output,[store.targets[sid] for sid in ids],device,kd_enable,arm)
            if not torch.isfinite(total):raise RuntimeError("nonfinite Stage S loss")
            total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(CFG["stage_s"]["gradient_clip_norm"]))
            if not torch.isfinite(norm):raise RuntimeError("nonfinite Stage S gradient")
            optimizer.step();scheduler.step();updates+=1;record.update({"status":"UPDATED","optimizer_update":updates,"total_loss":float(total.detach())});append(batch_log,record)
            sums["total"]+=float(total.detach())*n_valid
            for key,value in losses.items():sums[key]+=float(value.detach())*n_valid
            if updates<=5 or updates%50==0:print(json.dumps({"arm":arm,"stage":"S","epoch":epoch,"update":updates,"n_valid":n_valid,"loss":float(total.detach())}),flush=True)
        if traversal!=8836 or eligible!=8690 or invalid!=146:raise RuntimeError("conditional exposure mismatch")
        metrics,records_out,_=evaluate_model(model,source,val,"g0",device);row={"epoch":epoch,"optimizer_updates":updates,"train":{key:value/eligible for key,value in sums.items()}|{"traversal_exposures":traversal,"optimization_eligible_exposures":eligible,"invalid_g0_exposures":invalid,"empty_valid_g0_batches":empty},"validation":metrics,"seconds":time.time()-began};history.append(row);dump(history_path,history)
        root_eval=OUT/"evaluation"/arm;(root_eval/f"student_epoch_{epoch}_predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records_out));dump(root_eval/f"student_epoch_{epoch}_metrics.json",metrics)
        save_checkpoint(root/f"epoch_{epoch}.pt",model,optimizer,scheduler,{"arm":arm,"stage":"S","epoch":epoch,"optimizer_updates":updates,"copy_hash":copy_hash,"validation":metrics,"kd_enable":kd_enable})
    best=max(history,key=lambda row:(row["validation"]["mean_foreground_iou"],-row["epoch"]));selected=root/f"epoch_{best['epoch']}.pt";selector={"status":"COMPLETE","arm":arm,"primary":"all 1106 validation Fake canonical G0 mean FG IoU","detection_tie_break":"frozen P1 exact and therefore constant","final_tie_break":"earlier epoch","selected_epoch":best["epoch"],"selected_checkpoint":str(selected),"selected_metrics":best["validation"],"candidates":history,"train_valid_g0":8690,"formal_validation_population":1106,"internal_test_used":False,"official1000_used":False};dump(OUT/"selectors"/f"{arm}_student.json",selector);print(json.dumps(selector,indent=2))


def evaluate_selected(arm: str, device: torch.device):
    selector=json.loads((OUT/"selectors"/f"{arm}_student.json").read_text());model=common_model(arm,device);model.load_state_dict(torch.load(selector["selected_checkpoint"],map_location="cpu",weights_only=False)["model"]);source=evidence_model(topology(arm)["evidence"],device);val=FrozenStore(CFG,"val",keep_original=True);root=OUT/"final"/arm;root.mkdir(parents=True,exist_ok=True)
    result={};records_by={}
    for mode,condition in [("g0","matched"),("phrase","matched"),("tf","matched"),("g0","cross_image"),("g0","spatial_shuffle"),("g0","zero")]:
        metrics,records_out,collapse=evaluate_model(model,source,val,mode,device,condition=condition);name=("tf_full" if mode=="tf" else ("phrase_only" if mode=="phrase" else condition));result[name]=metrics;records_by[name]=records_out;(root/f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records_out));dump(root/f"{name}_metrics.json",metrics)
        if name=="matched":
            diagnostic=[row for row in records_out if row.get("valid_g0")]
            result["valid_g0_only_diagnostic"] = summarize(diagnostic)
            result["valid_g0_only_diagnostic"]["label"] = "VALID-G0-ONLY N=1,078; not headline canonical G0"
            dump(root/"slot_collapse_raw.json",collapse)
            if collapse:
                collapsed=[]
                for row in collapse:
                    active=int(np.asarray(row["active_slot_count"]).reshape(-1)[0])
                    overlap=np.asarray(row["mask_pairwise_soft_iou"])[0]
                    contribution=np.asarray(row["union_contribution"])[0]
                    offdiag=overlap[~np.eye(overlap.shape[0],dtype=bool)]
                    collapsed.append(active<=1 or (float(offdiag.mean())>=0.90 and int((contribution<0.001).sum())>=3))
                fraction=float(np.mean(collapsed)); state="TRUE" if fraction>=0.8 else ("PARTIAL" if fraction>=0.4 else "FALSE")
                result["slot_collapse"]={"SLOT_COLLAPSE":state,"collapsed_fraction":fraction,"n":len(collapsed)}
    result["matched_vs_cross"]=compare(records_by["matched"],records_by["cross_image"]);result["matched_vs_spatial_shuffle"]=compare(records_by["matched"],records_by["spatial_shuffle"])
    detection=json.loads((ROOT/"outputs/phase4b_global_fepn_injection/detection_metrics.json").read_text())["P1"]
    result["canonical_detection"]={**detection,"invariance":"exact by architecture: TF-FDG runs strictly downstream of frozen P1 generation/classification"}
    dump(root/"summary.json",result);print(json.dumps(result,indent=2))


def main():
    args=cli();device=torch.device(args.device);torch.cuda.set_device(device);seed_all()
    if args.mode=="preflight":preflight(args.arm,device)
    elif args.mode=="stage_t":stage_t(args.arm,device)
    elif args.mode=="qualify":qualify(args.arm,device)
    elif args.mode=="stage_s":stage_s(args.arm,device)
    else:evaluate_selected(args.arm,device)


if __name__=="__main__":main()
