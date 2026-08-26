#!/usr/bin/env python3
"""Phase 4A mandatory preflight and three-epoch standalone FEPN-v0 training."""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.fepn import FEPNv0
from tools.phase3c1 import binary_metrics, inverse_logits, probe_loss, summarize
from tools.phase4a import (
    BalancedBatchSampler, Phase4ADataset, classification_metrics, file_sha256,
    phase4a_collate, tensor_state_sha256,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4a_fepn_evidence_learnability.yaml")
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--physical-gpu", type=int, default=0)
    return parser.parse_args()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def model_and_device(config: dict, physical_gpu: int):
    expected = int(config["experiment"]["gpu"])
    if physical_gpu != expected:
        raise RuntimeError(f"Phase 4A is frozen to physical GPU {expected}, got {physical_gpu}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", str(physical_gpu)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES conflicts with requested physical GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    seed_all(int(config["experiment"]["seed"]))
    model = FEPNv0(config["preprocess"]["image_mean"], config["preprocess"]["image_std"]).to(device)
    return model, device


def group_grad_norm(model: torch.nn.Module, prefixes: tuple[str, ...]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes) and parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum())
    return total ** 0.5


ENCODER_PREFIXES = (
    "rgb_stem", "residual_stem", "fusion", "level1", "level2", "level3",
    "lateral1", "lateral2", "lateral3", "smooth1", "smooth2",
)


def optimizer_scheduler(model, config):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["optimizer"]["learning_rate"]),
        betas=tuple(map(float, config["optimizer"]["betas"])),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    warmup = int(config["optimizer"]["warmup_steps"])
    total = int(config["training"]["total_steps"])
    def schedule(completed):
        if completed < warmup:
            return float(completed + 1) / max(1, warmup)
        progress = float(completed - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def preflight(config, output: Path, model, device):
    train = Phase4ADataset(config, "train", return_original=True)
    val_ids = {str(row["sample_id"]) for row in Phase4ADataset(config, "val").rows}
    real = [i for i, row in enumerate(train.rows) if int(row["class_label"]) == 0][:16]
    fake = [i for i, row in enumerate(train.rows) if int(row["class_label"]) == 1][:16]
    indices = [value for pair in zip(real, fake) for value in pair]
    rows = [train[index] for index in indices]
    batch = phase4a_collate(rows)
    images = batch["images"].to(device)
    labels = batch["labels"].to(device)
    fake_positions = batch["fake_positions"].to(device)
    targets = batch["dense_targets"].to(device)
    sample_ids = batch["sample_ids"]
    integrity = {
        "sample_count": len(rows) == 32, "real_count": int((labels == 0).sum()) == 16,
        "fake_count": int((labels == 1).sum()) == 16,
        "unique_sample_ids": len(set(sample_ids)) == 32,
        "no_train_val_id_overlap": not (set(sample_ids) & val_ids),
        "image_shape_exact": tuple(images.shape[1:]) == (3, 336, 336),
        "fake_target_shape_exact": tuple(targets.shape[1:]) == (1, 336, 336),
        "fake_targets_nonempty": bool(targets.flatten(1).sum(1).gt(0).all()),
        "real_dense_targets_absent": all(row["dense_target"] is None for row in rows if row["label"] == 0),
        "files_resolve": all(Path(row["image_path"]).is_file() for row in rows),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"Phase 4A data preflight failed: {[k for k,v in integrity.items() if not v]}")

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        initial_output = model(images)
    residual = initial_output["residual_view"].float()
    residual_stats = {
        "mean": float(residual.mean()), "std": float(residual.std()),
        "min": float(residual.min()), "max": float(residual.max()),
        "finite": bool(torch.isfinite(residual).all()), "all_zero": bool(residual.abs().max() == 0),
        "near_constant": bool(residual.std() < 1e-6),
    }
    outputs = {
        "global_logits_finite": bool(torch.isfinite(initial_output["global_logits"]).all()),
        "dense_logits_finite": bool(torch.isfinite(initial_output["dense_logits"]).all()),
        "dense_feature_shape": list(initial_output["dense_features"].shape),
        "dense_logit_shape": list(initial_output["dense_logits"].shape),
    }

    model.train(); model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        value = model(images)
        global_loss = F.binary_cross_entropy_with_logits(value["global_logits"].float(), labels)
    global_loss.backward()
    global_grads = {
        "global_head": group_grad_norm(model, ("global_head",)),
        "encoder_and_stems": group_grad_norm(model, ENCODER_PREFIXES),
        "dense_head": group_grad_norm(model, ("dense_head",)),
    }
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        value = model(images)
        dense_losses = probe_loss(value["dense_logits"].index_select(0, fake_positions).float(), targets)
    dense_losses["total"].backward()
    dense_grads = {
        "dense_head": group_grad_norm(model, ("dense_head",)),
        "encoder_and_stems": group_grad_norm(model, ENCODER_PREFIXES),
        "global_head": group_grad_norm(model, ("global_head",)),
    }
    gradient_checks = {
        "global_loss_reaches_global_head": global_grads["global_head"] > 0,
        "global_loss_reaches_encoder_and_stems": global_grads["encoder_and_stems"] > 0,
        "global_loss_does_not_reach_dense_head": global_grads["dense_head"] == 0,
        "dense_loss_reaches_dense_head": dense_grads["dense_head"] > 0,
        "dense_loss_reaches_encoder_and_stems": dense_grads["encoder_and_stems"] > 0,
        "dense_loss_does_not_reach_global_head": dense_grads["global_head"] == 0,
    }
    model.zero_grad(set_to_none=True)

    saved = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    parameter_before = tensor_state_sha256(dict(model.named_parameters()))
    residual_before = tensor_state_sha256(dict(model.residual_operator.named_buffers()))
    optimizer, _ = optimizer_scheduler(model, config)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        value = model(images)
        one_global = F.binary_cross_entropy_with_logits(value["global_logits"].float(), labels)
        one_dense = probe_loss(value["dense_logits"].index_select(0, fake_positions).float(), targets)["total"]
        one_total = one_global + one_dense
    one_total.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    parameter_after = tensor_state_sha256(dict(model.named_parameters()))
    residual_after = tensor_state_sha256(dict(model.residual_operator.named_buffers()))
    one_step = {
        "trainable_parameters_changed": parameter_after != parameter_before,
        "fixed_residual_operator_unchanged": residual_after == residual_before,
        "loss_finite": bool(torch.isfinite(one_total)),
        "parameter_hash_before": parameter_before, "parameter_hash_after": parameter_after,
        "residual_hash_before": residual_before, "residual_hash_after": residual_after,
    }
    model.load_state_dict(saved, strict=True)
    restored = tensor_state_sha256(dict(model.named_parameters())) == parameter_before
    one_step["initial_state_restored_exactly"] = restored

    residual_ok = residual_stats["finite"] and not residual_stats["all_zero"] and not residual_stats["near_constant"]
    status = "PASS" if all(integrity.values()) and residual_ok and all(outputs[k] for k in ("global_logits_finite", "dense_logits_finite")) and all(gradient_checks.values()) and all(one_step[k] for k in ("trainable_parameters_changed", "fixed_residual_operator_unchanged", "loss_finite", "initial_state_restored_exactly")) else "FAIL"
    data_audit = {"status": status, "sample_ids": sample_ids, "integrity": integrity,
                  "residual_operator": "fixed reflected-padding depthwise Laplacian", "residual_stats": residual_stats,
                  "outputs": outputs, "test_loaded": False, "official1000_loaded": False}
    gradient_audit = {"status": status, "global_loss": float(global_loss.detach()), "dense_loss": float(dense_losses["total"].detach()),
                      "global_gradient_norms": global_grads, "dense_gradient_norms": dense_grads,
                      "checks": gradient_checks, "one_step": one_step}
    dump(output / "preflight_data_audit.json", data_audit)
    dump(output / "preflight_gradient_audit.json", gradient_audit)
    dump(output / "audits/preflight_data_audit.json", data_audit)
    dump(output / "audits/preflight_gradient_audit.json", gradient_audit)
    if status != "PASS":
        raise RuntimeError("GATE_PHASE4A_PREFLIGHT_FAILED")
    print(json.dumps({"status": status, "residual_stats": residual_stats, "global_grads": global_grads,
                      "dense_grads": dense_grads, "one_step": one_step}, indent=2))


def validation(model, config, device, epoch: int, output: Path):
    dataset = Phase4ADataset(config, "val", return_original=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=int(config["training"]["num_workers"]),
                        collate_fn=phase4a_collate, pin_memory=True, persistent_workers=True)
    model.eval(); class_logits=[]; class_labels=[]; dense_records=[]; low_logits=[]; dense_ids=[]
    feature_sum = feature_sq = None; feature_count = 0; spatial_variances=[]; pooled_norms={0:[],1:[]}
    with torch.no_grad():
        for batch in loader:
            images=batch["images"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value=model(images)
            logits=value["global_logits"].float().cpu(); labels=batch["labels"].numpy()
            class_logits.extend(logits.tolist()); class_labels.extend(labels.tolist())
            features=value["dense_features"].float()
            current_sum=features.sum(dim=(0,2,3)).cpu(); current_sq=features.square().sum(dim=(0,2,3)).cpu()
            feature_sum=current_sum if feature_sum is None else feature_sum+current_sum
            feature_sq=current_sq if feature_sq is None else feature_sq+current_sq
            feature_count += features.shape[0]*features.shape[2]*features.shape[3]
            spatial_variances.extend(features.var(dim=(2,3),unbiased=False).mean(dim=1).cpu().tolist())
            pooled=features.mean(dim=(2,3)).norm(dim=1).cpu().tolist()
            for label,norm in zip(labels.astype(int).tolist(),pooled): pooled_norms[label].append(norm)
            for local,label in enumerate(labels.astype(int).tolist()):
                if label != 1: continue
                low=value["dense_logits"][local,0].float().cpu()
                original=batch["original_masks"][local]
                original_logits=inverse_logits(low, batch["geometries"][local])
                metric=binary_metrics(original_logits, original)
                sid=batch["sample_ids"][local]
                dense_records.append({"sample_id":sid,"epoch":epoch,**metric})
                low_logits.append(low.to(torch.float16)); dense_ids.append(sid)
    classification=classification_metrics(np.asarray(class_logits),np.asarray(class_labels))
    dense=summarize(dense_records)
    totals={key:int(sum(row[key] for row in dense_records)) for key in ("tp","fp","fn","tn")}
    dense["global_foreground_iou"] = totals["tp"] / max(1, totals["tp"]+totals["fp"]+totals["fn"])
    dense["global_foreground_f1"] = 2*totals["tp"] / max(1, 2*totals["tp"]+totals["fp"]+totals["fn"])
    safety=classification["accuracy"]>=float(config["evaluation"]["global_safety_accuracy_min"]) and classification["roc_auc"]>=float(config["evaluation"]["global_safety_roc_auc_min"])
    mean=feature_sum/feature_count; variance=(feature_sq/feature_count-mean.square()).clamp_min(0)
    feature_statistics={"epoch":epoch,"channel_mean_mean":float(mean.mean()),"channel_mean_std":float(mean.std()),
      "channel_std_mean":float(variance.sqrt().mean()),"spatial_variance_mean":float(np.mean(spatial_variances)),
      "pooled_norm_real_mean":float(np.mean(pooled_norms[0])),"pooled_norm_fake_mean":float(np.mean(pooled_norms[1])),
      "finite":bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()),
      "noncollapsed":bool(float(variance.sqrt().mean())>1e-6 and float(np.mean(spatial_variances))>1e-8)}
    prediction_path=output/f"evaluation/epoch_{epoch}_predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row)+"\n" for row in dense_records),encoding="utf-8")
    torch.save({"sample_ids":dense_ids,"low_res_logits":torch.stack(low_logits)},output/f"evaluation/epoch_{epoch}_low_res_logits.pt")
    dump(output/f"evaluation/epoch_{epoch}_metrics.json",{"epoch":epoch,"dense":dense,"classification":classification,"global_safety_pass":safety,"feature_statistics":feature_statistics})
    return {"epoch":epoch,"dense":dense,"classification":classification,"global_safety_pass":safety,"feature_statistics":feature_statistics,
            "predictions":str(prediction_path),"low_res_logits":str(output/f"evaluation/epoch_{epoch}_low_res_logits.pt")}


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, global_step, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                "epoch":epoch,"global_step":global_step,"config":config},path)
    return file_sha256(path)


def train(config, output: Path, model, device):
    preflight_path=output/"preflight_gradient_audit.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text())["status"]!="PASS":
        raise RuntimeError("PASS Phase 4A preflight required")
    checkpoint_root=Path(config["experiment"]["checkpoint_root"])
    optimizer,scheduler=optimizer_scheduler(model,config); start_epoch=0; global_step=0
    metrics_path=output/"checkpoint_metrics.json"; checkpoint_metrics=[]
    last=checkpoint_root/"last.pt"
    if last.is_file():
        state=torch.load(last,map_location="cpu"); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start_epoch=int(state["epoch"]); global_step=int(state["global_step"])
        if metrics_path.is_file(): checkpoint_metrics=json.loads(metrics_path.read_text())["checkpoints"]
        checkpoint_metrics=[row for row in checkpoint_metrics if int(row["epoch"])<=start_epoch]
        if (output/"training/training_metrics.jsonl").is_file():
            log_path=output/"training/training_metrics.jsonl"
            kept=[line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip() and int(json.loads(line)["epoch"])<=start_epoch]
            log_path.write_text(("\n".join(kept)+"\n") if kept else "",encoding="utf-8")
        print(json.dumps({"resume_epoch":start_epoch,"global_step":global_step}),flush=True)
    if start_epoch==0 and not checkpoint_metrics:
        path=checkpoint_root/"epoch_0.pt"; digest=save_checkpoint(path,model,optimizer,scheduler,0,0,config)
        row=validation(model,config,device,0,output); row.update({"checkpoint":str(path),"sha256":digest,"role":"random_initialization_control"})
        checkpoint_metrics.append(row); dump(metrics_path,{"status":"IN_PROGRESS","checkpoints":checkpoint_metrics})
        save_checkpoint(last,model,optimizer,scheduler,0,0,config)
    train_dataset=Phase4ADataset(config,"train",return_original=False)
    training_log=output/"training/training_metrics.jsonl"
    max_epochs=int(config["training"]["max_epochs"]); seed=int(config["experiment"]["seed"])
    started=time.time()
    for epoch in range(start_epoch+1,max_epochs+1):
        sampler=BalancedBatchSampler(train_dataset.rows,int(config["training"]["batch_size"]),seed,epoch)
        if len(sampler)!=int(config["training"]["steps_per_epoch"]): raise RuntimeError("steps_per_epoch mismatch")
        loader=DataLoader(train_dataset,batch_sampler=sampler,num_workers=int(config["training"]["num_workers"]),
                          collate_fn=phase4a_collate,pin_memory=True,persistent_workers=True)
        model.train(); epoch_started=time.time(); counts={"real":0,"fake":0,"dense":0}
        sums={"global":0.0,"dense":0.0,"total":0.0}
        for step_in_epoch,batch in enumerate(loader,1):
            images=batch["images"].to(device,non_blocking=True); labels=batch["labels"].to(device,non_blocking=True)
            fake_positions=batch["fake_positions"].to(device,non_blocking=True); targets=batch["dense_targets"].to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                value=model(images); global_loss=F.binary_cross_entropy_with_logits(value["global_logits"].float(),labels)
                dense_loss=probe_loss(value["dense_logits"].index_select(0,fake_positions).float(),targets)["total"]
                total=global_loss+dense_loss
            total.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float(config["training"]["gradient_clip_norm"])))
            if not all(math.isfinite(value) for value in (float(global_loss.detach()),float(dense_loss.detach()),float(total.detach()),grad)):
                safety={"status":"SAFETY_STOP","category":"NaN_or_nonfinite_or_gradient_explosion","epoch":epoch,"step_in_epoch":step_in_epoch,"optimizer_step":global_step,
                        "global_loss":float(global_loss.detach()),"dense_loss":float(dense_loss.detach()),"total_loss":float(total.detach()),"gradient_norm":grad}
                dump(output/"training/safety_stop.json",safety); raise RuntimeError("PHASE4A_SAFETY_STOP_NONFINITE")
            optimizer.step(); scheduler.step(); global_step+=1
            real_count=int((labels==0).sum()); fake_count=int((labels==1).sum())
            counts["real"]+=real_count; counts["fake"]+=fake_count; counts["dense"]+=fake_count
            sums["global"]+=float(global_loss.detach())*len(labels); sums["dense"]+=float(dense_loss.detach())*fake_count; sums["total"]+=float(total.detach())*len(labels)
            record={"epoch":epoch,"step_in_epoch":step_in_epoch,"optimizer_step":global_step,"real_count":real_count,"fake_count":fake_count,
                    "dense_supervised_count":fake_count,"global_loss":float(global_loss.detach()),"dense_loss":float(dense_loss.detach()),
                    "total_loss":float(total.detach()),"gradient_norm_before_clip":grad,"learning_rate":float(optimizer.param_groups[0]["lr"])}
            append(training_log,record)
            if step_in_epoch<=3 or step_in_epoch%25==0: print(json.dumps(record),flush=True)
        if counts!={"real":8836,"fake":8836,"dense":8836}: raise RuntimeError(f"epoch population mismatch: {counts}")
        path=checkpoint_root/f"epoch_{epoch}.pt"; digest=save_checkpoint(path,model,optimizer,scheduler,epoch,global_step,config)
        save_checkpoint(last,model,optimizer,scheduler,epoch,global_step,config)
        row=validation(model,config,device,epoch,output); row.update({"checkpoint":str(path),"sha256":digest,"role":"formal_candidate",
             "training":{"real":counts["real"],"fake":counts["fake"],"dense":counts["dense"],"mean_global_loss":sums["global"]/17672,
                         "mean_dense_loss":sums["dense"]/8836,"mean_total_loss":sums["total"]/17672,"seconds":time.time()-epoch_started}})
        severe_collapse=(not row["feature_statistics"]["noncollapsed"] and row["classification"].get("logit_std",1.0)<1e-6)
        if severe_collapse:
            safety={"status":"SAFETY_STOP","category":"severe_collapse","epoch":epoch,"optimizer_step":global_step,
                    "feature_statistics":row["feature_statistics"],"classification":row["classification"]}
            dump(output/"training/safety_stop.json",safety); raise RuntimeError("PHASE4A_SAFETY_STOP_SEVERE_COLLAPSE")
        checkpoint_metrics=[x for x in checkpoint_metrics if int(x["epoch"])!=epoch]+[row]
        checkpoint_metrics.sort(key=lambda x:int(x["epoch"])); dump(metrics_path,{"status":"IN_PROGRESS","checkpoints":checkpoint_metrics})
        print(json.dumps({"epoch_complete":epoch,"metrics":row},ensure_ascii=False),flush=True)
    eligible=[row for row in checkpoint_metrics if row["global_safety_pass"]]
    pool=eligible if eligible else checkpoint_metrics
    selected=max(pool,key=lambda row:(row["dense"]["mean_foreground_iou"],row["dense"]["mean_foreground_f1"]))
    selector={"status":"COMPLETE","primary":"validation Fake mean foreground IoU","tie_break":"mean foreground F1",
              "global_safety":{"accuracy_min":config["evaluation"]["global_safety_accuracy_min"],"roc_auc_min":config["evaluation"]["global_safety_roc_auc_min"],
                               "eligible_epochs":[row["epoch"] for row in eligible]},
              "selected_epoch":selected["epoch"],"selected_checkpoint":selected["checkpoint"],"selected_sha256":selected["sha256"],
              "selected_dense":selected["dense"],"selected_classification":selected["classification"],"safety_pass":selected["global_safety_pass"]}
    dump(output/"selector_result.json",selector); dump(output/"checkpoint_metrics.json",{"status":"COMPLETE","checkpoints":checkpoint_metrics})
    dump(output/"training/run_summary.json",{"status":"COMPLETE","epochs":max_epochs,"optimizer_steps":global_step,"image_exposures":max_epochs*17672,
                                               "elapsed_seconds_this_invocation":time.time()-started,"selected_epoch":selected["epoch"]})


def main():
    args=parse_args(); config=yaml.safe_load((ROOT/args.config).read_text(encoding="utf-8")); output=ROOT/config["experiment"]["output_root"]
    model,device=model_and_device(config,args.physical_gpu)
    if args.mode=="preflight": preflight(config,output,model,device)
    else: train(config,output,model,device)


if __name__=="__main__": main()
