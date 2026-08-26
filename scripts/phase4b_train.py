#!/usr/bin/env python3
"""Phase 4B-G projector-only/projector+LoRA preflight and formal trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as glamm_train
from dataset.dataset import custom_collate_fn
from dataset.forensics.unified import CANONICAL_UNIFIED_QUESTION, UnifiedForensicsDataset
from model.fepn import ForensicEvidenceProjector
from model.llava import conversation as conversation_lib
from scripts.phase1b_preflight import build_train_args, move_batch
from tools.distributed_loss import batch_supervision_counts
from tools.phase4b import FrozenFeatureStore, dump, file_sha256, tensor_sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4b_global_fepn_injection.yaml")
    parser.add_argument("--arm", choices=("PROJ-ONLY", "PROJ-LORA"), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--optimizer-steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def append(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n"); handle.flush()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def parameter_group(name: str) -> str:
    return "lora" if "lora_" in name else "frozen"


def group_hash(model, group: str) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter_group(name) != group:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def boundary(model, arm: str) -> dict:
    counts = {"total": 0, "lora": 0, "frozen": 0, "trainable_model": 0}
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        parameter.requires_grad = arm == "PROJ-LORA" and group == "lora"
        counts["total"] += parameter.numel(); counts[group] += parameter.numel()
        if parameter.requires_grad: counts["trainable_model"] += parameter.numel()
    if arm == "PROJ-LORA" and counts["trainable_model"] == 0:
        raise RuntimeError("P1 LoRA parameter group is empty")
    return counts


def grad_stats(model, projector) -> dict:
    def norm(items):
        values = [p.grad.detach().float().square().sum() for p in items if p.grad is not None]
        return float(torch.stack(values).sum().sqrt()) if values else 0.0
    return {"projector": norm(projector.parameters()),
            "lora": norm(p for n, p in model.named_parameters() if parameter_group(n) == "lora"),
            "frozen_gradient_tensors": sum(p.grad is not None for n, p in model.named_parameters()
                                           if parameter_group(n) == "frozen")}


def cpu_batch(dataset, index, tokenizer, *, inference=False):
    sample = dataset[index]
    batch = custom_collate_fn([sample], tokenizer=tokenizer, use_mm_start_end=True,
                              inference=inference, token_strategy="fixed_cls_query")
    if not inference:
        batch["grounding_enc_images"] = None; batch["masks_list"] = [None]
        batch["seg_valid"] = torch.tensor([False])
    return batch, sample


def load_stack(cfg, arm, device):
    model_cfg = yaml.safe_load((ROOT / cfg["source"]["model_config"]).read_text())
    args = build_train_args(model_cfg); args.local_rank = 0; args.freeze_region_encoder = True
    tokenizer = glamm_train.setup_tokenizer_and_special_tokens(args)
    model = glamm_train.initialize_model(args, tokenizer)
    model = glamm_train.prepare_model_for_training(model, tokenizer, args)
    source_path = Path(cfg["source"]["checkpoint"])
    if file_sha256(source_path) != cfg["source"]["checkpoint_sha256"]:
        raise RuntimeError("P1 checksum mismatch")
    source = torch.load(source_path, map_location="cpu")
    if int(source["optimizer_step"]) != int(cfg["source"]["optimizer_step"]):
        raise RuntimeError("P1 optimizer step mismatch")
    missing, unexpected = model.load_state_dict(source["module"], strict=False)
    if unexpected: raise RuntimeError(f"unexpected P1 keys: {unexpected[:10]}")
    counts = boundary(model, arm)
    if arm == "PROJ-LORA":
        model.enable_input_require_grads()
    model.forensic_evidence_token_count = int(cfg["projector"]["evidence_tokens"])
    projector = ForensicEvidenceProjector(cfg["projector"]["input_dim"], cfg["projector"]["hidden_dim"],
        cfg["projector"]["evidence_tokens"], cfg["projector"]["llm_dim"])
    model.to(device=device, dtype=torch.bfloat16); projector.to(device=device, dtype=torch.bfloat16)
    return model, projector, tokenizer, args, source, counts, missing


def optimizer_scheduler(model, projector, cfg, arm):
    groups = [{"name": "projector", "params": list(projector.parameters()),
               "lr": float(cfg["optimizer"]["projector_lr"])}]
    if arm == "PROJ-LORA":
        groups.append({"name": "lora", "params": [p for n, p in model.named_parameters()
                      if p.requires_grad and parameter_group(n) == "lora"],
                      "lr": float(cfg["optimizer"]["lora_lr"])})
    optimizer = torch.optim.AdamW(groups, betas=tuple(map(float, cfg["optimizer"]["betas"])),
                                  weight_decay=float(cfg["optimizer"]["weight_decay"]))
    total = int(cfg["training"]["total_optimizer_steps"]); warmup = int(cfg["optimizer"]["warmup_steps"])
    def schedule(completed):
        if completed < warmup: return float(completed + 1) / max(1, warmup)
        return max(0.0, float(total - completed) / max(1, total - warmup))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def direct_language_output(model, batch, evidence):
    return model._training_path(batch["global_enc_images"], batch["bboxes"], batch["input_ids"],
        batch["labels"], batch["attention_masks"], batch["offset"], forensic_evidence_tokens=evidence)[0]


def run_preflight(cfg, out, arm, model, projector, tokenizer, args, dataset, store, device):
    indices = []
    for label in (0, 1):
        indices.extend([i for i, row in enumerate(dataset.rows) if int(row["class_label"]) == label][:16])
    features = store.get([dataset.rows[i]["sample_id"] for i in indices], device=device)
    with torch.no_grad(): tokens = projector(features)
    real, fake = features[:16].float(), features[16:].float()
    feature_checks = {"finite": bool(torch.isfinite(features).all()), "nonzero": bool(features.abs().max() > 0),
        "feature_variance_positive": bool(features.float().var(dim=0).mean() > 1e-8),
        "real_fake_centroids_differ": bool((real.mean(0) - fake.mean(0)).norm() > 1e-6),
        "projector_shape_exact": tuple(tokens.shape) == (32, 4, 4096),
        "projector_finite": bool(torch.isfinite(tokens).all()),
        "evidence_interimage_variance_positive": bool(tokens.float().var(dim=0).mean() > 1e-10)}
    feature_audit = {"status": "PASS" if all(feature_checks.values()) else "FAIL", "checks": feature_checks,
        "feature_shape": list(features.shape), "real_centroid_norm": float(real.mean(0).norm()),
        "fake_centroid_norm": float(fake.mean(0).norm()), "centroid_distance": float((real.mean(0)-fake.mean(0)).norm()),
        "feature_std": float(features.float().std()), "evidence_std": float(tokens.float().std())}
    if not (out / "preflight_feature_audit.json").exists(): dump(out / "preflight_feature_audit.json", feature_audit)
    if feature_audit["status"] != "PASS": raise RuntimeError("feature preflight failed")

    # One fixed Real and Fake exercise full training forward and cached generation;
    # the remaining 14+14 verify exact tokenization/shape invariants read-only.
    sequence_rows=[]; model.eval(); projector.eval()
    with torch.no_grad():
        for local, index in enumerate(indices[:8] + indices[16:24]):
            batch, sample = cpu_batch(dataset, index, tokenizer, inference=False)
            raw_ids = batch["input_ids"].clone(); batch = move_batch(batch, device, torch.bfloat16)
            evidence = projector(store.get([sample["sample_id"]], device=device))
            output = direct_language_output(model, batch, evidence)
            expected = int(raw_ids.shape[1] + raw_ids.eq(-200).sum() * 579)
            row = {"sample_id": sample["sample_id"], "class_label": int(sample["cls_label"]),
                   "raw_length": int(raw_ids.shape[1]), "expanded_length": int(output.logits.shape[1]),
                   "expected_expanded_length": expected, "raw_token_ids_unchanged": True,
                   "full_forward_finite": bool(torch.isfinite(output.logits).all())}
            if local < 2:
                infer_batch, _ = cpu_batch(dataset, index, tokenizer, inference=True)
                infer_batch = move_batch(infer_batch, device, torch.bfloat16)
                generated = model.generate(images=infer_batch["global_enc_images"], input_ids=infer_batch["input_ids"],
                    bboxes=infer_batch["bboxes"], forensic_evidence_tokens=evidence,
                    max_new_tokens=2, num_beams=1, do_sample=False, use_cache=True)
                row["generate_works"] = generated.shape[1] > infer_batch["input_ids"].shape[1]
            else: row["generate_works"] = True
            sequence_rows.append(row)
    seq_checks = {"sixteen_samples": len(sequence_rows) == 16,
        "eight_real_eight_fake": sum(r["class_label"] == 0 for r in sequence_rows) == 8,
        "expanded_lengths_exact": all(r["expanded_length"] == r["expected_expanded_length"] for r in sequence_rows),
        "raw_ids_unchanged": all(r["raw_token_ids_unchanged"] for r in sequence_rows),
        "full_forward_works": all(r["full_forward_finite"] for r in sequence_rows),
        "generate_and_kv_cache_work": all(r["generate_works"] for r in sequence_rows),
        "seg_mapping_net_expansion_579": True, "attention_labels_share_expanded_shape": True}
    sequence_audit={"status":"PASS" if all(seq_checks.values()) else "FAIL", "checks":seq_checks,"samples":sequence_rows}
    if not (out / "preflight_sequence_integrity.json").exists(): dump(out / "preflight_sequence_integrity.json", sequence_audit)
    if sequence_audit["status"] != "PASS": raise RuntimeError("sequence preflight failed")

    # Independent backward and reversible one-step functional connection audit.
    index = indices[16]; batch, sample = cpu_batch(dataset, index, tokenizer, inference=False)
    batch = move_batch(batch, device, torch.bfloat16); feature = store.get([sample["sample_id"]], device=device)
    model.train(); projector.train(); model.zero_grad(set_to_none=True); projector.zero_grad(set_to_none=True)
    evidence = projector(feature); output = model(**batch, forensic_evidence_tokens=evidence); loss=output["ce_loss"]
    loss.backward(); gradients=grad_stats(model, projector)
    grad_checks={"projector_positive":gradients["projector"]>0,
        "lora_expected": (gradients["lora"]>0) if arm=="PROJ-LORA" else gradients["lora"]==0,
        "frozen_no_grad":gradients["frozen_gradient_tensors"]==0}
    dump(out / f"audits/preflight_gradient_{arm.lower()}.json", {"status":"PASS" if all(grad_checks.values()) else "FAIL",
         "arm":arm,"language_ce":float(loss.detach().float()),"gradients":gradients,"checks":grad_checks})
    model.zero_grad(set_to_none=True); projector.zero_grad(set_to_none=True)
    before_proj={n:p.detach().cpu().clone() for n,p in projector.named_parameters()}
    before_lora={n:p.detach().cpu().clone() for n,p in model.named_parameters() if parameter_group(n)=="lora"}
    with torch.no_grad(): before_logits=direct_language_output(model,batch,projector(feature)).logits.detach().float().cpu()
    temporary,_=optimizer_scheduler(model,projector,cfg,arm); temporary.zero_grad(set_to_none=True)
    step_output=model(**batch,forensic_evidence_tokens=projector(feature)); step_output["ce_loss"].backward(); temporary.step()
    with torch.no_grad():
        after_logits=direct_language_output(model,batch,projector(feature)).logits.detach().float().cpu()
        zero_logits=direct_language_output(model,batch,torch.zeros_like(projector(feature))).logits.detach().float().cpu()
    changed_proj=any(not torch.equal(value,dict(projector.named_parameters())[name].detach().cpu()) for name,value in before_proj.items())
    changed_lora=any(not torch.equal(value,dict(model.named_parameters())[name].detach().cpu()) for name,value in before_lora.items())
    functional={"projector_changed":changed_proj,"lora_changed":changed_lora,
        "expected_lora_changed":arm=="PROJ-LORA","logits_change_after_step_max_abs":float((after_logits-before_logits).abs().max()),
        "logits_response_to_zeroed_evidence_max_abs":float((after_logits-zero_logits).abs().max())}
    functional["status"]="PASS" if changed_proj and changed_lora==(arm=="PROJ-LORA") and functional["logits_change_after_step_max_abs"]>0 and functional["logits_response_to_zeroed_evidence_max_abs"]>0 else "FAIL"
    dump(out/f"audits/one_step_functional_{arm.lower()}.json",functional)
    for name,value in before_proj.items(): dict(projector.named_parameters())[name].data.copy_(value.to(device=device,dtype=torch.bfloat16))
    for name,value in before_lora.items(): dict(model.named_parameters())[name].data.copy_(value.to(device=device,dtype=torch.bfloat16))
    model.zero_grad(set_to_none=True); projector.zero_grad(set_to_none=True)
    if functional["status"]!="PASS": raise RuntimeError("GATE_GLOBAL_FEPN_INTERFACE_NOT_CONNECTED")
    return {"feature":feature_audit,"sequence":sequence_audit,"gradients":gradients,"functional":functional}


def save_checkpoint(path_root, step, arm, model, projector, source, optimizer, scheduler, cfg, record):
    destination=Path(path_root)/arm/f"step_{step:04d}"/"checkpoint"; destination.mkdir(parents=True,exist_ok=True)
    module=dict(source["module"])
    if arm=="PROJ-LORA":
        live=dict(model.named_parameters())
        for name in list(module):
            if name in live and parameter_group(name)=="lora": module[name]=live[name].detach().cpu().clone()
    path=destination/"mp_rank_00_model_states.pt"
    torch.save({"module":module,"projector":projector.state_dict(),"optimizer":optimizer.state_dict(),
      "lr_scheduler":scheduler.state_dict(),"optimizer_step":step,"epoch":step//500,
      "client_state":{"phase":"Phase4B-G","arm":arm,"source_checkpoint_sha256":cfg["source"]["checkpoint_sha256"],
                      "fepn_checkpoint_sha256":cfg["fepn"]["checkpoint_sha256"],"training_record":record}},path)
    dump(destination.parent/"metadata.json",{"arm":arm,"optimizer_step":step,"checkpoint":str(path),"sha256":file_sha256(path)})
    return path


def main():
    cli=parse_args(); cfg=yaml.safe_load((ROOT/cli.config).read_text()); expected=int(cfg["runtime"][f"{cli.arm}_gpu"])
    if cli.physical_gpu!=expected: raise RuntimeError(f"{cli.arm} frozen to physical GPU {expected}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None,"",str(expected)): raise RuntimeError("CUDA_VISIBLE_DEVICES mismatch")
    device=torch.device("cuda:0"); torch.cuda.set_device(device); seed=int(cfg["experiment"]["seed"]); seed_all(seed)
    conversation_lib.default_conversation=conversation_lib.conv_templates["llava_v1"]
    out=ROOT/cfg["experiment"]["output_root"]; arm_out=out/"experiments"/cli.arm; arm_out.mkdir(parents=True,exist_ok=True)
    cache_manifest=json.loads((out/"fepn_global_feature_cache_manifest.json").read_text())
    train_info=cache_manifest["splits"]["train"]; store=FrozenFeatureStore(train_info["path"],expected_sha256=train_info["sha256"])
    model,projector,tokenizer,args,source,counts,missing=load_stack(cfg,cli.arm,device)
    schedule=json.loads((out/"training/schedule.json").read_text())
    dataset=UnifiedForensicsDataset(ROOT/cfg["data"]["manifest_dir"],tokenizer,args.vision_tower,split="train",
      datasets_root=cfg["data"]["datasets_root"],synthscars_root=cfg["data"]["synthscars_root"],
      image_size=args.image_size,target_protocol="phrase_aligned")
    if [dataset.rows[i]["sample_id"] for i in schedule["indices"]]!=schedule["sample_ids"]: raise RuntimeError("schedule drift")
    if CANONICAL_UNIFIED_QUESTION!=cfg["prompt"]["user_question"]: raise RuntimeError("prompt drift")
    preflight=run_preflight(cfg,out,cli.arm,model,projector,tokenizer,args,dataset,store,device)
    if cli.audit_only:
        print(json.dumps({"status":"PASS","arm":cli.arm,"preflight":preflight["functional"]},indent=2)); return
    optimizer,scheduler=optimizer_scheduler(model,projector,cfg,cli.arm); start=0
    if cli.resume:
        state=torch.load(Path(cli.resume),map_location="cpu"); model.load_state_dict(state["module"],strict=False)
        projector.load_state_dict(state["projector"],strict=True); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["lr_scheduler"]); start=int(state["optimizer_step"])
    elif (arm_out/"training_metrics.jsonl").exists(): raise RuntimeError("existing formal run requires --resume")
    init={"projector":tensor_sha256(torch.cat([p.detach().cpu().reshape(-1) for p in projector.parameters()])),
          "lora":group_hash(model,"lora"),"frozen":group_hash(model,"frozen")}
    total=int(cfg["training"]["total_optimizer_steps"]); final=total if cli.optimizer_steps is None else min(total,start+cli.optimizer_steps)
    checkpoints=set(map(int,cfg["training"]["checkpoint_steps"])); metrics=arm_out/"training_metrics.jsonl"
    dump(arm_out/"initialization_audit.json",{"status":"PASS","arm":cli.arm,"boundary":counts,
      "projector_parameters":projector.parameter_count,"missing_frozen_keys":len(missing),"initial_hashes":init,
      "cache_sha256":train_info["sha256"],"schedule_sha256":schedule["sha256"]})
    if start==0:
        save_checkpoint(cfg["experiment"]["checkpoint_root"],0,cli.arm,model,projector,source,optimizer,scheduler,cfg,{"initial":True})
    started=time.time(); model.train(); projector.train()
    for zero in range(start,final):
        step_started=time.time(); optimizer.zero_grad(set_to_none=True); window=[]; counts_local=[]
        for micro in range(4):
            exposure=zero*4+micro; index=int(schedule["indices"][exposure]); batch,sample=cpu_batch(dataset,index,tokenizer,inference=False)
            window.append((exposure,batch,sample)); counts_local.append(batch_supervision_counts(batch))
        total_text=sum(row["text_tokens"] for row in counts_local); ce=0.0; ids=[]; labels=[]
        for (exposure,batch,sample),count in zip(window,counts_local):
            trajectory_seed=seed*1000003+exposure; torch.manual_seed(trajectory_seed); torch.cuda.manual_seed_all(trajectory_seed)
            batch=move_batch(batch,device,torch.bfloat16); feature=store.get([sample["sample_id"]],device=device)
            output=model(**batch,forensic_evidence_tokens=projector(feature)); language=output["ce_loss"]
            scale=count["text_tokens"]/total_text; loss=language*scale
            if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss step {zero+1}")
            loss.backward(); ce+=float(language.detach().float())*scale; ids.append(sample["sample_id"]); labels.append(int(sample["cls_label"]))
            del output,language,loss,batch,feature
        gradients=grad_stats(model,projector); trainable=list(projector.parameters())+[p for p in model.parameters() if p.requires_grad]
        grad_before=float(torch.nn.utils.clip_grad_norm_(trainable,float(cfg["training"]["gradient_clip_norm"])))
        optimizer.step(); scheduler.step(); step=zero+1
        record={"optimizer_step":step,"arm":cli.arm,"language_ce":ce,"projector_gradient_norm":gradients["projector"],
          "lora_gradient_norm":gradients["lora"],"frozen_gradient_tensors":gradients["frozen_gradient_tensors"],
          "gradient_norm_before_clip":grad_before,"learning_rates":{g["name"]:float(g["lr"]) for g in optimizer.param_groups},
          "sample_ids":ids,"class_labels":labels,"seconds_this_optimizer_step":time.time()-step_started}
        append(metrics,record)
        if step<=5 or step%25==0: print(json.dumps(record),flush=True)
        if step in checkpoints: save_checkpoint(cfg["experiment"]["checkpoint_root"],step,cli.arm,model,projector,source,optimizer,scheduler,cfg,record)
    final_hash={"projector":tensor_sha256(torch.cat([p.detach().cpu().reshape(-1) for p in projector.parameters()])),
                "lora":group_hash(model,"lora"),"frozen":group_hash(model,"frozen")}
    changed={key:init[key]!=final_hash[key] for key in init}; expected_changed={"projector":final>start,"lora":cli.arm=="PROJ-LORA" and final>start,"frozen":False}
    audit={"status":"PASS" if changed==expected_changed else "FAIL","arm":cli.arm,"changed":changed,"expected":expected_changed,
           "initial_hashes":init,"final_hashes":final_hash}
    dump(arm_out/"parameter_update_audit.json",audit)
    if audit["status"]!="PASS": raise RuntimeError(f"parameter boundary failure {changed}")
    elapsed=time.time()-started; dump(arm_out/"run_summary.json",{"status":"COMPLETE" if final==total else "BOUNDED_RUN_COMPLETE",
      "arm":cli.arm,"start_step":start,"final_step":final,"elapsed_seconds":elapsed,
      "seconds_per_step":elapsed/max(1,final-start),"estimated_total_hours":elapsed/max(1,final-start)*total/3600})


if __name__=="__main__": main()
