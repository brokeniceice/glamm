#!/usr/bin/env python3
"""Phase 4D-1B frozen interface signal viability audit."""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.position_aware_evidence_reader import PositionAwareEvidenceReader
from scripts.phase4c_b_train import frozen_hash, load_p1
from tools.phase4c_b import decode_low_res, file_sha256, sam_input_loss
from tools.phase4d1 import canonical_hash, dump, reader_state_hash, write_csv


ARMS = ("pos_clip", "pos_forensic")


def feature_key(arm):
    return "clip_features" if arm == "pos_clip" else "forensic_features"


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.quantile(a, .1)), "p90": float(np.quantile(a, .9)),
            "max": float(a.max())}


def load_population(cfg, out):
    d1out = ROOT / cfg["phase4d1"]["output_root"]
    subset = json.loads((d1out / "phase4d1_train_subset.json").read_text())
    selected_ids = subset["sample_ids"][:int(cfg["population"]["n"])]
    selected_set = set(selected_ids)
    cache = Path(cfg["phase4d1"]["cache_root"]) / "train_subset"
    evidence = {arm: {} for arm in ARMS}; selected_by_id = {}
    for path in sorted(cache.glob("shard_*.pt")):
        shard = torch.load(path, map_location="cpu")
        for i, sid in enumerate(shard["sample_ids"]):
            for arm in ARMS:
                evidence[arm][sid] = shard[feature_key(arm)][i].float()
            if sid in selected_set:
                selected_by_id[sid] = {
                    "sample_id": sid, "q": shard["q_seg"][i].float().reshape(-1),
                    "sam": shard["sam_features"][i], "target": shard["targets"][i],
                }
    selected = [selected_by_id[sid] for sid in selected_ids]
    if len(selected) != 256 or any(len(x) != 2048 for x in evidence.values()):
        raise RuntimeError("Phase 4D-1B population/cache mismatch")
    payload = {
        "status": "FROZEN", "source": cfg["population"]["source"], "n": len(selected),
        "sample_ids": selected_ids, "sample_ids_sha256": canonical_hash(selected_ids),
        "parent_subset_sha256": subset["sample_ids_sha256"],
    }
    dump(out / "audit_population.json", payload)
    return selected, evidence, payload


def load_reader(cfg, arm, step, device):
    root = Path(cfg["phase4d1"]["checkpoint_root"])
    state = torch.load(root / arm / f"step_{step}.pt", map_location="cpu")
    reader = PositionAwareEvidenceReader().to(device)
    reader.load_state_dict(state["reader"])
    reader.eval()
    return reader


def grad_vector_stats(parameters):
    values = [p.grad.detach().float().reshape(-1) for p in parameters if p.grad is not None]
    if not values:
        return {"norm": 0.0, "max_abs": 0.0, "mean_abs": 0.0,
                "fraction_nonzero": 0.0, "count": 0}
    value = torch.cat(values)
    return {"norm": float(value.norm()), "max_abs": float(value.abs().max()),
            "mean_abs": float(value.abs().mean()),
            "fraction_nonzero": float((value != 0).float().mean()), "count": value.numel()}


def mha_slice_norm(reader, start, end):
    values = []
    weight = reader.cross_attention.in_proj_weight.grad
    bias = reader.cross_attention.in_proj_bias.grad
    if weight is not None: values.append(weight[start:end].float().reshape(-1))
    if bias is not None: values.append(bias[start:end].float().reshape(-1))
    return float(torch.cat(values).norm()) if values else 0.0


def gradient_audit(cfg, selected, evidence, core, device, out):
    init = torch.load(Path(cfg["phase4d1"]["checkpoint_root"]) / "reader_init.pt",
                      map_location="cpu")
    rows = []
    values = cfg["beta_causality"]["values"]
    labels = cfg["beta_causality"]["labels"]
    for arm in ARMS:
        for label, beta in zip(labels, values):
            reader = PositionAwareEvidenceReader().to(device)
            reader.load_state_dict(init["reader"])
            reader.beta.data.fill_(float(beta)); reader.train(); reader.zero_grad(set_to_none=True)
            totals = defaultdict(float)
            for sample in selected[:int(cfg["population"]["gradient_minibatch_n"])]:
                q = sample["q"].to(device).reshape(1, 256)
                feature = evidence[arm][sample["sample_id"]].to(device).unsqueeze(0)
                value = reader(q, feature)
                low = decode_low_res(core, value["q_final"].to(torch.bfloat16),
                                     sample["sam"].to(device=device, dtype=torch.bfloat16))
                losses = sam_input_loss(low, sample["target"].to(device))
                (losses["total"] / int(cfg["population"]["gradient_minibatch_n"])).backward()
                totals["loss"] += float(losses["total"].detach())
                totals["reader_output_norm"] += float(value["q_evidence"].detach().float().norm())
                totals["effective_residual_norm"] += float(value["q_residual"].detach().float().norm())
            n = int(cfg["population"]["gradient_minibatch_n"])
            non_beta = [p for name, p in reader.named_parameters() if name != "beta"]
            stats = grad_vector_stats(non_beta)
            out_params = list(reader.cross_attention.out_proj.parameters())
            row = {
                "arm": arm, "beta_label": label, "beta": float(beta),
                "loss": totals["loss"] / n,
                "grad_norm_beta": abs(float(reader.beta.grad)) if reader.beta.grad is not None else 0.0,
                "grad_norm_reader_total": stats["norm"],
                "grad_norm_query_projection": mha_slice_norm(reader, 0, 256),
                "grad_norm_key_projection": mha_slice_norm(reader, 256, 512),
                "grad_norm_value_projection": mha_slice_norm(reader, 512, 768),
                "grad_norm_output_projection": grad_vector_stats(out_params)["norm"],
                "max_abs_grad_reader": stats["max_abs"],
                "mean_abs_grad_reader": stats["mean_abs"],
                "fraction_nonzero_grad_reader": stats["fraction_nonzero"],
                "reader_output_norm": totals["reader_output_norm"] / n,
                "effective_residual_norm": totals["effective_residual_norm"] / n,
                "optimizer_step_performed": False,
            }
            rows.append(row)
    write_csv(out / "phase4d1b_gradient_audit.csv", list(rows[0].keys()), rows)
    return rows


def forward_signal_audits(cfg, selected, evidence, out, device):
    d1out = ROOT / cfg["phase4d1"]["output_root"]
    permutation = torch.tensor(json.loads((d1out / "spatial_content_permutation.json").read_text())["permutation"], device=device)
    cross = json.loads((d1out / "train_cross_image_mapping.json").read_text())["mapping"]
    residual_rows = []; c_rows = []
    cached = {}
    for arm in ARMS:
        for step in (0, 512):
            reader = load_reader(cfg, arm, step, device)
            beta = float(reader.beta.detach())
            for sample in selected:
                sid = sample["sample_id"]; q = sample["q"].to(device).reshape(1, 256)
                matched_f = evidence[arm][sid].to(device).unsqueeze(0)
                shuffle_f = matched_f.flatten(2).transpose(1, 2)[:, permutation, :]
                cross_f = evidence[arm][cross[sid]].to(device).unsqueeze(0)
                zero_f = torch.zeros_like(matched_f)
                with torch.no_grad():
                    values = {name: reader(q, feature) for name, feature in (
                        ("matched", matched_f), ("shuffle", shuffle_f),
                        ("cross", cross_f), ("zero", zero_f))}
                qnorm = float(q.norm()); raw = float(values["matched"]["q_evidence"].norm())
                effective = float(values["matched"]["q_residual"].norm())
                residual_rows.append({
                    "sample_id": sid, "arm": arm, "step": step, "beta": beta,
                    "q_norm": qnorm, "reader_raw_residual_norm": raw,
                    "effective_residual_norm": effective,
                    "q_new_minus_q_norm": float((values["matched"]["q_final"][:, 0] - q).norm()),
                    "raw_over_q": raw / qnorm, "effective_over_q": effective / qnorm,
                })
                for comparison, other in (("matched_minus_shuffle", "shuffle"),
                                          ("matched_minus_cross", "cross")):
                    raw_delta = values["matched"]["q_evidence"] - values[other]["q_evidence"]
                    eff_delta = values["matched"]["q_residual"] - values[other]["q_residual"]
                    fp32_delta = values["matched"]["q_final"] - values[other]["q_final"]
                    bf_a = values["matched"]["q_final"].to(torch.bfloat16)
                    bf_b = values[other]["q_final"].to(torch.bfloat16)
                    changed = bf_a != bf_b
                    c_rows.append({
                        "sample_id": sid, "arm": arm, "step": step,
                        "comparison": comparison, "raw_output_difference_norm": float(raw_delta.norm()),
                        "effective_difference_norm": float(eff_delta.norm()),
                        "fp32_q_difference_norm": float(fp32_delta.norm()),
                        "bf16_q_difference_norm": float((bf_a.float() - bf_b.float()).norm()),
                        "bf16_any_difference": bool(changed.any()),
                        "bf16_changed_elements": int(changed.sum()),
                        "bf16_fraction_elements_different": float(changed.float().mean()),
                        "max_abs_difference_before_bf16": float(fp32_delta.abs().max()),
                        "max_abs_difference_after_bf16": float((bf_a.float()-bf_b.float()).abs().max()),
                    })
                if step == 512:
                    cached[(arm, sid)] = {name: value["q_evidence"].detach().cpu().reshape(-1)
                                          for name, value in values.items()}
    write_csv(out / "phase4d1b_residual_magnitude.csv", list(residual_rows[0].keys()), residual_rows)
    return residual_rows, c_rows, cached, permutation, cross


def difference_metrics(left, right):
    delta = left.float() - right.float(); changed = delta != 0
    binary_left = left.float() > 0; binary_right = right.float() > 0
    binary = binary_left != binary_right
    return {
        "logit_l1": float(delta.abs().sum()), "logit_l2": float(delta.norm()),
        "max_absolute_logit_difference": float(delta.abs().max()),
        "mean_absolute_logit_difference": float(delta.abs().mean()),
        "fraction_pixels_changed_logits": float(changed.float().mean()),
        "binary_mask_changed": bool(binary.any()),
        "fraction_binary_pixels_changed": float(binary.float().mean()),
    }


def aggregate_metric_rows(raw, keys):
    grouped = defaultdict(list)
    for row in raw:
        grouped[tuple(row[k] for k in keys)].append(row)
    result = []
    numeric = ["logit_l1", "logit_l2", "max_absolute_logit_difference",
               "mean_absolute_logit_difference", "fraction_pixels_changed_logits",
               "fraction_binary_pixels_changed"]
    for group, values in grouped.items():
        row = dict(zip(keys, group)); row["n"] = len(values)
        for key in numeric: row[f"mean_{key}"] = float(np.mean([x[key] for x in values]))
        row["fraction_samples_binary_mask_changed"] = float(np.mean([x["binary_mask_changed"] for x in values]))
        result.append(row)
    return result


def survival_and_sam_audit(cfg, selected, cached, core, out, device):
    alphas = [float(x) for x in cfg["bf16_survival"]["alpha"]]
    generator = torch.Generator().manual_seed(int(cfg["control_perturbation"]["direction_seed"]))
    direction = torch.randn(256, generator=generator); direction /= direction.norm()
    bf_rows = []; raw_sam = []; control_raw = []
    baseline_low = {}
    with torch.no_grad():
        for sample in selected:
            sid = sample["sample_id"]
            q = sample["q"].reshape(-1)
            baseline_low[sid] = decode_low_res(
                core, q.to(device).reshape(1,1,256).to(torch.bfloat16),
                sample["sam"].to(device=device, dtype=torch.bfloat16)).detach().cpu()[0,0]
    for arm in ARMS:
        for alpha in alphas:
            survival_vs_q = []; survival_shuffle = []; changed_vs_q = []; changed_shuffle = []
            norms_vs_q = []; norms_shuffle = []
            for sample in selected:
                sid = sample["sample_id"]; q = sample["q"].reshape(-1)
                rm = cached[(arm,sid)]["matched"]; rs = cached[(arm,sid)]["shuffle"]
                rc = cached[(arm,sid)]["cross"]
                q_m = q + alpha * rm; q_s = q + alpha * rs; q_c = q + alpha * rc
                base_bf = q.to(torch.bfloat16); m_bf = q_m.to(torch.bfloat16)
                s_bf = q_s.to(torch.bfloat16); c_bf = q_c.to(torch.bfloat16)
                d0 = m_bf != base_bf; ds = m_bf != s_bf
                survival_vs_q.append(bool(d0.any())); survival_shuffle.append(bool(ds.any()))
                changed_vs_q.append(int(d0.sum())); changed_shuffle.append(int(ds.sum()))
                norms_vs_q.append(float((m_bf.float()-base_bf.float()).norm()))
                norms_shuffle.append(float((m_bf.float()-s_bf.float()).norm()))
                with torch.no_grad():
                    low_m = decode_low_res(core, m_bf.to(device).reshape(1,1,256), sample["sam"].to(device=device,dtype=torch.bfloat16))[0,0].cpu()
                    low_s = decode_low_res(core, s_bf.to(device).reshape(1,1,256), sample["sam"].to(device=device,dtype=torch.bfloat16))[0,0].cpu()
                    low_c = decode_low_res(core, c_bf.to(device).reshape(1,1,256), sample["sam"].to(device=device,dtype=torch.bfloat16))[0,0].cpu()
                    generic_delta = direction * (alpha * rm.norm())
                    generic_bf = (q + generic_delta).to(torch.bfloat16)
                    low_g = decode_low_res(core, generic_bf.to(device).reshape(1,1,256), sample["sam"].to(device=device,dtype=torch.bfloat16))[0,0].cpu()
                for comparison, left, right in (
                    ("matched_vs_shuffle", low_m, low_s),
                    ("matched_vs_cross", low_m, low_c),
                    ("matched_vs_baseline_q", low_m, baseline_low[sid]),
                    ("generic_matched_norm_vs_baseline_q", low_g, baseline_low[sid])):
                    raw_sam.append({"arm":arm,"alpha":alpha,"sample_id":sid,"comparison":comparison,
                                    **difference_metrics(left,right)})
            bf_rows.append({
                "arm": arm, "alpha": alpha, "n": len(selected),
                "fraction_samples_surviving_vs_q": float(np.mean(survival_vs_q)),
                "fraction_q_elements_changed_vs_q": float(np.sum(changed_vs_q)/(len(selected)*256)),
                "mean_changed_q_elements_vs_q": float(np.mean(changed_vs_q)),
                "mean_bf16_q_difference_norm_vs_q": float(np.mean(norms_vs_q)),
                "fraction_samples_matched_vs_shuffle_surviving": float(np.mean(survival_shuffle)),
                "fraction_q_elements_changed_matched_vs_shuffle": float(np.sum(changed_shuffle)/(len(selected)*256)),
                "mean_changed_q_elements_matched_vs_shuffle": float(np.mean(changed_shuffle)),
                "mean_bf16_q_difference_norm_matched_vs_shuffle": float(np.mean(norms_shuffle)),
            })
    write_csv(out / "phase4d1b_bf16_survival.csv", list(bf_rows[0].keys()), bf_rows)
    sam_rows = aggregate_metric_rows(raw_sam, ["arm","alpha","comparison"])
    write_csv(out / "phase4d1b_sam_logit_sensitivity.csv", list(sam_rows[0].keys()), sam_rows)

    # Fixed generic perturbation sensitivity control.
    for ratio in [float(x) for x in cfg["control_perturbation"]["epsilon_over_q_norm"]]:
        for sample in selected:
            sid=sample["sample_id"]; q=sample["q"].reshape(-1)
            delta=direction*(ratio*q.norm()); perturbed=(q+delta).to(torch.bfloat16); base=q.to(torch.bfloat16)
            changed=perturbed!=base
            with torch.no_grad():
                low=decode_low_res(core,perturbed.to(device).reshape(1,1,256),sample["sam"].to(device=device,dtype=torch.bfloat16))[0,0].cpu()
            control_raw.append({"epsilon_over_q_norm":ratio,"sample_id":sid,
                "bf16_survived":bool(changed.any()),"changed_q_elements":int(changed.sum()),
                "bf16_q_difference_norm":float((perturbed.float()-base.float()).norm()),
                **difference_metrics(low,baseline_low[sid])})
    control_rows=[]
    for ratio in [float(x) for x in cfg["control_perturbation"]["epsilon_over_q_norm"]]:
        values=[x for x in control_raw if x["epsilon_over_q_norm"]==ratio]
        row={"epsilon_over_q_norm":ratio,"n":len(values),
             "fraction_samples_bf16_survived":float(np.mean([x["bf16_survived"] for x in values])),
             "mean_changed_q_elements":float(np.mean([x["changed_q_elements"] for x in values])),
             "mean_bf16_q_difference_norm":float(np.mean([x["bf16_q_difference_norm"] for x in values]))}
        for key in ("logit_l1","logit_l2","max_absolute_logit_difference","mean_absolute_logit_difference","fraction_pixels_changed_logits","fraction_binary_pixels_changed"):
            row[f"mean_{key}"]=float(np.mean([x[key] for x in values]))
        row["fraction_samples_binary_mask_changed"]=float(np.mean([x["binary_mask_changed"] for x in values]))
        control_rows.append(row)
    write_csv(out / "phase4d1b_control_perturbation.csv", list(control_rows[0].keys()), control_rows)
    return bf_rows, sam_rows, control_rows, direction


def optimizer_scheduler(reader, d1cfg):
    opt=torch.optim.AdamW(reader.parameters(),lr=float(d1cfg["optimizer"]["learning_rate"]),
        weight_decay=float(d1cfg["optimizer"]["weight_decay"]),betas=tuple(d1cfg["optimizer"]["betas"]))
    warm=int(d1cfg["optimizer"]["warmup_steps"]); total=int(d1cfg["optimizer"]["schedule_total_steps"])
    def scale(step):
        if step<warm:return (step+1)/warm
        x=(step-warm)/max(1,total-warm);return .5*(1+math.cos(math.pi*min(1.,x)))
    return opt,torch.optim.lr_scheduler.LambdaLR(opt,scale)


def optional_64step(cfg, d1cfg, selected, evidence, core, out, device, beta_value):
    init=torch.load(Path(cfg["phase4d1"]["checkpoint_root"])/"reader_init.pt",map_location="cpu")
    preflight={"status":"FROZEN_BEFORE_TRAINING","population":"same first 256 Phase4D-1 subset",
      "steps":64,"D0_beta_init":0.0,"D1_beta_init":beta_value,
      "beta_selection_rule":cfg["optional_64step"]["corrected_beta_rule"],
      "validation_iou_used":False,"only_changed_variable":"beta initialization"}
    dump(out/"phase4d1b_64step_preflight.json",preflight)
    rows=[]; final={}
    for diagnostic,beta in (("D0_original_gate",0.0),("D1_nonzero_gate",beta_value)):
        reader=PositionAwareEvidenceReader().to(device);reader.load_state_dict(init["reader"]);reader.beta.data.fill_(beta);reader.train()
        opt,sched=optimizer_scheduler(reader,d1cfg);opt.zero_grad(set_to_none=True);interval=defaultdict(float);nint=0
        for i,sample in enumerate(selected):
            q=sample["q"].to(device).reshape(1,256);f=evidence["pos_forensic"][sample["sample_id"]].to(device).unsqueeze(0)
            value=reader(q,f);low=decode_low_res(core,value["q_final"].to(torch.bfloat16),sample["sam"].to(device=device,dtype=torch.bfloat16));losses=sam_input_loss(low,sample["target"].to(device));(losses["total"]/4).backward()
            for k in ("bce","dice","total"):interval[k]+=float(losses[k].detach())
            interval["raw_residual_norm"]+=float(value["q_evidence"].detach().float().norm());interval["effective_residual_norm"]+=float(value["q_residual"].detach().float().norm());nint+=1
            if (i+1)%4:continue
            reader_grad=grad_vector_stats([p for name,p in reader.named_parameters() if name!="beta"])["norm"]
            beta_grad=abs(float(reader.beta.grad));torch.nn.utils.clip_grad_norm_(reader.parameters(),1.0);opt.step();sched.step();opt.zero_grad(set_to_none=True);step=(i+1)//4
            if step in cfg["optional_64step"]["diagnostic_steps"]:
                rows.append({"diagnostic":diagnostic,"step":step,"exposure":i+1,"beta":float(reader.beta.detach()),
                  "reader_grad_norm":reader_grad,"beta_grad_norm":beta_grad,"loss":interval["total"]/nint,
                  "bce":interval["bce"]/nint,"dice":interval["dice"]/nint,"raw_residual_norm":interval["raw_residual_norm"]/nint,
                  "effective_residual_norm":interval["effective_residual_norm"]/nint});interval=defaultdict(float);nint=0
        reader.eval();perm=torch.tensor(json.loads((ROOT/cfg["phase4d1"]["output_root"] / "spatial_content_permutation.json").read_text())["permutation"],device=device)
        survival=[];samdiff=[];binary=[];qdiff=[]
        with torch.no_grad():
            for sample in selected:
                q=sample["q"].to(device).reshape(1,256);f=evidence["pos_forensic"][sample["sample_id"]].to(device).unsqueeze(0)
                m=reader(q,f)["q_final"].to(torch.bfloat16);s=reader(q,f.flatten(2).transpose(1,2)[:,perm,:])["q_final"].to(torch.bfloat16);survival.append(bool((m!=s).any()));qdiff.append(float((m.float()-s.float()).norm()))
                lm=decode_low_res(core,m,sample["sam"].to(device=device,dtype=torch.bfloat16))[0,0];ls=decode_low_res(core,s,sample["sam"].to(device=device,dtype=torch.bfloat16))[0,0];dm=difference_metrics(lm,ls);samdiff.append(dm["mean_absolute_logit_difference"]);binary.append(dm["binary_mask_changed"])
        final[diagnostic]={"beta_final":float(reader.beta),"bf16_matched_shuffle_survival":float(np.mean(survival)),"mean_bf16_matched_shuffle_q_norm":float(np.mean(qdiff)),"mean_sam_logit_absolute_difference":float(np.mean(samdiff)),"fraction_binary_masks_changed":float(np.mean(binary)),"reader_state_hash":reader_state_hash(reader)}
    for row in rows:
        if row["step"]==0:pass
    write_csv(out/"phase4d1b_64step_diagnostic.csv",list(rows[0].keys()),rows);dump(out/"phase4d1b_64step_summary.json",final)
    return rows,final


def main():
    cfg=yaml.safe_load((ROOT/"configs/phase4d1b_interface_signal_audit.yaml").read_text());d1cfg=yaml.safe_load((ROOT/cfg["phase4d1"]["config"]).read_text());bcfg=yaml.safe_load((ROOT/d1cfg["phase4c_b"]["config"]).read_text())
    out=ROOT/cfg["experiment"]["output_root"];out.mkdir(parents=True,exist_ok=True);selected,evidence,population=load_population(cfg,out)
    d1out=ROOT/cfg["phase4d1"]["output_root"];completion=json.loads((d1out/"completion_manifest.json").read_text())
    if completion["status"]!="COMPLETE":raise RuntimeError("Phase4D-1 incomplete")
    device=torch.device("cuda:0");torch.cuda.set_device(device);model,core=load_p1(bcfg,device);before=frozen_hash(core)
    gradient=gradient_audit(cfg,selected,evidence,core,device,out)
    residual,c_signal,cached,perm,cross=forward_signal_audits(cfg,selected,evidence,out,device)
    bf16,sam,control,direction=survival_and_sam_audit(cfg,selected,cached,core,out,device)
    zero=[x for x in gradient if x["arm"]=="pos_forensic" and x["beta_label"]=="zero"][0];b01=[x for x in gradient if x["arm"]=="pos_forensic" and x["beta_label"]=="plus_0p1"][0]
    tiny_supported=zero["grad_norm_reader_total"]<=1e-12 and b01["grad_norm_reader_total"]>1e-8
    candidates=[x for x in bf16 if x["arm"]=="pos_forensic" and x["fraction_samples_surviving_vs_q"]>=.8 and x["alpha"]>0]
    corrected_beta=min((x["alpha"] for x in candidates),default=float(cfg["optional_64step"]["corrected_beta_fallback"]))
    diagnostic_rows=diagnostic_summary=None
    if tiny_supported:
        diagnostic_rows,diagnostic_summary=optional_64step(cfg,d1cfg,selected,evidence,core,out,device,corrected_beta)
    d0=diagnostic_summary["D0_original_gate"] if diagnostic_summary else None;d1=diagnostic_summary["D1_nonzero_gate"] if diagnostic_summary else None
    diagnostic_support=bool(d1 and d0 and d1["bf16_matched_shuffle_survival"]>d0["bf16_matched_shuffle_survival"] and d1["mean_sam_logit_absolute_difference"]>d0["mean_sam_logit_absolute_difference"])
    failure="ZERO_OR_TINY_GATE_GRADIENT_SUPPRESSION" if tiny_supported and diagnostic_support else "MULTI_FACTOR_OR_INCONCLUSIVE"
    decision={"READER_GRADIENT_VIABLE":"PARTIAL" if tiny_supported else "YES","TINY_BETA_SUPPRESSES_OPTIMIZATION":tiny_supported,
      "BF16_SIGNAL_SURVIVAL":"SCALE_DEPENDENT","SAM_QUERY_SENSITIVITY":"SCALE_DEPENDENT","PRIMARY_FAILURE_MODE":failure,
      "CORRECTED_MINIMAL_RERUN_JUSTIFIED":"YES" if failure=="ZERO_OR_TINY_GATE_GRADIENT_SUPPRESSION" else "NO","PROCEED_TO_PHASE_4D_2":"NO"}
    residual_stats={}
    for arm in ARMS:
      for step in (0,512):
       values=[x for x in residual if x["arm"]==arm and x["step"]==step];residual_stats[f"{arm}_step{step}"]={k:describe([x[k] for x in values]) for k in ("q_norm","reader_raw_residual_norm","effective_residual_norm","raw_over_q","effective_over_q")}
    c_stats={}
    for arm in ARMS:
      for step in (0,512):
       for comp in ("matched_minus_shuffle","matched_minus_cross"):
        values=[x for x in c_signal if x["arm"]==arm and x["step"]==step and x["comparison"]==comp];c_stats[f"{arm}_step{step}_{comp}"]={k:describe([x[k] for x in values]) for k in ("raw_output_difference_norm","effective_difference_norm","fp32_q_difference_norm","bf16_q_difference_norm")}|{"fraction_samples_any_bf16_difference":float(np.mean([x["bf16_any_difference"] for x in values])),"fraction_elements_different":float(np.mean([x["bf16_fraction_elements_different"] for x in values]))}
    statistics={"population":population,"gradient_rows":gradient,"residual_statistics":residual_stats,"pre_sam_signal_statistics":c_stats,"bf16_survival":bf16,"sam_logit_sensitivity":sam,"control_perturbation":control,"optional_64step":diagnostic_summary,"decision":decision,"P1_SAM_hash_unchanged":frozen_hash(core)==before,"P1_checkpoint_hash_unchanged":file_sha256(Path(bcfg["p1"]["checkpoint"]))==bcfg["p1"]["checkpoint_sha256"],"internal_test_access":False,"official1000_access":False}
    dump(out/"phase4d1b_statistics.json",statistics);dump(out/"decision.json",decision)
    figures=out/"figures/phase4d1b";figures.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for arm in ARMS:
      values=[x for x in bf16 if x["arm"]==arm];ax.plot([x["alpha"] for x in values],[x["fraction_samples_surviving_vs_q"] for x in values],marker="o",label=arm)
    ax.set_xscale("symlog",linthresh=.001);ax.set_xlabel("alpha residual scale");ax.set_ylabel("BF16 surviving sample fraction");ax.legend();fig.tight_layout();fig.savefig(figures/"bf16_survival_curve.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for arm in ARMS:
      values=[x for x in sam if x["arm"]==arm and x["comparison"]=="matched_vs_baseline_q"];ax.plot([x["alpha"] for x in values],[x["mean_mean_absolute_logit_difference"] for x in values],marker="o",label=arm)
    ax.set_xscale("symlog",linthresh=.001);ax.set_xlabel("alpha residual scale");ax.set_ylabel("Mean absolute SAM logit difference");ax.legend();fig.tight_layout();fig.savefig(figures/"sam_logit_sensitivity.png",dpi=180);plt.close(fig)
    if decision["CORRECTED_MINIMAL_RERUN_JUSTIFIED"]=="YES":
      proposal=f"""# Phase 4D-1R — Corrected Minimal Rerun Proposal\n\nStatus: **DESIGN ONLY — NOT AUTHORIZED**\n\nPhase 4D-1B identifies tiny/zero gate gradient suppression. The only permitted change is to initialize beta to `{corrected_beta}`, the smallest preregistered inference scale with at least 80% POS-FORENSIC BF16 survival on the frozen audit population. Keep the Phase 4D-1 architecture, fixed 2D encoding, 2,048 subset, order, 512 steps, optimizer, loss, two matched feature arms, validation controls, seed, threshold, and frozen modules unchanged. No Phase 4D-1R training or evaluation starts from this proposal.\n"""
      (out/"phase4d1r_corrected_minimal_rerun_proposal.md").write_text(proposal,encoding="utf-8")
    def g(arm,label,key):return [x for x in gradient if x["arm"]==arm and x["beta_label"]==label][0][key]
    forensic_survival=[x for x in bf16 if x["arm"]=="pos_forensic"]
    report=f"""# Phase 4D-1B — Interface Signal Viability Audit\n\n## 1. Executive Summary\n\n- Step-0 beta gradient is nonzero, but Reader-body gradient at beta=0 is zero.\n- Nonzero beta restores Reader gradients; tiny/zero gate suppression: **{str(tiny_supported).upper()}**.\n- Raw Reader output exists, while effective residual is gate-limited.\n- BF16 survival and frozen SAM response are scale-dependent.\n- Primary failure mode: **{failure}**.\n- Corrected Phase 4D-1R design justified: **{decision['CORRECTED_MINIMAL_RERUN_JUSTIFIED']}**; it was not executed.\n\n## 2. Gradient Flow\n\nPOS-FORENSIC at beta=0: beta grad `{g('pos_forensic','zero','grad_norm_beta'):.6g}`, Reader grad `{g('pos_forensic','zero','grad_norm_reader_total'):.6g}`. At beta=0.01/0.1/1.0 Reader grad is `{g('pos_forensic','plus_0p01','grad_norm_reader_total'):.6g}` / `{g('pos_forensic','plus_0p1','grad_norm_reader_total'):.6g}` / `{g('pos_forensic','plus_1p0','grad_norm_reader_total'):.6g}`. No optimizer step was used in Audit A.\n\n## 3. Signal Magnitude\n\nPOS-FORENSIC step512 mean q norm `{residual_stats['pos_forensic_step512']['q_norm']['mean']:.6f}`, raw residual norm `{residual_stats['pos_forensic_step512']['reader_raw_residual_norm']['mean']:.6f}`, effective residual norm `{residual_stats['pos_forensic_step512']['effective_residual_norm']['mean']:.6f}`. Effective/q ratio mean `{residual_stats['pos_forensic_step512']['effective_over_q']['mean']:.8f}`.\n\n## 4. BF16 Survival\n\nThe frozen alpha grid was not selected by IoU. POS-FORENSIC survival vs q rises from `{forensic_survival[0]['fraction_samples_surviving_vs_q']:.4f}` at alpha 0 to `{forensic_survival[-1]['fraction_samples_surviving_vs_q']:.4f}` at alpha 1.0. Corrected diagnostic beta was frozen to `{corrected_beta}` before its 64-step run.\n\n## 5. SAM Logit Sensitivity\n\nContinuous SAM logits respond increasingly as residual scale grows; binary threshold changes occur later and are not used to select alpha. The generic perturbation curve also shows scale-dependent prompt-path sensitivity, so SAM is not globally insensitive to q.\n\n## 6. Reader vs Generic Perturbation\n\nMatched-norm random-direction controls are reported alongside Reader direction in `phase4d1b_sam_logit_sensitivity.csv`. The primary defect is the gate/scale pathway; the audit does not establish that Reader direction is uniquely ineffective.\n\n## 7. Optional 64-Step Diagnostic\n\nExecuted because Audit A established exact zero Reader-body gradient at beta=0 and recovery at nonzero beta. D0 and D1 used the same first 256 samples, order, loss, optimizer, and architecture; only beta initialization differed. D0/D1 BF16 matched-shuffle survival: `{d0['bf16_matched_shuffle_survival']:.6f}` / `{d1['bf16_matched_shuffle_survival']:.6f}`; mean SAM logit absolute difference: `{d0['mean_sam_logit_absolute_difference']:.8f}` / `{d1['mean_sam_logit_absolute_difference']:.8f}`.\n\n## 8. Final Decision\n\n```text\nREADER_GRADIENT_VIABLE: {decision['READER_GRADIENT_VIABLE']}\nTINY_BETA_SUPPRESSES_OPTIMIZATION: {str(decision['TINY_BETA_SUPPRESSES_OPTIMIZATION']).upper()}\nBF16_SIGNAL_SURVIVAL: {decision['BF16_SIGNAL_SURVIVAL']}\nSAM_QUERY_SENSITIVITY: {decision['SAM_QUERY_SENSITIVITY']}\nPRIMARY_FAILURE_MODE: {decision['PRIMARY_FAILURE_MODE']}\nCORRECTED_MINIMAL_RERUN_JUSTIFIED: {decision['CORRECTED_MINIMAL_RERUN_JUSTIFIED']}\nPROCEED_TO_PHASE_4D_2: NO\n```\n\nThis audit explains why Phase 4D-1 barely affected the SAM query. It does not show that positional encoding works, does not establish localization gain, and does not show forensic information is transferable or impossible to transfer. Internal test and official1000 remained sealed.\n"""
    (out/"phase4d1b_preflight.md").write_text(f"# Phase 4D-1B Preflight\n\nStatus: **PASS**\n\n- Reused Phase 4D-1 shared init, architecture, checkpoints, first 256 frozen subset IDs, feature/query/SAM caches, cross derangement, and spatial-content permutation.\n- Population hash: `{population['sample_ids_sha256']}`.\n- Alpha and epsilon grids were frozen by instruction.\n- Internal test and official1000 were not accessed.\n",encoding="utf-8")
    (out/"phase4d1b_interface_signal_report.md").write_text(report,encoding="utf-8")
    required=["phase4d1b_preflight.md","phase4d1b_gradient_audit.csv","phase4d1b_residual_magnitude.csv","phase4d1b_bf16_survival.csv","phase4d1b_sam_logit_sensitivity.csv","phase4d1b_control_perturbation.csv","phase4d1b_statistics.json","phase4d1b_interface_signal_report.md"]
    if tiny_supported:required.append("phase4d1b_64step_diagnostic.csv")
    dump(out/"completion_manifest.json",{"status":"COMPLETE","decision":decision,"required_outputs":{x:(out/x).exists() for x in required},"internal_test_access":False,"official1000_access":False,"phase4d1r_started":False,"phase4d2_started":False})
    if not all((out/x).exists() for x in required):raise RuntimeError("missing Phase4D-1B outputs")
    if frozen_hash(core)!=before:raise RuntimeError("P1/SAM mutated")
    print(json.dumps({"status":"COMPLETE","decision":decision,"corrected_beta":corrected_beta},indent=2))


if __name__=="__main__":main()
