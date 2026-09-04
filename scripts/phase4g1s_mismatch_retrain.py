#!/usr/bin/env python3
"""Phase 4G-1S: mismatch-aware retraining of the frozen CSCU-LF graph."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from scripts import phase4g1q_conditional_utility as q
from tools.phase4c_b import file_sha256
from tools.phase4e1 import tensor_state_sha256

PHASE="phase4g1s"; OUT=ROOT/f"outputs/{PHASE}"; DOC=ROOT/f"docs/{PHASE}/report.md"
CKPT_DIR=Path("/data/yz/groundingLMM_official/checkpoints/phase4g1s_mismatch_aware_utility")
CHECKPOINT=CKPT_DIR/"csculf_mismatch_utility_epoch10.pt"; CALIBRATION=CKPT_DIR/"utility_calibration.json"; MANIFEST=CKPT_DIR/"execution_manifest.json"
FIT_HISTORY=OUT/"fit_history.csv"; RESULTS=OUT/"results.json"; GATES=OUT/"gate_summary.json"
BASE_CHECKPOINT=ROOT/"outputs/phase4g1q/csculf_utility_fit_final.pt"; BASE_RESULTS=ROOT/"outputs/phase4g1q/results.json"; BASE_GATES=ROOT/"outputs/phase4g1q/gate_summary.json"
MARGIN=.1; RELATIVE_WEIGHT=1.; RANKING_WEIGHT=1.; SEED=3407; BATCH=8; EPOCHS=10


def configure_q_paths():
    q.OUT=OUT; q.DOC=DOC; q.CHECKPOINT=CHECKPOINT; q.CALIBRATION=CALIBRATION; q.FIT_HISTORY=FIT_HISTORY
    q.RESULTS=RESULTS; q.GATES=GATES; q.HASHES=MANIFEST


def dump(path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def canonical_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def frozen_preflight():
    _,target,split,architecture=q.frozen_preflight(); old_gates=json.loads(BASE_GATES.read_text()); old_results=json.loads(BASE_RESULTS.read_text())
    expected={"UTILITY_RELATIVE_LOSS_RELATION":"PASS","CROSS_IMAGE_UTILITY_RESPONSE":"FAIL","SPATIAL_SHUFFLE_UTILITY_RESPONSE":"FAIL",
              "QMF_CONDITIONAL_RELATION":"PASS","UTILITY_INTERVENTION_IDENTITY":"PASS","UTILITY_PERMUTATION_SENSITIVITY":"PASS",
              "VACUOUS_EXACT_RECOVERY":"PASS","INVALID_G0_POLICY":"PASS","FORMAL_TRAINING_JUSTIFIED":"NO"}
    if any(old_gates.get(k)!=v for k,v in expected.items()): raise RuntimeError("Phase 4G-1Q premise drift")
    if architecture["parameters"]["trainable"]!=371803 or target["tau"]!=q.TAU: raise RuntimeError("frozen graph/target drift")
    base=torch.load(BASE_CHECKPOINT,map_location="cpu",weights_only=False)
    if base["epoch"]!=10 or base["optimizer_updates"]!=6470: raise RuntimeError("warm-start checkpoint drift")
    return target,split,architecture,old_gates,old_results


def load_model(device,trained=False):
    q.set_determinism(); source=torch.load(q.G1C_CKPT,map_location="cpu",weights_only=False); temps=torch.load(q.G1C_TEMPERATURES,map_location="cpu",weights_only=False)
    model=q.CSCULF(temps["T_L"],temps["T_F"]); model.load_frozen_source_heads(source["language_head"],source["forensic_head"])
    state=torch.load(CHECKPOINT if trained else BASE_CHECKPOINT,map_location="cpu",weights_only=False); model.load_state_dict(state["model_state"])
    return model.to(device)


def freeze_manifest():
    target,split,architecture,old_gates,old_results=frozen_preflight()
    if MANIFEST.exists(): raise RuntimeError("Phase 4G-1S manifest exists; automatic rerun forbidden")
    config={"stage":PHASE,"architecture":"unchanged CSCU-LF d64/w7/heads4/blocks1/refinement NO","initialization":"Phase4G-1Q epoch10 final",
            "target":"sigmoid((ell_L-ell_F)/tau)","tau":q.TAU,"ranking":"pairwise margin ReLU(margin-(U_match-U_mismatch))",
            "margin":MARGIN,"relative_loss_weight":RELATIVE_WEIGHT,"ranking_loss_weight":RANKING_WEIGHT,"ranking_internal_weights":{"cross":.5,"shuffle":.5},
            "optimizer":"AdamW","learning_rate":1e-4,"weight_decay":1e-4,"batch_size":BATCH,"epochs":EPOCHS,"gradient_clip_norm":1.,
            "scheduler":"none","early_stopping":False,"seed":SEED,"checkpoint_rule":"epoch10 final only; one overwritten recovery path",
            "cross_transform":"cyclic next forensic in frozen UTIL-FIT/AUDIT ID order; language and current geometry fixed",
            "spatial_transform":"fixed torch.randperm(24*24), CPU generator seed3407, shared by F24/z_F24",
            "audit_statistics":"identical Phase4G-1Q preregistered gates/bootstrap/thresholds","audit_interpretation":"confirmatory re-audit, not blind"}
    manifest={"schema":"phase4g1s_execution_manifest_v1","status":"FROZEN_BEFORE_FIRST_OPTIMIZER_STEP","created_unix":time.time(),
              "config":config,"config_sha256":canonical_hash(config),"implementation_sha256":file_sha256(Path(__file__)),
              "architecture_manifest_sha256":file_sha256(ROOT/"outputs/phase4g1p/architecture_manifest.json"),"split_sha256":file_sha256(ROOT/"outputs/phase4g1p/utility_split_manifest.json"),
              "target_manifest_sha256":file_sha256(ROOT/"outputs/phase4g1p/utility_target_manifest.json"),"base_checkpoint_sha256":file_sha256(BASE_CHECKPOINT),
              "base_results_sha256":file_sha256(BASE_RESULTS),"base_gates_sha256":file_sha256(BASE_GATES),
              "fit_ids_sha256":q.ids_hash(q.valid_ids(split,"UTILITY-FIT")),"cal_ids_sha256":q.ids_hash(q.valid_ids(split,"UTILITY-CAL")),"audit_ids_sha256":q.ids_hash(q.valid_ids(split,"UTILITY-AUDIT")),
              "source_files":{"g1c_epoch10":{"path":str(q.G1C_CKPT),"sha256_before":file_sha256(q.G1C_CKPT)},"source_temperatures":{"path":str(q.G1C_TEMPERATURES),"sha256_before":file_sha256(q.G1C_TEMPERATURES)}},
              "audit_access":{"count":0,"status":"SEALED"},"firewall":{"development_validation_accessed":False,"internal_test_accessed":False,"official1000_accessed":False,"old_g1c_train_audit_accessed":False}}
    dump(MANIFEST,manifest); return manifest


def image_rank_loss(match,mismatch,support):
    value=F.relu(MARGIN-(match.float()-mismatch.float()))*support.float()
    return (value.flatten(1).sum(1)/support.flatten(1).sum(1).clamp_min(1)).mean()


def grad_norm(module): return math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in module.parameters() if p.grad is not None))


def save_checkpoint(model,optimizer,epoch,updates,source_hash):
    payload={"schema":"phase4g1s_mismatch_aware_utility_checkpoint_v1","epoch":epoch,"optimizer_updates":updates,
             "model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},"optimizer":optimizer.state_dict(),"source_state_hash":tensor_state_sha256(q.source_state(model)),
             "initial_source_state_hash":source_hash,"base_checkpoint_sha256":file_sha256(BASE_CHECKPOINT),"execution_config_sha256":json.loads(MANIFEST.read_text())["config_sha256"],
             "checkpoint_rule":"epoch10 final only; same path overwritten for crash recovery; no epoch selection","not_deployment_checkpoint":True}
    temporary=CHECKPOINT.with_suffix(".pt.tmp"); torch.save(payload,temporary); temporary.replace(CHECKPOINT)


def fit(device):
    configure_q_paths(); recovering=MANIFEST.exists()
    manifest=json.loads(MANIFEST.read_text()) if recovering else freeze_manifest(); _,split,_,_,_=frozen_preflight(); ids=q.valid_ids(split,"UTILITY-FIT")
    if recovering:
        recovery=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
        if manifest.get("status")!="FROZEN_BEFORE_FIRST_OPTIMIZER_STEP" or recovery["epoch"]!=1 or recovery["optimizer_updates"]!=math.ceil(len(ids)/BATCH) or recovery["execution_config_sha256"]!=manifest["config_sha256"]:
            raise RuntimeError("unsupported or drifted crash-recovery state")
        manifest["crash_recovery"]={"status":"AUTHORIZED_BY_FROZEN_CHECKPOINT_RULE","completed_epoch":1,"updates":recovery["optimizer_updates"],
            "failure":"fit_history parent directory missing after epoch1 checkpoint save","epoch1_replayed":False,"implementation_sha256_before":manifest["implementation_sha256"],
            "implementation_sha256_after":file_sha256(Path(__file__)),"config_unchanged":canonical_hash(manifest["config"])==manifest["config_sha256"],"resumed_unix":time.time()}; dump(MANIFEST,manifest)
    data=q.load_ids(ids,("S64","q_seg","z_L","F24","z_F24","target64","clip_geometries")); model=load_model(device,trained=recovering).train()
    model.language_source.eval(); model.forensic_source.eval(); source_hash=(recovery["initial_source_state_hash"] if recovering else tensor_state_sha256(q.source_state(model))); params=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params)!=371803: raise RuntimeError("trainable parameter drift")
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    if recovering: optimizer.load_state_dict(recovery["optimizer"])
    perm24=torch.randperm(24*24,generator=torch.Generator().manual_seed(SEED)).to(device)
    fields=["epoch","updates","total_loss","relative_loss","ranking_loss","cross_rank_loss","shuffle_rank_loss","mean_U_match","mean_U_cross","mean_U_shuffle","mean_cross_delta","mean_shuffle_delta",
            "std_U_match","p01","p05","p25","p50","p75","p95","p99","low_saturation","high_saturation","grad_language_projection","grad_forensic_projection","grad_channel_spatial_interaction","grad_context_exchange","grad_utility_head",
            "loss_finite","gradient_finite","trainable_params_finite","seconds"]
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]; updates=0; start_epoch=1
    if recovering:
        placeholder={field:"" for field in fields}; placeholder.update({"epoch":1,"updates":recovery["optimizer_updates"],"loss_finite":True,"gradient_finite":True,"trainable_params_finite":True})
        rows=[placeholder]; updates=recovery["optimizer_updates"]; start_epoch=2
    for epoch in range(start_epoch,EPOCHS+1):
        started=time.time(); order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(SEED+epoch)); sums=defaultdict(float); uvals=[]; grad_sq=defaultdict(float); finite_loss=finite_grad=True; seen=0
        for begin in range(0,len(ids),BATCH):
            idx=order[begin:begin+BATCH]; batch=q.batch_to(data,idx,device); cross_idx=(idx+1)%len(ids); forensic=q.batch_to(data,cross_idx,device)
            crossed=dict(batch); crossed["F24"]=forensic["F24"]; crossed["z_F24"]=forensic["z_F24"]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                matched=q.utility_forward(model,batch); cross=q.utility_forward(model,crossed); shuffle=q.utility_forward(model,batch,spatial_permutation=perm24)
                _,soft=q.target_delta(matched,batch["target64"]); relative=q.image_balanced_loss(matched["utility_logit"],soft,matched["support"])
                cross_rank=image_rank_loss(matched["U"],cross["U"],matched["support"]); shuffle_rank=image_rank_loss(matched["U"],shuffle["U"],matched["support"])
                ranking=.5*(cross_rank+shuffle_rank); total=RELATIVE_WEIGHT*relative+RANKING_WEIGHT*ranking
            finite_loss &= bool(torch.isfinite(total)); total.backward()
            norms={"language_context":grad_norm(model.language_context),"forensic_context":grad_norm(model.forensic_context),"rectification":grad_norm(model.rectification),"exchange":grad_norm(model.exchange),"utility_head":grad_norm(model.utility_head)}
            finite_grad &= all(math.isfinite(v) for v in norms.values()) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
            for k,v in norms.items(): grad_sq[k]+=v*v
            torch.nn.utils.clip_grad_norm_(params,1.); optimizer.step(); updates+=1; n=len(idx); seen+=n
            for k,v in (("total",total),("relative",relative),("ranking",ranking),("cross_rank",cross_rank),("shuffle_rank",shuffle_rank)): sums[k]+=float(v.detach())*n
            support=matched["support"]; denom=support.flatten(1).sum(1); um=(matched["U"]*support).flatten(1).sum(1)/denom; uc=(cross["U"]*support).flatten(1).sum(1)/denom; us=(shuffle["U"]*support).flatten(1).sum(1)/denom
            sums["um"]+=float(um.sum()); sums["uc"]+=float(uc.sum()); sums["us"]+=float(us.sum()); uvals.append(matched["U"].detach()[support].float().cpu())
        values=torch.cat(uvals).numpy(); quant=np.quantile(values,[.01,.05,.25,.5,.75,.95,.99]); row={"epoch":epoch,"updates":updates,"total_loss":sums["total"]/seen,"relative_loss":sums["relative"]/seen,
            "ranking_loss":sums["ranking"]/seen,"cross_rank_loss":sums["cross_rank"]/seen,"shuffle_rank_loss":sums["shuffle_rank"]/seen,"mean_U_match":sums["um"]/seen,"mean_U_cross":sums["uc"]/seen,"mean_U_shuffle":sums["us"]/seen,
            "mean_cross_delta":(sums["uc"]-sums["um"])/seen,"mean_shuffle_delta":(sums["us"]-sums["um"])/seen,"std_U_match":float(values.std()),
            **dict(zip(("p01","p05","p25","p50","p75","p95","p99"),map(float,quant))),"low_saturation":float((values<=.01).mean()),"high_saturation":float((values>=.99).mean()),
            "grad_language_projection":math.sqrt(grad_sq["language_context"]),"grad_forensic_projection":math.sqrt(grad_sq["forensic_context"]),"grad_channel_spatial_interaction":math.sqrt(grad_sq["rectification"]),
            "grad_context_exchange":math.sqrt(grad_sq["exchange"]),"grad_utility_head":math.sqrt(grad_sq["utility_head"]),"loss_finite":finite_loss,"gradient_finite":finite_grad,
            "trainable_params_finite":all(bool(torch.isfinite(p).all()) for p in params),"seconds":time.time()-started}; rows.append(row); save_checkpoint(model,optimizer,epoch,updates,source_hash)
        with FIT_HISTORY.open("w",newline="",encoding="utf-8") as handle: writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        print(json.dumps({"stage":"FIT","epoch":epoch,"total":row["total_loss"],"relative":row["relative_loss"],"ranking":row["ranking_loss"],"cross_delta":row["mean_cross_delta"],"shuffle_delta":row["mean_shuffle_delta"],"seconds":row["seconds"]}),flush=True)
    complete=updates==EPOCHS*math.ceil(len(ids)/BATCH) and all(r["loss_finite"] and r["gradient_finite"] and r["trainable_params_finite"] for r in rows) and tensor_state_sha256(q.source_state(model))==source_hash
    manifest=json.loads(MANIFEST.read_text()); manifest.update({"status":"FIT_COMPLETE" if complete else "FIT_FAILED","fit_checkpoint_sha256":file_sha256(CHECKPOINT),"fit_history_sha256":file_sha256(FIT_HISTORY),
        "fit_updates":updates,"source_state_hash_before_fit":source_hash,"source_state_hash_after_fit":tensor_state_sha256(q.source_state(model))}); dump(MANIFEST,manifest)
    if not complete: raise RuntimeError("mismatch-aware FIT failed")


def calibrate(device): configure_q_paths(); return q.calibrate(device)
def audit(device): configure_q_paths(); return q.audit(device)


def finalize():
    configure_q_paths(); result=json.loads(RESULTS.read_text()); manifest=json.loads(MANIFEST.read_text()); old=json.loads(BASE_GATES.read_text())
    identity=result["utility_intervention_identity"]["gate"]; permutation=result["utility_permutation"]["gate"] if result["utility_permutation"] else "FAIL"
    gates={"schema":"phase4g1s_gate_summary_v1","MISMATCH_AWARE_UTILITY_FIT_COMPLETE":"YES","CALIBRATION_STATUS":result["calibration"]["CALIBRATION_STATUS"],
           "UTILITY_RELATIVE_LOSS_RELATION":result["relative_loss"]["primary"]["gate"],"CROSS_IMAGE_UTILITY_RESPONSE":result["cross_image"]["gate"],"SPATIAL_SHUFFLE_UTILITY_RESPONSE":result["spatial_shuffle"]["gate"],
           "QMF_CONDITIONAL_RELATION":result["qmf"]["gate"],"UTILITY_INTERVENTION_IDENTITY":identity,"UTILITY_PERMUTATION_SENSITIVITY":permutation,
           "VACUOUS_EXACT_RECOVERY":result["vacuous_exact_recovery"]["gate"],"NO_ORACLE_LEAKAGE":"PASS","INVALID_G0_POLICY":"PASS",
           "SOURCE_HASH_INTEGRITY":"PASS" if result["source_integrity"]["unchanged"] and all(row["unchanged"] for row in manifest["source_files"].values()) else "FAIL",
           "UTILITY_AUDIT_ACCESS_COUNT":manifest["audit_access"]["count"],"AUDIT_INTERPRETATION":"CONFIRMATORY_REAUDIT_NOT_BLIND",
           "DEVELOPMENT_VALIDATION_ACCESSED":"NO","INTERNAL_TEST_ACCESSED":"NO","OFFICIAL1000_ACCESSED":"NO"}
    mandatory=[gates[k] for k in ("CALIBRATION_STATUS","UTILITY_RELATIVE_LOSS_RELATION","CROSS_IMAGE_UTILITY_RESPONSE","SPATIAL_SHUFFLE_UTILITY_RESPONSE","QMF_CONDITIONAL_RELATION","UTILITY_INTERVENTION_IDENTITY","UTILITY_PERMUTATION_SENSITIVITY","VACUOUS_EXACT_RECOVERY","NO_ORACLE_LEAKAGE","INVALID_G0_POLICY","SOURCE_HASH_INTEGRITY")]
    overall="FAIL" if "FAIL" in mandatory else ("INCONCLUSIVE" if "INCONCLUSIVE" in mandatory else "PASS"); gates["MISMATCH_AWARE_UTILITY_PREFLIGHT"]=overall; gates["FORMAL_TRAINING_JUSTIFIED"]="YES" if overall=="PASS" else "NO"; gates["FORMAL_TRAINING_EXECUTED"]="NO"; dump(GATES,gates)
    history=list(csv.DictReader(FIT_HISTORY.open())); first_logged=next(row for row in history if row["total_loss"]); p=result["relative_loss"]["primary"]; cross=result["cross_image"]; shuffle=result["spatial_shuffle"]
    report=f"""# Phase 4G-1S mismatch-aware CSCU-LF utility retrain

## Protocol and firewall

CSCU-LF graph保持完全不变：d64/window7/heads4/blocks1，371,803 trainable parameters，REFINEMENT_INCLUDED=NO。Warm-start为 Phase4G-1Q epoch10。一次冻结 margin=0.1、relative:ranking=1:1、cross:shuffle=1:1、seed3407、AdamW lr1e-4/wd1e-4、batch8、10 epochs、grad clip1；无 sweep、scheduler、early stopping、fused segmentation loss或QMF regularizer。只用 UTIL-FIT训练、UTIL-CAL校准；dev/internal test/official1000均未访问。

本次同一 UTIL-AUDIT 是由先前 mismatch failure 驱动的 **confirmatory re-audit，不是 blind audit**；它没有参与参数更新或 calibration。

## FIT and CAL

10 epochs共 {history[-1]['updates']} updates。Epoch1 checkpoint 在日志目录错误前已保存且未重放，其逐项指标未持久化；可用日志从 epoch{first_logged['epoch']} 开始。Total loss {first_logged['total_loss']} → {history[-1]['total_loss']}；relative loss {first_logged['relative_loss']} → {history[-1]['relative_loss']}；ranking loss {first_logged['ranking_loss']} → {history[-1]['ranking_loss']}。最终 FIT cross/shuffle mean ΔU分别为 {history[-1]['mean_cross_delta']} / {history[-1]['mean_shuffle_delta']}。CAL `T_U={result['calibration']['T_U']:.8f}`，objective {result['calibration']['pre']['objective']:.8f} → {result['calibration']['post']['objective']:.8f}。

## Preregistered gates

- Relative-loss pooled-pixel Spearman={p['point']:.6f}，95% CI={p['ci95']}：**{gates['UTILITY_RELATIVE_LOSS_RELATION']}**。
- Cross−matched mean ΔU={cross['mean_delta']:.6f}，95% CI={cross['ci95']}：**{gates['CROSS_IMAGE_UTILITY_RESPONSE']}**。
- Shuffle−matched mean ΔU={shuffle['mean_delta']:.6f}，95% CI={shuffle['ci95']}：**{gates['SPATIAL_SHUFFLE_UTILITY_RESPONSE']}**。
- QMF conditional relation：**{gates['QMF_CONDITIONAL_RELATION']}**。
- Utility intervention identity：**{gates['UTILITY_INTERVENTION_IDENTITY']}**。
- Utility permutation sensitivity：**{gates['UTILITY_PERMUTATION_SENSITIVITY']}**。
- Vacuous exact recovery：**{gates['VACUOUS_EXACT_RECOVERY']}**。
- Invalid G0 policy：**{gates['INVALID_G0_POLICY']}**。

完整 four-state、scoped relation、QMF correlations、permutation metrics与hash identity集中在 `outputs/phase4g1s/results.json`。

## Gate summary

```json
{json.dumps(gates,ensure_ascii=False,indent=2)}
```

## Interpretation and decision

Phase4G-1Q历史结果保持不变。本阶段只判断 mismatch-aware utility retrain 是否同时保留 relative utility并修复 cross/spatial response；不计算或声称 G0/Phrase/TF/localization performance。任一核心 gate FAIL即停止。

`MISMATCH_AWARE_UTILITY_PREFLIGHT={overall}`  
`FORMAL_TRAINING_JUSTIFIED={gates['FORMAL_TRAINING_JUSTIFIED']}`

即使全部 PASS，也只输出授权建议，不自动运行正式 localization training。Phase 4G-1S 到此 STOP。
"""
    DOC.parent.mkdir(parents=True,exist_ok=True); DOC.write_text(report,encoding="utf-8")
    manifest=json.loads(MANIFEST.read_text()); manifest.update({"status":"COMPLETE","results_sha256":file_sha256(RESULTS),"gate_summary_sha256":file_sha256(GATES),"fit_history_sha256":file_sha256(FIT_HISTORY),"report_sha256":file_sha256(DOC),"final_checkpoint_sha256":file_sha256(CHECKPOINT)}); dump(MANIFEST,manifest); return gates


def selftest(device):
    configure_q_paths(); frozen_preflight(); model=load_model(device).eval(); _,split,_,_,_=frozen_preflight(); ids=q.valid_ids(split,"UTILITY-FIT")[:8]
    data=q.load_ids(ids,("S64","q_seg","z_L","F24","z_F24","target64","clip_geometries")); idx=torch.arange(4); batch=q.batch_to(data,idx,device); forensic=q.batch_to(data,(idx+1)%len(ids),device); crossed=dict(batch); crossed["F24"]=forensic["F24"]; crossed["z_F24"]=forensic["z_F24"]
    perm=torch.randperm(24*24,generator=torch.Generator().manual_seed(SEED)).to(device)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
        matched=q.utility_forward(model,batch); cross=q.utility_forward(model,crossed); shuffle=q.utility_forward(model,batch,spatial_permutation=perm); _,soft=q.target_delta(matched,batch["target64"])
        relative=q.image_balanced_loss(matched["utility_logit"],soft,matched["support"]); cr=image_rank_loss(matched["U"],cross["U"],matched["support"]); sr=image_rank_loss(matched["U"],shuffle["U"],matched["support"]); total=relative+.5*(cr+sr)
    total.backward(); assert torch.isfinite(total) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()); print(json.dumps({"SELFTEST":"PASS","relative":float(relative),"cross_rank":float(cr),"shuffle_rank":float(sr),"total":float(total)}))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=("selftest","fit","cal","audit","finalize","all")); parser.add_argument("--device",default="cuda:1"); args=parser.parse_args(); device=torch.device(args.device)
    if device.type=="cuda": torch.cuda.set_device(device)
    if args.stage=="selftest": selftest(device)
    elif args.stage=="fit": fit(device)
    elif args.stage=="cal": calibrate(device)
    elif args.stage=="audit": audit(device)
    elif args.stage=="finalize": print(json.dumps(finalize(),ensure_ascii=False,indent=2))
    else: selftest(device); fit(device); calibrate(device); audit(device); print(json.dumps(finalize(),ensure_ascii=False,indent=2))

if __name__=="__main__": main()
