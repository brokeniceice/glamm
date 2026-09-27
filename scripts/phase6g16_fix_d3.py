#!/usr/bin/env python3
"""Repair Phase6G.16 D3 with strict within-support norm-preserving shuffles."""
from pathlib import Path
import json, sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import phase6g10_train_arm as g10
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts.phase6g13_utility_train import load_head
from scripts.phase6g16_type2_utility import (OUT, SEED, TypeIIUtility, bootstrap_mean, dump,
    finalize, load_collapsed, rect_forward, seed_all)
from scripts.phase6e2_c1_specific_r1_train import c1_language_batch, load_c1_cache, phase4f_spatial_batch

def main():
    device=torch.device("cuda:0");torch.cuda.set_device(device);seed_all()
    _,collapsed,_=load_collapsed(device);fusion=g10.load_fusion(device);projection=g10.load_projection(device)
    head=load_head(device);sam=g10.load_sam_runtime(hd.CFG,device);p4=g10.Phase4FStore(hd.CFG,"val")
    fs=g10.Store("val");dev=g10.load_dev("g0");cache=load_c1_cache("val",dev["sample_ids"])
    model=TypeIIUtility().to(device).eval();state=torch.load(OUT/"A1_type2_correction_aware/selected_checkpoint.pt",map_location="cpu",weights_only=False);model.load_state_dict(state["model"])
    per_abs=[];per_signed=[];per=[];max_global=0.;max_channel=0.;support_errors=0
    valid=[i for i,v in enumerate(cache["valid"]) if bool(v)]
    with torch.no_grad():
        for n,i in enumerate(valid,1):
            sid=dev["sample_ids"][i];s64,_,_,sc,cc=phase4f_spatial_batch(p4,[sid],device)
            f=g10.fused_evidence(fusion,projection,fs,[sid],device);b,_=c1_language_batch(dev,cache,torch.tensor([i]),p4,sam,device)
            _,_,c,support=rect_forward(collapsed,s64,f,sc,cc);mask=support[0,0].flatten().bool();idx=mask.nonzero(as_tuple=False).flatten()
            perm=torch.randperm(len(idx),generator=torch.Generator(device="cpu").manual_seed(SEED+1+i)).to(device)
            csp=c.clone();cf=c.flatten(2);sf=csp.flatten(2);sf[0,:,idx]=cf[0,:,idx.index_select(0,perm)]
            # Exact support occupancy and tight numerical norm invariants.
            invalid=(~mask);support_errors+=int(bool((sf[0,:,invalid]!=0).any()) != bool((cf[0,:,invalid]!=0).any()))
            ge=float((csp.float().norm()-c.float().norm()).abs());ce=float((csp.float().square().sum((2,3)).sqrt()-c.float().square().sum((2,3)).sqrt()).abs().max())
            max_global=max(max_global,ge);max_channel=max(max_channel,ce)
            ur=model(s64,c,b["q_seg"],support)["U"][support.bool()].float();us=model(s64,csp,b["q_seg"],support)["U"][support.bool()].float();d=us-ur
            ma=float(d.abs().mean());sd=float(d.mean());per_abs.append(ma);per_signed.append(sd);per.append({"sample_id":sid,"mean_abs_dU":ma,"signed_dU":sd,"global_norm_abs_error":ge,"max_channel_norm_abs_error":ce})
            if n%100==0:print(json.dumps({"stage":"D3","done":n,"total":len(valid)}),flush=True)
    metric={"unit":"per-image mean over valid support pixels","n":len(per_abs),"mean_abs_dU":float(np.mean(per_abs)),"median_abs_dU":float(np.median(per_abs)),"bootstrap_95_ci":bootstrap_mean(per_abs),"signed_dU":float(np.mean(per_signed))}
    inv={"valid_samples":len(valid),"support_occupancy_errors":support_errors,"max_global_norm_abs_error":max_global,"max_per_channel_norm_abs_error":max_channel,"passed":support_errors==0 and max_global<=1e-3 and max_channel<=1e-3}
    if not inv["passed"]:raise RuntimeError(f"D3 invariants failed: {inv}")
    path=OUT/"diagnostics/correction_sensitivity.json";sens=json.loads(path.read_text());sens["identity_sensitivity"]["spatial_shuffle"]=metric;sens["spatial_shuffle_contract"]="deterministic permutation strictly within each sample's M_valid=1 cells";sens["spatial_shuffle_invariants"]=inv;sens["spatial_shuffle_per_sample"]=per
    dump(path,sens);dump(OUT/"correction_sensitivity.json",sens);finalize(device)
    print(json.dumps({"status":"COMPLETE_STOP","D3":"CORRECTED","metric":metric,"invariants":inv}),flush=True)
if __name__=="__main__":main()
