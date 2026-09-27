#!/usr/bin/env python3
"""Recompute only Phase6G.16 selected-A1 post-hoc diagnostics."""
from pathlib import Path
import json, sys, torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts import phase6g10_train_arm as g10
from scripts.phase6g13_utility_train import load_head
from scripts.phase6g16_type2_utility import (OUT, TypeIIUtility, diagnostics, dump, finalize,
    load_collapsed, seed_all)
from scripts import phase4hd_rectifier_unfreeze_control as hd
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache

def main():
    device=torch.device("cuda:0");torch.cuda.set_device(device);seed_all()
    _,collapsed,_=load_collapsed(device);fusion=g10.load_fusion(device);projection=g10.load_projection(device)
    head=load_head(device);sam=g10.load_sam_runtime(hd.CFG,device);p4=g10.Phase4FStore(hd.CFG,"val")
    fs=g10.Store("val");dev=g10.load_dev("g0");cache=load_c1_cache("val",dev["sample_ids"])
    model=TypeIIUtility().to(device);state=torch.load(OUT/"A1_type2_correction_aware/selected_checkpoint.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(state["model"]);sens=diagnostics(model,collapsed,sam,fusion,projection,head,p4,fs,dev,cache,device)
    sens["spatial_shuffle_contract"]="within M_valid=1 only; exact per-channel/global norm preservation"
    dump(OUT/"diagnostics/correction_sensitivity.json",sens);dump(OUT/"correction_sensitivity.json",sens);finalize(device)
    print(json.dumps({"status":"COMPLETE_STOP","spatial_shuffle":"CORRECTED"}))
if __name__=="__main__":main()
