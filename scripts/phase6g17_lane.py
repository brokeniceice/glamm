#!/usr/bin/env python3
"""Run one Phase6G.17 arm on an independently assigned GPU."""
import argparse,sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import phase4g1q_conditional_utility as q,phase4hd_rectifier_unfreeze_control as hd,phase6g10_train_arm as g10
from scripts.phase6g13_utility_train import load_head
from scripts.phase6g16_type2_utility import load_collapsed,seed_all
from scripts.phase6g17_type3_incremental import init_model,train
from scripts.phase6e2_c1_specific_r1_train import load_c1_cache

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=('H0_zero_C','H1_real_C'),required=True);a=ap.parse_args()
    device=torch.device('cuda:0');torch.cuda.set_device(device);seed_all();_,collapsed,_=load_collapsed(device);fusion=g10.load_fusion(device);projection=g10.load_projection(device);head=load_head(device);sam=g10.load_sam_runtime(hd.CFG,device)
    p4t,p4v=g10.Phase4FStore(hd.CFG,'train'),g10.Phase4FStore(hd.CFG,'val');fst,fsv=g10.Store('train'),g10.Store('val');dev=g10.load_dev('g0');tc=load_c1_cache('train',p4t.sample_ids);vc=load_c1_cache('val',dev['sample_ids']);data=q.load_ids(p4t.sample_ids,('valid_g0','S64','q_seg','z_L','F24','z_F24','target64','clip_geometries'));data['sample_ids']=p4t.sample_ids
    train(a.arm,init_model(device),collapsed,sam,fusion,projection,head,p4t,p4v,fst,fsv,data,tc,dev,vc,device)
if __name__=='__main__':main()
