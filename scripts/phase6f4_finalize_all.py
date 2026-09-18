#!/usr/bin/env python3
"""Consolidate Full-FOV I2 and staged internal/Official1000 comparisons."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/phase6f4_full_fov_comparison';DOC=ROOT/'docs/phase6f4_full_fov_training_results.md'
def load(p):return json.load(open(p))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)
def main():
 i2=load(ROOT/'outputs/phase6f4_full_fov_i2/summary.json');ic=load(ROOT/'outputs/phase6f4_full_fov_i2/comparison.json');io=load(ROOT/'outputs/phase6f4_full_fov_i2/official1000/official1000_results.json')
 st=load(ROOT/'outputs/phase6f4_full_fov_staged/summary.json');sc=load(ROOT/'outputs/phase6f4_full_fov_staged/comparison.json');so=load(ROOT/'outputs/phase6f4_full_fov_staged/official1000/official1000_results.json')
 result={'schema':'phase6f4_full_fov_training_comparison_v1','status':'COMPLETE_STOP','arms':{'current_new_r1':'Phase6E2 selected baseline','full_fov_i2':{'selector':i2,'internal_validation':ic,'official1000':io},'full_fov_staged':{'selector':st,'internal_validation':sc,'official1000':so}},'selection_firewall':'both checkpoints selected on internal validation before Official1000','test_accessed':False,'ood_accessed':False};dump(OUT/'phase6f4_full_fov_training_results.json',result)
 def line(name,c,o):return f"| {name} | {c['overall']['A1']['mean_fg_iou']:.6f} | {c['overall']['A1']['mean_fg_f1']:.6f} | {o['metrics']['mean_foreground_iou']:.6f} | {o['metrics']['mean_foreground_f1']:.6f} |"
 DOC.write_text('# Phase 6F.4 — Full-FOV training results\n\n| Arm | Internal-val mean IoU | Internal-val mean F1 | Official1000 mean IoU | Official1000 mean F1 |\n|---|---:|---:|---:|---:|\n'+line('Full-FOV I2',ic,io)+'\n'+line('Full-FOV staged',sc,so)+'\n\nBoth selectors were frozen using internal-validation canonical G0 only. Coverage, aspect-ratio, tile-count strata and paired bootstrap CIs versus current selected new R1 are preserved in each `comparison.json`. No internal test or OOD was accessed.\n\n`COMPLETE_STOP`\n')
if __name__=='__main__':main()
