#!/usr/bin/env python3
"""Attach supplemental I1 to the Phase6E.2 machine result and report."""
from __future__ import annotations
import json, os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/phase6e2_c1_specific_r1'
MAIN=BASE/'phase6e2_c1_specific_r1.json'
I1=BASE/'i1/phase6e2_i1_random_utility.json'
DOC=ROOT/'docs/phase6e2_c1_specific_r1.md'
MARK='## Supplemental I1 — random utility + Phase4F epoch9 rectifier'

def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)

def main():
    result=json.loads(MAIN.read_text()); i1=json.loads(I1.read_text())
    result['supplemental_I1_random_utility_trained_rectifier']={
        'arm_metrics':i1['arms']['C1+I1_random_utility_R1'],'paired_statistics':i1['paired_statistics'],
        'decisions':i1['decisions'],'initialization':i1['provenance']['initialization'],
        'selector':i1['provenance']['selector'],'result_path':str(I1.resolve())}
    result['status']='COMPLETE_WITH_I1_I2';atomic(MAIN,result)
    text=DOC.read_text(); m=i1['arms']['C1+I1_random_utility_R1']
    appendix=(f'\n{MARK}\n\nI1仅将utility恢复为seed-3407随机初始化；rectifier严格复用Phase4F selected epoch9。其余训练、validation selector与Official1000协议均与Phase6E.2一致。\n\n'
        '| Arm | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 | SEG trigger |\n'
        '|---|---:|---:|---:|---:|---:|\n'
        f"| C1+I1 random-utility R1 | {m['mean_foreground_iou']:.6f} | {m['mean_foreground_f1']:.6f} | {m['global_foreground_iou']:.6f} | {m['global_foreground_f1']:.6f} | {m['seg_trigger_rate']:.6f} |\n")
    if MARK not in text: DOC.write_text(text+appendix)

if __name__=='__main__':main()
