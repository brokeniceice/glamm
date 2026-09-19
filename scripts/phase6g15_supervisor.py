#!/usr/bin/env python3
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; PY='/home/yz/miniconda3/envs/glamm_official/bin/python'
def lane(items,device,log):
    with open(log,'a') as h:
        for arm in items: subprocess.run([PY,str(ROOT/'scripts/phase6g15_residual_staged_optimization.py'),'--arm',arm,'--device',device],cwd=ROOT,env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':'max_split_size_mb:128'},stdout=h,stderr=subprocess.STDOUT,check=True)
def main():
    out=ROOT/'outputs/phase6g15_residual_staged_optimization'; out.mkdir(parents=True,exist_ok=True)
    p0=subprocess.Popen([PY,__file__,'--lane','A1,C1','--device','cuda:0','--log',str(out/'lane0.log')],cwd=ROOT)
    p2=subprocess.Popen([PY,__file__,'--lane','A2,C2','--device','cuda:2','--log',str(out/'lane2.log')],cwd=ROOT)
    if p0.wait() or p2.wait(): raise RuntimeError('Phase6G.15 lane failed')
    subprocess.run([PY,str(ROOT/'scripts/phase6g15_finalize.py')],cwd=ROOT,check=True)
if __name__=='__main__':
    if '--lane' in sys.argv:
        import argparse; p=argparse.ArgumentParser();p.add_argument('--lane');p.add_argument('--device');p.add_argument('--log');a=p.parse_args();lane(a.lane.split(','),a.device,a.log)
    else: main()
