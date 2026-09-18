#!/usr/bin/env python3
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PY='/home/yz/miniconda3/envs/glamm_official/bin/python'
def run(args):subprocess.run([PY,*map(str,args)],cwd=ROOT,check=True)
def main():
 run([ROOT/'scripts/phase6g1_block17_adapter.py'])
 for stage in ('rectifier','utility','joint'):run([ROOT/'scripts/phase6g1_block17_staged_r1.py','--stage',stage,'--device','cuda:0'])
if __name__=='__main__':main()
