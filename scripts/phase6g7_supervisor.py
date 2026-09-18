#!/usr/bin/env python3
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PY='/home/yz/miniconda3/envs/glamm_official/bin/python'
def main():
 for stage in ('rectifier','utility','joint'):
  subprocess.run([PY,str(ROOT/'scripts/phase6g7_staged_r1.py'),'--stage',stage,'--device','cuda:0'],cwd=ROOT,check=True)
 print(json.dumps({'status':'COMPLETE_STOP'}))
if __name__=='__main__':main()
