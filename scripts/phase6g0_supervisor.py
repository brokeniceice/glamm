#!/usr/bin/env python3
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PY='/home/yz/miniconda3/envs/glamm_official/bin/python';SCRIPT=ROOT/'scripts/phase6g0_multilevel_dense_clip.py'
def run(*args):subprocess.run([PY,str(SCRIPT),*args],cwd=ROOT,check=True)
def main():
 run('protocol');run('cache','--split','train','--device','cuda:0');run('cache','--split','val','--device','cuda:0')
 for layer in ('early','middle','late','current_hidden_minus2'):run('train','--layer',layer,'--device','cuda:0')
 run('analyze')
if __name__=='__main__':main()
