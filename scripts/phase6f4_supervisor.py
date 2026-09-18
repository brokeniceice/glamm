#!/usr/bin/env python3
"""Detached sequential supervisor for Phase 6F.4 cache -> train -> finalize."""
from __future__ import annotations
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PY='/home/yz/miniconda3/envs/glamm_official/bin/python'

def run(args):
 subprocess.run([PY,*map(str,args)],cwd=ROOT,check=True)

def main():
 cache=ROOT/'scripts/phase6f4_cache_full_fov.py'
 run([cache,'--split','train','--device','cuda:0'])
 run([cache,'--split','val','--device','cuda:0'])
 run([ROOT/'scripts/phase6f4_full_fov_i2_train.py'])

if __name__=='__main__':main()
