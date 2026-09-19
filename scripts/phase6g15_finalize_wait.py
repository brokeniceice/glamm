#!/usr/bin/env python3
"""Wait for all Phase6G.15 arm summaries after a GPU migration, then finalize."""
import subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/phase6g15_residual_staged_optimization';PY='/home/yz/miniconda3/envs/glamm_official/bin/python'
while not all((OUT/'arms'/a/'summary.json').exists() for a in ('A1','A2','C1','C2')): time.sleep(30)
subprocess.run([PY,str(ROOT/'scripts/phase6g15_finalize.py')],cwd=ROOT,check=True)
