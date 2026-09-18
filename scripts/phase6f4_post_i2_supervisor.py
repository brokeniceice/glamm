#!/usr/bin/env python3
"""Wait for I2, then run the authorized post-selection chain on one GPU."""
import subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PY='/home/yz/miniconda3/envs/glamm_official/bin/python'
def run(args):subprocess.run([PY,*map(str,args)],cwd=ROOT,check=True)
def main():
 gate=ROOT/'outputs/phase6f4_full_fov_i2/summary.json'
 while not gate.exists():time.sleep(30)
 if not (ROOT/'outputs/phase6f4_full_fov_i2/official1000/status.json').exists():run([ROOT/'scripts/phase6f4_full_fov_official1000.py','--checkpoint',ROOT/'outputs/phase6f4_full_fov_i2/selected_checkpoint.pt','--selector',ROOT/'outputs/phase6f4_full_fov_i2/selector.json','--output-dir',ROOT/'outputs/phase6f4_full_fov_i2/official1000','--arm-name','Full-FOV-I2','--device','cuda:0'])
 for stage in ('rectifier','utility','joint'):
  if not (ROOT/f'outputs/phase6f4_full_fov_staged/{stage}/summary.json').exists():run([ROOT/'scripts/phase6f4_full_fov_staged_train.py','--stage',stage,'--device','cuda:0'])
 if not (ROOT/'outputs/phase6f4_full_fov_staged/official1000/status.json').exists():run([ROOT/'scripts/phase6f4_full_fov_official1000.py','--checkpoint',ROOT/'outputs/phase6f4_full_fov_staged/joint/selected_checkpoint.pt','--selector',ROOT/'outputs/phase6f4_full_fov_staged/joint/selector.json','--output-dir',ROOT/'outputs/phase6f4_full_fov_staged/official1000','--arm-name','Full-FOV-staged','--device','cuda:0'])
 if not (ROOT/'outputs/phase6f4_full_fov_comparison/phase6f4_full_fov_training_results.json').exists():run([ROOT/'scripts/phase6f4_finalize_all.py'])
if __name__=='__main__':main()
