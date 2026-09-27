#!/bin/bash
set -euo pipefail
root=/home/yz/groundingLMM_official
while [ ! -f "$root/outputs/phase6g17_type3_incremental/H0_zero_C/summary.json" ] || [ ! -f "$root/outputs/phase6g17_type3_incremental/H1_real_C/summary.json" ]; do sleep 30; done
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /home/yz/miniconda3/envs/glamm_official/bin/python "$root/scripts/phase6g17_type3_incremental.py" >> "$root/outputs/phase6g17_type3_incremental/finalize.log" 2>&1
