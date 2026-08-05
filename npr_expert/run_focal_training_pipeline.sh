#!/usr/bin/env bash
set -euo pipefail

source /home/yz/miniconda3/etc/profile.d/conda.sh
conda activate glamm_official
cd /home/yz/groundingLMM_official

python -u -m npr_expert.cache_focal_features \
  --device cuda:0 \
  --batch-size 4 \
  --num-workers 8 \
  --save-every 500

exec python -u -m npr_expert.train \
  --device cuda:0 \
  --batch-size 64 \
  --epochs 32 \
  --focal-cache /data/yz/myLISA_storage/checkpoints/FOCAL/focal_vit_l_mean_max_all.pt \
  --run-name official_npr_srm_focal
