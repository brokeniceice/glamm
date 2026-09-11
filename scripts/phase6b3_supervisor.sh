#!/usr/bin/env bash
set -euo pipefail
cd /home/yz/groundingLMM_official
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
for dataset in aigi_holmes genimage loki raise998; do
  test "$(/home/yz/miniconda3/envs/glamm_official/bin/python -c "import json;print(json.load(open('outputs/phase6b3_fusion_ood/features/$dataset/complete.json'))['status'])")" = COMPLETE
done
/home/yz/miniconda3/envs/glamm_official/bin/python scripts/phase6b3_fusion_ood.py finalize
