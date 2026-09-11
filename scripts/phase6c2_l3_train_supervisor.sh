#!/usr/bin/env bash
set -uo pipefail
repo=/home/yz/groundingLMM_official
log="$repo/outputs/phase6c2_multiseg_training/logs/l3_train.log"
done_file="$repo/outputs/phase6c2_multiseg_training/l3_training/train_complete.json"
cd "$repo" || exit 2
while true; do
  if /home/yz/miniconda3/envs/glamm_official/bin/python - "$done_file" <<'PY'
import json,sys
try:
    x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get('status')=='COMPLETE' else 1)
except (FileNotFoundError,json.JSONDecodeError): raise SystemExit(1)
PY
  then exit 0; fi
  CUDA_VISIBLE_DEVICES=0,1 /home/yz/miniconda3/envs/glamm_official/bin/torchrun \
    --standalone --nproc_per_node=2 scripts/phase6c2_l3_train.py --resume >>"$log" 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then continue; fi
  printf '%s supervisor retry after exit=%s\n' "$(date -u +%FT%TZ)" "$code" >>"$log"
  sleep 30
done
