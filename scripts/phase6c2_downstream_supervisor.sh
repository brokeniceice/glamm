#!/usr/bin/env bash
set -uo pipefail
repo=/home/yz/groundingLMM_official
log="$repo/outputs/phase6c2_multiseg_training/logs/l3_downstream.log"
cd "$repo" || exit 2
wait_json() {
  while ! /home/yz/miniconda3/envs/glamm_official/bin/python - "$1" <<'PY'
import json,sys
try: x=json.load(open(sys.argv[1]));raise SystemExit(0 if x.get('status')=='COMPLETE' else 1)
except Exception:raise SystemExit(1)
PY
  do sleep 30; done
}
retry() { until "$@" >>"$log" 2>&1; do printf '%s retry: %s\n' "$(date -u +%FT%TZ)" "$*" >>"$log"; sleep 30; done; }
wait_json outputs/phase6c2_multiseg_training/l3_training/train_complete.json
if [ ! -f outputs/phase6c2_multiseg_training/l2_val_g0_cache/complete.json ]; then
  retry env CUDA_VISIBLE_DEVICES=0,1 /home/yz/miniconda3/envs/glamm_official/bin/torchrun --standalone --nproc_per_node=2 scripts/phase6c2_l2_g0_cache.py --batch-size 2
fi
(
 retry env CUDA_VISIBLE_DEVICES=0 /home/yz/miniconda3/envs/glamm_official/bin/python scripts/phase6c2_l3_evaluate.py --mode g0 --device cuda:0 --epochs 1 3 5 7 9
) & p0=$!
(
 retry env CUDA_VISIBLE_DEVICES=1 /home/yz/miniconda3/envs/glamm_official/bin/python scripts/phase6c2_l3_evaluate.py --mode g0 --device cuda:0 --epochs 2 4 6 8 10
) & p1=$!
wait "$p0"; wait "$p1"
retry /home/yz/miniconda3/envs/glamm_official/bin/python scripts/phase6c2_l3_select.py
selected=$(/home/yz/miniconda3/envs/glamm_official/bin/python - <<'PY'
import json;print(json.load(open('outputs/phase6c2_multiseg_training/l3_selector.json'))['selected_epoch'])
PY
)
retry env CUDA_VISIBLE_DEVICES=0 /home/yz/miniconda3/envs/glamm_official/bin/python scripts/phase6c2_l3_evaluate.py --mode tf --device cuda:0 --epochs "$selected"
retry /home/yz/miniconda3/envs/glamm_official/bin/python scripts/phase6c2_finalize.py
