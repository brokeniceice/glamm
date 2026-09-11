#!/usr/bin/env bash
set -euo pipefail
root="/home/yz/groundingLMM_official"
python="/home/yz/miniconda3/envs/glamm_official/bin/python"
out="$root/outputs/final_evaluation"
status="$out/supervisor_status.json"
mkdir -p "$out/logs"

write_status(){ "$python" - "$status" "$1" "$2" <<'PY'
import json,os,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1]);v={"schema":"final_evaluation_supervisor_v1","status":sys.argv[2],"stage":sys.argv[3],"pid":os.getppid(),"updated_at_utc":datetime.now(timezone.utc).isoformat(),"sequence":["classification","localization","tables_report"]};t=p.with_suffix('.json.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
PY
}
gpu_free(){ local gpu="$1"; local used; used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"; [[ "$used" -lt 2048 ]]; }
wait_gpu(){ local gpu="$1"; while ! gpu_free "$gpu"; do write_status WAITING "gpu_${gpu}"; sleep 60; done; }

cd "$root"
write_status WAITING data_gate
while true; do
 state="$($python - <<'PY'
import json
try: print(json.load(open('outputs/final_eval_datasets/data_pipeline_status.json')).get('status','MISSING'))
except Exception: print('MISSING')
PY
)"
 [[ "$state" == COMPLETE ]] && break
 [[ "$state" == FAILED ]] && { write_status FAILED data_gate; exit 1; }
 sleep 30
done

# Physical GPU 2 is currently free. Run both frozen classifiers sequentially
# there so another user's active GPU-1 training is never disturbed.
wait_gpu 2
write_status RUNNING classification_r1
CUDA_VISIBLE_DEVICES=2 "$python" scripts/final_eval_classification.py --model r1 --datasets aigi_holmes genimage loki raise998 --device cuda:0 >>"$out/logs/classification_r1.log" 2>&1
write_status RUNNING classification_legion_retrained
CUDA_VISIBLE_DEVICES=2 /home/yz/miniconda3/envs/legion/bin/python scripts/final_eval_classification.py --model legion_retrained --datasets aigi_holmes genimage loki raise998 --device cuda:0 >>"$out/logs/classification_legion_retrained.log" 2>&1

write_status RUNNING localization
while [[ ! -f scripts/final_eval_localization_supervisor.py ]]; do
  write_status WAITING localization_supervisor_ready
  sleep 60
done
"$python" scripts/final_eval_localization_supervisor.py >>"$out/logs/localization.log" 2>&1
write_status RUNNING tables_report
while [[ ! -f scripts/final_eval_finalize.py ]]; do
  write_status WAITING finalizer_ready
  sleep 60
done
"$python" scripts/final_eval_finalize.py >>"$out/logs/finalize.log" 2>&1
write_status COMPLETE all
