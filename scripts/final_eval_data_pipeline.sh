#!/usr/bin/env bash
set -euo pipefail

root="/home/yz/groundingLMM_official"
python="/home/yz/miniconda3/envs/glamm_official/bin/python"
status="$root/outputs/final_eval_datasets/data_pipeline_status.json"

write_status() {
  "$python" - "$status" "$1" "$2" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, stage = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
value = {"schema":"final_eval_data_pipeline_v1", "status":state, "stage":stage,
         "pid":os.getppid(), "updated_at_utc":datetime.now(timezone.utc).isoformat()}
tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,indent=2)+"\n"); os.replace(tmp,path)
PY
}

cd "$root"
write_status WAITING extraction
while true; do
  state="$($python - <<'PY'
import json
try: print(json.load(open('outputs/final_eval_datasets/extraction_status.json')).get('status','MISSING'))
except Exception: print('MISSING')
PY
)"
  [[ "$state" == COMPLETE ]] && break
  [[ "$state" == FAILED ]] && { write_status FAILED extraction; exit 1; }
  sleep 30
done

# A leakage override never removes or rewrites audit findings.  It only allows
# the already-audited manifests to proceed when explicitly authorized.
override="$($python - <<'PY'
import json
from pathlib import Path
try:
    approval=json.load(open('outputs/final_eval_datasets/leakage_override.json'))
    prepared=json.load(open('outputs/final_eval_datasets/downloaded_prepare_status.json'))
    audit=json.load(open('outputs/final_eval_datasets/leakage_audit/summary.json'))
    ok=(approval.get('status')=='ACTIVE' and prepared.get('status')=='COMPLETE'
        and audit.get('status')=='BLOCKED_OVERLAP')
    print('YES' if ok else 'NO')
except Exception:
    print('NO')
PY
)"
if [[ "$override" == YES ]]; then
  write_status COMPLETE data_gate_override
  exit 0
fi

write_status RUNNING existing_manifests
"$python" scripts/final_eval_prepare_existing.py
write_status RUNNING xaigd
PYTHONPATH=/data/yz/myLISA_storage/AIGC/.tools/pyarrow23 "$python" scripts/final_eval_prepare_xaigd.py
write_status RUNNING downloaded_manifests
"$python" scripts/final_eval_prepare_downloaded.py
write_status RUNNING leakage_audit
"$python" scripts/final_eval_leakage_audit.py
write_status COMPLETE data_gate
