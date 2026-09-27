#!/usr/bin/env bash
set -euo pipefail

repo=/home/yz/groundingLMM_official
out="$repo/outputs/phase6i_jepa_discrepancy"
python=/home/yz/miniconda3/envs/glamm_official/bin/python
cd "$repo"
mkdir -p "$out/logs"

CUDA_VISIBLE_DEVICES=1 "$python" -m scripts.phase6i_jepa_discrepancy fake >"$out/logs/fake.log" 2>&1 &
fake_pid=$!
CUDA_VISIBLE_DEVICES=2 "$python" -m scripts.phase6i_jepa_discrepancy real >"$out/logs/real.log" 2>&1 &
real_pid=$!

fake_ok=0
real_ok=0
if wait "$fake_pid"; then fake_ok=1; fi
if wait "$real_pid"; then real_ok=1; fi
if [[ "$fake_ok" -ne 1 || "$real_ok" -ne 1 ]]; then
  "$python" -c 'import json,pathlib; p=pathlib.Path("outputs/phase6i_jepa_discrepancy/pipeline_status.json"); d=json.loads(p.read_text()); d.update(status="WORKER_FAILED", fake_complete=pathlib.Path("outputs/phase6i_jepa_discrepancy/worker_fake.json").exists(), real_complete=pathlib.Path("outputs/phase6i_jepa_discrepancy/worker_real.json").exists()); p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")'
  exit 1
fi

"$python" -m scripts.phase6i_jepa_discrepancy finalize >"$out/logs/finalize.log" 2>&1
