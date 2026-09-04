#!/usr/bin/env bash
set -euo pipefail

root="/data/yz/myLISA_storage/AIGC"
out="/home/yz/groundingLMM_official/outputs/final_eval_datasets"
status="$out/extraction_status.json"
python="/home/yz/miniconda3/envs/glamm_official/bin/python"

write_status() {
  "$python" - "$status" "$1" "$2" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path, state, stage = map(Path, [sys.argv[1], sys.argv[1], sys.argv[1]]) if False else (Path(sys.argv[1]), sys.argv[2], sys.argv[3])
payload = {"schema": "final_eval_extraction_v1", "status": state, "stage": stage,
           "updated_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getppid()}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY
}

mkdir -p "$out" "$root/AIGI-Holmes/raw/extracted" "$root/GenImage/raw/extracted" "$root/PAL4VST/raw/extracted"
write_status RUNNING AIGI-Holmes
unzip -q -n "$root/AIGI-Holmes/raw/TestSet.zip" -d "$root/AIGI-Holmes/raw/extracted"
write_status RUNNING GenImage
unzip -q -n "$root/GenImage/raw/huggingface_test/genimage_test.zip" -d "$root/GenImage/raw/extracted"
write_status RUNNING PAL4VST
# Keep the pristine archive in raw; extracted content is also raw and preserves
# the official directory structure.  The evaluator manifest later selects test only.
unzip -q -n "$root/PAL4VST/raw/specific_tasks.zip" -d "$root/PAL4VST/raw/extracted"
write_status COMPLETE all
