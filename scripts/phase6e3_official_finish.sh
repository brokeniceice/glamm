#!/usr/bin/env bash
set -euo pipefail

unit=codex-c1-native-r1-official.service
root=/home/yz/groundingLMM_official
python=/home/yz/miniconda3/envs/glamm_official/bin/python

while systemctl --user is-active --quiet "$unit"; do
    sleep 30
done

result=$(systemctl --user show "$unit" --property=Result --value)
test "$result" = success
cd "$root"
"$python" -c 'import json; from pathlib import Path; p=Path("outputs/phase6e3_c1_native_staged/official1000_results.json"); s=json.loads(p.read_text()); assert s["status"] == "COMPLETE", s'
exec "$python" -u scripts/phase6e3_official_compare.py
