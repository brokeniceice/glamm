#!/usr/bin/env bash
set -euo pipefail

status=/home/yz/groundingLMM_official/outputs/final_eval_datasets/downloads/status.json
if test -f "$status"; then
  /home/yz/miniconda3/envs/glamm_official/bin/python -m json.tool "$status"
else
  echo "status file has not been created yet"
fi
