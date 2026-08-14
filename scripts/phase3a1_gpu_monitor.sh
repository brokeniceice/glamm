#!/usr/bin/env bash
set -euo pipefail

cd /home/yz/groundingLMM_official
log="outputs/phase3a1_paired_control/c0_training/logs/gpu_process_monitor.log"
while true; do
  {
    date -u '+timestamp=%Y-%m-%dT%H:%M:%SZ'
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits
  } >>"$log" 2>&1
  sleep 10
done
