#!/usr/bin/env bash
set -euo pipefail

cd /home/yz/groundingLMM_official
run="outputs/phase3a1_paired_control/c0_training/run"
checkpoint="/data/yz/groundingLMM_official/checkpoints/phase3a1_paired_control/c0"
if [[ -e "$run/metrics.jsonl" || -e "$checkpoint/best" || -e "$checkpoint/last" ]]; then
  echo "Refusing to overwrite an existing formal Phase 3A.1 C0 trajectory." >&2
  exit 2
fi
if tmux has-session -t phase3a1_c0 2>/dev/null; then
  echo "tmux session phase3a1_c0 already exists." >&2
  exit 3
fi
mkdir -p "$run" "$checkpoint" "outputs/phase3a1_paired_control/c0_training/logs"
tmux new-session -d -s phase3a1_c0 "/home/yz/groundingLMM_official/scripts/run_phase3a1_c0_worker.sh"
if ! tmux has-session -t phase3a1_gpu_monitor 2>/dev/null; then
  tmux new-session -d -s phase3a1_gpu_monitor "/home/yz/groundingLMM_official/scripts/phase3a1_gpu_monitor.sh"
fi
tmux display-message -p -t phase3a1_c0 '#{pane_pid}' \
  >outputs/phase3a1_paired_control/c0_training/logs/formal_training.pid
