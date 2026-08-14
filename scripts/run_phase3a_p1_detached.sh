#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/yz/groundingLMM_official"
output_root="$repo_root/outputs/phase3a_phrase_grounding"
training_root="$output_root/training/p1"
checkpoint_root="$repo_root/checkpoints/phase3a_phrase_grounding/p1"
log_dir="$output_root/logs"

if [[ -e "$training_root/metrics.jsonl" || -e "$checkpoint_root/best" || -e "$checkpoint_root/last" ]]; then
  echo "Refusing to overwrite an existing formal Phase 3A P1 trajectory." >&2
  exit 2
fi

mkdir -p "$training_root" "$checkpoint_root" "$log_dir"
cd "$repo_root"
if tmux has-session -t phase3a_p1 2>/dev/null; then
  echo "tmux session phase3a_p1 already exists." >&2
  exit 3
fi
tmux new-session -d -s phase3a_p1 "$repo_root/scripts/run_phase3a_p1_worker.sh"
pane_pid=$(tmux display-message -p -t phase3a_p1 '#{pane_pid}')
printf '%s\n' "$pane_pid" >"$log_dir/p1_training.pid"
echo "Phase 3A P1 tmux session started; pane PID: $pane_pid"
