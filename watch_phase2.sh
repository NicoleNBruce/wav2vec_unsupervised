#!/usr/bin/env bash
set -euo pipefail

# Live monitor for Phase 2 (GAN training)
# Usage:
#   bash watch_phase2.sh
#   REFRESH=5 bash watch_phase2.sh

REFRESH="${REFRESH:-3}"
MAX_UPDATE="${MAX_UPDATE:-3000}"

LOG_FILE="/home/nikki/wav2vec_unsupervised/data/results/timit/training1.log"
PIPELINE_LOG="/home/nikki/wav2vec_unsupervised/data/logs/timit/pipeline.log"
CKPT="/home/nikki/wav2vec_unsupervised/data/checkpoints/timit/progress.checkpoint"

format_duration() {
  local total="$1"
  if [[ "$total" -lt 0 ]]; then
    echo "unknown"
    return
  fi
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  printf "%02dh:%02dm:%02ds" "$h" "$m" "$s"
}

last_num_updates() {
  if [[ ! -f "$LOG_FILE" ]]; then
    echo 0
    return
  fi
  grep -oE '"num_updates"[[:space:]]*:[[:space:]]*"?[0-9]+"?' "$LOG_FILE" | tail -n 1 | grep -oE '[0-9]+' || echo 0
}

first_update_epoch() {
  if [[ ! -f "$LOG_FILE" ]]; then
    echo 0
    return
  fi
  local ts
  ts=$(grep -m1 -E '"num_updates"[[:space:]]*:[[:space:]]*"?[0-9]+"?' "$LOG_FILE" | sed -E 's/^\[([^]]+)\].*/\1/' || true)
  if [[ -z "$ts" ]]; then
    echo 0
    return
  fi
  date -d "$ts" +%s 2>/dev/null || echo 0
}

render_bar() {
  local current="$1"
  local total="$2"
  local width=32
  if [[ "$total" -le 0 ]]; then
    total=1
  fi
  if [[ "$current" -lt 0 ]]; then
    current=0
  fi
  if [[ "$current" -gt "$total" ]]; then
    current="$total"
  fi
  local percent=$((current * 100 / total))
  local filled=$((percent * width / 100))
  local empty=$((width - filled))
  local left right
  left=$(printf '%*s' "$filled" '' | tr ' ' '#')
  right=$(printf '%*s' "$empty" '' | tr ' ' '-')
  printf "[%s%s] %3d%% (%d/%d)" "$left" "$right" "$percent" "$current" "$total"
}

clear
while true; do
  printf "\033[H\033[2J"
  echo "=== Phase 2 Live Monitor ==="
  date
  echo

  updates=$(last_num_updates)
  start_epoch=$(first_update_epoch)
  now_epoch=$(date +%s)
  eta="unknown"
  elapsed="unknown"

  if [[ "$start_epoch" -gt 0 && "$updates" -gt 0 ]]; then
    run_elapsed=$((now_epoch - start_epoch))
    if [[ "$run_elapsed" -gt 0 ]]; then
      # Integer arithmetic is enough for dashboard ETA.
      remaining_updates=$((MAX_UPDATE - updates))
      if [[ "$remaining_updates" -lt 0 ]]; then
        remaining_updates=0
      fi
      eta_seconds=$((remaining_updates * run_elapsed / updates))
      eta=$(format_duration "$eta_seconds")
      elapsed=$(format_duration "$run_elapsed")
    fi
  fi

  echo "[Training Progress]"
  render_bar "$updates" "$MAX_UPDATE"
  echo
  echo "Elapsed: $elapsed"
  echo "ETA:     $eta"
  echo

  echo "[Processes]"
  pgrep -af "fairseq-hydra-train|run_gans.sh|hydra_train.py" || echo "No Phase 2 process found"
  echo

  echo "[Checkpoint Tail]"
  if [[ -f "$CKPT" ]]; then
    tail -n 8 "$CKPT"
  else
    echo "Checkpoint file not found: $CKPT"
  fi
  echo

  echo "[Pipeline Log Tail]"
  if [[ -f "$PIPELINE_LOG" ]]; then
    tail -n 6 "$PIPELINE_LOG"
  else
    echo "Pipeline log not found: $PIPELINE_LOG"
  fi
  echo

  echo "[Training Log Tail]"
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 20 "$LOG_FILE"
  else
    echo "Training log not found yet: $LOG_FILE"
  fi
  echo

  echo "Refreshing every ${REFRESH}s. Press Ctrl+C to exit."
  sleep "$REFRESH"
done
