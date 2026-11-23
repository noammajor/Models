#!/bin/bash
# =============================================================================
# run_after.sh — start a command once earlier jobs have fully finished.
#
#   ./scripts/run_after.sh [--pid PID]... [--match PATTERN]... [--gpu N]... -- <command...>
#
#   --pid PID        wait until that process exits (use the chain's parent bash,
#                    which lives for the whole chain, not one python run)
#   --match PATTERN  wait until no process of yours matches `pgrep -f PATTERN`
#                    e.g. --match 'lejepa forecast 4 123'
#   --gpu N          wait until none of your processes use GPU N, seen free on
#                    3 checks in a row, so the few seconds between two runs of
#                    a chain do not count as finished
#
# Conditions combine (all must hold). Polls every POLL seconds (default 60).
# Run it detached like any job:
#   setsid nohup ./scripts/run_after.sh --gpu 5 -- bash -c 'CUDA_VISIBLE_DEVICES=5 ./scripts/...' \
#     > logs/after_gpu5.out 2>&1 < /dev/null &
# =============================================================================
set -u
POLL=${POLL:-60}
PIDS=(); PATS=(); GPUS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --pid)   PIDS+=("$2"); shift 2 ;;
    --match) PATS+=("$2"); shift 2 ;;
    --gpu)   GPUS+=("$2"); shift 2 ;;
    --)      shift; break ;;
    *) echo "unknown option: $1 (expected --pid / --match / --gpu, then -- <command>)" >&2; exit 1 ;;
  esac
done
[ $# -gt 0 ] || { echo "no command given after --" >&2; exit 1; }
[ ${#PIDS[@]} -gt 0 ] || [ ${#PATS[@]} -gt 0 ] || [ ${#GPUS[@]} -gt 0 ] \
  || { echo "nothing to wait for: give --pid, --match or --gpu" >&2; exit 1; }

ME=$(id -un); SELF=$$

busy_gpu() {   # 0 if any of my processes is on GPU $1
  local uuid
  uuid=$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader 2>/dev/null) || return 1
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | while IFS=', ' read -r g p; do
        [ "$g" = "$uuid" ] && [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$ME" ] && echo y
      done | grep -q y
}

busy() {
  local p
  for p in "${PIDS[@]+"${PIDS[@]}"}"; do kill -0 "$p" 2>/dev/null && return 0; done
  for p in "${PATS[@]+"${PATS[@]}"}"; do
    pgrep -u "$ME" -f -- "$p" | grep -vx "$SELF" | grep -q . && return 0
  done
  for p in "${GPUS[@]+"${GPUS[@]}"}"; do busy_gpu "$p" && return 0; done
  return 1
}

echo "[$(date '+%m-%d %H:%M:%S')] waiting: pids=[${PIDS[*]-}] match=[${PATS[*]-}] gpus=[${GPUS[*]-}]"
free=0
while [ $free -lt 3 ]; do
  if busy; then free=0; else free=$((free + 1)); fi
  [ $free -lt 3 ] && sleep "$POLL"
done
echo "[$(date '+%m-%d %H:%M:%S')] all clear, starting: $*"
exec "$@"
