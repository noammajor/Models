#!/bin/bash
# =============================================================================
# launch.sh — fan a protocol script out over the GPUs.
#
# Every run_*.sh in this directory is single-GPU and sequential by design. This
# expands one of them over the seven objectives and five seeds and keeps one job
# per GPU running until the queue drains.
#
# The protocol's arguments are given with two placeholders:
#
#   %M   the model        (from $MODELS, default: all seven objectives)
#   %S   the seed         (from $SEEDS,  default: the paper's five)
#
# Usage:
#   ./scripts/launch.sh "<gpus>" <script> <args with %M and %S>
#
#   # the main Monash forecasting table, 35 jobs over 8 GPUs
#   ./scripts/launch.sh "0 1 2 3 4 5 6 7" ./scripts/run_per_model.sh %M forecast %S
#
#   # one model, one seed per GPU
#   SEEDS="123 456 789 1337 2003" MODELS=mae \
#     ./scripts/launch.sh "0 1 2 3 4" ./scripts/run_equal_budget.sh %M classify %S
#
#   # the hybrid corpus, classification
#   ./scripts/launch.sh "0 1 2 3" ./scripts/run_corpus.sh %M hybrid classify %S
#
#   # the random control, which takes no model
#   MODELS=- ./scripts/launch.sh "0 1 2 3 4" ./scripts/run_random_baseline.sh forecast %S
#
# DRY_RUN=1 prints the jobs instead of running them.
# Each job's own stdout goes to its protocol's log directory; what appears here
# is only the progress line per job.
# =============================================================================
set -u

GPU_LIST=${1:?gpus, e.g. "0 1 2 3"}; shift
SCRIPT=${1:?protocol script}; shift
[ $# -gt 0 ] || { echo "no protocol arguments given" >&2; exit 1; }

MODELS=${MODELS:-"dino jepa lejepa ntp mae diffusion softclt"}
SEEDS=${SEEDS:-"123 456 789 1337 2003"}
DRY_RUN=${DRY_RUN:-0}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

read -r -a GPUS <<< "$GPU_LIST"
NGPU=${#GPUS[@]}
[ "$NGPU" -gt 0 ] || { echo "no GPUs given" >&2; exit 1; }

# ── build the job list ───────────────────────────────────────────────────────
# MODELS=- for protocols that take no model (the random control).
i=0
declare -a QUEUE
for m in $MODELS; do
  for s in $SEEDS; do
    args=""
    for a in "$@"; do
      a=${a//%M/$m}
      a=${a//%S/$s}
      args="$args $a"
    done
    QUEUE[$i]="$args"
    i=$((i + 1))
  done
done

echo "${#QUEUE[@]} jobs over $NGPU GPUs ($GPU_LIST)"

# ── deal the jobs round-robin, then run each GPU's share sequentially ────────
for g in $(seq 0 $((NGPU - 1))); do
  gpu=${GPUS[$g]}
  (
    for i in $(seq $g $NGPU $((${#QUEUE[@]} - 1))); do
      job="${QUEUE[$i]}"
      echo "  gpu $gpu  ←  $SCRIPT$job"
      [ "$DRY_RUN" = 1 ] || CUDA_VISIBLE_DEVICES=$gpu $SCRIPT $job
    done
  ) &
done

wait
echo "all jobs finished"
