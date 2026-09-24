#!/bin/bash
# =============================================================================
# run_random_baseline.sh — the untrained-encoder control.
#
# The same transformer encoder, randomly initialised and never pre-trained, then
# frozen and probed exactly like every pre-trained backbone. It is the floor
# every SSL objective in the paper has to clear: whatever a linear head can read
# off random features is not evidence that the objective learned anything.
#
# There is no pre-training stage, so this runs the downstream half only. Two
# random encoders are available:
#
#   mae_random   randomly initialised PatchTST encoder  (the paper's Random row)
#   jepa_random  randomly initialised JEPA encoder
#
# Usage:
#   ./scripts/run_random_baseline.sh <task> <seed> [<seed> ...]
#
#     task  forecast | anomaly | classify
#
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_random_baseline.sh forecast 123
#   MODEL=jepa_random CUDA_VISIBLE_DEVICES=1 ./scripts/run_random_baseline.sh classify 123 456
#
# The seed matters more here than anywhere else — it IS the encoder — so run all
# five: 123, 456, 789, 1337, 2003.
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TASK=${1:?task}; shift
SEEDS=${@:-$ALL_SEEDS}

M=${MODEL:-mae_random}

ts_batch 64
ts_task "$TASK"
ts_log random_baseline "$M" "$TASK"

FLAGS="--model $M $ENCODER_FLAGS --num_patches $PATCHES"
case "$M" in mae_random) FLAGS="$FLAGS --step_size 16" ;; esac

echo "[random] $M / $TASK  (no pre-training, patches=$PATCHES)"
ts_eval_only "$M" "$TASK" $SEEDS
