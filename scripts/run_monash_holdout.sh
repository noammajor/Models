#!/bin/bash
# =============================================================================
# run_monash_holdout.sh — pre-train on Monash with one source held out, then
# forecast that source only. Tests transfer to a series family the encoder
# never saw: electricity_hourly is the same data as the electricity.csv
# forecasting benchmark, so the full-Monash electricity score is partly
# in-distribution.
#
# Everything else is run_per_model.sh: 8 layers, d_model 128, per-objective LR
# and batch, 20 epochs, 336-step context.
#
#   HOLDOUT        Monash .tsf substring held out      (default electricity_hourly)
#   HOLDOUT_EVAL   forecasting set(s) evaluated        (default electricity)
#
# Checkpoints:  <usual dir>..._monash_no_electricity_hourly...  (TS_MONASH_EXCLUDE)
# Logs:         logs/monash_no_electricity_hourly/<model>_forecast/
#
# Usage:
#   ./scripts/run_monash_holdout.sh <model> <seed> [<seed> ...]
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_monash_holdout.sh dino 123
#   SKIP_PRETRAIN=1 ./scripts/run_monash_holdout.sh dino 123      (re-evaluate)
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

M=${1:?model}; shift
SEEDS=${@:-$ALL_SEEDS}

HOLDOUT=${HOLDOUT:-electricity_hourly}
export TS_MONASH_EXCLUDE=$HOLDOUT
FORECAST_DATASETS=${HOLDOUT_EVAL:-electricity}
_tag=$(echo "$HOLDOUT" | tr -c '[:alnum:]\n' '_' | sed 's/^_*//; s/_*$//')

EPOCHS=20
ts_budget "$M"          # → LR, BATCH
ts_batch $BATCH
export TS_CKPT_TAG=monash

ts_task forecast
ts_log "monash_no_${_tag}" "$M" forecast

FLAGS="--model $M --pretrain_source monash $ENCODER_FLAGS"
FLAGS="$FLAGS --num_patches $PATCHES --lr $LR --epochs $EPOCHS --ckpt_tag monash"
case "$M" in mae|patchtst) FLAGS="$FLAGS --step_size 16" ;; esac

echo "[holdout] $M  pre-train Monash minus '$HOLDOUT'  ->  forecast $DATASETS  (lr=$LR batch=$BATCH)"
if [ "${SKIP_PRETRAIN:-0}" = 1 ]; then
  ts_eval_only "$M" forecast $SEEDS
else
  ts_sweep "$M" forecast $SEEDS
fi
