#!/bin/bash
# =============================================================================
# run_depth_sweep.sh — the encoder-depth ablation (Table 4, lower panel; Fig. 8).
#
# run_per_model.sh at a different encoder depth and nothing else: same corpus,
# context, epochs, learning rate and batch size, so the 8-layer point of this
# sweep IS the main Monash table and the comparisons in the paper are like for
# like. The paper sweeps N in {2, 4, 8, 12, 24}, single-seed.
#
# The predictor of JEPA and LE-JEPA is half the encoder depth throughout, as in
# the main configuration (8 layers -> 4).
#
# Usage:
#   ./scripts/run_depth_sweep.sh <model> <task> <layers> [<seed>]
#
#     model   dino | jepa | lejepa | ntp | mae | diffusion | softclt | autotcl
#     task    forecast | anomaly | classify
#     layers  2 | 4 | 8 | 12 | 24
#
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_depth_sweep.sh mae forecast 2
#   for N in 2 4 12 24; do ./scripts/run_depth_sweep.sh mae forecast $N; done
#
# Learning rate: each objective's tuned 8-layer rate (ts_budget) scaled by
# depth, so the 8-layer point IS the main table and deeper encoders train with
# a proportionally lower rate:
#
#   layers   2     4     8    12    24
#   scale   x6    x2    x1  x0.4  x0.2
#
# LR=<value> overrides the result with a fixed rate; LR_SCALE=<factor> overrides
# only the depth factor.
#
# SKIP_PRETRAIN=1 runs the downstream half only, against the backbone an earlier
# call of this script already pre-trained at the same depth / seed / context.
# Forecasting and anomaly detection share the 336-step backbone, so:
#   ./scripts/run_depth_sweep.sh mae forecast 2 123                   (pre-trains)
#   SKIP_PRETRAIN=1 ./scripts/run_depth_sweep.sh mae anomaly 2 123    (re-uses it)
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

M=${1:?model}; TASK=${2:?task}; LAYERS=${3:?layers}; SEED=${4:-123}

case "$LAYERS" in 2|4|8|12|24) ;; *) echo "unusual depth: $LAYERS" >&2 ;; esac

EPOCHS=20
LR_OVERRIDE=${LR:-}             # capture before ts_budget sets LR
ts_budget "$M"                  # → LR (tuned, 8-layer), BATCH

# Depth scaling of the tuned rate (see header). Unlisted depths keep x1.
case "$LAYERS" in
  2)  LR_SCALE_DEFAULT=6   ;;
  4)  LR_SCALE_DEFAULT=2   ;;
  8)  LR_SCALE_DEFAULT=1   ;;
  12) LR_SCALE_DEFAULT=0.4 ;;
  24) LR_SCALE_DEFAULT=0.2 ;;
  *)  LR_SCALE_DEFAULT=1   ;;
esac
LR_SCALE=${LR_SCALE:-$LR_SCALE_DEFAULT}
LR=$(awk -v b="$LR" -v f="$LR_SCALE" 'BEGIN { printf "%.4g", b * f }')
LR=${LR_OVERRIDE:-$LR}          # an explicit LR=<value> wins over everything
ts_batch $BATCH

ts_task "$TASK"
ts_log "depth_${LAYERS}layers" "$M" "$TASK"

FLAGS="--model $M --pretrain_source monash --encoder_layers $LAYERS --embed_dim 128"
FLAGS="$FLAGS --num_patches $PATCHES --lr $LR --epochs $EPOCHS --ckpt_tag depth$LAYERS"
export TS_CKPT_TAG=depth$LAYERS
export TS_SAVE_EPOCH_CKPTS=0    # keep only the best checkpoint of each pre-training
# Predictor depth tracks the encoder, as at 8 layers.
case "$M" in jepa|lejepa) FLAGS="$FLAGS --predictor_layers $((LAYERS / 2))" ;; esac
case "$M" in mae|patchtst) FLAGS="$FLAGS --step_size 16" ;; esac

echo "[depth:$LAYERS] $M / $TASK  (lr=$LR = tuned x$LR_SCALE, batch=$BATCH epochs=$EPOCHS patches=$PATCHES)"
if [ "${SKIP_PRETRAIN:-0}" = 1 ]; then
  ts_eval_only "$M" "$TASK" $SEED
else
  ts_sweep "$M" "$TASK" $SEED
fi
