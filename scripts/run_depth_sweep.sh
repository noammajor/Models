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
#     model   dino | jepa | lejepa | ntp | mae | diffusion | softclt
#     task    forecast | classify
#     layers  2 | 4 | 8 | 12 | 24
#
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_depth_sweep.sh mae forecast 2
#   for N in 2 4 12 24; do ./scripts/run_depth_sweep.sh mae forecast $N; done
#
# LR=<value> overrides the per-objective learning rate, should the depth sweep
# need a depth-specific one; the default keeps it fixed so that depth is the
# only variable.
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

M=${1:?model}; TASK=${2:?task}; LAYERS=${3:?layers}; SEED=${4:-123}

case "$LAYERS" in 2|4|8|12|24) ;; *) echo "unusual depth: $LAYERS" >&2 ;; esac

EPOCHS=20
ts_budget "$M"                  # → LR, BATCH
LR=${LR:-$LR}                   # allow an explicit override
ts_batch $BATCH

ts_task "$TASK"
ts_log "depth_${LAYERS}layers" "$M" "$TASK"

FLAGS="--model $M --pretrain_source monash --encoder_layers $LAYERS --embed_dim 128"
FLAGS="$FLAGS --num_patches $PATCHES --lr $LR --epochs $EPOCHS --ckpt_tag depth$LAYERS"
export TS_CKPT_TAG=depth$LAYERS
# Predictor depth tracks the encoder, as at 8 layers.
case "$M" in jepa|lejepa) FLAGS="$FLAGS --predictor_layers $((LAYERS / 2))" ;; esac
case "$M" in mae|patchtst) FLAGS="$FLAGS --step_size 16" ;; esac

echo "[depth:$LAYERS] $M / $TASK  (lr=$LR batch=$BATCH epochs=$EPOCHS patches=$PATCHES)"
ts_sweep "$M" "$TASK" $SEED
