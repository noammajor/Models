#!/bin/bash
# =============================================================================
# run_long_pretrain.sh — the pre-training length ablation (40 epochs).
#
# run_per_model.sh with the epoch count doubled and nothing else touched: same
# corpus, encoder, context, learning rate and batch size. Answers whether the
# ranking of the objectives is an artefact of the 20-epoch budget.
#
# Checkpoints pick up an "_ep40" tag automatically (any epoch count other than
# the default 20 does), so the 20-epoch backbones are left intact.
#
# Usage:
#   ./scripts/run_long_pretrain.sh <model> <task> <seed> [<seed> ...]
#
#     model  dino | jepa | lejepa | ntp | mae | diffusion | softclt
#     task   forecast | anomaly | classify   (each pre-trains first)
#
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_long_pretrain.sh ntp forecast 123
#   EPOCHS=80 CUDA_VISIBLE_DEVICES=1 ./scripts/run_long_pretrain.sh mae forecast 123
#
# The paper uses seeds 123, 456, 789, 1337, 2003.
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

M=${1:?model}; TASK=${2:?task}; shift 2
SEEDS=${@:-$ALL_SEEDS}

EPOCHS=${EPOCHS:-40}
ts_budget "$M"          # → LR, BATCH — the same tuned values as the Monash table
ts_batch $BATCH

ts_task "$TASK"
ts_log "long_pretrain_ep$EPOCHS" "$M" "$TASK"

FLAGS="--model $M --pretrain_source monash $ENCODER_FLAGS"
FLAGS="$FLAGS --num_patches $PATCHES --lr $LR --epochs $EPOCHS"
case "$M" in mae|patchtst) FLAGS="$FLAGS --step_size 16" ;; esac

echo "[long-pretrain] $M / $TASK  (lr=$LR batch=$BATCH epochs=$EPOCHS patches=$PATCHES)"
ts_sweep "$M" "$TASK" $SEEDS
