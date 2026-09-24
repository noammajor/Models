#!/bin/bash
# =============================================================================
# run_per_model.sh — the per-model (tuned) protocol of the paper.
#
# Counterpart to run_equal_budget.sh. The encoder, epochs and context are the
# same for every objective; the learning rate and batch size are each
# objective's own tuned values rather than a shared budget. This is the main
# Monash table, and the backbone the corpus / epoch / probe ablations vary from.
#
#   encoder      8 layers, d_model 128, 16 heads, d_ff 512, patch length 16
#   pre-training per-objective LR and batch (see ts_budget in _common.sh),
#                20 epochs, Monash
#   context      336 steps (21 patches) for forecasting and anomaly detection
#                1152 steps (72 patches) for classification
#   checkpoints  tagged "_monash" so they never collide with the equal-budget runs
#
# Usage:
#   ./scripts/run_per_model.sh <model> <task> <seed> [<seed> ...]
#
#     model  dino | jepa | lejepa | ntp | mae | diffusion | softclt
#     task   forecast | anomaly | classify   (each pre-trains first)
#
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_per_model.sh ntp forecast 123
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_per_model.sh mae classify 123 456
#
# The paper uses seeds 123, 456, 789, 1337, 2003.
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

M=${1:?model}; TASK=${2:?task}; shift 2
SEEDS=${@:-$ALL_SEEDS}

EPOCHS=20
ts_budget "$M"          # → LR, BATCH
ts_batch $BATCH
export TS_CKPT_TAG=monash

ts_task "$TASK"
ts_log per_model "$M" "$TASK"

FLAGS="--model $M --pretrain_source monash $ENCODER_FLAGS"
FLAGS="$FLAGS --num_patches $PATCHES --lr $LR --epochs $EPOCHS --ckpt_tag monash"
# MAE patches non-overlappingly; without this it would inherit an overlapping stride
case "$M" in mae|patchtst) FLAGS="$FLAGS --step_size 16" ;; esac

echo "[per-model] $M / $TASK  (lr=$LR batch=$BATCH epochs=$EPOCHS patches=$PATCHES)"
ts_sweep "$M" "$TASK" $SEEDS
