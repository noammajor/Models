#!/bin/bash
# =============================================================================
# run_equal_budget.sh — the equal pre-training budget protocol of the paper.
#
# Every objective is pre-trained with the SAME learning rate, batch size, number
# of epochs, context length and encoder architecture; only the SSL objective
# differs. Downstream, each objective keeps its own tuned probe learning rate
# (classification and anomaly detection use 1e-3 for all objectives).
#
#   encoder      8 layers, d_model 128, 16 heads, d_ff 512, patch length 16
#   pre-training LR 1e-4, batch 64, 20 epochs, Monash
#   context      336 steps (21 patches) for forecasting and anomaly detection
#                1152 steps (72 patches) for classification
#   checkpoints  tagged "_equal" so they never collide with other protocols
#
# Usage:
#   ./scripts/run_equal_budget.sh <model> <task> <seed> [<seed> ...]
#
#     model  dino | jepa | lejepa | ntp | mae | diffusion | softclt
#     task   forecast | anomaly | classify   (each pre-trains first)
#
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_equal_budget.sh ntp forecast 123
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_equal_budget.sh mae classify 123 456
#
# Reproducing a full table means running all seven models for the task, one
# seed per GPU; the paper uses seeds 123, 456, 789, 1337, 2003.
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

M=${1:?model}; TASK=${2:?task}; shift 2
SEEDS=${@:-$ALL_SEEDS}

# ── the equal budget ─────────────────────────────────────────────────────────
LR=1e-4          # identical pre-training LR for every objective
BATCH=64         # identical pre-training batch for every objective
EPOCHS=20        # identical number of pre-training epochs
ts_batch $BATCH
export TS_CKPT_TAG=equal

ts_task "$TASK"
ts_log equal_budget "$M" "$TASK"

FLAGS="--model $M --pretrain_source monash $ENCODER_FLAGS"
FLAGS="$FLAGS --num_patches $PATCHES --lr $LR --epochs $EPOCHS --ckpt_tag equal"
# MAE patches non-overlappingly; without this it would inherit an overlapping stride
case "$M" in mae|patchtst) FLAGS="$FLAGS --step_size 16" ;; esac

echo "[equal-budget] $M / $TASK  (lr=$LR batch=$BATCH epochs=$EPOCHS patches=$PATCHES)"
ts_sweep "$M" "$TASK" $SEEDS
