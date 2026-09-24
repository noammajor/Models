#!/bin/bash
# =============================================================================
# run_probe_head.sh — the evaluation-protocol ablation.
#
# All three variants read the SAME per-model Monash backbone that
# run_per_model.sh produced (--ckpt_tag monash); nothing is pre-trained here.
# Only what sits on top of the frozen features changes:
#
#   linear    one Linear layer on a frozen encoder — the paper's main protocol
#   mlp       one hidden layer on a frozen encoder — how much of the gap between
#             objectives a non-linear probe can close
#   finetune  the whole encoder unfrozen — how much of the pre-training survives
#             when the downstream task is free to overwrite it
#
# Run run_per_model.sh <model> <task> first: without those checkpoints there is
# nothing to probe.
#
# Usage:
#   ./scripts/run_probe_head.sh <model> <head> <task> <seed> [<seed> ...]
#
#     model  dino | jepa | lejepa | ntp | mae | diffusion | softclt
#     head   linear | mlp | finetune
#     task   forecast | anomaly | classify
#
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_probe_head.sh ntp mlp forecast 123
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_probe_head.sh mae finetune classify 123 456
#
# The paper uses seeds 123, 456, 789, 1337, 2003.
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

M=${1:?model}; HEAD=${2:?head}; TASK=${3:?task}; shift 3
SEEDS=${@:-$ALL_SEEDS}

case "$HEAD" in
  linear)   HEAD_FLAGS="--head linear" ;;
  mlp)      HEAD_FLAGS="--head mlp" ;;
  finetune) HEAD_FLAGS="--head linear --finetune" ;;
  *) echo "unknown head: $HEAD  (linear | mlp | finetune)" >&2; exit 1 ;;
esac

ts_budget "$M"          # → BATCH; LR is unused, nothing is pre-trained here
ts_batch $BATCH
export TS_CKPT_TAG=monash

ts_task "$TASK"
ts_log "probe_$HEAD" "$M" "$TASK"

FLAGS="--model $M --pretrain_source monash $ENCODER_FLAGS"
FLAGS="$FLAGS --num_patches $PATCHES --ckpt_tag monash $HEAD_FLAGS"
case "$M" in mae|patchtst) FLAGS="$FLAGS --step_size 16" ;; esac

echo "[probe:$HEAD] $M / $TASK  (per-model Monash backbone, patches=$PATCHES)"
ts_eval_only "$M" "$TASK" $SEEDS
