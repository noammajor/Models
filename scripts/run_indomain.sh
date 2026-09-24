#!/bin/bash
# =============================================================================
# run_indomain.sh — the in-domain pre-training ablation.
#
# Instead of pre-training once on a large corpus and transferring, each encoder
# is pre-trained on the forecasting dataset it is then evaluated on. Same
# encoder, budget, context and epochs as run_per_model.sh — only the corpus is
# the target dataset itself, which is two to four orders of magnitude smaller
# than Monash.
#
# Unlike the other protocols this loops over datasets in the outer position:
# one pre-training run per dataset, each followed by forecasting on that same
# dataset. Checkpoints are tagged by dataset ("_etth1", "_weather", …), so they
# collide with nothing.
#
# Forecasting only. The five datasets are the ones large enough to pre-train on:
#   etth1 etth2 ettm1 ettm2 weather
# (In-domain pre-training on a classification or anomaly dataset is implemented
# for SoftCLT alone, via TS_INDOMAIN="classification:<name>"; it is not part of
# this sweep.)
#
# Usage:
#   ./scripts/run_indomain.sh <model> <seed> [<seed> ...]
#
#     model  dino | jepa | lejepa | ntp | mae | diffusion | softclt
#
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_indomain.sh ntp 123
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_indomain.sh softclt 123 456
#
# Restrict the datasets by exporting IN_DOMAIN_DATASETS first.
# The paper uses seeds 123, 456, 789, 1337, 2003.
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

M=${1:?model}; shift
SEEDS=${@:-$ALL_SEEDS}

EPOCHS=20
PATCHES=21              # 336-step context, as in the forecasting tables
ts_budget "$M"          # → LR, BATCH — the same tuned values as the Monash table
ts_batch $BATCH

ts_log indomain "$M" forecast

FLAGS="--model $M $ENCODER_FLAGS --num_patches $PATCHES --lr $LR --epochs $EPOCHS"
case "$M" in mae|patchtst) FLAGS="$FLAGS --step_size 16" ;; esac

echo "[in-domain] $M  (lr=$LR batch=$BATCH epochs=$EPOCHS patches=$PATCHES)"

for S in $SEEDS; do
  for D in $IN_DOMAIN_DATASETS; do
    echo "[$(date +%H:%M:%S)] $M / in-domain $D / seed $S"

    ts_py "$OUT/pretrain_${D}_seed$S.log" \
      $FLAGS --pretrain_dataset "$D" --task pretrain --seed "$S" \
      || { echo "  pre-training failed, skipping $D"; continue; }

    ts_py "$OUT/${M}_seed${S}_$D.log" \
      $FLAGS --pretrain_dataset "$D" --task forecast --forecast_dataset "$D" --seed "$S"
  done
done
