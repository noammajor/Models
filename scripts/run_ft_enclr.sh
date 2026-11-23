#!/bin/bash
# =============================================================================
# run_ft_enclr.sh — robustness of full fine-tuning to the ENCODER learning rate,
# on the per-model 8-layer Monash backbone (the main-table checkpoint).
#
# Same as run_probe_head.sh <model> finetune forecast: the pre-trained encoder is
# loaded and trained end to end with a linear forecasting head. The head keeps
# each objective's usual forecasting LR; only the encoder's LR is swept, through
# TS_FT_ENCODER_LR (read by every objective's fine-tuning optimiser; MAE gets a
# head / encoder parameter-group split for it).
#
#   ENC_LRS              encoder LRs   (default "1e-3 5e-4 5e-5 1e-5 1e-6")
#   FORECAST_DATASETS    datasets      (default "etth1 etth2 ettm1 ettm2")
#   CKPT_TAG             backbone tag  (default "depth8", the 8-layer depth-sweep backbone;
#                        "monash" has no DINO/SoftCLT/Diffusion checkpoint on the server)
#
# Logs: logs/ft_enclr/<model>_forecast/<model>_seed<seed>_enc<lr>_<dataset>.log
#
# Usage:
#   ./scripts/run_ft_enclr.sh <model> [<seed> ...]          (default seed 123)
#   CUDA_VISIBLE_DEVICES=6 ./scripts/run_ft_enclr.sh jepa 123
#     model  dino | jepa | lejepa | ntp | mae | diffusion | softclt | autotcl
# =============================================================================
FORECAST_DATASETS=${FORECAST_DATASETS:-"etth1 etth2 ettm1 ettm2"}
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

M=${1:?model}; shift
SEEDS=${@:-123}
ENC_LRS=${ENC_LRS:-"1e-3 5e-4 5e-5 1e-5 1e-6"}
CKPT_TAG=${CKPT_TAG:-depth8}

ts_budget "$M"          # → BATCH (nothing is pre-trained here)
ts_batch $BATCH
export TS_CKPT_TAG=$CKPT_TAG
ts_task forecast
ts_log ft_enclr "$M" forecast

FLAGS="--model $M --pretrain_source monash $ENCODER_FLAGS"
FLAGS="$FLAGS --num_patches $PATCHES --ckpt_tag $CKPT_TAG --head linear --finetune"
case "$M" in mae|patchtst) FLAGS="$FLAGS --step_size 16" ;; esac

echo "[ft-enclr] $M  8-layer Monash backbone ($CKPT_TAG), encoder LRs: $ENC_LRS  datasets: $DATASETS  seeds: $SEEDS"
for seed in $SEEDS; do
  for dataset in $DATASETS; do
    for lr in $ENC_LRS; do
      echo "[$(date +%H:%M:%S)] $M seed $seed  $dataset  encoder_lr $lr"
      export TS_FT_ENCODER_LR=$lr
      ts_py "$OUT/${M}_seed${seed}_enc${lr}_${dataset}.log" \
        $FLAGS --task forecast $DS_ARG "$dataset" --seed "$seed"
    done
  done
done
