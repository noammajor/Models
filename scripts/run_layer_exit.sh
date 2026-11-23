#!/bin/bash
# =============================================================================
# run_layer_exit.sh — layer-wise linear probing / early exit of ONE pre-trained
# 24-layer backbone (the depth-sweep checkpoint, logs/depth_24layers/).
#
# The frozen 24-layer encoder is cut after its first k blocks (TS_EXIT_LAYER=k:
# every weight still loads strictly, blocks k+1..24 are skipped in forward) and
# the usual forecasting linear probe is trained on that output. Head, readout,
# splits, horizons and training budget are exactly the normal forecasting
# protocol, so the only thing that changes between k is the depth read out.
# k=24 is the full encoder (identical to the depth-sweep 24-layer run).
#
#   EXIT_LAYERS          blocks kept            (default "2 4 8 12"; k=24 is the
#                        depth-sweep 24-layer run, which analyze_layer_exit.py reuses)
#   FORECAST_DATASETS    datasets               (default "etth1 etth2")
#
# Logs: logs/layer_exit/<model>_forecast/<model>_seed<seed>_exit<k>_<dataset>.log
#
# Usage:
#   ./scripts/run_layer_exit.sh <model> [<seed> ...]        (default seeds 123 456)
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_layer_exit.sh jepa 123 456
#     model  dino | jepa | lejepa | ntp | mae | diffusion | softclt
# =============================================================================
FORECAST_DATASETS=${FORECAST_DATASETS:-"etth1 etth2"}
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

M=${1:?model}; shift
SEEDS=${@:-"123 456"}
EXIT_LAYERS=${EXIT_LAYERS:-"2 4 8 12"}   # 24 = the full encoder: logs/depth_24layers already has it

# Exactly the flags run_depth_sweep.sh used for the 24-layer backbone, so the
# same checkpoint dir is resolved (ckpt_tag depth24, LR = tuned x0.2).
LAYERS=24
ts_budget "$M"
LR=$(awk -v b="$LR" 'BEGIN { printf "%.4g", b * 0.2 }')
ts_batch $BATCH
ts_task forecast
ts_log layer_exit "$M" forecast

FLAGS="--model $M --pretrain_source monash --encoder_layers $LAYERS --embed_dim 128"
FLAGS="$FLAGS --num_patches $PATCHES --lr $LR --epochs 20 --ckpt_tag depth$LAYERS"
export TS_CKPT_TAG=depth$LAYERS
case "$M" in jepa|lejepa) FLAGS="$FLAGS --predictor_layers $((LAYERS / 2))" ;; esac
case "$M" in mae|patchtst) FLAGS="$FLAGS --step_size 16" ;; esac

echo "[layer-exit] $M  24-layer backbone, exits: $EXIT_LAYERS  datasets: $DATASETS  seeds: $SEEDS"
for seed in $SEEDS; do
  for dataset in $DATASETS; do
    for k in $EXIT_LAYERS; do
      echo "[$(date +%H:%M:%S)] $M seed $seed  $dataset  exit after block $k"
      export TS_EXIT_LAYER=$k
      ts_py "$OUT/${M}_seed${seed}_exit${k}_${dataset}.log" \
        $FLAGS --task forecast $DS_ARG "$dataset" --seed "$seed"
    done
  done
done
