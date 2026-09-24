#!/bin/bash
# =============================================================================
# run_corpus.sh — the pre-training corpus ablation.
#
# Identical to run_per_model.sh in every respect except the data the encoder is
# pre-trained on, so any difference downstream is attributable to the corpus.
#
#   synthetic        the full synthetic corpus
#   synthetic_small  a Monash-sized random subset of it, so corpus SIZE is held
#                    fixed and only the data-generating process changes
#                    (build it first with scripts/subsample_synthetic.py)
#   hybrid           Monash + the full synthetic corpus
#
# Checkpoints separate themselves by source: "_synthetic", "_synthetic_monashsize"
# and "_monash_synthetic", so none of these collide with the Monash runs or with
# each other and no --ckpt_tag is needed.
#
# Usage:
#   ./scripts/run_corpus.sh <model> <corpus> <task> <seed> [<seed> ...]
#
#     model   dino | jepa | lejepa | ntp | mae | diffusion | softclt
#     corpus  synthetic | synthetic_small | hybrid
#     task    forecast | anomaly | classify   (each pre-trains first)
#
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_corpus.sh ntp synthetic forecast 123
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_corpus.sh softclt hybrid classify 123 456
#
# The paper uses seeds 123, 456, 789, 1337, 2003.
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

M=${1:?model}; CORPUS=${2:?corpus}; TASK=${3:?task}; shift 3
SEEDS=${@:-$ALL_SEEDS}

# Where the Monash-sized synthetic subset was written by subsample_synthetic.py.
# Override by exporting SYNTH_SMALL_DIR before calling this script.
SYNTH_SMALL_DIR=${SYNTH_SMALL_DIR:-/home/shared/datasets/synthetic_data_TS_monashsize}

case "$CORPUS" in
  synthetic)       SRC="--pretrain_source synthetic" ;;
  synthetic_small) SRC="--pretrain_source synthetic --synthetic_data_dir $SYNTH_SMALL_DIR" ;;
  hybrid)          SRC="--pretrain_source monash+synthetic" ;;
  *) echo "unknown corpus: $CORPUS  (synthetic | synthetic_small | hybrid)" >&2; exit 1 ;;
esac

EPOCHS=20
ts_budget "$M"          # → LR, BATCH — the same tuned values as the Monash table
ts_batch $BATCH

ts_task "$TASK"
ts_log "corpus_$CORPUS" "$M" "$TASK"

FLAGS="--model $M $SRC $ENCODER_FLAGS"
FLAGS="$FLAGS --num_patches $PATCHES --lr $LR --epochs $EPOCHS"
case "$M" in mae|patchtst) FLAGS="$FLAGS --step_size 16" ;; esac

echo "[corpus:$CORPUS] $M / $TASK  (lr=$LR batch=$BATCH epochs=$EPOCHS patches=$PATCHES)"
ts_sweep "$M" "$TASK" $SEEDS
