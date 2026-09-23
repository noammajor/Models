#!/bin/bash
# =============================================================================
# run_equal_budget.sh — the equal pre-training budget protocol of the paper.
#
# Every objective is pre-trained with the SAME learning rate, batch size, number
# of epochs, context length and encoder architecture; only the SSL objective
# differs. Downstream, each objective keeps its own tuned linear-probe learning
# rate (classification and anomaly detection use 1e-3 for all objectives).
#
#   encoder      8 layers, d_model 128, 16 heads, d_ff 512, patch length 16
#   pre-training LR 1e-4, batch 64, 20 epochs
#   context      336 steps (21 patches) for forecasting and anomaly detection
#                1152 steps (72 patches) for classification
#   checkpoints  tagged "_equal" so they never collide with other protocols
#
# Usage:
#   ./scripts/run_equal_budget.sh <model> <task> <seed> [<seed> ...]
#
#     model  dino | jepa | lejepa | ntp | patchtst | timedart | softclt
#     task   forecast | anomaly | classify   (each pre-trains first)
#
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_equal_budget.sh ntp forecast 123
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_equal_budget.sh patchtst classify 123 456
#
# Reproducing a full table means running all seven models for the task, one
# seed per GPU; the paper uses seeds 123, 456, 789, 1337, 2003.
# =============================================================================
set -u

M=${1:?model}; TASK=${2:?task}; shift 2
SEEDS=${@:?seeds}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="logs/equal_budget/${M}_${TASK}"
mkdir -p "$OUT"

# ── the equal budget ─────────────────────────────────────────────────────────
LR=1e-4          # identical pre-training LR for every objective
BATCH=64         # identical pre-training batch for every objective
EPOCHS=20        # identical number of pre-training epochs
export TS_PRETRAIN_BS=$BATCH    # jepa / lejepa / ntp
export TS_PATCHTST_BS=$BATCH    # patchtst
export TS_TIMEDART_BS=$BATCH    # timedart
export TS_SOFTCLT_BS=$BATCH     # softclt
export TS_DINO_BS=$BATCH        # dino
export TS_CKPT_TAG=equal

# context: classification uses the full 1152-step window, the rest 336
case "$TASK" in
  classify) PATCHES=72 ;;
  *)        PATCHES=21 ;;
esac

FLAGS="--model $M --pretrain_source monash --encoder_layers 8 --embed_dim 128"
FLAGS="$FLAGS --num_patches $PATCHES --lr $LR --epochs $EPOCHS --ckpt_tag equal"
# MAE patches non-overlappingly; without this it would inherit an overlapping stride
[ "$M" = "patchtst" ] && FLAGS="$FLAGS --step_size 16"

FORECAST_DATASETS="etth1 etth2 ettm1 ettm2 weather electricity traffic"
ANOMALY_DATASETS="MSL PSM SMAP SMD SWaT"
CLASSIFY_DATASETS="EthanolConcentration FaceDetection Handwriting Heartbeat \
JapaneseVowels SelfRegulationSCP1 SelfRegulationSCP2 SpokenArabicDigits UWaveGestureLibrary"

for S in $SEEDS; do
  echo "[equal-budget] $M / $TASK / seed $S  (lr=$LR batch=$BATCH epochs=$EPOCHS patches=$PATCHES)"

  python -u Train_and_downstream.py $FLAGS --task pretrain --seed "$S" \
    > "$OUT/pretrain_cw$((PATCHES * 16))_seed$S.log" 2>&1 || { echo "  pretrain failed, skipping seed $S"; continue; }

  case "$TASK" in
    forecast) DATASETS="$FORECAST_DATASETS"; ARG="--forecast_dataset" ;;
    anomaly)  DATASETS="$ANOMALY_DATASETS";  ARG="--anomaly_dataset" ;;
    classify) DATASETS="$CLASSIFY_DATASETS"; ARG="--classification_dataset" ;;
    *) echo "unknown task: $TASK"; exit 1 ;;
  esac

  for D in $DATASETS; do
    python -u Train_and_downstream.py $FLAGS --task "$TASK" $ARG "$D" --seed "$S" \
      > "$OUT/${M}_seed${S}_$D.log" 2>&1
  done
done
