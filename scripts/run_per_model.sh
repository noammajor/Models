#!/bin/bash
# =============================================================================
# run_per_model.sh — the per-model (tuned) protocol of the paper.
#
# Counterpart to run_equal_budget.sh. The encoder, epochs and context are the
# same for every objective; the learning rate and batch size are each
# objective's own tuned values rather than a shared budget.
#
#   encoder      8 layers, d_model 128, 16 heads, d_ff 512, patch length 16
#   epochs       20
#   context      336 steps (21 patches) for forecasting and anomaly detection
#                1152 steps (72 patches) for classification
#   LR / batch   per objective, see the table below
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
set -u

M=${1:?model}; TASK=${2:?task}; shift 2
SEEDS=${@:?seeds}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="logs/per_model/${M}_${TASK}"
mkdir -p "$OUT"

# ── per-objective learning rate and batch size ───────────────────────────────
case "$M" in
  dino)            LR=5e-4; BATCH=128 ;;
  jepa)            LR=3e-3; BATCH=64  ;;   # SGD with Nesterov momentum
  lejepa)          LR=5e-4; BATCH=64  ;;
  ntp)             LR=1e-4; BATCH=64  ;;
  mae|patchtst)    LR=1e-4; BATCH=64  ;;
  diffusion|timedart) LR=1e-4; BATCH=64 ;;
  softclt)         LR=2e-4; BATCH=64  ;;
  *) echo "unknown model: $M"; exit 1 ;;
esac

EPOCHS=20
export TS_PRETRAIN_BS=$BATCH    # jepa / lejepa / ntp
export TS_PATCHTST_BS=$BATCH    # mae
export TS_TIMEDART_BS=$BATCH    # diffusion
export TS_SOFTCLT_BS=$BATCH     # softclt
export TS_DINO_BS=$BATCH        # dino
export TS_CKPT_TAG=monash

# context: classification uses the full 1152-step window, the rest 336
case "$TASK" in
  classify) PATCHES=72 ;;
  *)        PATCHES=21 ;;
esac

FLAGS="--model $M --pretrain_source monash --encoder_layers 8 --embed_dim 128"
FLAGS="$FLAGS --num_patches $PATCHES --lr $LR --epochs $EPOCHS --ckpt_tag monash"
# MAE patches non-overlappingly; without this it would inherit an overlapping stride
case "$M" in mae|patchtst) FLAGS="$FLAGS --step_size 16" ;; esac

FORECAST_DATASETS="etth1 etth2 ettm1 ettm2 weather electricity traffic"
ANOMALY_DATASETS="MSL PSM SMAP SMD SWaT"
CLASSIFY_DATASETS="EthanolConcentration FaceDetection Handwriting Heartbeat \
JapaneseVowels SelfRegulationSCP1 SelfRegulationSCP2 SpokenArabicDigits UWaveGestureLibrary"

for S in $SEEDS; do
  echo "[per-model] $M / $TASK / seed $S  (lr=$LR batch=$BATCH epochs=$EPOCHS patches=$PATCHES)"

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
