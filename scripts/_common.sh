#!/bin/bash
# =============================================================================
# _common.sh — pieces shared by every experiment script. Sourced, never run.
#
# Each run_*.sh in this directory declares only what its protocol changes; the
# dataset lists, the per-objective budget and the pretrain→downstream loop all
# live here, so two protocols can be compared by diffing their scripts.
# =============================================================================

# Fail on undefined variables; do NOT set -e — a single dataset that OOMs should
# not abort the remaining datasets or seeds.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# ── the evaluation suites ────────────────────────────────────────────────────
# Each is overridable from the environment, so a single dataset can be run to
# check a protocol before committing GPUs to the whole suite:
#   FORECAST_DATASETS=etth1 ./scripts/run_per_model.sh mae forecast 123
FORECAST_DATASETS=${FORECAST_DATASETS:-"etth1 etth2 ettm1 ettm2 weather electricity traffic"}
ANOMALY_DATASETS=${ANOMALY_DATASETS:-"MSL PSM SMAP SMD SWaT"}
CLASSIFY_DATASETS=${CLASSIFY_DATASETS:-"EthanolConcentration FaceDetection Handwriting Heartbeat \
JapaneseVowels SelfRegulationSCP1 SelfRegulationSCP2 SpokenArabicDigits UWaveGestureLibrary"}

# Forecasting datasets small enough to pre-train on by themselves (in-domain).
IN_DOMAIN_DATASETS=${IN_DOMAIN_DATASETS:-"etth1 etth2 ettm1 ettm2 weather"}

# The five seeds every table in the paper averages over.
ALL_SEEDS=${ALL_SEEDS:-"123 456 789 1337 2003"}

# The shared encoder: 8 layers, d_model 128, 16 heads, d_ff 512, patch length 16.
ENCODER_FLAGS="--encoder_layers 8 --embed_dim 128"

# DRY_RUN=1 prints the commands a protocol would run and executes nothing.
DRY_RUN=${DRY_RUN:-0}

# ts_py <logfile> <args...> — one Train_and_downstream call. Returns its status.
ts_py() {
  local log=$1; shift
  if [ "$DRY_RUN" = 1 ]; then
    echo "  python -u Train_and_downstream.py $*   > $log"
    return 0
  fi
  python -u Train_and_downstream.py "$@" > "$log" 2>&1
}

# ── ts_budget <model> → LR, BATCH ────────────────────────────────────────────
# The per-objective tuned values. Every protocol except run_equal_budget.sh
# pre-trains with these, so the corpus / in-domain / epoch ablations differ from
# the main table in one thing only.
ts_budget() {
  case "$1" in
    dino)               LR=5e-4; BATCH=128 ;;
    jepa)               LR=3e-3; BATCH=64  ;;   # SGD with Nesterov momentum
    lejepa)             LR=5e-4; BATCH=64  ;;
    ntp)                LR=1e-4; BATCH=64  ;;
    mae|patchtst)       LR=1e-4; BATCH=64  ;;
    diffusion|timedart) LR=1e-4; BATCH=64  ;;
    softclt)            LR=2e-4; BATCH=64  ;;
    *) echo "unknown model: $1" >&2; exit 1 ;;
  esac
}

# ── ts_batch <n> → export the batch size to every implementation ─────────────
# The objectives live in separate sub-projects with their own configs; these are
# the env overrides each one reads.
ts_batch() {
  export TS_PRETRAIN_BS=$1    # jepa / lejepa / ntp
  export TS_MAE_BS=$1         # mae
  export TS_DIFFUSION_BS=$1   # diffusion
  export TS_SOFTCLT_BS=$1     # softclt
  export TS_DINO_BS=$1        # dino
}

# ── ts_task <task> → PATCHES, DATASETS, DS_ARG ───────────────────────────────
# Classification reads the full 1152-step window (72 patches); forecasting and
# anomaly detection use the 336-step context (21 patches).
ts_task() {
  case "$1" in
    forecast) PATCHES=21; DATASETS="$FORECAST_DATASETS"; DS_ARG="--forecast_dataset" ;;
    anomaly)  PATCHES=21; DATASETS="$ANOMALY_DATASETS";  DS_ARG="--anomaly_dataset" ;;
    classify) PATCHES=72; DATASETS="$CLASSIFY_DATASETS"; DS_ARG="--classification_dataset" ;;
    *) echo "unknown task: $1  (forecast | anomaly | classify)" >&2; exit 1 ;;
  esac
}

# ── ts_log <protocol> <model> <task> → OUT ───────────────────────────────────
ts_log() {
  OUT="logs/$1/${2}_${3}"
  mkdir -p "$OUT"
}

# ── ts_sweep <model> <task> <seed...> ────────────────────────────────────────
# Pre-train, then evaluate every dataset of the task. Expects FLAGS, OUT and
# PATCHES to be set; $FLAGS must already carry --model, the encoder, the budget
# and whatever the protocol overrides.
ts_sweep() {
  local model=$1 task=$2; shift 2
  local seed dataset
  for seed in "$@"; do
    echo "[$(date +%H:%M:%S)] $model / $task / seed $seed"

    ts_py "$OUT/pretrain_cw$((PATCHES * 16))_seed$seed.log" \
      $FLAGS --task pretrain --seed "$seed" \
      || { echo "  pre-training failed, skipping seed $seed"; continue; }

    for dataset in $DATASETS; do
      ts_py "$OUT/${model}_seed${seed}_$dataset.log" \
        $FLAGS --task "$task" $DS_ARG "$dataset" --seed "$seed"
    done
  done
}

# ── ts_eval_only <model> <task> <seed...> ────────────────────────────────────
# The downstream half alone, for protocols that re-use an existing backbone.
ts_eval_only() {
  local model=$1 task=$2; shift 2
  local seed dataset
  for seed in "$@"; do
    echo "[$(date +%H:%M:%S)] $model / $task / seed $seed"
    for dataset in $DATASETS; do
      ts_py "$OUT/${model}_seed${seed}_$dataset.log" \
        $FLAGS --task "$task" $DS_ARG "$dataset" --seed "$seed"
    done
  done
}
