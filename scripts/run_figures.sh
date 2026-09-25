#!/bin/bash
# =============================================================================
# run_figures.sh — the figures and embedding-space diagnostics of the paper.
#
# None of these train anything: each runs forward passes through the frozen
# backbones run_per_model.sh produced, so run that first for the task a figure
# reads (forecast context for isotropy and drift, classify for t-SNE and the
# Gaussianity panels).
#
#   isotropy      effective rank and participation ratio of the frozen per-patch
#                 embeddings, scattered against the forecasting gain over the
#                 random encoder
#   drift         latent drift 1 - cos(z_clean, z_perturbed) under circular
#                 time-shift and random time-masking
#   tsne          t-SNE of classification embeddings, one figure per objective
#   gaussianity   whether the LE-JEPA embedding distribution is the isotropic
#                 Gaussian SIGReg pushes for
#
# Usage:
#   ./scripts/run_figures.sh [isotropy | drift | tsne | gaussianity | all]
#
#   ./scripts/run_figures.sh all
#   SEED=456 GPU=3 ./scripts/run_figures.sh isotropy
#   DRY_RUN=1 ./scripts/run_figures.sh all
#
# The four scripts ship with different defaults for seed, GPU and pre-training
# source — t-SNE defaults to the hybrid corpus, drift to seed 1337 — so this
# passes all of them explicitly rather than inheriting whichever each one has.
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

STAGE=${1:-all}
case "$STAGE" in isotropy|drift|tsne|gaussianity|all) ;; *)
  echo "unknown figure: $STAGE  (isotropy | drift | tsne | gaussianity | all)" >&2; exit 1 ;;
esac

# ── what every figure is pinned to ───────────────────────────────────────────
SEED=${SEED:-123}               # the paper's first seed
LAYERS=${LAYERS:-8}
SOURCE=${SOURCE:-monash}        # the per-model Monash backbones
GPU=${GPU:-0}
OUT=${OUT:-plots}

# Both name sets work; these are the paper's.
MODELS=${MODELS:-"dino jepa lejepa ntp mae diffusion softclt"}

# Forecasting datasets the embedding statistics are computed over, and the
# classification datasets t-SNE is drawn on.
ISO_DATASETS=${ISO_DATASETS:-"etth1 weather"}
TSNE_DATASETS=${TSNE_DATASETS:-"JapaneseVowels Heartbeat SpokenArabicDigits"}
GAUSS_DATASET=${GAUSS_DATASET:-SpokenArabicDigits}

DRY_RUN=${DRY_RUN:-0}
run() { echo "    $*"; [ "$DRY_RUN" = 1 ] || "$@"; }

if [ "$STAGE" = isotropy ] || [ "$STAGE" = all ]; then
  echo "[figures] isotropy vs forecasting gain  (seed $SEED, $SOURCE)"
  # --ckpt_override MODEL=PATH pins an individual backbone when its checkpoint
  # is not where the default resolver looks.
  run python -u Visuals/e4_isotropy.py \
    --models $MODELS --datasets $ISO_DATASETS \
    --encoder_layers "$LAYERS" --seed "$SEED" --pretrain_source "$SOURCE" \
    --gpu "$GPU" --output_dir "$OUT"
fi

if [ "$STAGE" = drift ] || [ "$STAGE" = all ]; then
  echo "[figures] latent drift under perturbation  (seed $SEED, $SOURCE)"
  run python -u Visuals/latent_drift.py \
    --models $MODELS --datasets $ISO_DATASETS --context forecast \
    --encoder_layers "$LAYERS" --seed "$SEED" --pretrain_source "$SOURCE" \
    --gpu "$GPU" --output_dir "$OUT/latent_drift"
fi

if [ "$STAGE" = tsne ] || [ "$STAGE" = all ]; then
  echo "[figures] t-SNE of classification embeddings  (seed $SEED, $SOURCE)"
  # Reads the cw1152 classification backbones, so run the classify task first.
  # Note the script's own default source is monash+synthetic; $SOURCE overrides it.
  run python -u Visuals/tsne_embeddings.py \
    --models random $MODELS --datasets $TSNE_DATASETS \
    --encoder_layers "$LAYERS" --seed "$SEED" --pretrain_source "$SOURCE" \
    --gpu "$GPU" --output_dir "$OUT"
fi

if [ "$STAGE" = gaussianity ] || [ "$STAGE" = all ]; then
  echo "[figures] LE-JEPA embedding Gaussianity  ($GAUSS_DATASET)"
  # LE-JEPA only, and it takes a checkpoint rather than a model and seed.
  # Pass CKPT=<path> to point it at a specific backbone.
  _CKPT_ARG=""
  [ -n "${CKPT:-}" ] && _CKPT_ARG="--ckpt $CKPT"
  run python -u Visuals/lejepa_gaussianity.py \
    --dataset "$GAUSS_DATASET" --encoder_layers "$LAYERS" \
    $_CKPT_ARG --output_dir "$OUT"
fi

echo "[figures] done — written to $OUT/"
