#!/bin/bash
# =============================================================================
# run_synthetic_generation.sh — build the synthetic pre-training corpora.
#
# Produces the two corpora the ablations pre-train on:
#
#   full    the synthetic corpus itself, written as GluonTS .arrow files into
#           $OUT_DIR. Two generators, both Gaussian-process based:
#             LMC_Synth.py     multivariate, via the Linear Coregionalization
#                              Model — the bulk of the corpus
#             kernel-synth.py  univariate, the Chronos kernel-synth procedure
#           The dataloader discovers every .arrow file in the directory and
#           handles univariate [T] and multivariate [C, T] targets alike, so the
#           two sit side by side.
#
#   subset  a random subset of those same series whose total filtered timestep
#           count matches Monash. "Synthetic vs Monash" otherwise confounds the
#           data-generating process with corpus size; this holds size fixed so
#           only the process differs. run_corpus.sh synthetic_small uses it.
#
# Usage:
#   ./scripts/run_synthetic_generation.sh [full | subset | all]     (default: all)
#
#   # the corpus as used in the paper, on 16 cores
#   JOBS=16 ./scripts/run_synthetic_generation.sh full
#
#   # just the Monash-sized subset, from an already-generated corpus
#   ./scripts/run_synthetic_generation.sh subset
#
# Every parameter below can be overridden from the environment, e.g.
#   N_SERIES=2000 OUT_DIR=/tmp/synth ./scripts/run_synthetic_generation.sh full
#
# Generation is CPU-bound and takes hours at the paper's settings: each series
# is a draw from a GP prior, and the LMC pass draws several latent series per
# output series. Raise JOBS to the core count and run it detached.
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

GEN="scripts/synthetic_data_generation"
STAGE=${1:-all}

# ── where the corpora go ─────────────────────────────────────────────────────
# Defaults match synthetic_data_dir in data_paths.py; SUBSET_DIR is what
# run_corpus.sh expects for the synthetic_small protocol.
OUT_DIR=${OUT_DIR:-/home/shared/datasets/synthetic_data_TS}
SUBSET_DIR=${SUBSET_DIR:-/home/shared/datasets/synthetic_data_TS_monashsize}

# ── LMC_Synth: the multivariate bulk of the corpus ───────────────────────────
N_SERIES=${N_SERIES:-4000}          # -N  rows written
LENGTH=${LENGTH:-2500}              # -L  timesteps per series
CHANNELS=${CHANNELS:-160}           # -C  channels per series (each counts as its own
                                    #     series downstream — the loaders split them),
                                    #     so a row carries 160 x 2500 = 400k timesteps
                                    #     and 4000 rows give the ~1.61 B corpus
MAX_KERNELS=${MAX_KERNELS:-5}       # -J  max base kernels per latent function
DIRICHLET_MIN=${DIRICHLET_MIN:-0.1} # -M  lower bound of the mixing-weight alpha
DIRICHLET_MAX=${DIRICHLET_MAX:-1.0} # -X  upper bound
WEIBULL_SHAPE=${WEIBULL_SHAPE:-1.5} # -W  shape of the latent-count distribution
WEIBULL_SCALE=${WEIBULL_SCALE:-2.0} # -Z  scale

# ── kernel-synth: the univariate component ───────────────────────────────────
KS_SERIES=${KS_SERIES:-4000}
KS_LENGTH=${KS_LENGTH:-2500}

JOBS=${JOBS:-16}                    # -P  parallel workers, set to the core count

# ── subset: match Monash's filtered size exactly ─────────────────────────────
# 445,011,429 is Monash's timestep count under the same >=512 filter the
# pre-training loaders apply; recount it with count_dataset_sizes.py if the
# Monash directory changes.
MONASH_TIMESTEPS=${MONASH_TIMESTEPS:-445011429}
MIN_LEN=${MIN_LEN:-512}
SUBSET_SEED=${SUBSET_SEED:-42}

DRY_RUN=${DRY_RUN:-0}
run() { echo "  $*"; [ "$DRY_RUN" = 1 ] || "$@"; }

case "$STAGE" in full|subset|all) ;; *)
  echo "unknown stage: $STAGE  (full | subset | all)" >&2; exit 1 ;;
esac

# ── 1. generate ──────────────────────────────────────────────────────────────
if [ "$STAGE" = full ] || [ "$STAGE" = all ]; then
  mkdir -p "$OUT_DIR"
  echo "[synthetic] generating into $OUT_DIR  ($JOBS workers)"

  echo "[synthetic] LMC_Synth: $N_SERIES series x $LENGTH steps x $CHANNELS channels"
  run python -u "$GEN/LMC_Synth.py" \
    -N "$N_SERIES" -L "$LENGTH" -C "$CHANNELS" -J "$MAX_KERNELS" -P "$JOBS" \
    -M "$DIRICHLET_MIN" -X "$DIRICHLET_MAX" \
    -W "$WEIBULL_SHAPE" -Z "$WEIBULL_SCALE" \
    -O LMC_synth_MTS.arrow -D "$OUT_DIR/" \
    || { echo "LMC_Synth failed" >&2; exit 1; }

  echo "[synthetic] kernel-synth: $KS_SERIES series x $KS_LENGTH steps (univariate)"
  run python -u "$GEN/kernel-synth.py" \
    -N "$KS_SERIES" -L "$KS_LENGTH" -J "$MAX_KERNELS" -P "$JOBS" \
    -O kernel_synth.arrow -D "$OUT_DIR" \
    || { echo "kernel-synth failed" >&2; exit 1; }

  echo "[synthetic] corpus size:"
  run python -u scripts/count_dataset_sizes.py --synthetic_dir "$OUT_DIR" --min_len "$MIN_LEN"
fi

# ── 2. draw the Monash-sized subset ──────────────────────────────────────────
if [ "$STAGE" = subset ] || [ "$STAGE" = all ]; then
  echo "[synthetic] drawing a Monash-sized subset into $SUBSET_DIR"
  echo "            target $MONASH_TIMESTEPS timesteps at min_len $MIN_LEN, seed $SUBSET_SEED"
  run python -u scripts/subsample_synthetic.py \
    --synthetic_dir "$OUT_DIR" \
    --out_dir "$SUBSET_DIR" \
    --target_timesteps "$MONASH_TIMESTEPS" \
    --min_len "$MIN_LEN" \
    --seed "$SUBSET_SEED" \
    || { echo "subsample failed" >&2; exit 1; }
fi

echo "[synthetic] done"
echo "  full corpus   --pretrain_source synthetic                        ($OUT_DIR)"
echo "  Monash-sized  ./scripts/run_corpus.sh <model> synthetic_small …  ($SUBSET_DIR)"
echo "  hybrid        --pretrain_source monash+synthetic"
