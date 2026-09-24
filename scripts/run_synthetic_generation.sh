#!/bin/bash
# =============================================================================
# run_synthetic_generation.sh — build every synthetic corpus from scratch.
#
# Three corpora, each into the directory data_paths.py points at. The script
# reads those paths from data_paths.py, so moving the data means editing that
# file and nothing here.
#
#   full    synthetic_data_dir       what --pretrain_source synthetic reads
#           4,000 LMC rows + 4,000 kernel-synth rows = 8,000 rows, ~1.61 B timesteps
#
#   mix     synthetic_mix_data_dir   what --pretrain_source monash+synthetic reads
#           2,900 + 2,900 = 5,800 rows, ~1.17 B timesteps. The hybrid corpus
#           deliberately pairs Monash with a SMALLER synthetic half, so the
#           hybrid runs are not simply Monash plus the whole synthetic corpus.
#
#   subset  a Monash-sized random draw from `full`, for run_corpus.sh
#           synthetic_small. "Synthetic vs Monash" otherwise confounds the
#           data-generating process with corpus size; this holds size fixed so
#           only the process differs.
#
# Both generators are Gaussian-process based and are called in equal numbers:
#   LMC_Synth.py     multivariate, via the Linear Coregionalization Model.
#                    160 channels x 2,500 steps = 400k timesteps per row, which
#                    is why it dominates the totals.
#   kernel-synth.py  univariate, the Chronos kernel-synth procedure.
#                    2,500 timesteps per row.
# The loader discovers every .arrow file in a directory and handles univariate
# [T] and multivariate [C, T] targets alike, so the two sit side by side.
#
# Usage:
#   ./scripts/run_synthetic_generation.sh [full | mix | subset | all]   (default: all)
#
#   JOBS=16 ./scripts/run_synthetic_generation.sh all     # everything, on 16 cores
#   ./scripts/run_synthetic_generation.sh subset          # redraw the subset only
#   DRY_RUN=1 ./scripts/run_synthetic_generation.sh all   # print, generate nothing
#
# Any parameter below can be overridden from the environment:
#   N_FULL=500 OUT_DIR=/tmp/synth ./scripts/run_synthetic_generation.sh full
#
# Generation is CPU-bound and takes hours at these sizes: every series is a draw
# from a GP prior, and each LMC row draws several latent series and mixes them.
# Set JOBS to the core count and run it detached.
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

GEN="scripts/synthetic_data_generation"
STAGE=${1:-all}
case "$STAGE" in full|mix|subset|all) ;; *)
  echo "unknown stage: $STAGE  (full | mix | subset | all)" >&2; exit 1 ;;
esac

# ── destinations, straight from data_paths.py ────────────────────────────────
eval "$(python3 -c "
from data_paths import DATA_PATHS as D
print(f'''_SYNTH_DIR=\"{D['synthetic_data_dir']}\"''')
print(f'''_MIX_DIR=\"{D['synthetic_mix_data_dir']}\"''')
")" || { echo "could not read data_paths.py" >&2; exit 1; }

OUT_DIR=${OUT_DIR:-$_SYNTH_DIR}
MIX_DIR=${MIX_DIR:-$_MIX_DIR}
# run_corpus.sh synthetic_small looks here; keep the two in step if you move it.
SUBSET_DIR=${SUBSET_DIR:-${_SYNTH_DIR}_monashsize}

# ── corpus sizes: rows PER GENERATOR, so a corpus is twice this ──────────────
N_FULL=${N_FULL:-4000}              # full → 8,000 rows, ~1.61 B timesteps
N_MIX=${N_MIX:-2900}                # mix  → 5,800 rows, ~1.17 B timesteps

# ── generator settings, shared by both corpora ───────────────────────────────
LENGTH=${LENGTH:-2500}              # -L  timesteps per series. The generator builds
                                    #     an L x L covariance matrix, so this cannot
                                    #     go much above a few thousand.
CHANNELS=${CHANNELS:-160}           # -C  channels per LMC row. Each channel counts as
                                    #     its own series downstream (the loaders split
                                    #     them), so a row is 160 x 2500 = 400k timesteps.
MAX_KERNELS=${MAX_KERNELS:-5}       # -J  max base kernels composed per latent function
DIRICHLET_MIN=${DIRICHLET_MIN:-0.1} # -M  lower bound of the mixing-weight alpha;
DIRICHLET_MAX=${DIRICHLET_MAX:-1.0} # -X  upper bound. Lower alpha → sparser mixtures.
WEIBULL_SHAPE=${WEIBULL_SHAPE:-1.5} # -W  shape of the latent-count distribution;
WEIBULL_SCALE=${WEIBULL_SCALE:-2.0} # -Z  scale. Higher scale → more latents per row.

JOBS=${JOBS:-16}                    # -P  parallel workers; set to the core count

# ── subset: match Monash's filtered size exactly ─────────────────────────────
# 445,011,429 is Monash's timestep count under the >=512 filter the pre-training
# loaders apply. Recount with count_dataset_sizes.py if the Monash dir changes.
MONASH_TIMESTEPS=${MONASH_TIMESTEPS:-445011429}
MIN_LEN=${MIN_LEN:-512}
SUBSET_SEED=${SUBSET_SEED:-42}

DRY_RUN=${DRY_RUN:-0}
run() { echo "    $*"; [ "$DRY_RUN" = 1 ] || "$@"; }

# ── generate <n-per-generator> <target dir> <label> ──────────────────────────
generate() {
  local n=$1 dir=$2 label=$3
  [ "$DRY_RUN" = 1 ] || mkdir -p "$dir" || { echo "cannot create $dir" >&2; exit 1; }
  echo "[synthetic] $label → $dir"
  echo "            $n LMC rows ($CHANNELS ch x $LENGTH steps) + $n kernel-synth rows ($LENGTH steps)"

  run python -u "$GEN/LMC_Synth.py" \
    -N "$n" -L "$LENGTH" -C "$CHANNELS" -J "$MAX_KERNELS" -P "$JOBS" \
    -M "$DIRICHLET_MIN" -X "$DIRICHLET_MAX" \
    -W "$WEIBULL_SHAPE" -Z "$WEIBULL_SCALE" \
    -O LMC_synth_MTS.arrow -D "$dir/" \
    || { echo "LMC_Synth failed" >&2; exit 1; }

  run python -u "$GEN/kernel-synth.py" \
    -N "$n" -L "$LENGTH" -J "$MAX_KERNELS" -P "$JOBS" \
    -O kernel_synth.arrow -D "$dir" \
    || { echo "kernel-synth failed" >&2; exit 1; }

  run python -u scripts/count_dataset_sizes.py --only "$dir" --min_len "$MIN_LEN"
}

if [ "$STAGE" = full ] || [ "$STAGE" = all ]; then
  generate "$N_FULL" "$OUT_DIR" "full corpus"
fi

if [ "$STAGE" = mix ] || [ "$STAGE" = all ]; then
  generate "$N_MIX" "$MIX_DIR" "hybrid mix"
fi

# ── the Monash-sized draw from the full corpus ───────────────────────────────
if [ "$STAGE" = subset ] || [ "$STAGE" = all ]; then
  echo "[synthetic] Monash-sized subset → $SUBSET_DIR"
  echo "            target $MONASH_TIMESTEPS timesteps at min_len $MIN_LEN, seed $SUBSET_SEED"
  run python -u scripts/subsample_synthetic.py \
    --synthetic_dir "$OUT_DIR" \
    --out_dir "$SUBSET_DIR" \
    --target_timesteps "$MONASH_TIMESTEPS" \
    --min_len "$MIN_LEN" \
    --seed "$SUBSET_SEED" \
    || { echo "subsample failed" >&2; exit 1; }
fi

echo
echo "[synthetic] done. Pre-train against them with:"
echo "  --pretrain_source synthetic          → $OUT_DIR"
echo "  --pretrain_source monash+synthetic   → $MIX_DIR"
echo "  run_corpus.sh <model> synthetic_small → $SUBSET_DIR"
