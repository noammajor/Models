#!/bin/bash
# =============================================================================
# run_lejepa_ablation.sh — the two LE-JEPA ablations of the paper.
#
#   aug     the augmentation-family study (Appendix F.1, Tables 35 and 36).
#           Each family replaces the two DWT views with a different pair of
#           transforms; everything else is the per-model protocol.
#             dwt         soft-threshold / high-perturb wavelet views (default)
#             galilean    global temporal rescaling / local Lorentz transforms
#             polar       global polar warp / local rotation in value-time
#             hyperbolic  global hyperbolic warp / local Mobius shift
#
#   sigreg  the SIGReg strength sweep (Appendix F.3, Table 41), over
#             lambda in {0, 0.002, 0.008, 0.02, 0.05}
#           lambda = 0 removes SIGReg and collapses the representation, which is
#           the point of including it.
#
# Both tag their checkpoints by the variant, so the sweeps never overwrite the
# main LE-JEPA backbone or each other. Table 36 is single-seed (123); Table 41
# reports the same.
#
# Usage:
#   ./scripts/run_lejepa_ablation.sh aug    <family> <task> [<seed>]
#   ./scripts/run_lejepa_ablation.sh sigreg <lambda> <task> [<seed>]
#
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_lejepa_ablation.sh aug hyperbolic forecast
#   for L in 0 0.002 0.008 0.02 0.05; do
#     ./scripts/run_lejepa_ablation.sh sigreg $L classify
#   done
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

KIND=${1:?aug | sigreg}; VARIANT=${2:?family or lambda}; TASK=${3:?task}; SEED=${4:-123}
M=lejepa

case "$KIND" in
  aug)
    case "$VARIANT" in dwt|galilean|polar|hyperbolic) ;;
      *) echo "unknown family: $VARIANT  (dwt | galilean | polar | hyperbolic)" >&2; exit 1 ;;
    esac
    export TS_LEJEPA_AUG=$VARIANT
    LABEL="aug_$VARIANT"
    ;;
  sigreg)
    export TS_LEJEPA_LAMBDA=$VARIANT
    LABEL="sigreg_$VARIANT"
    ;;
  *) echo "unknown ablation: $KIND  (aug | sigreg)" >&2; exit 1 ;;
esac

EPOCHS=20
ts_budget "$M"          # → LR 5e-4, BATCH 128
ts_batch $BATCH

ts_task "$TASK"
ts_log "lejepa_$LABEL" "$M" "$TASK"

FLAGS="--model $M --pretrain_source monash $ENCODER_FLAGS"
FLAGS="$FLAGS --num_patches $PATCHES --lr $LR --epochs $EPOCHS"

echo "[lejepa:$LABEL] $TASK  (lr=$LR batch=$BATCH epochs=$EPOCHS patches=$PATCHES)"
ts_sweep "$M" "$TASK" $SEED
