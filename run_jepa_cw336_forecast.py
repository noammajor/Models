#!/usr/bin/env python3
"""
run_jepa_cw336_forecast.py — Pretrain JEPA-simple (8 layers, 336-ts context) on Monash,
then zero-shot forecast on ETTh1, ETTh2, ETTm1, ETTm2, and Weather.

Context window: 21 patches × 16 = 336 timesteps  (vs. default 32 × 16 = 512)
Checkpoint:     output_model/classification/JEPA_layers8_cw336/

Usage:
    python run_jepa_cw336_forecast.py
    python run_jepa_cw336_forecast.py --gpu 2
    python run_jepa_cw336_forecast.py --skip_pretrain        # forecast only (checkpoint must exist)
    python run_jepa_cw336_forecast.py --dry_run
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

ENCODER_LAYERS   = 8
PREDICTOR_LAYERS = 4          # half of encoder layers
NUM_PATCHES      = 21         # 21 × 16 = 336 timesteps
PATCH_SIZE       = 16
CONTEXT_WINDOW   = NUM_PATCHES * PATCH_SIZE   # 336
LR               = 5e-4

FORECAST_DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather"]

CKPT_DIR = f"output_model/classification/JEPA_layers{ENCODER_LAYERS}_cw{CONTEXT_WINDOW}"


def main():
    parser = argparse.ArgumentParser(
        description=f"JEPA {ENCODER_LAYERS}-layer, {CONTEXT_WINDOW}-ts context — Monash pretrain + forecast"
    )
    parser.add_argument("--gpu",          type=int,  default=0)
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Skip pretraining and go straight to forecasting (checkpoint must exist)")
    parser.add_argument("--dry_run",      action="store_true",
                        help="Print config and exit without running anything")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("=" * 60)
    print("  JEPA-simple  cw336 forecast experiment")
    print(f"  encoder_layers  : {ENCODER_LAYERS}")
    print(f"  predictor_layers: {PREDICTOR_LAYERS}")
    print(f"  num_patches     : {NUM_PATCHES}  (context = {CONTEXT_WINDOW} timesteps)")
    print(f"  pretrain_source : monash")
    print(f"  lr              : {LR}")
    print(f"  forecast        : {FORECAST_DATASETS}")
    print(f"  checkpoint dir  : {CKPT_DIR}")
    print(f"  GPU             : {args.gpu}")
    if args.skip_pretrain:
        print("  mode            : forecast only (skip pretrain)")
    if args.dry_run:
        print("  DRY RUN")
    print("=" * 60)

    if args.dry_run:
        return

    from Train_and_downstream import run

    # ── 1. Pretrain ───────────────────────────────────────────────────────────
    if not args.skip_pretrain:
        print("\n\n" + "=" * 60)
        print("  PHASE 1 — Pretraining")
        print("=" * 60)
        run(
            model           = "jepa_simple",
            task            = "pretrain",
            encoder_layers  = ENCODER_LAYERS,
            predictor_layers= PREDICTOR_LAYERS,
            num_patches     = NUM_PATCHES,
            pretrain_source = "monash",
            lr              = LR,
        )
        print(f"\nPretraining done. Checkpoint in: {CKPT_DIR}/")

    # ── 2. Forecast each dataset ──────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  PHASE 2 — Zero-shot forecasting")
    print("=" * 60)

    results = {}
    for dataset in FORECAST_DATASETS:
        print(f"\n--- Forecasting: {dataset} ---")
        try:
            result = run(
                model            = "jepa_simple",
                task             = "forecast",
                forecast_dataset = dataset,
                encoder_layers   = ENCODER_LAYERS,
                num_patches      = NUM_PATCHES,
                pretrain_source  = "monash",
            )
            # result = (best_ckpt, best_mse, cls_acc=None, anom=None)
            best_ckpt, best_mse = result[0], result[1]
            results[dataset] = best_mse
            print(f"  {dataset}: MSE={best_mse:.4f}  (best ckpt={best_ckpt})")
        except Exception as e:
            import traceback
            print(f"  [ERROR] {dataset}: {e}")
            traceback.print_exc()
            results[dataset] = None

    # ── 3. Summary ────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print(f"  JEPA  layers={ENCODER_LAYERS}  cw={CONTEXT_WINDOW}  monash")
    print("=" * 60)
    for ds, mse in results.items():
        val = f"{mse:.4f}" if mse is not None else "FAILED"
        print(f"  {ds:<12} MSE = {val}")
    print()


if __name__ == "__main__":
    main()
