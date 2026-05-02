#!/usr/bin/env python3
"""
run_layer_sweep.py — Pretrain all models across multiple layer depths.

For each encoder layer config [2, 4, 8, 12, 24], all models are launched in
parallel (one per GPU). The script waits for all to finish before moving
to the next layer config.

Predictor layers (JEPA models) are always half the encoder layers:
    encoder  2  →  predictor  1
    encoder  4  →  predictor  2
    encoder  8  →  predictor  4
    encoder 12  →  predictor  6
    encoder 24  →  predictor 12

Checkpoints are saved under layer-suffixed paths, e.g.:
    output_model/JEPA_layers8/
    output_model/LE-JEPA_layers8/
    checkpoints_layers8/

Usage:
    python run_layer_sweep.py                          # all models, all layer configs
    python run_layer_sweep.py --layers 8 12            # specific layer counts
    python run_layer_sweep.py --models dino lejepa     # specific models
    python run_layer_sweep.py --layers 8 --dry_run     # print commands only
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

LAYER_CONFIGS = [2, 4, 8, 12, 24]

# Per-layer learning rate: larger models need lower LR to stay stable.
# jepa/lejepa use SGD+OneCycleLR (max_lr), so these are conservative.
# dino/ntp/patchtst use Adam, so they can afford slightly higher LR.
LAYER_LR = {
    2:  3e-3,
    4:  1e-3,
    8:  5e-4,
    12: 2e-4,
    24: 1e-4,
}

# DINO uses AdamW with cosine schedule — separate LR tuning.
DINO_LAYER_LR = {
    2:  3e-3,
    4:  1e-3,
    8:  5e-4,
    12: 2e-4,
    24: 1e-4,
}

# NTP, PatchTST, and TimeDart use Adam — lower LR, scale down aggressively with depth.
NTP_PATCHTST_TIMEDART_LAYER_LR = {
    2:  3e-4,
    4:  1e-4,
    8:  5e-5,
    12: 2e-5,
    24: 1e-5,
}

# Model → GPU assignment (change to fit your server)
MODEL_GPU = {
    "dino":       0,
    "jepa":       1,
    "lejepa":     2,
    "patchtst":   3,
    "ntp":        4,
    "timedart":   5,
}

ALL_MODELS = list(MODEL_GPU.keys())


def predictor_layers_for(encoder_layers: int) -> int:
    return max(1, encoder_layers // 2)


def launch_model(model: str, encoder_layers: int, gpu: int,
                 log_dir: Path, dry_run: bool, pretrain_source: str = None,
                 log_tag: str = ""):
    pred_layers = predictor_layers_for(encoder_layers)
    # For unlisted depths, fall back to the 8-layer LR (mid-range default).
    if model == "dino":
        lr = DINO_LAYER_LR.get(encoder_layers, DINO_LAYER_LR[8])
    elif model in ("ntp", "patchtst", "timedart"):
        lr = NTP_PATCHTST_TIMEDART_LAYER_LR.get(encoder_layers,
                                                NTP_PATCHTST_TIMEDART_LAYER_LR[8])
    else:
        lr = LAYER_LR.get(encoder_layers, LAYER_LR[8])
    src_tag   = f"_{pretrain_source}" if pretrain_source and pretrain_source != "monash" else ""
    tag_suffix = f"_{log_tag}" if log_tag else ""
    log_path  = log_dir / f"{model}{src_tag}_layers{encoder_layers}{tag_suffix}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "Train_and_downstream.py"),
        "--model",            model,
        "--pretrain_only",    "true",
        "--encoder_layers",   str(encoder_layers),
        "--predictor_layers", str(pred_layers),
        "--lr",               str(lr),
    ]
    if pretrain_source is not None:
        cmd += ["--pretrain_source", pretrain_source]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"  [{model:12s}] GPU={gpu}  layers={encoder_layers}  pred={pred_layers}"
          f"  lr={lr}  log={log_path.relative_to(ROOT)}")

    if dry_run:
        print(f"    CMD: {' '.join(cmd)}")
        return None

    fh = open(log_path, "w")
    fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    proc._log_fh = fh   # keep file handle alive
    proc._label  = f"{model}/layers{encoder_layers}"
    return proc


def run_sweep(models: list, layer_configs: list, dry_run: bool,
              pretrain_source: str = None, gpu_overrides: dict = None,
              log_tag: str = ""):
    log_dir = ROOT / "logs" / "layer_sweep"
    gpu_overrides = gpu_overrides or {}

    for n_layers in layer_configs:
        print(f"\n{'='*60}")
        print(f"  LAYER CONFIG: encoder={n_layers}  predictor={predictor_layers_for(n_layers)}")
        print(f"{'='*60}")

        procs = []
        for model in models:
            gpu = gpu_overrides.get(model, MODEL_GPU[model])
            proc = launch_model(model, n_layers, gpu, log_dir, dry_run, pretrain_source, log_tag)
            if proc is not None:
                procs.append(proc)

        if dry_run or not procs:
            continue

        print(f"\n  Waiting for {len(procs)} processes to finish …")
        failed = []
        for proc in procs:
            proc.wait()
            proc._log_fh.close()
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"    {proc._label}: {status}")
            if proc.returncode != 0:
                failed.append(proc._label)

        if failed:
            print(f"\n  WARNING: {len(failed)} job(s) failed: {failed}")
        else:
            print(f"\n  All {len(procs)} jobs completed successfully.")

    print(f"\n\nSweep complete. Logs in {log_dir.relative_to(ROOT)}/")


def main():
    parser = argparse.ArgumentParser(description="Layer-depth pretraining sweep")
    parser.add_argument("--layers",  nargs="+", type=int, default=LAYER_CONFIGS,
                        help=f"Encoder layer counts to sweep (default: {LAYER_CONFIGS})")
    parser.add_argument("--models",  nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                        metavar="MODEL")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--pretrain_source", type=str, default=None,
                        choices=["monash", "synthetic", "monash+synthetic"],
                        help="Override pretrain data source. Checkpoints saved to a separate folder.")
    parser.add_argument("--gpu_override", type=int, default=None,
                        help="Override GPU for all models in this run")
    parser.add_argument("--log_tag", type=str, default="",
                        help="Suffix appended to training log filename (e.g. 'physical_2' → dino_layers8_physical_2.log)")
    args = parser.parse_args()

    gpu_overrides = {}
    if args.gpu_override is not None:
        for m in (args.models or ALL_MODELS):
            gpu_overrides[m] = args.gpu_override

    print(f"Layer sweep")
    print(f"  Models:  {args.models}")
    print(f"  Layers:  {args.layers}")
    if args.pretrain_source:
        print(f"  Pretrain source: {args.pretrain_source}")
    effective_gpus = {m: gpu_overrides.get(m, MODEL_GPU[m]) for m in args.models}
    print(f"  GPU map: {effective_gpus}")
    if args.dry_run:
        print("  DRY RUN — no processes will be started\n")

    run_sweep(args.models, args.layers, args.dry_run,
              pretrain_source=args.pretrain_source,
              gpu_overrides=gpu_overrides,
              log_tag=args.log_tag)


if __name__ == "__main__":
    main()
