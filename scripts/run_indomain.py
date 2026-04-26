#!/usr/bin/env python3
"""
run_indomain.py — In-domain pretraining + forecasting sweep.

For each model × each in-domain dataset:
  1. Pretrain the model on that dataset (no Monash / synthetic data)
  2. Evaluate forecasting on the same dataset (pred_lens 96/192/336/720)

Each model runs on its own GPU in a separate thread; datasets are processed
sequentially within each thread.

Logs:
  logs/indomain/{dataset}/pretrain_{model}.log
  logs/indomain/{dataset}/forecast_{model}.log

Usage:
    python run_indomain.py                                 # all models, all datasets
    python run_indomain.py --models dino patchtst          # specific models
    python run_indomain.py --datasets etth1 ettm2          # specific datasets
    python run_indomain.py --gpu_override 0                # all models on one GPU
    python run_indomain.py --skip_pretrain                 # use existing checkpoints
    python run_indomain.py --seed 42
    python run_indomain.py --dry_run
"""

import argparse
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

ENCODER_LAYERS = 8

IN_DOMAIN_DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather"]

# Per-model LR overrides (same as run_seed_analysis.py)
MODEL_LR = {
    "dino":        5e-4,
    "jepa": 5e-4,
    "lejepa":      5e-4,
    "patchtst":    5e-5,
    "ntp":         5e-5,
    "timedart":    5e-5,
}

MODEL_GPU = {
    "dino":        0,
    "jepa": 1,
    "lejepa":      2,
    "patchtst":    3,
    "ntp":         4,
    "timedart":    5,
}

ALL_MODELS = list(MODEL_GPU.keys())

_python = str(Path(sys.executable).parent / "python")


# ── subprocess launcher ────────────────────────────────────────────────────────

def _launch(cmd: list, gpu: int, log_path: Path, dry_run: bool, label: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"  [{label}] GPU={gpu}  log={log_path.relative_to(ROOT)}")
    if dry_run:
        print(f"    CMD: {' '.join(cmd)}")
        return None
    fh = open(log_path, "w")
    fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    proc._log_fh = fh
    proc._label  = label
    return proc


# ── command builders ───────────────────────────────────────────────────────────

def pretrain_cmd(model: str, dataset: str, seed: int = None) -> list:
    pred_layers = max(1, ENCODER_LAYERS // 2)
    lr = MODEL_LR[model]
    cmd = [
        _python, str(ROOT / "Train_and_downstream.py"),
        "--model",            model,
        "--pretrain_only",    "true",
        "--pretrain_dataset", dataset,
        "--encoder_layers",   str(ENCODER_LAYERS),
        "--predictor_layers", str(pred_layers),
        "--lr",               str(lr),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


def forecast_cmd(model: str, dataset: str, seed: int = None) -> list:
    cmd = [
        _python, str(ROOT / "Train_and_downstream.py"),
        "--model",            model,
        "--task",             "forecast",
        "--pretrain_dataset", dataset,
        "--forecast_dataset", dataset,
        "--encoder_layers",   str(ENCODER_LAYERS),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


# ── per-model pipeline (runs in its own thread) ────────────────────────────────

def _run_model_pipeline(model: str, datasets: list, gpu: int, seed: int,
                        skip_pretrain: bool, dry_run: bool, log_base: Path):

    def _step(proc):
        if proc:
            proc.wait()
            proc._log_fh.close()
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"    {proc._label}: {status}")

    for dataset in datasets:
        if not skip_pretrain:
            _step(_launch(
                pretrain_cmd(model, dataset, seed),
                gpu,
                log_base / dataset / f"pretrain_{model}.log",
                dry_run,
                f"pretrain/{model}/{dataset}",
            ))

        _step(_launch(
            forecast_cmd(model, dataset, seed),
            gpu,
            log_base / dataset / f"forecast_{model}.log",
            dry_run,
            f"forecast/{model}/{dataset}",
        ))

    print(f"  [{model}] pipeline complete.")


# ── main sweep ─────────────────────────────────────────────────────────────────

def run_indomain(models, datasets, gpu_override, seed, skip_pretrain, dry_run):
    log_base = ROOT / "logs" / "indomain"

    print(f"\n{'='*60}")
    print(f"  IN-DOMAIN SWEEP")
    print(f"  models:   {models}")
    print(f"  datasets: {datasets}")
    print(f"  layers:   {ENCODER_LAYERS}")
    if seed is not None:
        print(f"  seed:     {seed}")
    if skip_pretrain:
        print(f"  [skip_pretrain=True]")
    if dry_run:
        print(f"  [DRY RUN]")
    print(f"{'='*60}\n")

    threads = []
    for model in models:
        gpu = gpu_override if gpu_override is not None else MODEL_GPU[model]
        t = threading.Thread(
            target=_run_model_pipeline,
            args=(model, datasets, gpu, seed, skip_pretrain, dry_run, log_base),
            name=f"{model}",
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"\nIn-domain sweep complete. Logs in {log_base.relative_to(ROOT)}/")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="In-domain pretraining + forecasting sweep"
    )
    parser.add_argument("--models",   nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                        metavar="MODEL",   help=f"Models to run (default: all)")
    parser.add_argument("--datasets", nargs="+", default=IN_DOMAIN_DATASETS,
                        choices=IN_DOMAIN_DATASETS, metavar="DATASET",
                        help=f"Datasets (default: {IN_DOMAIN_DATASETS})")
    parser.add_argument("--gpu_override", type=int, default=None,
                        help="Run all models on this GPU")
    parser.add_argument("--seed",         type=int, default=None,
                        help="Random seed (also suffixes checkpoint paths with _seedN)")
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Skip pretraining — use existing checkpoints")
    parser.add_argument("--dry_run",       action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    effective_gpus = {
        m: (args.gpu_override if args.gpu_override is not None else MODEL_GPU[m])
        for m in args.models
    }
    print(f"GPU map: {effective_gpus}")

    run_indomain(
        models=args.models,
        datasets=args.datasets,
        gpu_override=args.gpu_override,
        seed=args.seed,
        skip_pretrain=args.skip_pretrain,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
