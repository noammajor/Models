#!/usr/bin/env python3
"""
run_seed_analysis.py — Reproducibility sweep across 5 random seeds.

For each seed, all models are pretrained at 8 encoder layers, then evaluated
on forecasting, classification, and anomaly detection.

Checkpoints are saved with a _seedN suffix (e.g. JEPA_monash_synthetic_layers8_seed2/).

Results saved to:
  results/seed_analysis_forecast.csv
  results/seed_analysis_classification.csv
  results/seed_analysis_anomaly.csv

Usage:
    python run_seed_analysis.py                                          # all seeds, all models
    python run_seed_analysis.py --seeds 0 1 2                            # specific seeds
    python run_seed_analysis.py --models dino npt                        # specific models
    python run_seed_analysis.py --pretrain_source synthetic              # pretrain data source
    python run_seed_analysis.py --skip_pretrain                          # forecasting/cls/anom only
    python run_seed_analysis.py --gpu_override 3                         # all models on one GPU
    python run_seed_analysis.py --dry_run
"""

import argparse
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

SEEDS            = [2003, 123, 456, 789, 1337]
ENCODER_LAYERS   = 8
PRETRAIN_SOURCE  = "monash+synthetic"
LR               = 5e-4   # LAYER_LR[8] from run_layer_sweep.py
CLS_NUM_PATCHES  = 72     # 72 × 16 = 1152 ts — covers longest UEA dataset (SelfRegulationSCP2)

FORECAST_DATASETS     = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic"]
CLASSIFICATION_DATASETS = [
    "EthanolConcentration", "FaceDetection", "Handwriting", "Heartbeat",
    "JapaneseVowels", "PEMS-SF", "SelfRegulationSCP1", "SelfRegulationSCP2",
    "SpokenArabicDigits", "UWaveGestureLibrary",
]
ANOMALY_DATASETS = ["SMD", "MSL", "SMAP", "SWaT", "PSM"]

# Per-model LR overrides (from run_layer_sweep.py LAYER_LR / NPT_PATCHTST_LAYER_LR)
MODEL_LR = {
    "dino":        5e-4,
    "jepa_simple": 5e-4,
    "lejepa":      5e-4,
    "patchtst":    5e-5,
    "npt":         5e-5,
    "timedart":    5e-5,
}

MODEL_GPU = {
    "dino":        0,
    "jepa_simple": 1,
    "lejepa":      2,
    "patchtst":    3,
    "npt":         4,
    "timedart":    5,
}

ALL_MODELS = list(MODEL_GPU.keys())

_python = str(Path(sys.executable).parent / "python")


# ── subprocess launcher ───────────────────────────────────────────────────────

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


# ── phase builders ────────────────────────────────────────────────────────────

def pretrain_cls_cmd(model: str, seed: int, pretrain_source: str) -> list:
    """Separate pretrain with longer context (num_patches=72) needed for classification."""
    pred_layers = max(1, ENCODER_LAYERS // 2)
    lr = MODEL_LR[model]
    return [
        _python, str(ROOT / "Train_and_downstream.py"),
        "--model",            model,
        "--pretrain_only",    "true",
        "--encoder_layers",   str(ENCODER_LAYERS),
        "--predictor_layers", str(pred_layers),
        "--num_patches",      str(CLS_NUM_PATCHES),
        "--lr",               str(lr),
        "--pretrain_source",  pretrain_source,
        "--seed",             str(seed),
    ]


def pretrain_cmd(model: str, seed: int, gpu: int, pretrain_source: str) -> list:
    pred_layers = max(1, ENCODER_LAYERS // 2)
    lr = MODEL_LR[model]
    return [
        _python, str(ROOT / "Train_and_downstream.py"),
        "--model",            model,
        "--pretrain_only",    "true",
        "--encoder_layers",   str(ENCODER_LAYERS),
        "--predictor_layers", str(pred_layers),
        "--lr",               str(lr),
        "--pretrain_source",  pretrain_source,
        "--seed",             str(seed),
    ]


def forecast_cmd(model: str, dataset: str, seed: int, gpu: int, pretrain_source: str) -> list:
    return [
        _python, str(ROOT / "Train_and_downstream.py"),
        "--model",           model,
        "--task",            "forecast",
        "--forecast_dataset", dataset,
        "--encoder_layers",  str(ENCODER_LAYERS),
        "--pretrain_source", pretrain_source,
        "--seed",            str(seed),
    ]


def classify_cmd(model: str, dataset: str, seed: int, gpu: int, pretrain_source: str) -> list:
    return [
        _python, str(ROOT / "Train_and_downstream.py"),
        "--model",                    model,
        "--task",                     "classify",
        "--classification_dataset",   dataset,
        "--encoder_layers",           str(ENCODER_LAYERS),
        "--num_patches",              str(CLS_NUM_PATCHES),
        "--pretrain_source",          pretrain_source,
        "--seed",                     str(seed),
    ]


def anomaly_cmd(model: str, dataset: str, seed: int, gpu: int, pretrain_source: str) -> list:
    return [
        _python, str(ROOT / "Train_and_downstream.py"),
        "--model",           model,
        "--task",            "anomaly",
        "--anomaly_dataset", dataset,
        "--encoder_layers",  str(ENCODER_LAYERS),
        "--pretrain_source", pretrain_source,
        "--seed",            str(seed),
    ]


# ── per-model pipeline (runs in its own thread) ───────────────────────────────

def _run_model_pipeline(model: str, seed: int, gpu: int, pretrain_source: str,
                        skip_pretrain: bool, dry_run: bool, log_base: Path):
    def _step(proc):
        if proc:
            proc.wait()
            proc._log_fh.close()
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"    {proc._label}: {status}")

    if not skip_pretrain:
        # 1a — standard pretrain (forecasting/anomaly context)
        _step(_launch(pretrain_cmd(model, seed, gpu, pretrain_source),
                      gpu, log_base / f"seed{seed}" / f"pretrain_{model}.log",
                      dry_run, f"pretrain/{model}/seed{seed}"))

        # 1b — classification pretrain (num_patches=72)
        _step(_launch(pretrain_cls_cmd(model, seed, pretrain_source),
                      gpu, log_base / f"seed{seed}" / f"pretrain_cls_{model}.log",
                      dry_run, f"pretrain_cls/{model}/seed{seed}"))

    # 2 — forecasting (all datasets in one subprocess)
    _step(_launch(forecast_cmd(model, ",".join(FORECAST_DATASETS), seed, gpu, pretrain_source),
                  gpu, log_base / f"seed{seed}" / f"forecast_{model}.log",
                  dry_run, f"forecast/{model}/seed{seed}"))

    # 3 — classification (sequential per dataset)
    for dataset in CLASSIFICATION_DATASETS:
        _step(_launch(classify_cmd(model, dataset, seed, gpu, pretrain_source),
                      gpu, log_base / f"seed{seed}" / f"cls_{model}_{dataset}.log",
                      dry_run, f"cls/{model}/{dataset}/seed{seed}"))

    # 4 — anomaly (all datasets in parallel within this model)
    procs = []
    for dataset in ANOMALY_DATASETS:
        proc = _launch(anomaly_cmd(model, dataset, seed, gpu, pretrain_source),
                       gpu, log_base / f"seed{seed}" / f"anom_{model}_{dataset}.log",
                       dry_run, f"anom/{model}/{dataset}/seed{seed}")
        if proc:
            procs.append(proc)
    for proc in procs:
        _step(proc)

    print(f"  [{model}/seed{seed}] pipeline complete.")


# ── main sweep ────────────────────────────────────────────────────────────────

def run_seed_analysis(seeds, models, gpu_override, skip_pretrain, dry_run, pretrain_source=PRETRAIN_SOURCE):
    log_base = ROOT / "logs" / "seed_analysis"

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}  (encoder_layers={ENCODER_LAYERS}, src={pretrain_source})")
        print(f"{'='*60}")

        threads = []
        for model in models:
            gpu = gpu_override if gpu_override is not None else MODEL_GPU[model]
            t = threading.Thread(
                target=_run_model_pipeline,
                args=(model, seed, gpu, pretrain_source, skip_pretrain, dry_run, log_base),
                name=f"{model}/seed{seed}",
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        print(f"\n  Seed {seed} complete.")

    print(f"\n\nSeed analysis complete. Logs in {log_base.relative_to(ROOT)}/")
    print(f"Results: results/seed_analysis_*.csv")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed analysis sweep — 5 seeds × all models × all tasks")
    parser.add_argument("--seeds",        nargs="+", type=int, default=SEEDS,
                        help=f"Seeds to sweep (default: {SEEDS})")
    parser.add_argument("--models",       nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                        metavar="MODEL",  help=f"Models to run (default: all)")
    parser.add_argument("--pretrain_source", type=str, default=PRETRAIN_SOURCE,
                        choices=["monash", "synthetic", "monash+synthetic"],
                        help=f"Pretrain data source (default: {PRETRAIN_SOURCE})")
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Skip pretraining — use existing checkpoints")
    parser.add_argument("--gpu_override", type=int, default=None,
                        help="Run all models on this GPU")
    parser.add_argument("--dry_run",      action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    print(f"Seed analysis")
    print(f"  Seeds:          {args.seeds}")
    print(f"  Models:         {args.models}")
    print(f"  Encoder layers: {ENCODER_LAYERS}")
    print(f"  Pretrain source:{args.pretrain_source}")
    effective_gpus = {m: (args.gpu_override if args.gpu_override is not None else MODEL_GPU[m])
                      for m in args.models}
    print(f"  GPU map:        {effective_gpus}")
    if args.skip_pretrain:
        print(f"  [skip_pretrain=True]")
    if args.dry_run:
        print("  DRY RUN — no processes will be started\n")

    run_seed_analysis(
        seeds=args.seeds,
        models=args.models,
        gpu_override=args.gpu_override,
        skip_pretrain=args.skip_pretrain,
        dry_run=args.dry_run,
        pretrain_source=args.pretrain_source,
    )


if __name__ == "__main__":
    main()
