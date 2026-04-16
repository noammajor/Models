#!/usr/bin/env python3
"""
run_layer_classification.py — Classification evaluation across layer-sweep checkpoints.

Runs all models (dino, jepa_simple, lejepa, patchtst, npt) using their best
checkpoint for each encoder layer config on all classification datasets.

Each model runs as a subprocess on its assigned GPU (matching run_layer_forecast.py).

Results saved to:
  results/layer_classification.csv           (standard checkpoints)
  results/layer_classification_cw1152.csv    (--num_patches 72, classification encoder)

Usage:
    python run_layer_classification.py
    python run_layer_classification.py --layers 2 4 8
    python run_layer_classification.py --models dino jepa_simple
    python run_layer_classification.py --datasets EthanolConcentration SelfRegulationSCP2
    python run_layer_classification.py --gpu_override 5
    python run_layer_classification.py --dry_run

    # Use classification-encoder checkpoints (pretrained with pretrain_cls_encoder.py):
    python run_layer_classification.py --num_patches 72 --layers 8
"""

import argparse
import csv
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

LAYER_CONFIGS           = [2, 4, 8, 12, 24]
CLASSIFICATION_DATASETS = [
    # Legacy non-UEA datasets
    "HAR", "Epilepsy-2", "eeg_no_big",
    # UEA datasets (sorted by T ascending)
    "JapaneseVowels", "SpokenArabicDigits", "FaceDetection",
    "Handwriting", "PEMS-SF", "UWaveGestureLibrary",
    "Heartbeat", "SelfRegulationSCP1", "SelfRegulationSCP2",
    "EthanolConcentration",
]

MODEL_GPU = {
    "dino":        0,
    "jepa_simple": 1,
    "lejepa":      2,
    "patchtst":    3,
    "npt":         4,
}
ALL_MODELS = list(MODEL_GPU.keys())


# ── logging: tee stdout to file ───────────────────────────────────────────────

class _Tee:
    def __init__(self, original, file_handle):
        self._orig = original
        self._file = file_handle

    def write(self, data):
        self._orig.write(data)
        self._orig.flush()
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._orig.flush()
        self._file.flush()

    def fileno(self):
        return self._orig.fileno()


@contextmanager
def log_to_file(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
        orig = sys.stdout
        sys.stdout = _Tee(orig, fh)
        try:
            yield
        finally:
            sys.stdout = orig


# ── single model+layer worker ─────────────────────────────────────────────────

def run_model_worker(model: str, encoder_layers: int, gpu: int,
                     datasets: list, out_csv: Path, num_patches: int = None):
    _cw_tag  = f"_cw{num_patches * 16}" if num_patches is not None else ""
    log_base = ROOT / "logs" / f"layer_classification{_cw_tag}"
    fieldnames = ["model", "encoder_layers", "num_patches", "dataset", "accuracy", "timestamp"]

    existing = set()
    rows = []
    if out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                rows.append(row)
                existing.add((row["model"], int(row["encoder_layers"]),
                              row.get("num_patches", ""), row["dataset"]))

    print(f"\n{'='*60}")
    print(f"  {model.upper()}  layers={encoder_layers}"
          + (f"  num_patches={num_patches}" if num_patches else ""))
    print(f"{'='*60}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for dataset in datasets:
        key = (model, encoder_layers, str(num_patches) if num_patches else "", dataset)
        if key in existing:
            print(f"\n  {model}/layers{encoder_layers}/{dataset} — already complete, skipping.")
            continue

        print(f"\n--- {model} / layers={encoder_layers} / {dataset} ---")
        log_path = log_base / f"layers{encoder_layers}" / model / f"{dataset}.log"

        with log_to_file(log_path):
            try:
                from Train_and_downstream import run

                result = run(
                    model=model,
                    task="classify",
                    classification_dataset=dataset,
                    encoder_layers=encoder_layers,
                    num_patches=num_patches,
                )

                # Return value varies by model:
                #   dino:        (best_ckpt, best_mse, cls_acc, anom_result)
                #   jepa_simple: (best_ckpt, best_mse, cls_acc, anom_result)
                #   lejepa:      (best_ckpt, best_mse, cls_acc, anom_result)
                #   patchtst:    (mse, mae, cls_acc)  or  cls_acc float
                #   npt:         (mse, mae, cls_acc)  or  cls_acc float
                if isinstance(result, tuple):
                    cls_acc = result[2]
                else:
                    cls_acc = result

            except Exception as e:
                print(f"[ERROR] {model}/layers{encoder_layers}/{dataset}: {e}")
                import traceback; traceback.print_exc()
                cls_acc = None

        ts = datetime.now().isoformat(timespec="seconds")
        rows.append({
            "model":          model,
            "encoder_layers": encoder_layers,
            "num_patches":    str(num_patches) if num_patches is not None else "",
            "dataset":        dataset,
            "accuracy":       f"{cls_acc:.6f}" if cls_acc is not None else "N/A",
            "timestamp":      ts,
        })
        existing.add(key)

        out_csv.parent.mkdir(exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved → {out_csv}")

    print(f"\nDone: {model} layers={encoder_layers}")


# ── parallel launcher ─────────────────────────────────────────────────────────

def launch_worker(model: str, encoder_layers: int, gpu: int,
                  datasets: list, log_dir: Path, out_csv: Path, dry_run: bool,
                  num_patches: int = None):
    log_path = log_dir / f"{model}_layers{encoder_layers}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, __file__,
        "--_worker",
        "--model",          model,
        "--encoder_layers", str(encoder_layers),
        "--gpu",            str(gpu),
        "--datasets",       *datasets,
        "--out_csv",        str(out_csv),
    ]
    if num_patches is not None:
        cmd += ["--num_patches", str(num_patches)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    _np_str = f"  num_patches={num_patches}" if num_patches else ""
    print(f"  [{model:14s}] GPU={gpu}  layers={encoder_layers}{_np_str}"
          f"  log={log_path.relative_to(ROOT)}")

    if dry_run:
        print(f"    CMD: {' '.join(cmd)}")
        return None

    fh = open(log_path, "w")
    fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    proc._log_fh = fh
    proc._label  = f"{model}/layers{encoder_layers}"
    return proc


def run_classification_sweep(models: list, layer_configs: list, datasets: list,
                              gpu_override: int, dry_run: bool, num_patches: int = None):
    _cw_tag = f"_cw{num_patches * 16}" if num_patches is not None else ""
    log_dir = ROOT / "logs" / f"layer_classification{_cw_tag}"
    out_csv = ROOT / "results" / f"layer_classification{_cw_tag}.csv"

    print(f"Layer classification sweep")
    print(f"  Models:      {models}")
    print(f"  Layers:      {layer_configs}")
    print(f"  Datasets:    {datasets}")
    if num_patches is not None:
        print(f"  num_patches: {num_patches}  (context = {num_patches * 16} ts, classification encoder)")
    if dry_run:
        print("  DRY RUN\n")

    for n_layers in layer_configs:
        print(f"\n{'='*60}")
        print(f"  encoder_layers={n_layers}")
        print(f"{'='*60}")

        procs = []
        for model in models:
            gpu = gpu_override if gpu_override is not None else MODEL_GPU[model]
            proc = launch_worker(model, n_layers, gpu, datasets, log_dir, out_csv, dry_run,
                                 num_patches=num_patches)
            if proc is not None:
                procs.append(proc)

        if dry_run or not procs:
            continue

        print(f"\n  Waiting for {len(procs)} workers …")
        failed = []
        for proc in procs:
            proc.wait()
            proc._log_fh.close()
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"    {proc._label}: {status}")
            if proc.returncode != 0:
                failed.append(proc._label)

        if failed:
            print(f"\n  WARNING: {len(failed)} failed: {failed}")
        else:
            print(f"\n  All {len(procs)} workers completed.")

    print(f"\n\nClassification sweep complete. Results in {out_csv}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Layer-sweep classification evaluation")
    parser.add_argument("--layers",      nargs="+", type=int, default=LAYER_CONFIGS)
    parser.add_argument("--models",      nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                        metavar="MODEL")
    parser.add_argument("--datasets",    nargs="+", default=CLASSIFICATION_DATASETS)
    parser.add_argument("--gpu_override", type=int, default=None,
                        help="Override GPU for all models")
    parser.add_argument("--num_patches", type=int, default=None,
                        help="Use classification-encoder checkpoints (e.g. 72 for cw1152). "
                             "Must match what was used in pretrain_cls_encoder.py.")
    parser.add_argument("--dry_run",     action="store_true")

    # internal worker mode
    parser.add_argument("--_worker",        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model",          type=str,            help=argparse.SUPPRESS)
    parser.add_argument("--encoder_layers", type=int,            help=argparse.SUPPRESS)
    parser.add_argument("--gpu",            type=int,            help=argparse.SUPPRESS)
    parser.add_argument("--out_csv",        type=str,            help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args._worker:
        run_model_worker(
            model=args.model,
            encoder_layers=args.encoder_layers,
            gpu=args.gpu,
            datasets=args.datasets,
            out_csv=Path(args.out_csv),
            num_patches=args.num_patches,
        )
        return

    run_classification_sweep(
        args.models, args.layers, args.datasets,
        args.gpu_override, args.dry_run,
        num_patches=args.num_patches,
    )


if __name__ == "__main__":
    main()
