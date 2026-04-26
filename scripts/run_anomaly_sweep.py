#!/usr/bin/env python3
"""
run_anomaly_sweep.py — Anomaly detection evaluation across models and datasets.

For each model, loads the best pretrained checkpoint (layer-sweep or default),
runs frozen-encoder + trained-decoder reconstruction on standard anomaly datasets,
and reports F1, precision, recall, accuracy.

Decoder training mirrors the reference Exp_Anomaly_Detection pattern:
  • 80/20 train/val split from the normal (train) data
  • validation-based early stopping (patience=3)
  • cosine LR decay

Results saved to:
  results/anomaly_sweep.csv

Usage:
    python run_anomaly_sweep.py
    python run_anomaly_sweep.py --models jepa lejepa --layers 8
    python run_anomaly_sweep.py --datasets MSL SMAP SMD
    python run_anomaly_sweep.py --gpu_override 4
    python run_anomaly_sweep.py --pretrain_source monash+synthetic --layers 8
    python run_anomaly_sweep.py --dry_run
"""

import argparse
import csv
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# Standard anomaly detection benchmark datasets
ANOMALY_DATASETS = ["SMD", "MSL", "SMAP", "SWaT", "PSM"]

LAYER_CONFIGS = [2, 4, 8, 12, 24]

MODEL_GPU = {
    "dino":            0,
    "jepa":     1,
    "lejepa":          2,
    "patchtst":        3,
    "ntp":             4,
    "timedart":        5,
    "patchtst_random": 5,
}
ALL_MODELS = list(MODEL_GPU.keys())


# ── logging ──────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, original, file_handle):
        self._orig = original
        self._file = file_handle

    def write(self, data):
        self._orig.write(data); self._orig.flush()
        self._file.write(data); self._file.flush()

    def flush(self):
        self._orig.flush(); self._file.flush()

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


# ── single worker ─────────────────────────────────────────────────────────────

def run_model_worker(model: str, encoder_layers: int, gpu: int,
                     datasets: list, out_csv: Path,
                     pretrain_source: str = "monash"):
    log_base   = ROOT / "logs" / "anomaly_sweep"
    fieldnames = ["model", "encoder_layers", "pretrain_source",
                  "dataset", "f1", "precision", "recall", "accuracy", "timestamp"]

    existing = set()
    rows = []
    if out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                rows.append(row)
                if row.get("f1", "N/A") not in ("N/A", "", None):
                    existing.add((row["model"], int(row["encoder_layers"]),
                                  row.get("pretrain_source", "monash"), row["dataset"]))

    print(f"\n{'='*60}")
    print(f"  {model.upper()}  layers={encoder_layers}  src={pretrain_source}")
    print(f"{'='*60}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for dataset in datasets:
        key = (model, encoder_layers, pretrain_source, dataset)
        if key in existing:
            print(f"\n  {model}/layers{encoder_layers}/{dataset} — already complete, skipping.")
            continue

        print(f"\n--- {model} / layers={encoder_layers} / {dataset} ---")
        log_path = log_base / f"layers{encoder_layers}" / pretrain_source / model / f"{dataset}.log"

        f1 = prec = rec = acc = None
        with log_to_file(log_path):
            try:
                from Train_and_downstream import run

                result = run(
                    model           = model,
                    task            = "anomaly",
                    anomaly_dataset = dataset,
                    encoder_layers  = encoder_layers,
                    pretrain_source = pretrain_source,
                )

                # result shape varies by model but last element is anom_result dict
                anom = result[-1] if isinstance(result, tuple) else result
                if isinstance(anom, dict):
                    f1   = anom.get("f1")
                    prec = anom.get("precision")
                    rec  = anom.get("recall")
                    acc  = anom.get("accuracy")

            except Exception as e:
                print(f"[ERROR] {model}/layers{encoder_layers}/{dataset}: {e}")
                import traceback; traceback.print_exc()

        def _fmt(v):
            return f"{v:.6f}" if v is not None else "N/A"

        ts = datetime.now().isoformat(timespec="seconds")
        rows.append({
            "model":          model,
            "encoder_layers": encoder_layers,
            "pretrain_source": pretrain_source,
            "dataset":        dataset,
            "f1":             _fmt(f1),
            "precision":      _fmt(prec),
            "recall":         _fmt(rec),
            "accuracy":       _fmt(acc),
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
                  datasets: list, log_dir: Path, out_csv: Path,
                  dry_run: bool, pretrain_source: str = "monash"):
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
        "--pretrain_source", pretrain_source,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"  [{model:14s}] GPU={gpu}  layers={encoder_layers}  src={pretrain_source}"
          f"  log={log_path.relative_to(ROOT)}")

    if dry_run:
        print(f"    CMD: {' '.join(cmd)}")
        return None

    fh   = open(log_path, "w")
    fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    proc._log_fh = fh
    proc._label  = f"{model}/layers{encoder_layers}"
    return proc


def run_anomaly_sweep(models: list, layer_configs: list, datasets: list,
                      gpu_override: int, dry_run: bool,
                      pretrain_source: str = "monash"):
    log_dir = ROOT / "logs" / "anomaly_sweep"
    out_csv = ROOT / "results" / "anomaly_sweep.csv"

    print("Anomaly detection sweep")
    print(f"  Models:          {models}")
    print(f"  Layers:          {layer_configs}")
    print(f"  Datasets:        {datasets}")
    print(f"  pretrain_source: {pretrain_source}")
    if dry_run:
        print("  DRY RUN\n")

    for n_layers in layer_configs:
        print(f"\n{'='*60}")
        print(f"  encoder_layers={n_layers}")
        print(f"{'='*60}")

        procs = []
        for model in models:
            gpu  = gpu_override if gpu_override is not None else MODEL_GPU[model]
            proc = launch_worker(model, n_layers, gpu, datasets, log_dir, out_csv,
                                 dry_run, pretrain_source=pretrain_source)
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

    print(f"\n\nAnomaly sweep complete. Results in {out_csv}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anomaly detection sweep")
    parser.add_argument("--layers",          nargs="+", type=int, default=LAYER_CONFIGS)
    parser.add_argument("--models",          nargs="+", default=ALL_MODELS, choices=ALL_MODELS, metavar="MODEL")
    parser.add_argument("--datasets",        nargs="+", default=ANOMALY_DATASETS)
    parser.add_argument("--pretrain_source", type=str,  default="monash",
                        help="monash | synthetic | monash+synthetic")
    parser.add_argument("--gpu_override",    type=int,  default=None)
    parser.add_argument("--dry_run",         action="store_true")

    # internal worker mode
    parser.add_argument("--_worker",        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model",          type=str,            help=argparse.SUPPRESS)
    parser.add_argument("--encoder_layers", type=int,            help=argparse.SUPPRESS)
    parser.add_argument("--gpu",            type=int,            help=argparse.SUPPRESS)
    parser.add_argument("--out_csv",        type=str,            help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args._worker:
        run_model_worker(
            model           = args.model,
            encoder_layers  = args.encoder_layers,
            gpu             = args.gpu,
            datasets        = args.datasets,
            out_csv         = Path(args.out_csv),
            pretrain_source = args.pretrain_source,
        )
        return

    run_anomaly_sweep(
        args.models, args.layers, args.datasets,
        args.gpu_override, args.dry_run,
        pretrain_source=args.pretrain_source,
    )


if __name__ == "__main__":
    main()
