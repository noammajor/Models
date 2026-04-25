#!/usr/bin/env python3
"""
run_finetune_8layers.py — Fine-tuning evaluation for all three downstream tasks
                          using 8-layer pretrained backbones.

Runs forecasting, anomaly detection, and classification for all models
in parallel (one subprocess per model) using their 8-layer pretrained
checkpoints (skip_train=True).

Results saved to:
  results/finetune_8layers_forecast.csv
  results/finetune_8layers_classification.csv
  results/finetune_8layers_anomaly.csv

Usage:
    python run_finetune_8layers.py
    python run_finetune_8layers.py --models dino lejepa timedart
    python run_finetune_8layers.py --pretrain_source monash+synthetic
    python run_finetune_8layers.py --gpu_override 0
    python run_finetune_8layers.py --tasks forecast anomaly
    python run_finetune_8layers.py --forecast_datasets etth1 etth2
    python run_finetune_8layers.py --dry_run
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

ENCODER_LAYERS = 8

FORECAST_DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather"]
PRED_LENS         = [96, 192, 336, 720]

CLASSIFICATION_DATASETS = [
    "JapaneseVowels", "SpokenArabicDigits", "FaceDetection",
    "Handwriting", "PEMS-SF", "UWaveGestureLibrary",
    "Heartbeat", "SelfRegulationSCP1", "SelfRegulationSCP2",
    "EthanolConcentration",
]

ANOMALY_DATASETS = ["SMD", "MSL", "SMAP", "SWaT", "PSM"]

# Per-dataset forecast batch sizes (same as run_layer_forecast.py)
DATASET_FORECAST_BS = {
    "etth1":       256,
    "etth2":       256,
    "ettm1":       256,
    "ettm2":       256,
    "weather":     128,
}

# Model → GPU assignment
MODEL_GPU = {
    "dino":            0,
    "jepa_simple":     1,
    "lejepa":          2,
    "patchtst":        3,
    "npt":             4,
    "timedart":        5,
    "patchtst_random": 6,
}
ALL_MODELS = list(MODEL_GPU.keys())
ALL_TASKS  = ["forecast", "classify", "anomaly"]


# ── logging ───────────────────────────────────────────────────────────────────

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


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _load_csv(path: Path, fieldnames: list) -> tuple:
    """Return (rows, existing_keys_set). existing_keys = frozenset of (model, dataset, ...)."""
    rows = []
    existing = set()
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                # back-fill any missing columns added later
                for fn in fieldnames:
                    row.setdefault(fn, "N/A")
                rows.append(row)
    return rows, existing


def _save_csv(path: Path, rows: list, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── result unpacking helpers ──────────────────────────────────────────────────

def _unpack_forecast(model: str, result) -> tuple:
    """
    Extract (mse, mae) from a runner's return value.

    Runner return layouts (when task="forecast", cls/anom are None):
      dino / jepa_simple / patchtst : (ckpt,  mse,  cls_acc, anom)
      lejepa                        : (mse,   mae,  cls_acc, anom)
      npt                           : (mse,   cls_acc, anom)
      timedart                      : (pred,  mse,  mae, cls_acc, anom)
    """
    if result is None:
        return None, None
    if isinstance(result, (int, float)):
        return float(result), None
    if not isinstance(result, tuple) or len(result) == 0:
        return None, None

    if model == "lejepa":
        mse = result[0]
        mae = result[1] if len(result) > 1 else None
    elif model == "npt":
        mse = result[0]
        mae = None
    elif model == "timedart":
        mse = result[1] if len(result) > 1 else None
        mae = result[2] if len(result) > 2 else None
    else:
        # dino, jepa_simple, patchtst: (ckpt, mse, cls_acc, anom)
        mse = result[1] if len(result) > 1 else None
        mae = None

    return (float(mse) if mse is not None else None,
            float(mae) if mae is not None else None)


def _unpack_classify(model: str, result) -> float | None:
    """
    Extract accuracy from a runner's return value.

    Runner return layouts (when task="classify", mse/anom are None):
      dino / jepa_simple / patchtst : (ckpt,  None, cls_acc, None)
      lejepa                        : (None,  None, cls_acc, None)
      npt                           : (None,  cls_acc, None)
      timedart                      : (pred,  None, None, cls_acc, None)
    """
    if result is None:
        return None
    if isinstance(result, (int, float)):
        return float(result)
    if not isinstance(result, tuple) or len(result) == 0:
        return None

    if model == "timedart":
        acc = result[3] if len(result) > 3 else None
    elif model == "npt":
        acc = result[1] if len(result) > 1 else None
    else:
        # dino, jepa_simple, patchtst, lejepa: (*, *, cls_acc, anom)
        acc = result[2] if len(result) > 2 else None

    return float(acc) if acc is not None else None


# ── single worker (called in subprocess) ─────────────────────────────────────

def run_model_worker(model: str, gpu: int, tasks: list,
                     forecast_datasets: list, classification_datasets: list,
                     anomaly_datasets: list, pretrain_source: str,
                     out_forecast_csv: Path, out_cls_csv: Path, out_anom_csv: Path):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"\n{'='*60}")
    print(f"  MODEL: {model.upper()}  layers={ENCODER_LAYERS}  src={pretrain_source}")
    print(f"  tasks={tasks}")
    print(f"{'='*60}")

    from Train_and_downstream import run

    # ── forecasting ────────────────────────────────────────────────────────────
    if "forecast" in tasks:
        fc_fields = ["model", "encoder_layers", "pretrain_source",
                     "dataset", "pred_len", "mse", "mae", "timestamp"]
        fc_rows, _ = _load_csv(out_forecast_csv, fc_fields)
        fc_existing = {
            (r["model"], r["encoder_layers"], r.get("pretrain_source", "monash"),
             r["dataset"], int(r["pred_len"]))
            for r in fc_rows
            if r.get("mse", "N/A") not in ("N/A", "", None)
        }

        for dataset in forecast_datasets:
            for pred_len in PRED_LENS:
                key = (model, str(ENCODER_LAYERS), pretrain_source, dataset, pred_len)
                if key in fc_existing:
                    print(f"\n  [forecast] {model}/{dataset}/pl{pred_len} — already done, skipping.")
                    continue

                print(f"\n--- [forecast] {model} / {dataset} / pred_len={pred_len} ---")
                log_path = (ROOT / "logs" / "finetune_8layers" / "forecast"
                            / pretrain_source / model / f"{dataset}_pl{pred_len}.log")

                if dataset in DATASET_FORECAST_BS:
                    os.environ["TS_FORECAST_BS"] = str(DATASET_FORECAST_BS[dataset])
                else:
                    os.environ.pop("TS_FORECAST_BS", None)

                mse = mae = None
                with log_to_file(log_path):
                    try:
                        result = run(
                            model           = model,
                            task            = "forecast",
                            forecast_dataset= dataset,
                            pred_lens       = [pred_len],
                            encoder_layers  = ENCODER_LAYERS,
                            pretrain_source = pretrain_source,
                            gpu             = gpu,
                            linear_probe    = False,
                        )
                        mse, mae = _unpack_forecast(model, result)
                    except Exception as e:
                        print(f"[ERROR] {model}/{dataset}/pl{pred_len}: {e}")
                        import traceback; traceback.print_exc()

                ts = datetime.now().isoformat(timespec="seconds")
                fc_rows.append({
                    "model":          model,
                    "encoder_layers": ENCODER_LAYERS,
                    "pretrain_source": pretrain_source,
                    "dataset":        dataset,
                    "pred_len":       pred_len,
                    "mse":            f"{mse:.6f}" if mse is not None else "N/A",
                    "mae":            f"{mae:.6f}" if mae is not None else "N/A",
                    "timestamp":      ts,
                })
                _save_csv(out_forecast_csv, fc_rows, fc_fields)
                print(f"  Saved → {out_forecast_csv}")

    # ── classification ─────────────────────────────────────────────────────────
    if "classify" in tasks:
        cls_fields = ["model", "encoder_layers", "pretrain_source",
                      "dataset", "accuracy", "timestamp"]
        cls_rows, _ = _load_csv(out_cls_csv, cls_fields)
        cls_existing = {
            (r["model"], r["encoder_layers"], r.get("pretrain_source", "monash"), r["dataset"])
            for r in cls_rows
            if r.get("accuracy", "N/A") not in ("N/A", "", None)
        }

        for dataset in classification_datasets:
            key = (model, str(ENCODER_LAYERS), pretrain_source, dataset)
            if key in cls_existing:
                print(f"\n  [classify] {model}/{dataset} — already done, skipping.")
                continue

            print(f"\n--- [classify] {model} / {dataset} ---")
            log_path = (ROOT / "logs" / "finetune_8layers" / "classify"
                        / pretrain_source / model / f"{dataset}.log")

            acc = None
            with log_to_file(log_path):
                try:
                    result = run(
                        model                  = model,
                        task                   = "classify",
                        classification_dataset = dataset,
                        encoder_layers         = ENCODER_LAYERS,
                        pretrain_source        = pretrain_source,
                        gpu                    = gpu,
                        linear_probe           = False,
                    )
                    acc = _unpack_classify(model, result)
                except Exception as e:
                    print(f"[ERROR] {model}/{dataset}: {e}")
                    import traceback; traceback.print_exc()

            ts = datetime.now().isoformat(timespec="seconds")
            cls_rows.append({
                "model":          model,
                "encoder_layers": ENCODER_LAYERS,
                "pretrain_source": pretrain_source,
                "dataset":        dataset,
                "accuracy":       f"{acc:.6f}" if acc is not None else "N/A",
                "timestamp":      ts,
            })
            _save_csv(out_cls_csv, cls_rows, cls_fields)
            print(f"  Saved → {out_cls_csv}")

    # ── anomaly detection ──────────────────────────────────────────────────────
    if "anomaly" in tasks:
        anom_fields = ["model", "encoder_layers", "pretrain_source",
                       "dataset", "f1", "precision", "recall", "accuracy", "timestamp"]
        anom_rows, _ = _load_csv(out_anom_csv, anom_fields)
        anom_existing = {
            (r["model"], r["encoder_layers"], r.get("pretrain_source", "monash"), r["dataset"])
            for r in anom_rows
            if r.get("f1", "N/A") not in ("N/A", "", None)
        }

        for dataset in anomaly_datasets:
            key = (model, str(ENCODER_LAYERS), pretrain_source, dataset)
            if key in anom_existing:
                print(f"\n  [anomaly] {model}/{dataset} — already done, skipping.")
                continue

            print(f"\n--- [anomaly] {model} / {dataset} ---")
            log_path = (ROOT / "logs" / "finetune_8layers" / "anomaly"
                        / pretrain_source / model / f"{dataset}.log")

            f1 = prec = rec = acc = None
            with log_to_file(log_path):
                try:
                    result = run(
                        model           = model,
                        task            = "anomaly",
                        anomaly_dataset = dataset,
                        encoder_layers  = ENCODER_LAYERS,
                        pretrain_source = pretrain_source,
                        gpu             = gpu,
                    )
                    # result last element is the anom dict
                    anom = result[-1] if isinstance(result, tuple) else result
                    if isinstance(anom, dict):
                        f1   = anom.get("f1")
                        prec = anom.get("precision")
                        rec  = anom.get("recall")
                        acc  = anom.get("accuracy")
                except Exception as e:
                    print(f"[ERROR] {model}/{dataset}: {e}")
                    import traceback; traceback.print_exc()

            def _fmt(v): return f"{v:.6f}" if v is not None else "N/A"
            ts = datetime.now().isoformat(timespec="seconds")
            anom_rows.append({
                "model":          model,
                "encoder_layers": ENCODER_LAYERS,
                "pretrain_source": pretrain_source,
                "dataset":        dataset,
                "f1":             _fmt(f1),
                "precision":      _fmt(prec),
                "recall":         _fmt(rec),
                "accuracy":       _fmt(acc),
                "timestamp":      ts,
            })
            _save_csv(out_anom_csv, anom_rows, anom_fields)
            print(f"  Saved → {out_anom_csv}")

    print(f"\nDone: {model}  tasks={tasks}")


# ── parallel launcher ─────────────────────────────────────────────────────────

def launch_worker(model: str, gpu: int, tasks: list,
                  forecast_datasets: list, classification_datasets: list,
                  anomaly_datasets: list, pretrain_source: str,
                  out_forecast_csv: Path, out_cls_csv: Path, out_anom_csv: Path,
                  log_dir: Path, dry_run: bool):
    log_path = log_dir / f"{model}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, __file__,
        "--_worker",
        "--model",           model,
        "--gpu",             str(gpu),
        "--tasks",           *tasks,
        "--forecast_datasets",       *forecast_datasets,
        "--classification_datasets", *classification_datasets,
        "--anomaly_datasets",        *anomaly_datasets,
        "--pretrain_source", pretrain_source,
        "--out_forecast_csv",  str(out_forecast_csv),
        "--out_cls_csv",       str(out_cls_csv),
        "--out_anom_csv",      str(out_anom_csv),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"  [{model:16s}] GPU={gpu}  tasks={tasks}  log={log_path.relative_to(ROOT)}")

    if dry_run:
        print(f"    CMD: {' '.join(cmd)}")
        return None

    fh   = open(log_path, "w")
    fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    proc._log_fh = fh
    proc._label  = model
    return proc


def run_finetune_sweep(models: list, tasks: list,
                       forecast_datasets: list, classification_datasets: list,
                       anomaly_datasets: list, pretrain_source: str,
                       gpu_override: int, dry_run: bool):
    src_tag = f"_{pretrain_source.replace('+', '_')}" if pretrain_source != "monash" else ""
    log_dir = ROOT / "logs" / "finetune_8layers"

    out_forecast_csv = ROOT / "results" / f"finetune_8layers_forecast{src_tag}.csv"
    out_cls_csv      = ROOT / "results" / f"finetune_8layers_classification{src_tag}.csv"
    out_anom_csv     = ROOT / "results" / f"finetune_8layers_anomaly{src_tag}.csv"

    print(f"\nFine-tuning sweep — {ENCODER_LAYERS}-layer backbones")
    print(f"  Models:       {models}")
    print(f"  Tasks:        {tasks}")
    print(f"  Pretrain src: {pretrain_source}")
    if "forecast" in tasks:
        print(f"  Forecast:     {forecast_datasets}  pred_lens={PRED_LENS}")
    if "classify" in tasks:
        print(f"  Classify:     {classification_datasets}")
    if "anomaly" in tasks:
        print(f"  Anomaly:      {anomaly_datasets}")
    if dry_run:
        print("  DRY RUN\n")

    procs = []
    for model in models:
        gpu  = gpu_override if gpu_override is not None else MODEL_GPU[model]
        proc = launch_worker(
            model=model, gpu=gpu, tasks=tasks,
            forecast_datasets=forecast_datasets,
            classification_datasets=classification_datasets,
            anomaly_datasets=anomaly_datasets,
            pretrain_source=pretrain_source,
            out_forecast_csv=out_forecast_csv,
            out_cls_csv=out_cls_csv,
            out_anom_csv=out_anom_csv,
            log_dir=log_dir,
            dry_run=dry_run,
        )
        if proc is not None:
            procs.append(proc)

    if dry_run or not procs:
        return

    print(f"\nWaiting for {len(procs)} workers …")
    failed = []
    for proc in procs:
        proc.wait()
        proc._log_fh.close()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  {proc._label}: {status}")
        if proc.returncode != 0:
            failed.append(proc._label)

    if failed:
        print(f"\nWARNING: {len(failed)} workers failed: {failed}")
    else:
        print(f"\nAll {len(procs)} workers completed.")

    print(f"\nResults:")
    if "forecast" in tasks:
        print(f"  Forecast      → {out_forecast_csv}")
    if "classify" in tasks:
        print(f"  Classification → {out_cls_csv}")
    if "anomaly" in tasks:
        print(f"  Anomaly        → {out_anom_csv}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=f"Fine-tuning sweep using {ENCODER_LAYERS}-layer pretrained backbones"
    )
    parser.add_argument("--models",   nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                        metavar="MODEL", help=f"Models to run. Default: all. Choices: {ALL_MODELS}")
    parser.add_argument("--tasks",    nargs="+", default=ALL_TASKS,
                        choices=ALL_TASKS, metavar="TASK",
                        help="Tasks to run. Default: all. Choices: forecast classify anomaly")
    parser.add_argument("--pretrain_source", type=str, default="monash",
                        choices=["monash", "synthetic", "monash+synthetic"],
                        help="Pretrain data source (default: monash)")
    parser.add_argument("--gpu_override", type=int, default=None,
                        help="Override GPU for all models")
    parser.add_argument("--dry_run",  action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--forecast_datasets",       nargs="+", default=FORECAST_DATASETS)
    parser.add_argument("--classification_datasets", nargs="+", default=CLASSIFICATION_DATASETS)
    parser.add_argument("--anomaly_datasets",        nargs="+", default=ANOMALY_DATASETS)

    # internal worker flags
    parser.add_argument("--_worker",          action="store_true",  help=argparse.SUPPRESS)
    parser.add_argument("--model",            type=str,             help=argparse.SUPPRESS)
    parser.add_argument("--gpu",              type=int,             help=argparse.SUPPRESS)
    parser.add_argument("--out_forecast_csv", type=str,             help=argparse.SUPPRESS)
    parser.add_argument("--out_cls_csv",      type=str,             help=argparse.SUPPRESS)
    parser.add_argument("--out_anom_csv",     type=str,             help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args._worker:
        run_model_worker(
            model                   = args.model,
            gpu                     = args.gpu,
            tasks                   = args.tasks,
            forecast_datasets       = args.forecast_datasets,
            classification_datasets = args.classification_datasets,
            anomaly_datasets        = args.anomaly_datasets,
            pretrain_source         = args.pretrain_source,
            out_forecast_csv        = Path(args.out_forecast_csv),
            out_cls_csv             = Path(args.out_cls_csv),
            out_anom_csv            = Path(args.out_anom_csv),
        )
        return

    run_finetune_sweep(
        models                  = args.models,
        tasks                   = args.tasks,
        forecast_datasets       = args.forecast_datasets,
        classification_datasets = args.classification_datasets,
        anomaly_datasets        = args.anomaly_datasets,
        pretrain_source         = args.pretrain_source,
        gpu_override            = args.gpu_override,
        dry_run                 = args.dry_run,
    )


if __name__ == "__main__":
    main()
