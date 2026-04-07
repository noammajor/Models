#!/usr/bin/env python3
"""
run_layer_forecast.py — Forecasting evaluation across layer-sweep checkpoints.

For each encoder layer config [2, 6, 12, 24], all models are evaluated in
parallel (one per GPU) on all datasets using the tournament checkpoint search:
  pred96 (all ckpts) → top3 → pred192 → top2 → pred336 → best → pred720

Also includes patchtst_random as a baseline (no layer sweep — runs once).

Results saved per model to:
  results/layer_forecast_{model}_layers{N}.csv

Usage:
    python run_layer_forecast.py                        # all models, all layers
    python run_layer_forecast.py --layers 6 12
    python run_layer_forecast.py --models dino lejepa
    python run_layer_forecast.py --dry_run
"""

import argparse
import csv
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy 

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
 # noqa: ensure numpy is in sys.modules before lazy imports

LAYER_CONFIGS = [2, 4, 8, 12, 24]
DATASETS      = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic"]
PRED_LENS     = [96, 192, 336, 720]

# Model → GPU assignment (match run_layer_sweep.py)
MODEL_GPU = {
    "dino":            0,
    "jepa_simple":     1,
    "lejepa":          2,
    "patchtst":        3,
    "npt":             4,
    "patchtst_random": 5,
}

ALL_MODELS = list(MODEL_GPU.keys())


# ── checkpoint discovery (layer-aware) ───────────────────────────────────────

def discover_checkpoints(model: str, encoder_layers: int) -> list:
    """Return sorted list of checkpoint epoch numbers for a given layer config."""
    suffix = f"_layers{encoder_layers}"

    if model == "dino":
        ckpt_dir = ROOT / f"checkpoints{suffix}"
        found = sorted(
            int(p.stem.replace("checkpoint", ""))
            for p in ckpt_dir.glob("checkpoint*.pth")
            if p.stem.replace("checkpoint", "").isdigit()
        ) if ckpt_dir.exists() else []
        return found or []

    elif model == "jepa_simple":
        import re as _re
        ckpt_dir = ROOT / "output_model" / f"JEPA{suffix}"
        found = sorted(
            int(m.group(1))
            for p in ckpt_dir.glob("_epoch*best_model.pt")
            for m in [_re.search(r'_epoch(\d+)best_model', p.stem)]
            if m
        ) if ckpt_dir.exists() else []
        return found or []

    elif model == "lejepa":
        import re as _re
        ckpt_dir = ROOT / "output_model" / f"LE-JEPA{suffix}"
        found = sorted(
            int(m.group(1))
            for p in ckpt_dir.glob("_epoch*best_model.pt")
            for m in [_re.search(r'_epoch(\d+)best_model', p.stem)]
            if m
        ) if ckpt_dir.exists() else []
        return found or []

    elif model == "npt":
        import re as _re, importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("config_ntp", ROOT / "NPT" / "config_ntp.py")
        _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        _cfg = _mod.config
        # override n_layers to build correct filename prefix
        _cfg = dict(_cfg); _cfg['n_layers'] = encoder_layers
        _cfg['pretrained_model_id'] = encoder_layers  # mirrors run_ntp save-side logic
        _prefix = (f"ntp_pretrained"
                   f"_patch{_cfg['patch_size']}"
                   f"_patches{_cfg['ratio_patches']}"
                   f"_epochs{_cfg['num_epochs']}"
                   f"_model{_cfg['pretrained_model_id']}_epoch")
        save_dir = ROOT / "NPT" / "saved_models" / "monash" / "ntp" / f"layers{encoder_layers}"
        found = sorted(set(
            int(m.group(1))
            for p in save_dir.glob("*.pt")
            if p.stem.startswith(_prefix)
            for m in [_re.search(r'_epoch(\d+)$', p.stem)]
            if m
        )) if save_dir.exists() else []
        return found or [None]

    elif model == "patchtst":
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("config_patchtst",
                    ROOT / "PatchTST_self_supervised" / "config_patchtst.py")
        _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        _cfg = dict(_mod.config); _cfg['n_layers'] = encoder_layers
        _cfg['pretrained_model_id'] = encoder_layers  # mirrors run_patchtst save-side logic
        _prefix = (f"patchtst_pretrained"
                   f"_cw{_cfg['context_points']}"
                   f"_patch{_cfg['patch_len']}"
                   f"_stride{_cfg['stride']}"
                   f"_epochs-pretrain{_cfg['n_epochs_pretrain']}"
                   f"_mask{_cfg['mask_ratio']}"
                   f"_model{_cfg['pretrained_model_id']}_")
        save_dir = ROOT / "PatchTST_self_supervised" / "saved_models" / "monash" / "masked_patchtst" / "based_model" / f"layers{encoder_layers}"
        found = set()
        for p in save_dir.glob("*.pth"):
            if p.stem.startswith(_prefix):
                sfx = p.stem[len(_prefix):]
                if sfx.isdigit():
                    found.add(int(sfx))
        return sorted(found) or [None]

    elif model == "patchtst_random":
        return [None]

    return [None]


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


# ── single checkpoint evaluation ─────────────────────────────────────────────

def eval_checkpoint(model: str, dataset: str, pred_len: int, ckpt,
                    gpu: int, log_dir: Path, encoder_layers: int) -> float:
    ckpt_tag = str(ckpt) if ckpt is not None else "best"
    log_path = log_dir / f"pred{pred_len}_ckpt{ckpt_tag}.log"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    with log_to_file(log_path):
        try:
            from Train_and_downstream import run

            if model in ("jepa_simple", "lejepa", "dino"):
                result = run(
                    model=model,
                    skip_train=True,
                    forecast_dataset=dataset,
                    pred_lens=[pred_len],
                    checkpoints=[ckpt],
                    encoder_layers=encoder_layers,
                )
                return result[1] if result else None

            elif model == "npt":
                result = run(
                    model="npt",
                    skip_train=True,
                    forecast_dataset=dataset,
                    pred_len=pred_len,
                    checkpoints=[ckpt] if ckpt is not None else None,
                    encoder_layers=encoder_layers,
                )
                return result[0] if isinstance(result, tuple) else result

            elif model == "patchtst":
                result = run(
                    model="patchtst",
                    skip_train=True,
                    forecast_dataset=dataset,
                    pred_len=pred_len,
                    checkpoints=[ckpt] if ckpt is not None else None,
                    encoder_layers=encoder_layers,
                )
                return result[0] if isinstance(result, tuple) else result

            elif model == "patchtst_random":
                result = run(
                    model="patchtst_random",
                    skip_train=True,
                    forecast_dataset=dataset,
                    pred_len=pred_len,
                    encoder_layers=encoder_layers,
                )
                return result[0] if isinstance(result, tuple) else result

        except Exception as e:
            print(f"[ERROR] {model}/{dataset}/pred{pred_len}/ckpt{ckpt_tag}: {e}")
            import traceback; traceback.print_exc()
            return None


# ── tournament checkpoint search ─────────────────────────────────────────────

def checkpoint_search(model: str, dataset: str, all_checkpoints: list,
                      gpu: int, log_base: Path, encoder_layers: int) -> dict:
    log_dir = log_base / f"layers{encoder_layers}" / model / dataset
    results = {}

    if not all_checkpoints:
        print(f"  [WARN] No checkpoints found for {model} layers={encoder_layers}")
        return results

    print(f"\n  [Search] pred_len=96 — {len(all_checkpoints)} checkpoints: {all_checkpoints}")
    mses_96 = {}
    for ckpt in all_checkpoints:
        mse = eval_checkpoint(model, dataset, 96, ckpt, gpu, log_dir, encoder_layers)
        if mse is not None:
            mses_96[ckpt] = mse
            print(f"    ckpt {ckpt}: MSE={mse:.4f}")

    if not mses_96:
        return results

    top3 = sorted(mses_96, key=mses_96.get)[:3]
    results[96] = mses_96[top3[0]]
    print(f"  → Top-3: {top3}  (best MSE={results[96]:.4f})")

    print(f"\n  [Search] pred_len=192 — top-3: {top3}")
    mses_192 = {}
    for ckpt in top3:
        mse = eval_checkpoint(model, dataset, 192, ckpt, gpu, log_dir, encoder_layers)
        if mse is not None:
            mses_192[ckpt] = mse
            print(f"    ckpt {ckpt}: MSE={mse:.4f}")

    if not mses_192:
        return results

    top2 = sorted(mses_192, key=mses_192.get)[:2]
    results[192] = mses_192[top2[0]]
    print(f"  → Top-2: {top2}  (best MSE={results[192]:.4f})")

    print(f"\n  [Search] pred_len=336 — top-2: {top2}")
    mses_336 = {}
    for ckpt in top2:
        mse = eval_checkpoint(model, dataset, 336, ckpt, gpu, log_dir, encoder_layers)
        if mse is not None:
            mses_336[ckpt] = mse
            print(f"    ckpt {ckpt}: MSE={mse:.4f}")

    if not mses_336:
        return results

    best = min(mses_336, key=mses_336.get)
    results[336] = mses_336[best]
    print(f"  → Best: ckpt {best}  (MSE={results[336]:.4f})")

    print(f"\n  [Search] pred_len=720 — best ckpt: {best}")
    mse = eval_checkpoint(model, dataset, 720, best, gpu, log_dir, encoder_layers)
    if mse is not None:
        results[720] = mse
        print(f"    MSE={mse:.4f}")

    return results


# ── per-model worker (runs as subprocess) ────────────────────────────────────

def eval_best(model: str, dataset: str, pred_len: int,
              gpu: int, log_dir: Path, encoder_layers: int) -> float:
    """Evaluate all pred_lens using only the best checkpoint (no tournament)."""
    log_path = log_dir / f"pred{pred_len}_ckptbest.log"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_to_file(log_path):
        try:
            from Train_and_downstream import run
            if model in ("jepa_simple", "lejepa", "dino"):
                result = run(model=model, skip_train=True, forecast_dataset=dataset,
                             pred_lens=[pred_len], checkpoints=["best"],
                             encoder_layers=encoder_layers)
                return result[1] if result else None
            elif model == "npt":
                result = run(model="npt", skip_train=True, forecast_dataset=dataset,
                             pred_len=pred_len, checkpoints=None,
                             encoder_layers=encoder_layers)
                return result[0] if isinstance(result, tuple) else result
        except Exception as e:
            print(f"[ERROR] {model}/{dataset}/pred{pred_len}/best: {e}")
            import traceback; traceback.print_exc()
            return None


def run_model_worker(model: str, encoder_layers: int, gpu: int,
                     datasets: list, out_csv: Path, best_only: bool = False):
    """Run full forecasting for one model at one layer config. Called in subprocess."""
    log_base = ROOT / "logs" / "layer_forecast"
    fieldnames = ["model", "encoder_layers", "dataset", "pred_len", "mse", "timestamp"]

    existing = set()
    rows = []
    if out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                rows.append(row)
                existing.add((row["model"], int(row["encoder_layers"]),
                               row["dataset"], int(row["pred_len"])))

    all_checkpoints = discover_checkpoints(model, encoder_layers)
    print(f"\n{'='*60}")
    print(f"  {model.upper()}  layers={encoder_layers}  checkpoints={all_checkpoints}")
    print(f"{'='*60}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for dataset in datasets:
        already_done = [(model, encoder_layers, dataset, pl) in existing for pl in PRED_LENS]
        if all(already_done):
            print(f"\n  {model}/layers{encoder_layers}/{dataset} — already complete, skipping.")
            continue

        print(f"\n--- {model} / layers={encoder_layers} / {dataset} ---")

        try:
            if best_only:
                log_dir = log_base / f"layers{encoder_layers}" / model / dataset
                mses = {}
                for pl in PRED_LENS:
                    mse = eval_best(model, dataset, pl, gpu, log_dir, encoder_layers)
                    if mse is not None:
                        mses[pl] = mse
            else:
                mses = checkpoint_search(model, dataset, all_checkpoints,
                                         gpu, log_base, encoder_layers)
        except Exception as e:
            print(f"  [ERROR] {model}/layers{encoder_layers}/{dataset}: {e}")
            import traceback; traceback.print_exc()
            continue

        ts = datetime.now().isoformat(timespec="seconds")
        for pred_len, mse in mses.items():
            key = (model, encoder_layers, dataset, pred_len)
            if key not in existing:
                rows.append({
                    "model":          model,
                    "encoder_layers": encoder_layers,
                    "dataset":        dataset,
                    "pred_len":       pred_len,
                    "mse":            f"{mse:.6f}" if mse is not None else "N/A",
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
                  datasets: list, log_dir: Path, dry_run: bool,
                  best_only: bool = False):
    out_csv = ROOT / "results" / f"layer_forecast_{model}_layers{encoder_layers}.csv"
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
    if best_only:
        cmd.append("--best_only")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"  [{model:14s}] GPU={gpu}  layers={encoder_layers}"
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


def run_forecast_sweep(models: list, layer_configs: list,
                       datasets: list, dry_run: bool, gpu_override: int = None,
                       best_only: bool = False):
    log_dir = ROOT / "logs" / "layer_forecast"

    # patchtst_random runs once (no layer sweep)
    random_models  = [m for m in models if m == "patchtst_random"]
    layered_models = [m for m in models if m != "patchtst_random"]

    for n_layers in layer_configs:
        print(f"\n{'='*60}")
        print(f"  FORECASTING  encoder_layers={n_layers}")
        print(f"{'='*60}")

        procs = []
        for model in layered_models + random_models:
            gpu  = gpu_override if gpu_override is not None else MODEL_GPU[model]
            proc = launch_worker(model, n_layers, gpu, datasets, log_dir, dry_run, best_only=best_only)
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

    print(f"\n\nForecast sweep complete. Results in results/layer_forecast_*.csv")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Layer-sweep forecasting evaluation")
    parser.add_argument("--layers",   nargs="+", type=int, default=LAYER_CONFIGS)
    parser.add_argument("--models",   nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                        metavar="MODEL")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,   metavar="DATASET")
    parser.add_argument("--dry_run",  action="store_true")
    parser.add_argument("--gpu_override", type=int, default=None,
                        help="Override GPU for all models in this run")
    parser.add_argument("--best_only", action="store_true",
                        help="Skip tournament — evaluate only the best checkpoint for all pred_lens")

    # internal worker mode
    parser.add_argument("--_worker",        action="store_true",  help=argparse.SUPPRESS)
    parser.add_argument("--model",          type=str,             help=argparse.SUPPRESS)
    parser.add_argument("--encoder_layers", type=int,             help=argparse.SUPPRESS)
    parser.add_argument("--gpu",            type=int,             help=argparse.SUPPRESS)
    parser.add_argument("--out_csv",        type=str,             help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args._worker:
        run_model_worker(
            model=args.model,
            encoder_layers=args.encoder_layers,
            gpu=args.gpu,
            datasets=args.datasets,
            out_csv=Path(args.out_csv),
            best_only=args.best_only,
        )
        return

    print(f"Layer forecast sweep")
    print(f"  Models:   {args.models}")
    print(f"  Layers:   {args.layers}")
    print(f"  Datasets: {args.datasets}")
    if args.dry_run:
        print("  DRY RUN\n")

    run_forecast_sweep(args.models, args.layers, args.datasets, args.dry_run,
                       gpu_override=args.gpu_override, best_only=args.best_only)


if __name__ == "__main__":
    main()
