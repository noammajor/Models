#!/usr/bin/env python3
"""
run_layer_forecast.py — Forecasting evaluation across layer-sweep checkpoints.

For each encoder layer config [2, 4, 8, 12, 24], all models are evaluated in
parallel (one per GPU) on all datasets using the tournament checkpoint search:
  pred96 (all ckpts) → top3 → pred192 → top2 → pred336 → best → pred720

Also includes patchtst_random as a baseline (no layer sweep — runs once).

Results saved per model to:
  results/layer_forecast_{model}_layers{N}.csv

Usage:
    python run_layer_forecast.py                        # all models, all layers
    python run_layer_forecast.py --layers 4 12
    python run_layer_forecast.py --models dino lejepa
    python run_layer_forecast.py --linear_probe false   # fine-tune mode
    python run_layer_forecast.py --pred_lens 96 192     # only --best_only mode
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

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from Train_and_downstream import run


def _str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "y")


LAYER_CONFIGS = [2, 4, 8, 12, 24]
DATASETS      = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic"]
PRED_LENS     = [96, 192, 336, 720]

# Model → GPU assignment (match run_layer_sweep.py)
MODEL_GPU = {
    "dino":            0,
    "jepa":            7,
    "lejepa":          2,
    "patchtst":        3,
    "ntp":             4,
    "patchtst_random": 5,
    "timedart":        6,
}

ALL_MODELS = list(MODEL_GPU.keys())


# ── checkpoint discovery (layer-aware) ───────────────────────────────────────

def discover_checkpoints(model: str, encoder_layers: int,
                         pretrain_source: str = None,
                         output_dir: str = None) -> list:
    """Return sorted list of checkpoint epoch numbers for a given layer config."""
    suffix = f"_layers{encoder_layers}"
    src_tag = f"_{pretrain_source.replace('+', '_')}" if pretrain_source and pretrain_source != "monash" else ""

    if model == "dino":
        if output_dir is not None:
            base = output_dir.rstrip('/')
            if base.startswith('./'):
                base = base[2:]
            ckpt_dir = Path(base + suffix) if Path(base).is_absolute() else (ROOT / (base + suffix))
        else:
            ckpt_dir = ROOT / f"checkpoints{src_tag}{suffix}"
        found = sorted(
            int(p.stem.replace("checkpoint", ""))
            for p in ckpt_dir.glob("checkpoint*.pth")
            if p.stem.replace("checkpoint", "").isdigit()
        ) if ckpt_dir.exists() else []
        return found or []

    elif model == "jepa":
        import re as _re
        ckpt_dir = ROOT / "output_model" / f"JEPA{src_tag}{suffix}"
        found = sorted(
            int(m.group(1))
            for p in ckpt_dir.glob("_epoch*best_model.pt")
            for m in [_re.search(r'_epoch(\d+)best_model', p.stem)]
            if m
        ) if ckpt_dir.exists() else []
        return found or []

    elif model == "lejepa":
        import re as _re
        ckpt_dir = ROOT / "output_model" / f"LE-JEPA{src_tag}{suffix}"
        found = sorted(
            int(m.group(1))
            for p in ckpt_dir.glob("_epoch*best_model.pt")
            for m in [_re.search(r'_epoch(\d+)best_model', p.stem)]
            if m
        ) if ckpt_dir.exists() else []
        return found or []

    elif model == "ntp":
        import re as _re, importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("config_ntp", ROOT / "NTP" / "config_ntp.py")
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
        _ntp_src = pretrain_source if pretrain_source else "monash"
        save_dir = ROOT / "NTP" / "saved_models" / _ntp_src / "ntp" / f"layers{encoder_layers}"
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
        _ptst_src = pretrain_source if pretrain_source else "monash"
        save_dir = ROOT / "PatchTST_self_supervised" / "saved_models" / _ptst_src / "masked_patchtst" / "based_model" / f"layers{encoder_layers}"
        found = set()
        for p in save_dir.glob("*.pth"):
            if p.stem.startswith(_prefix):
                sfx = p.stem[len(_prefix):]
                if sfx.isdigit():
                    found.add(int(sfx))
        return sorted(found) or [None]

    elif model == "patchtst_random":
        return [None]

    elif model == "timedart":
        # TimeDart only saves ckpt_best.pth — no per-epoch tournament search
        return ["best"]

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
                    gpu: int, log_dir: Path, encoder_layers: int,
                    pretrain_source: str = None,
                    linear_probe: bool = True,
                    head_type: str = "linear",
                    output_dir: str = None):
    """Returns (mse, mae) tuple — mae may be None for models that don't expose it."""
    ckpt_tag = str(ckpt) if ckpt is not None else "best"
    log_path = log_dir / f"pred{pred_len}_ckpt{ckpt_tag}.log"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    with log_to_file(log_path):
        try:
            if model in ("jepa", "lejepa", "dino"):
                result = run(
                    model=model,
                    skip_train=True,
                    forecast_dataset=dataset,
                    pred_lens=[pred_len],
                    checkpoints=[ckpt],
                    encoder_layers=encoder_layers,
                    linear_probe=linear_probe,
                    head_type=head_type,
                    output_dir=output_dir,
                )
                return (result[1], None) if result else None

            elif model in ("ntp", "patchtst", "patchtst_random"):
                kwargs = dict(
                    model=model,
                    skip_train=True,
                    forecast_dataset=dataset,
                    pred_len=pred_len,
                    encoder_layers=encoder_layers,
                    pretrain_source=pretrain_source,
                    linear_probe=linear_probe,
                    head_type=head_type,
                )
                if model != "patchtst_random":
                    kwargs["checkpoints"] = [ckpt] if ckpt is not None else None
                result = run(**kwargs)
                if isinstance(result, tuple) and len(result) >= 2:
                    return (result[0], result[1])  # (mse, mae)
                return (result, None) if result is not None else None

            elif model == "timedart":
                result = run(
                    model="timedart",
                    skip_train=True,
                    forecast_dataset=dataset,
                    pred_lens=[pred_len],
                    encoder_layers=encoder_layers,
                    gpu=gpu,
                    pretrain_source=pretrain_source,
                    linear_probe=linear_probe,
                    head_type=head_type,
                )
                if isinstance(result, tuple) and len(result) >= 3:
                    return (result[1], result[2])
                return (result[1], None) if isinstance(result, tuple) else None

        except Exception as e:
            print(f"[ERROR] {model}/{dataset}/pred{pred_len}/ckpt{ckpt_tag}: {e}")
            import traceback; traceback.print_exc()
            return None


# ── tournament checkpoint search ─────────────────────────────────────────────

def checkpoint_search(model: str, dataset: str, all_checkpoints: list,
                      gpu: int, log_base: Path, encoder_layers: int,
                      pretrain_source: str = None,
                      linear_probe: bool = True,
                      head_type: str = "linear",
                      output_dir: str = None) -> dict:
    log_dir = log_base / f"layers{encoder_layers}" / model / dataset
    results = {}

    if not all_checkpoints:
        print(f"  [WARN] No checkpoints found for {model} layers={encoder_layers}")
        return results

    def _eval(pl, ck):
        return eval_checkpoint(model, dataset, pl, ck, gpu, log_dir, encoder_layers,
                               pretrain_source, linear_probe=linear_probe,
                               head_type=head_type,
                               output_dir=output_dir)

    def _mse(val):
        """Extract MSE from a (mse, mae) tuple or scalar — for ranking."""
        return val[0] if isinstance(val, tuple) else val

    print(f"\n  [Search] pred_len=96 — {len(all_checkpoints)} checkpoints: {all_checkpoints}")
    mses_96 = {}
    for ckpt in all_checkpoints:
        val = _eval(96, ckpt)
        if val is not None and _mse(val) is not None:
            mses_96[ckpt] = val
            print(f"    ckpt {ckpt}: MSE={_mse(val):.4f}")

    if not mses_96:
        return results

    top3 = sorted(mses_96, key=lambda c: _mse(mses_96[c]))[:3]
    results[96] = mses_96[top3[0]]
    print(f"  → Top-3: {top3}  (best MSE={_mse(results[96]):.4f})")

    print(f"\n  [Search] pred_len=192 — top-3: {top3}")
    mses_192 = {}
    for ckpt in top3:
        val = _eval(192, ckpt)
        if val is not None and _mse(val) is not None:
            mses_192[ckpt] = val
            print(f"    ckpt {ckpt}: MSE={_mse(val):.4f}")

    if not mses_192:
        return results

    top2 = sorted(mses_192, key=lambda c: _mse(mses_192[c]))[:2]
    results[192] = mses_192[top2[0]]
    print(f"  → Top-2: {top2}  (best MSE={_mse(results[192]):.4f})")

    print(f"\n  [Search] pred_len=336 — top-2: {top2}")
    mses_336 = {}
    for ckpt in top2:
        val = _eval(336, ckpt)
        if val is not None and _mse(val) is not None:
            mses_336[ckpt] = val
            print(f"    ckpt {ckpt}: MSE={_mse(val):.4f}")

    if not mses_336:
        return results

    best = min(mses_336, key=lambda c: _mse(mses_336[c]))
    results[336] = mses_336[best]
    print(f"  → Best: ckpt {best}  (MSE={_mse(results[336]):.4f})")

    print(f"\n  [Search] pred_len=720 — best ckpt: {best}")
    val = _eval(720, best)
    if val is not None and _mse(val) is not None:
        results[720] = val
        print(f"    MSE={_mse(val):.4f}")

    return results


# ── per-model worker (runs as subprocess) ────────────────────────────────────

def eval_best(model: str, dataset: str, pred_len: int,
              gpu: int, log_dir: Path, encoder_layers: int,
              pretrain_source: str = None,
              linear_probe: bool = True,
              head_type: str = "linear",
              output_dir: str = None):
    """Evaluate using only the best checkpoint. Returns (mse, mae) tuple."""
    log_path = log_dir / f"pred{pred_len}_ckptbest.log"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_to_file(log_path):
        try:
            if model in ("jepa", "lejepa", "dino"):
                result = run(model=model, skip_train=True, forecast_dataset=dataset,
                             pred_lens=[pred_len], checkpoints=["best"],
                             encoder_layers=encoder_layers,
                             pretrain_source=pretrain_source,
                             linear_probe=linear_probe,
                             head_type=head_type,
                             output_dir=output_dir)
                return (result[1], None) if result else None
            elif model in ("ntp", "patchtst", "patchtst_random"):
                kwargs = dict(model=model, skip_train=True, forecast_dataset=dataset,
                              pred_len=pred_len, encoder_layers=encoder_layers,
                              pretrain_source=pretrain_source,
                              linear_probe=linear_probe,
                              head_type=head_type)
                if model != "patchtst_random":
                    kwargs["checkpoints"] = None
                result = run(**kwargs)
                if isinstance(result, tuple) and len(result) >= 2:
                    return (result[0], result[1])
                return (result, None) if result is not None else None
            elif model == "timedart":
                result = run(model="timedart", skip_train=True, forecast_dataset=dataset,
                             pred_lens=[pred_len], encoder_layers=encoder_layers, gpu=gpu,
                             pretrain_source=pretrain_source,
                             linear_probe=linear_probe,
                             head_type=head_type)
                if isinstance(result, tuple) and len(result) >= 3:
                    return (result[1], result[2])
                return (result[1], None) if isinstance(result, tuple) else None
        except Exception as e:
            print(f"[ERROR] {model}/{dataset}/pred{pred_len}/best: {e}")
            import traceback; traceback.print_exc()
            return None


def run_model_worker(model: str, encoder_layers: int, gpu: int,
                     datasets: list, out_csv: Path, best_only: bool = False,
                     pretrain_source: str = None,
                     linear_probe: bool = True,
                     head_type: str = "linear",
                     pred_lens: list = None,
                     output_dir: str = None,
                     log_tag: str = ""):
    """Run full forecasting for one model at one layer config. Called in subprocess."""
    if pred_lens is None:
        pred_lens = PRED_LENS
    src_tag = f"_{pretrain_source.replace('+', '_')}" if pretrain_source and pretrain_source != "monash" else ""
    _log_tag = f"_{log_tag}" if log_tag else ""
    log_base = ROOT / "logs" / f"layer_forecast{src_tag}{_log_tag}"
    fieldnames = ["model", "encoder_layers", "dataset", "pred_len", "mse", "mae", "timestamp"]

    existing = set()
    rows = []
    if out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                if "mae" not in row:
                    row["mae"] = "N/A"
                rows.append(row)
                existing.add((row["model"], int(row["encoder_layers"]),
                               row["dataset"], int(row["pred_len"])))

    all_checkpoints = discover_checkpoints(model, encoder_layers, pretrain_source,
                                           output_dir=output_dir)
    print(f"\n{'='*60}")
    print(f"  {model.upper()}  layers={encoder_layers}  checkpoints={all_checkpoints}")
    print(f"{'='*60}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for dataset in datasets:
        already_done = [(model, encoder_layers, dataset, pl) in existing for pl in pred_lens]
        if all(already_done):
            print(f"\n  {model}/layers{encoder_layers}/{dataset} — already complete, skipping.")
            continue

        print(f"\n--- {model} / layers={encoder_layers} / {dataset} ---")

        try:
            if best_only:
                log_dir = log_base / f"layers{encoder_layers}" / model / dataset
                mses = {}
                for pl in pred_lens:
                    val = eval_best(model, dataset, pl, gpu, log_dir, encoder_layers,
                                    pretrain_source, linear_probe=linear_probe,
                                    head_type=head_type,
                                    output_dir=output_dir)
                    if val is not None:
                        mses[pl] = val
            else:
                mses = checkpoint_search(model, dataset, all_checkpoints,
                                         gpu, log_base, encoder_layers, pretrain_source,
                                         linear_probe=linear_probe,
                                         head_type=head_type,
                                         output_dir=output_dir)
        except Exception as e:
            print(f"  [ERROR] {model}/layers{encoder_layers}/{dataset}: {e}")
            import traceback; traceback.print_exc()
            continue

        ts = datetime.now().isoformat(timespec="seconds")
        for pred_len, val in mses.items():
            key = (model, encoder_layers, dataset, pred_len)
            if key not in existing:
                if isinstance(val, tuple):
                    mse_val, mae_val = val
                else:
                    mse_val, mae_val = val, None
                rows.append({
                    "model":          model,
                    "encoder_layers": encoder_layers,
                    "dataset":        dataset,
                    "pred_len":       pred_len,
                    "mse":            f"{mse_val:.6f}" if mse_val is not None else "N/A",
                    "mae":            f"{mae_val:.6f}" if mae_val is not None else "N/A",
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
                  best_only: bool = False, pretrain_source: str = None,
                  linear_probe: bool = True,
                  head_type: str = "linear",
                  pred_lens: list = None, out_csv_override: Path = None,
                  output_dir: str = None, log_tag: str = ""):
    src_tag = f"_{pretrain_source.replace('+', '_')}" if pretrain_source and pretrain_source != "monash" else ""
    _log_tag = f"_{log_tag}" if log_tag else ""
    out_csv = out_csv_override if out_csv_override is not None else \
              ROOT / "results" / f"layer_forecast_{model}{src_tag}{_log_tag}_layers{encoder_layers}.csv"
    log_path = log_dir / f"{model}{src_tag}{_log_tag}_layers{encoder_layers}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, __file__,
        "--_worker",
        "--model",          model,
        "--encoder_layers", str(encoder_layers),
        "--gpu",            str(gpu),
        "--datasets",       *datasets,
        "--out_csv",        str(out_csv),
        "--linear_probe",   str(linear_probe).lower(),
        "--head",           head_type,
    ]
    if best_only:
        cmd.append("--best_only")
    if pretrain_source is not None:
        cmd += ["--pretrain_source", pretrain_source]
    if pred_lens is not None:
        cmd += ["--pred_lens", *[str(p) for p in pred_lens]]
    if output_dir is not None:
        cmd += ["--output_dir", output_dir]
    if log_tag:
        cmd += ["--log_tag", log_tag]

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
                       best_only: bool = False, pretrain_source: str = None,
                       linear_probe: bool = True,
                       head_type: str = "linear",
                       pred_lens: list = None, out_csv_override: Path = None,
                       output_dir: str = None, log_tag: str = ""):
    _log_tag = f"_{log_tag}" if log_tag else ""
    log_dir = ROOT / "logs" / f"layer_forecast{_log_tag}"

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
            proc = launch_worker(model, n_layers, gpu, datasets, log_dir, dry_run,
                                 best_only=best_only, pretrain_source=pretrain_source,
                                 linear_probe=linear_probe,
                                 head_type=head_type,
                                 pred_lens=pred_lens, out_csv_override=out_csv_override,
                                 output_dir=output_dir, log_tag=log_tag)
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
    parser.add_argument("--pretrain_source", type=str, default=None,
                        choices=["monash", "synthetic", "monash+synthetic"],
                        help="Pretrain data source for checkpoint lookup (default: monash)")
    parser.add_argument("--linear_probe", type=_str2bool, default=True,
                        help="True (head only; default) or False (fine-tune backbone + head)")
    parser.add_argument("--head", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Downstream head type: 'linear' (single Linear) or 'mlp' (1-hidden-layer MLP)")
    parser.add_argument("--pred_lens", nargs="+", type=int, default=None,
                        help=f"Forecast horizons (default: {PRED_LENS}). "
                             "Only takes effect with --best_only; tournament search uses fixed [96,192,336,720].")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Output CSV path (default: results/layer_forecast_{model}{src_tag}{log_tag}_layers{N}.csv per model)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override DINO config's output_dir (base path before _layers suffix). "
                             "E.g. './checkpoints_phys', './checkpoints_physical_2', './checkpoints_physical_3'.")
    parser.add_argument("--log_tag", type=str, default="",
                        help="Suffix for log dir / csv / per-worker log filename (e.g. 'physical_2')")

    # internal worker mode
    parser.add_argument("--_worker",        action="store_true",  help=argparse.SUPPRESS)
    parser.add_argument("--model",          type=str,             help=argparse.SUPPRESS)
    parser.add_argument("--encoder_layers", type=int,             help=argparse.SUPPRESS)
    parser.add_argument("--gpu",            type=int,             help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args._worker:
        run_model_worker(
            model=args.model,
            encoder_layers=args.encoder_layers,
            gpu=args.gpu,
            datasets=args.datasets,
            out_csv=Path(args.out_csv),
            best_only=args.best_only,
            pretrain_source=args.pretrain_source,
            linear_probe=args.linear_probe,
            head_type=args.head,
            pred_lens=args.pred_lens,
            output_dir=args.output_dir,
            log_tag=args.log_tag,
        )
        return

    print(f"Layer forecast sweep")
    print(f"  Models:       {args.models}")
    print(f"  Layers:       {args.layers}")
    print(f"  Datasets:     {args.datasets}")
    print(f"  linear_probe: {args.linear_probe}")
    print(f"  head_type:    {args.head}")
    if args.pretrain_source:
        print(f"  Pretrain src: {args.pretrain_source}")
    if args.pred_lens:
        print(f"  pred_lens:    {args.pred_lens}")
    if args.dry_run:
        print("  DRY RUN\n")

    run_forecast_sweep(args.models, args.layers, args.datasets, args.dry_run,
                       gpu_override=args.gpu_override, best_only=args.best_only,
                       pretrain_source=args.pretrain_source,
                       linear_probe=args.linear_probe,
                       head_type=args.head,
                       pred_lens=args.pred_lens,
                       out_csv_override=Path(args.out_csv) if args.out_csv else None,
                       output_dir=args.output_dir, log_tag=args.log_tag)


if __name__ == "__main__":
    main()
