#!/usr/bin/env python3
"""
run_forecasting.py — Checkpoint search + full forecasting evaluation.

Checkpoint search pipeline (per model per dataset):
  1. Evaluate ALL saved checkpoints at pred_len=96  → keep top-3 by MSE
  2. Evaluate top-3 at pred_len=192                 → keep top-2
  3. Evaluate top-2 at pred_len=336                 → keep top-1
  4. Evaluate top-1 at pred_len=720

All screen output is also written to:
  logs/forecasting/{model}/{dataset}/pred{pred_len}_ckpt{N}.log

Results summary saved to:
  results/forecasting_results.csv

Usage:
    python run_forecasting.py                        # all models, all datasets, GPU 0
    python run_forecasting.py --gpu 1
    python run_forecasting.py --models jepa dino
    python run_forecasting.py --datasets etth1 etth2

Models run in order: jepa, patchtst, ntp, dino
Datasets in order:   etth1, etth2, ettm1, ettm2, weather, electricity, traffic
"""

import argparse
import csv
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

MODELS   = ["jepa", "lejepa", "patchtst", "ntp", "dino", "patchtst_random"]
DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic"]
PRED_LENS = [96, 192, 336, 720]


# ── logging: tee stdout to file ───────────────────────────────────────────────

class _Tee:
    """Write to both the original stdout and a file simultaneously."""
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
    """Context manager: everything printed inside goes to screen AND log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
        orig = sys.stdout
        sys.stdout = _Tee(orig, fh)
        try:
            yield
        finally:
            sys.stdout = orig


# ── checkpoint discovery ──────────────────────────────────────────────────────

def discover_checkpoints(model: str) -> list:
    """Return sorted list of available checkpoint epoch numbers for each model."""
    if model == "dino":
        ckpt_dir = ROOT / "checkpoints"
        found = sorted(
            int(p.stem.replace("checkpoint", ""))
            for p in ckpt_dir.glob("checkpoint*.pth")
            if p.stem.replace("checkpoint", "").isdigit()
        )
        return found or []

    elif model == "jepa":
        import re as _re
        ckpt_dir = ROOT / "output_model" / "JEPA"
        found = sorted(
            int(m.group(1))
            for p in ckpt_dir.glob("_epoch*best_model.pt")
            for m in [_re.search(r'_epoch(\d+)best_model', p.stem)]
            if m
        )
        return found or []

    elif model == "ntp":
        import re as _re
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("config_ntp", ROOT / "NTP" / "config_ntp.py")
        _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        _cfg = _mod.config
        _prefix = (f"ntp_pretrained"
                   f"_patch{_cfg['patch_size']}"
                   f"_patches{_cfg['ratio_patches']}"
                   f"_epochs{_cfg['num_epochs']}"
                   f"_model{_cfg.get('pretrained_model_id', 1)}_epoch")
        save_dir = ROOT / "NTP" / "saved_models"
        found = sorted(set(
            int(m.group(1))
            for p in save_dir.rglob("*.pt")
            if p.stem.startswith(_prefix)
            for m in [_re.search(r'_epoch(\d+)$', p.stem)]
            if m
        ))
        return found or [None]

    elif model == "patchtst":
        # Per-epoch checkpoints — filter to the config's cw/patch/stride variant
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("config_patchtst",
                    ROOT / "PatchTST_self_supervised" / "config_patchtst.py")
        _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        _cfg = _mod.config
        _prefix = (f"patchtst_pretrained"
                   f"_cw{_cfg['context_points']}"
                   f"_patch{_cfg['patch_len']}"
                   f"_stride{_cfg['stride']}"
                   f"_epochs-pretrain{_cfg['n_epochs_pretrain']}"
                   f"_mask{_cfg['mask_ratio']}"
                   f"_model{_cfg['pretrained_model_id']}_")
        save_dir = ROOT / "PatchTST_self_supervised" / "saved_models"
        found = set()
        for p in save_dir.rglob("*.pth"):
            if p.stem.startswith(_prefix):
                suffix = p.stem[len(_prefix):]
                if suffix.isdigit():
                    found.add(int(suffix))
        return sorted(found) or [None]

    elif model == "lejepa":
        import re as _re
        ckpt_dir = ROOT / "output_model" / "LE-JEPA"
        found = sorted(
            int(m.group(1))
            for p in ckpt_dir.glob("_epoch*best_model.pt")
            for m in [_re.search(r'_epoch(\d+)best_model', p.stem)]
            if m
        )
        return found or []

    elif model == "patchtst_random":
        return [None]

    else:
        return [None]


# ── single checkpoint evaluation ─────────────────────────────────────────────

def eval_checkpoint(model: str, dataset: str, pred_len: int, ckpt,
                    gpu: int, log_dir: Path) -> float:
    """
    Evaluate one checkpoint at one pred_len. Returns MSE (or None on failure).
    All output is teed to logs/forecasting/{model}/{dataset}/pred{pred_len}_ckpt{ckpt}.log
    """
    ckpt_tag = str(ckpt) if ckpt is not None else "best"
    log_path = log_dir / f"pred{pred_len}_ckpt{ckpt_tag}.log"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    with log_to_file(log_path):
        try:
            from Train_and_downstream import run

            if model in ("jepa", "lejepa", "dino"):
                result = run(
                    model=model,
                    skip_train=True,
                    forecast_dataset=dataset,
                    pred_lens=[pred_len],
                    checkpoints=["best"] if ckpt is None else [ckpt],
                )
                # result is (best_ckpt, best_mse)
                return result[1] if result else None

            elif model == "ntp":
                result = run(
                    model="ntp",
                    skip_train=True,
                    forecast_dataset=dataset,
                    pred_len=pred_len,
                    checkpoints=[ckpt] if ckpt is not None else None,
                )
                return result[0] if isinstance(result, tuple) else result

            elif model == "patchtst":
                return run(
                    model="patchtst",
                    skip_train=True,
                    forecast_dataset=dataset,
                    pred_len=pred_len,
                    checkpoints=[ckpt] if ckpt is not None else None,
                )

            elif model == "patchtst_random":
                return run(
                    model="patchtst_random",
                    skip_train=True,
                    forecast_dataset=dataset,
                    pred_len=pred_len,
                )

        except Exception as e:
            print(f"[ERROR] {model}/{dataset}/pred{pred_len}/ckpt{ckpt_tag}: {e}")
            import traceback; traceback.print_exc()
            return None


# ── checkpoint search ─────────────────────────────────────────────────────────

def checkpoint_search(model: str, dataset: str, all_checkpoints: list,
                      gpu: int, log_base: Path) -> dict:
    """
    Run the checkpoint search. Returns {pred_len: best_mse}.
    """
    log_dir = log_base / model / dataset
    results = {}

    if not all_checkpoints:
        print(f"  [WARN] No checkpoints found for {model}")
        return results

    # ── Round 1: all checkpoints at pred_len=96 ───────────────────────────────
    print(f"\n  [Search] pred_len=96 — evaluating {len(all_checkpoints)} checkpoints: {all_checkpoints}")
    mses_96 = {}
    for ckpt in all_checkpoints:
        mse = eval_checkpoint(model, dataset, 96, ckpt, gpu, log_dir)
        if mse is not None:
            mses_96[ckpt] = mse
            print(f"    checkpoint {ckpt}: MSE={mse:.4f}")

    if not mses_96:
        return results

    top3 = sorted(mses_96, key=mses_96.get)[:3]
    results[96] = mses_96[top3[0]]
    print(f"  → Top-3: {top3}  (best MSE={results[96]:.4f})")

    # ── Round 2: top-3 at pred_len=192 ───────────────────────────────────────
    print(f"\n  [Search] pred_len=192 — evaluating top-3 checkpoints: {top3}")
    mses_192 = {}
    for ckpt in top3:
        mse = eval_checkpoint(model, dataset, 192, ckpt, gpu, log_dir)
        if mse is not None:
            mses_192[ckpt] = mse
            print(f"    checkpoint {ckpt}: MSE={mse:.4f}")

    if not mses_192:
        return results

    top2 = sorted(mses_192, key=mses_192.get)[:2]
    results[192] = mses_192[top2[0]]
    print(f"  → Top-2: {top2}  (best MSE={results[192]:.4f})")

    # ── Round 3: top-2 at pred_len=336 ───────────────────────────────────────
    print(f"\n  [Search] pred_len=336 — evaluating top-2 checkpoints: {top2}")
    mses_336 = {}
    for ckpt in top2:
        mse = eval_checkpoint(model, dataset, 336, ckpt, gpu, log_dir)
        if mse is not None:
            mses_336[ckpt] = mse
            print(f"    checkpoint {ckpt}: MSE={mse:.4f}")

    if not mses_336:
        return results

    best = min(mses_336, key=mses_336.get)
    results[336] = mses_336[best]
    print(f"  → Best: checkpoint {best}  (MSE={results[336]:.4f})")

    # ── Round 4: best checkpoint at pred_len=720 ─────────────────────────────
    print(f"\n  [Search] pred_len=720 — best checkpoint: {best}")
    mse = eval_checkpoint(model, dataset, 720, best, gpu, log_dir)
    if mse is not None:
        results[720] = mse
        print(f"    MSE={mse:.4f}")

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Checkpoint search + forecasting evaluation")
    parser.add_argument("--models",    nargs="+", default=MODELS,   choices=MODELS,   metavar="MODEL")
    parser.add_argument("--datasets",  nargs="+", default=DATASETS, choices=DATASETS, metavar="DATASET")
    parser.add_argument("--gpu",       type=int,  default=0)
    parser.add_argument("--best_only", action="store_true",
                        help="Skip tournament — evaluate only the best checkpoint for all pred_lens")
    parser.add_argument("--out_csv",   type=str,  default=None,
                        help="Output CSV path (default: results/forecasting_results.csv)")
    args = parser.parse_args()

    results_dir = ROOT / "results"
    log_base    = ROOT / "logs" / "forecasting"
    results_dir.mkdir(exist_ok=True)
    log_base.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.out_csv) if args.out_csv else results_dir / "forecasting_results.csv"
    fieldnames = ["model", "dataset", "pred_len", "mse", "timestamp"]

    # Load existing results to allow resuming interrupted runs
    existing = set()
    rows = []
    if out_csv.exists():
        with open(out_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                existing.add((row["model"], row["dataset"], int(row["pred_len"])))

    print(f"\nForecasting checkpoint search")
    print(f"Models:   {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"GPU:      {args.gpu}")
    print(f"Logs  →   logs/forecasting/{{model}}/{{dataset}}/")
    print(f"Results → {out_csv.resolve()}\n")

    for model in args.models:
        all_checkpoints = discover_checkpoints(model)
        print(f"\n{'='*60}")
        print(f"  MODEL: {model.upper()}   checkpoints={all_checkpoints}")
        print(f"{'='*60}")

        for dataset in args.datasets:
            already_done = [pl for pl in PRED_LENS if (model, dataset, pl) in existing]
            if len(already_done) == len(PRED_LENS):
                print(f"\n  {model}/{dataset} — already complete, skipping.")
                continue

            print(f"\n--- {model} / {dataset} ---")

            try:
                if model == "patchtst_random" or args.best_only:
                    log_dir = log_base / model / dataset
                    mses = {}
                    for pl in PRED_LENS:
                        if (model, dataset, pl) in existing:
                            continue
                        mse = eval_checkpoint(model, dataset, pl, None, args.gpu, log_dir)
                        if mse is not None:
                            mses[pl] = mse
                            print(f"    pred_len={pl}: MSE={mse:.4f}")
                else:
                    mses = checkpoint_search(model, dataset, all_checkpoints, args.gpu, log_base)
            except Exception as e:
                print(f"  [ERROR] {model}/{dataset}: {e}")
                import traceback; traceback.print_exc()
                continue

            ts = datetime.now().isoformat(timespec="seconds")
            for pred_len, mse in mses.items():
                if (model, dataset, pred_len) not in existing:
                    rows.append({
                        "model":     model,
                        "dataset":   dataset,
                        "pred_len":  pred_len,
                        "mse":       f"{mse:.6f}" if mse is not None else "N/A",
                        "timestamp": ts,
                    })
                    existing.add((model, dataset, pred_len))

            # Write after each dataset so partial results are not lost
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Saved → {out_csv}")

    print(f"\n\nDone. Results in {out_csv}")


if __name__ == "__main__":
    main()
