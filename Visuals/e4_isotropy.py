#!/usr/bin/env python3
"""
E4 — Isotropy across all seven objectives (no training).

Computes the effective rank and participation ratio of the frozen per-patch
embeddings (in R^D, the space the forecasting head consumes) for every objective,
on ETTh1 and Weather windows, then scatters effective rank against forecasting Δ
(mean over the 7 forecast datasets of model MSE − random-encoder MSE; negative Δ =
beats the random encoder).

No training: forward passes through the frozen Monash-pretrained encoders only.

Reuses the tested encoder builders / checkpoint resolution / per-model forwards in
tsne_embeddings.py via its RETURN_PATCH_EMB mode, pointed at the forecast backbone
(cw336, 21 patches) so the encoder matches the forecasting-Δ axis.

Usage (single seed 123, Monash):
    python3 e4_isotropy.py --gpu 0 --seed 123 --pretrain_source monash \
        --datasets etth1 weather --models dino jepa lejepa ntp patchtst timedart softclt random
"""
import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import torch

_VIS = Path(__file__).parent.resolve()
_ROOT = _VIS.parent
for _p in (str(_VIS), str(_ROOT), str(_ROOT / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tsne_embeddings as T   # reuse extractors / ckpt paths / constants

# Forecast-backbone context: cw336 = 21 patches × 16. This is the encoder that
# produces the forecasts, so effective rank is measured on the same representation.
FORECAST_NUM_PATCHES = 21
PATCH_SIZE = T.PATCH_SIZE   # 16

# ── forecasting-window loader (ETTh1 / Weather) ────────────────────────────────

class _ForecastWindowDataset(torch.utils.data.Dataset):
    """ETTh1/Weather CSV → fixed-length windows as (patches [P,PL,C], label, mask [P]).

    StandardScaler fit on the train split (first 60%); windows taken over the whole
    series with 50%-overlap stride. Labels are dummy (0) and the padding mask is all
    real (True) — the extractors only need the (patches, labels, mask) interface.
    """
    def __init__(self, dataset_name: str, num_patches: int, patch_size: int, stride_frac=0.5):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from dataset_registry import get_dataset_info

        info = get_dataset_info(dataset_name)
        df = pd.read_csv(info["csv_path"])
        df = df.drop(columns=[info.get("timestamp_col", "date")], errors="ignore")
        data = df.values.astype(np.float32)          # [T, C]
        T_total = len(data)
        train_len = int(T_total * 0.6)
        scaler = StandardScaler().fit(data[:train_len])
        data = scaler.transform(data).astype(np.float32)

        self.cw = num_patches * patch_size
        self.P = num_patches
        self.PL = patch_size
        stride = max(1, int(self.cw * stride_frac))
        self.windows = [data[i:i + self.cw]
                        for i in range(0, T_total - self.cw + 1, stride)]
        if not self.windows:
            self.windows = [np.pad(data, ((0, self.cw - T_total), (0, 0)), mode="edge")]
        self.C = data.shape[1]
        print(f"    [{dataset_name}] {len(self.windows)} windows of cw={self.cw}, C={self.C}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]                                   # [cw, C]
        patches = torch.from_numpy(w).reshape(self.P, self.PL, self.C).float()
        mask = torch.ones(self.P, dtype=torch.bool)
        return patches, torch.tensor(0), mask


def _build_loader(dataset_name, num_patches, patch_size, batch_size):
    ds = _ForecastWindowDataset(dataset_name, num_patches, patch_size)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                       num_workers=0, drop_last=False)


# ── isotropy metrics ───────────────────────────────────────────────────────────

def rank_metrics(X: np.ndarray, max_samples: int = 40000):
    """Effective rank and participation ratio of per-patch embeddings X [N, D].

    Both use the eigenvalues of the (centered) embedding covariance, obtained via SVD.
      effective_rank   = exp(entropy of normalized eigenvalues)      (Roy & Vetterli)
      participation_ratio = (Σλ)² / Σλ²                               (in [1, D])
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] > max_samples:
        idx = np.random.RandomState(0).choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
    X = X - X.mean(axis=0, keepdims=True)
    # singular values of centered X → eigenvalues of covariance ∝ s²
    s = np.linalg.svd(X, compute_uv=False)
    ev = s ** 2
    tot = ev.sum()
    if tot <= 0:
        return float("nan"), float("nan"), X.shape[1]
    p = ev / tot
    p_nz = p[p > 0]
    eff_rank = float(np.exp(-(p_nz * np.log(p_nz)).sum()))
    participation = float((ev.sum() ** 2) / (ev ** 2).sum())
    return eff_rank, participation, X.shape[1]


# ── forecasting Δ (mean over 7 datasets of model − random MSE) ──────────────────

_FORECAST_DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic"]

_MSE_REGEX = {
    "dino":     r'Mean MSE:\s*([0-9.]+)',
    "softclt":  r'Mean MSE:\s*([0-9.]+)',
    "ntp":      r'(?m)^\s*MSE\s*:\s*([0-9.]+)',
    "jepa":     r'MSE\s*—\s*P2P:\s*([0-9.]+)',
    "lejepa":   r'(?m)^\s*MSE:\s*([0-9.]+)',
    "patchtst": r'MSE=([0-9.]+)',
    "random":   r'MSE=([0-9.]+)',
    "timedart": r'test MSE=\s*([0-9.]+)',
}

# Candidate dirs to search for <model>_seed<seed>_<ds>.log (first match wins).
# Regular-Monash linear-probe forecast logs: dino/ntp/patchtst/timedart under
# monash_forecast_seed; jepa/lejepa/softclt under their *_seed_sweep/forecast; the
# random-encoder baseline under forecast_random_lp.
_FORECAST_DIRS = [
    "logs/testing_data/monash_forecast_seed",
    "logs/testing_data/jepa_seed_sweep/forecast",
    "logs/testing_data/lejepa_seed_sweep/forecast",
    "logs/testing_data/softclt_seed_sweep/forecast",
    "logs/testing_data/forecast_random_lp",
]


def _avg_last4(vals):
    return sum(vals[-4:]) / 4 if len(vals) >= 4 else None


def _parse_forecast_mse(model, seed, ds):
    rx = _MSE_REGEX[model]
    for d in _FORECAST_DIRS:
        for f in glob.glob(str(_ROOT / d / f"{model}_seed{seed}_{ds}.log")):
            vals = [float(x) for x in re.findall(rx, open(f).read())]
            v = _avg_last4(vals)
            if v is not None:
                return v
    return None


def forecast_delta(model, seed):
    """Δ = mean over the 7 datasets of (model MSE − random MSE). None if insufficient."""
    diffs = []
    for ds in _FORECAST_DATASETS:
        m = _parse_forecast_mse(model, seed, ds)
        r = _parse_forecast_mse("random", seed, ds)
        if m is not None and r is not None:
            diffs.append(m - r)
    return (float(np.mean(diffs)), len(diffs)) if diffs else (float("nan"), 0)


# ── extractor dispatch (mirrors tsne_embeddings.main) ──────────────────────────

def _run_extractor(model_name, ckpt, loader, encoder_layers, device):
    ex = T._EXTRACTORS[model_name]
    if model_name == "dino":
        return ex(ckpt, loader, device)
    if model_name == "timedart":
        return ex(ckpt, loader, encoder_layers, device)
    if model_name == "jepa":
        return ex(ckpt, loader, encoder_layers, device, model_name="jepa")
    return ex(ckpt, loader, encoder_layers, device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=T.ALL_MODELS, choices=T.ALL_MODELS)
    ap.add_argument("--datasets", nargs="+", default=["etth1", "weather"])
    ap.add_argument("--encoder_layers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--pretrain_source", type=str, default="monash")
    ap.add_argument("--num_patches", type=int, default=FORECAST_NUM_PATCHES,
                    help="Forecast-backbone context in patches (cw = num_patches*16). Default 21 (cw336).")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output_dir", type=str, default="plots")
    ap.add_argument("--suffix", type=str, default="")
    args = ap.parse_args()

    # Point the shared extractors at the forecast backbone and per-patch mode.
    T.NUM_PATCHES = args.num_patches
    T.CW = args.num_patches * PATCH_SIZE
    T.RETURN_PATCH_EMB = True

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  backbone cw={T.CW} ({T.NUM_PATCHES} patches)")
    print(f"Models: {args.models}  Datasets: {args.datasets}  seed={args.seed}\n")

    # Build loaders once (shared across models).
    loaders = {}
    for ds in args.datasets:
        loaders[ds] = _build_loader(ds, args.num_patches, PATCH_SIZE, args.batch_size)

    rows = []  # (model, eff_rank_mean, pr_mean, D, delta, n_delta)
    for model in args.models:
        print(f"\n{'='*60}\n  MODEL: {model}\n{'='*60}")
        ckpt = T._ckpt_path(model, args.encoder_layers, args.seed, args.pretrain_source)
        print(f"  Checkpoint: {ckpt}  exists={Path(str(ckpt)).exists()}")
        eff_ranks, prs, Ds = [], [], []
        for ds in args.datasets:
            try:
                embs, _ = _run_extractor(model, ckpt, loaders[ds], args.encoder_layers, device)
                er, pr, D = rank_metrics(embs)
                print(f"    {ds}: embs={embs.shape}  eff_rank={er:.2f}  PR={pr:.2f}  D={D}")
                eff_ranks.append(er); prs.append(pr); Ds.append(D)
            except Exception as e:
                import traceback; print(f"    ERROR on {ds}: {e}"); traceback.print_exc()
        if not eff_ranks:
            continue
        er_m = float(np.mean(eff_ranks)); pr_m = float(np.mean(prs)); D = Ds[0]
        delta, n_delta = forecast_delta(model, args.seed)
        print(f"  → eff_rank={er_m:.2f}  PR={pr_m:.2f}  D={D}  Δ={delta:.4f} (n={n_delta} datasets)")
        rows.append((model, er_m, pr_m, D, delta, n_delta))

    # ── write CSV ──
    csv_path = out_dir / f"e4_isotropy{args.suffix}.csv"
    with open(csv_path, "w") as fh:
        fh.write("model,eff_rank,participation_ratio,D,forecast_delta,n_delta_datasets\n")
        for m, er, pr, D, dl, nd in rows:
            fh.write(f"{m},{er:.4f},{pr:.4f},{D},{dl:.6f},{nd}\n")
    print(f"\nSaved → {csv_path}")

    # ── scatter: effective rank (x) vs forecasting Δ (y) ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for m, er, pr, D, dl, nd in rows:
            if np.isnan(dl):
                continue
            disp = T.MODEL_DISPLAY.get(m, m)
            ax.scatter(er, dl, s=90, zorder=3)
            ax.annotate(disp, (er, dl), xytext=(5, 4), textcoords="offset points", fontsize=10)
        ax.axhline(0.0, color="grey", lw=1, ls="--", zorder=1)  # Δ=0: ties the random encoder
        ax.set_xlabel("Effective rank of frozen per-patch embeddings (R$^D$)")
        ax.set_ylabel(r"Forecasting $\Delta$ MSE vs random encoder (mean over 7 datasets)")
        ax.set_title("E4 — Isotropy vs forecasting performance")
        fig.tight_layout()
        fig_path = out_dir / f"e4_isotropy{args.suffix}.png"
        fig.savefig(fig_path, dpi=200)
        print(f"Saved → {fig_path}")
    except Exception as e:
        print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    main()
