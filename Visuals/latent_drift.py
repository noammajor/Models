#!/usr/bin/env python
"""
Temporal-perturbation / latent-drift experiment.

For each frozen encoder we embed a set of windows (clean) and the same windows
under a temporal perturbation (circular time-shift or random time-masking), and
measure the latent drift  D = 1 - cos(z_clean, z_perturbed), averaged over windows.

Reuses the per-model extractors + checkpoint-path logic from tsne_embeddings.py,
so it covers all 7 models and both contexts:
  --context forecast   -> cw336  (num_patches=21)
  --context classify   -> cw1152 (num_patches=72)

Data are raw forecast CSVs (e.g. weather, etth1): we z-score each channel over
the whole series and cut non-overlapping windows of length = context.

Example:
  python Visuals/latent_drift.py \
      --datasets weather etth1 --context forecast classify \
      --models dino jepa lejepa patchtst ntp timedart softclt \
      --pretrain_source monash --seed 1337 \
      --data_dir /home/shared/datasets/forecasting \
      --shifts 0 1 2 4 8 16 32 --mask_p 0.0 0.1 0.2 \
      --gpu 7 --output_dir plots/latent_drift
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Visuals"))
import tsne_embeddings as T   # reuse _ckpt_path, _EXTRACTORS, globals

# Each model dir ships its own generic package names (models/, src/, utils/, …);
# they collide in sys.modules when we load several models in one process. Reset
# them before every model so each extractor's imports resolve against its own dir.
_BASE_SYS_PATH = list(sys.path)
_COLLIDING_TOP = {"models", "src", "utils", "data", "data_loaders", "layers",
                  "configs", "config", "model", "shared", "TimeDART", "exp", "dataset"}

def reset_model_imports():
    for name in list(sys.modules):
        if name.split(".")[0] in _COLLIDING_TOP:
            sys.modules.pop(name, None)
    sys.path[:] = _BASE_SYS_PATH

PATCH = 16
CONTEXT_NP = {"forecast": 21, "classify": 72}   # num_patches per context (× PATCH = 336 / 1152)


# ── data: raw forecast CSV → windows [N, L, C] ────────────────────────────────

def load_series(name: str, data_dir: str) -> np.ndarray:
    """Return z-scored multivariate series [T, C] for a forecast dataset."""
    d = Path(data_dir)
    # case-insensitive match on the stem (etth1 -> ETTh1.csv, weather -> weather.csv)
    cands = [p for p in d.rglob("*.csv") if p.stem.lower() == name.lower()]
    if not cands:
        raise FileNotFoundError(f"No CSV for '{name}' under {data_dir}")
    df = pd.read_csv(cands[0])
    # drop a leading date/time column if present
    if df.columns[0].lower() in ("date", "time", "timestamp") or df.dtypes[0] == object:
        df = df.iloc[:, 1:]
    x = df.values.astype(np.float32)                 # [T, C]
    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-8
    return (x - mu) / sd


def make_windows(series: np.ndarray, L: int, n_max: int, stride: int) -> np.ndarray:
    T_, C = series.shape
    starts = list(range(0, T_ - L + 1, stride))
    W = np.stack([series[s:s + L] for s in starts])  # [N, L, C]
    if len(W) > n_max:
        idx = np.linspace(0, len(W) - 1, n_max).astype(int)
        W = W[idx]
    return W


def patchify(W: np.ndarray, patch: int) -> np.ndarray:
    n, L, C = W.shape
    P = L // patch
    return W[:, :P * patch].reshape(n, P, patch, C)


# ── temporal perturbations (applied on windows [N, L, C]) ─────────────────────

def perturb_shift(W: np.ndarray, k: int) -> np.ndarray:
    return np.roll(W, shift=k, axis=1) if k else W


def perturb_mask(W: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    if p <= 0:
        return W
    Wm = W.copy()
    m = rng.random((W.shape[0], W.shape[1])) < p          # [N, L] mask over time
    Wm[m] = 0.0
    return Wm


# ── loader matching the extractor interface: (patches, labels, padding_mask) ──

class ArrLoader:
    def __init__(self, patches: np.ndarray, batch_size: int = 64):
        self.p = torch.tensor(patches, dtype=torch.float32)
        self.bs = batch_size

    def __iter__(self):
        P = self.p.shape[1]
        for i in range(0, len(self.p), self.bs):
            b = self.p[i:i + self.bs]
            yield (b,
                   torch.zeros(len(b), dtype=torch.long),
                   torch.ones(len(b), P, dtype=torch.bool))

    @property
    def dataset(self):
        return self.p

    def __len__(self):
        return (len(self.p) + self.bs - 1) // self.bs


def run_extractor(model_name, ckpt, loader, encoder_layers, device):
    ex = T._EXTRACTORS[model_name]
    if model_name == "dino":
        embs, _ = ex(ckpt, loader, device)
    elif model_name == "jepa":
        embs, _ = ex(ckpt, loader, encoder_layers, device, model_name="jepa")
    else:  # lejepa, ntp, patchtst, softclt, timedart
        embs, _ = ex(ckpt, loader, encoder_layers, device)
    return np.asarray(embs, dtype=np.float64)


def cosine_drift(zc: np.ndarray, zp: np.ndarray) -> float:
    num = (zc * zp).sum(1)
    den = np.linalg.norm(zc, axis=1) * np.linalg.norm(zp, axis=1) + 1e-12
    return float(np.mean(1.0 - num / den))


# Patch-token axis in each extractor's flattened embedding, so we can mean-pool over
# patches (order-free) and remove the token-slot permutation effect from shift drift.
#   last  : [.., P]         (dino/ntp/patchtst/timedart -> C*d_model*P)
#   mid   : [C, P, E]       (jepa/lejepa -> C*P*embed_dim)
#   first : [P, ..]         (softclt -> P*C*d_model)
POOL_LAYOUT = {"dino":"last","ntp":"last","patchtst":"last","timedart":"last",
               "jepa":"mid","lejepa":"mid","softclt":"first"}

def mean_pool(emb: np.ndarray, model: str, C: int, P: int) -> np.ndarray:
    """Mean over the P patch tokens -> order-free per-window vector."""
    N, tot = emb.shape
    layout = POOL_LAYOUT.get(model, "last")
    try:
        if layout == "last":
            return emb.reshape(N, tot // P, P).mean(2)
        if layout == "first":
            return emb.reshape(N, P, tot // P).mean(1)
        if layout == "mid":
            E = tot // (C * P)
            return emb.reshape(N, C, P, E).mean(2).reshape(N, -1)
    except Exception:
        pass
    return emb  # fallback: no pooling


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--context",  nargs="+", default=["forecast", "classify"],
                    choices=["forecast", "classify"])
    ap.add_argument("--models",   nargs="+", default=T.ALL_MODELS, choices=T.ALL_MODELS)
    ap.add_argument("--pretrain_source", type=str, default="monash")
    ap.add_argument("--seed",     type=int, default=1337)
    ap.add_argument("--encoder_layers", type=int, default=8)
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Directory containing the forecast CSVs (weather.csv, etth1.csv, …)")
    ap.add_argument("--n_windows", type=int, default=300)
    ap.add_argument("--stride",   type=int, default=48, help="window stride when cutting series")
    ap.add_argument("--shifts",   nargs="+", type=int, default=[0, 1, 2, 4, 8, 16, 32])
    ap.add_argument("--mask_p",   nargs="+", type=float, default=[0.0, 0.1, 0.2])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--output_dir", type=str, default="plots/latent_drift")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []  # csv rows

    for ctx in args.context:
        np_ = CONTEXT_NP[ctx]
        L   = np_ * PATCH
        # point tsne_embeddings globals at this context so _ckpt_path + extractors agree
        T.NUM_PATCHES = np_
        T.CW = L
        T.PATCH_SIZE = PATCH
        print(f"\n########## CONTEXT={ctx}  L={L}  num_patches={np_} ##########")

        for ds in args.datasets:
            series = load_series(ds, args.data_dir)
            W = make_windows(series, L, args.n_windows, args.stride)   # [N, L, C]
            clean_patches = patchify(W, PATCH)
            print(f"  {ds}: {W.shape[0]} windows × L{L} × C{W.shape[2]}")

            for m in args.models:
                reset_model_imports()
                ckpt = T._ckpt_path(m, args.encoder_layers, args.seed, args.pretrain_source)
                print(f"    [{ctx}/{ds}] {m}: {ckpt}")
                Cn = W.shape[2]; Pn = np_
                try:
                    zc = run_extractor(m, ckpt, ArrLoader(clean_patches, args.batch_size),
                                       args.encoder_layers, device)
                except Exception as e:
                    print(f"      ERROR loading {m}: {e}")
                    continue
                zc_pool = mean_pool(zc, m, Cn, Pn)
                def record(pert, sev, Wp):
                    zp = run_extractor(m, ckpt, ArrLoader(Wp, args.batch_size),
                                       args.encoder_layers, device)
                    zp_pool = mean_pool(zp, m, Cn, Pn)
                    rows.append(dict(context=ctx, dataset=ds, model=m, rep="flat",
                                     perturbation=pert, severity=sev,
                                     drift=cosine_drift(zc, zp)))
                    rows.append(dict(context=ctx, dataset=ds, model=m, rep="pooled",
                                     perturbation=pert, severity=sev,
                                     drift=cosine_drift(zc_pool, zp_pool)))
                for k in args.shifts:
                    if k == 0: continue
                    record("shift", k, patchify(perturb_shift(W, k), PATCH))
                for p in args.mask_p:
                    if p == 0: continue
                    record("mask", p, patchify(perturb_mask(W, p, rng), PATCH))

    df = pd.DataFrame(rows)
    if 'rep' not in df.columns: df['rep']='flat'
    csv_path = outdir / f"latent_drift_{args.pretrain_source}_seed{args.seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}  ({len(df)} rows)")

    # ── figures: drift vs shift, one panel per (context, dataset), curve per model ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sh = df[(df.perturbation == "shift") & (df.rep == "flat")]
        panels = sh[["context", "dataset"]].drop_duplicates().values.tolist()
        ncol = len(panels) or 1
        fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.6), squeeze=False)
        for ax, (ctx, ds) in zip(axes[0], panels):
            sub = sh[(sh.context == ctx) & (sh.dataset == ds)]
            for m in args.models:
                mm = sub[sub.model == m].sort_values("severity")
                if len(mm):
                    ax.plot(mm.severity, mm.drift, marker="o", label=m)
            ax.set_title(f"{ctx} / {ds}"); ax.set_xlabel("time shift (steps)")
            ax.set_ylabel("latent drift  (1-cos)")
        axes[0][-1].legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig_path = outdir / f"latent_drift_shift_{args.pretrain_source}_seed{args.seed}.png"
        fig.savefig(fig_path, dpi=150)
        print(f"Saved {fig_path}")
    except Exception as e:
        print(f"[figure skipped] {e}")

    # ── console summary table (mean drift at shift=8 and mask=0.2) ──
    def cell(ctx, m, pert, sev):
        s = df[(df.context == ctx) & (df.model == m) & (df.perturbation == pert)
               & (df.rep == "flat") & (np.isclose(df.severity, sev))]
        return f"{s.drift.mean():.3f}" if len(s) else "--"
    print("\n=== mean latent drift (avg over datasets) ===")
    for ctx in args.context:
        print(f"[{ctx}]  model      shift@8   mask@0.2")
        for m in args.models:
            print(f"        {m:9s}  {cell(ctx, m, 'shift', 8):>7}   {cell(ctx, m, 'mask', 0.2):>7}")


if __name__ == "__main__":
    main()
