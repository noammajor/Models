#!/usr/bin/env python3
"""
lejepa_gaussianity.py — Gaussianity diagnostic for the LE-JEPA embedding space.

Loads a frozen LE-JEPA classification backbone, encodes a UEA dataset
(default: SpokenArabicDigits), pools patch-level embeddings into [N, D]
(N = samples · n_vars · num_patches, D = embed_dim), and produces a
3-panel paper figure probing whether the embedding distribution behaves
like an isotropic standard normal — the prior SIGReg pushes for.

Panels:
  1. Random 1-D projection histograms (4 unit-norm directions) vs N(0,1)
  2. Singular-value spectrum of the centred embedding covariance (log scale)
  3. 2D PCA scatter coloured by class label

Default paths assume the user's Mac layout. Override via CLI flags.

Usage:
    python Visuals/lejepa_gaussianity.py
    python Visuals/lejepa_gaussianity.py --dataset SpokenArabicDigits \\
        --ckpt "/Users/noammajor/Downloads/Models/LE-JEPA Backbones/classification/LE-JEPA_layers8_cw1152/best_model.pt" \\
        --cls_dir "/Users/noammajor/Desktop/data/Classification data" \\
        --output_dir Visuals/plots
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT        = Path(__file__).parent.parent.resolve()
PATCH_SIZE  = 16
NUM_PATCHES = 72                  # cw1152 / patch_size = 72
CW          = NUM_PATCHES * PATCH_SIZE

DEFAULT_CKPT    = ROOT / "output_model" / "classification" / "LE-JEPA_layers8_cw1152" / "best_model.pt"
DEFAULT_CLS_DIR = "/home/shared/datasets/Classification_TS"


def _add_path(*dirs):
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)


def _load_config(path):
    spec = importlib.util.spec_from_file_location("_cfg", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.config)


# ── data loading ──────────────────────────────────────────────────────────────

def _build_loader(cls_dir: str, dataset_name: str, batch_size: int = 32):
    _add_path(ROOT / "shared")
    from data_loaders.data_puller import make_uea_dataloaders
    import torch.utils.data

    target_T = NUM_PATCHES * PATCH_SIZE

    def _patch_collate(batch):
        xs, ys, orig_lens = zip(*batch)
        orig_lens = torch.stack(orig_lens)
        max_t = max(x.shape[0] for x in xs)
        xs = torch.stack([F.pad(x, (0, 0, 0, max_t - x.shape[0])) for x in xs])
        B_, T_, C_ = xs.shape
        if T_ != target_T:
            idx = torch.linspace(0, T_ - 1, target_T).long()
            xs  = xs[:, idx, :]
            patch_idx    = idx[torch.arange(NUM_PATCHES) * PATCH_SIZE]
            padding_mask = patch_idx.unsqueeze(0) < orig_lens.unsqueeze(1)
        else:
            patch_starts = torch.arange(NUM_PATCHES) * PATCH_SIZE
            padding_mask = patch_starts.unsqueeze(0) < orig_lens.unsqueeze(1)
        xs = xs.reshape(B_, NUM_PATCHES, PATCH_SIZE, C_)
        return xs, torch.stack(ys), padding_mask

    tr_raw, _, te_raw, n_classes = make_uea_dataloaders(cls_dir, dataset_name, batch_size)
    combined = torch.utils.data.ConcatDataset([tr_raw.dataset, te_raw.dataset])
    loader   = torch.utils.data.DataLoader(combined, batch_size=batch_size,
                                           shuffle=False, collate_fn=_patch_collate)
    class_names = [str(c) for c in getattr(tr_raw.dataset, "class_names", []) or []] or None
    return loader, n_classes, class_names


# ── encoder loading & embedding extraction ────────────────────────────────────

def _instance_norm(x, eps=1e-6):
    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2),  keepdim=True) + eps
    return (x - mean) / std, mean, std


@torch.no_grad()
def _extract_lejepa_patch_level(ckpt: Path, loader, encoder_layers: int, device):
    """
    Returns:
      patch_embs        : [N_patches, D]   N_patches = sum_b (B_b * C_b * P)
      sample_labels     : [N_samples]      one label per *sample* (not per patch)
      patch_to_sample   : [N_patches]      maps each patch row back to its sample idx
                                           (same digit label expanded across C*P)
    """
    lejepa_dir = ROOT / "LE-JEPA"
    _add_path(str(lejepa_dir), str(ROOT / "shared"))

    spec = importlib.util.spec_from_file_location("lejepa_encoder", lejepa_dir / "Encoder.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    LeJepaEncoder = mod.Encoder

    cfg = _load_config(lejepa_dir / "config_lejepa.py")
    cfg["num_encoder_layers"] = encoder_layers
    embed_dim = cfg["encoder_embed_dim"]

    encoder = LeJepaEncoder(
        num_patches    = NUM_PATCHES,
        dim_in         = PATCH_SIZE,
        embed_dim      = embed_dim,
        nhead          = cfg["nhead"],
        num_layers     = encoder_layers,
        mlp_ratio      = cfg["mlp_ratio"],
        drop_rate      = cfg["drop_rate"],
        attn_drop_rate = cfg["attn_drop_rate"],
        pe             = "sincos",
        learn_pe       = False,
        res_attention  = True,
    ).to(device)

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    raw = torch.load(ckpt, map_location="cpu")
    encoder.load_state_dict(raw["encoder"], strict=False)
    encoder.eval()
    print(f"  [lejepa] loaded encoder from {ckpt.name}  (D={embed_dim}, layers={encoder_layers})")

    all_patch_embs   = []
    sample_labels    = []
    patch_to_sample  = []
    sample_offset    = 0

    for patches, labels, padding_mask in loader:
        patches      = patches.float().to(device)
        padding_mask = padding_mask.to(device)
        B, P, PL, C  = patches.shape

        ctx_norm, _, _ = _instance_norm(patches)
        out  = encoder(ctx_norm, padding_mask=padding_mask)
        enc  = out["data_patches"]                       # [B*C, P, D]
        enc  = enc.reshape(B, C, P, embed_dim)           # [B, C, P, D]
        enc  = enc.reshape(B * C * P, embed_dim).cpu().numpy()

        all_patch_embs.append(enc)
        sample_labels.append(labels.numpy())
        # each sample contributes C*P patch rows
        sample_idx = np.repeat(np.arange(B) + sample_offset, C * P)
        patch_to_sample.append(sample_idx)
        sample_offset += B

    return (
        np.concatenate(all_patch_embs, axis=0),
        np.concatenate(sample_labels,  axis=0),
        np.concatenate(patch_to_sample, axis=0),
    )


# ── plotting ──────────────────────────────────────────────────────────────────

def _normal_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def make_figure(embs: np.ndarray,
                sample_labels: np.ndarray,
                patch_to_sample: np.ndarray,
                output_path: Path,
                dataset_name: str,
                rng_seed: int = 42,
                n_proj: int = 4):
    """
    embs            : [N, D] patch-level embeddings
    sample_labels   : [N_samples]
    patch_to_sample : [N] → sample idx for each patch row
    """
    rng = np.random.default_rng(rng_seed)
    N, D = embs.shape
    print(f"  Embedding cloud: N={N} patches, D={D}")

    # Standardise globally: per-coord z-score so that "Gaussianity vs N(0,1)"
    # is the right comparison (LE-JEPA's mean and scale are not enforced to 0/1
    # — only the *shape* of the distribution is).
    mu  = embs.mean(axis=0, keepdims=True)
    sig = embs.std(axis=0, keepdims=True) + 1e-8
    Z   = (embs - mu) / sig                                     # [N, D]

    # ── Random unit-norm 1-D projections ─────────────────────────────────────
    A = rng.standard_normal((D, max(n_proj, 8)))
    A = A / np.linalg.norm(A, axis=0, keepdims=True)            # unit cols
    proj = Z @ A                                                # [N, K]

    # ── Singular value spectrum ──────────────────────────────────────────────
    Zc = Z - Z.mean(axis=0, keepdims=True)
    # subsample if N is huge to keep SVD tractable
    n_svd = min(N, 20000)
    if N > n_svd:
        sub = rng.choice(N, size=n_svd, replace=False)
        Zc_sub = Zc[sub]
    else:
        Zc_sub = Zc
    s = np.linalg.svd(Zc_sub, compute_uv=False)
    s2 = (s**2) / max(Zc_sub.shape[0] - 1, 1)                   # eigvals of cov

    # ── PCA scatter (sample-level, mean over patches) ────────────────────────
    n_samples = sample_labels.shape[0]
    sample_embs = np.zeros((n_samples, D), dtype=np.float32)
    counts      = np.zeros(n_samples, dtype=np.int64)
    np.add.at(sample_embs, patch_to_sample, embs)
    np.add.at(counts,       patch_to_sample, 1)
    sample_embs = sample_embs / np.maximum(counts[:, None], 1)
    pca = PCA(n_components=2, random_state=rng_seed).fit_transform(sample_embs)

    # ── Build figure ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          11,
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    fig.suptitle(
        "LE-JEPA embedding-space",
        fontsize=14, fontweight="bold", y=1.02,
    )

    bins = np.linspace(-5, 5, 80)
    xs = np.linspace(-5, 5, 400)

    # Panel 1: random 1-D projections (4 directions, layered hists)
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for k in range(n_proj):
        ax.hist(proj[:, k], bins=bins, density=True, alpha=0.35,
                color=cmap(k), edgecolor="none",
                label=f"slice {k+1}")
    ax.plot(xs, _normal_pdf(xs), color="black", linewidth=1.6,
            label=r"$\mathcal{N}(0,1)$")
    ax.set_xlim(-5, 5)
    ax.set_xlabel(r"$\langle a_k, z\rangle$  (unit-norm $a_k$)")
    ax.set_ylabel("density")
    ax.set_title("(1) Random 1-D projections (SIGReg slices)", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, ncol=2)

    # Panel 2: singular value spectrum
    ax = axes[1]
    ax.semilogy(np.arange(1, len(s2) + 1), s2, "o-", markersize=3,
                color="#3b6ea5", label="empirical eigvals")
    ax.axhline(1.0, color="crimson", linewidth=1.2, linestyle="--",
               label=r"isotropic baseline ($\lambda_i = 1$)")
    # Effective rank
    p = s2 / s2.sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    ax.set_xlabel("index $i$")
    ax.set_ylabel(r"$\lambda_i$  (cov eigenvalue)")
    ax.set_title(f"(2) Covariance spectrum  (eff. rank = {eff_rank:.1f} / {D})",
                 fontweight="bold")
    ax.legend(fontsize=9, frameon=False)

    # Panel 3: PCA scatter coloured by class
    ax = axes[2]
    classes = np.unique(sample_labels)
    cmap6   = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")
    for ci, c in enumerate(classes):
        mask = sample_labels == c
        ax.scatter(pca[mask, 0], pca[mask, 1],
                   s=14, alpha=0.75, linewidths=0,
                   color=cmap6(int(c) % 20),
                   label=str(c))
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("(3) PCA of sample-mean embeddings", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, ncol=2, title="class")

    fig.tight_layout(pad=1.5, rect=(0, 0, 1, 0.99))
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"\nSaved figure → {output_path}")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)

    return {
        "N": int(N), "D": int(D),
        "eff_rank": eff_rank,
        "proj_mean":  float(proj.mean()),
        "proj_std":   float(proj.std()),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LE-JEPA Gaussianity diagnostic figure")
    parser.add_argument("--ckpt",        type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--cls_dir",     type=str, default=DEFAULT_CLS_DIR)
    parser.add_argument("--dataset",     type=str, default="SpokenArabicDigits")
    parser.add_argument("--encoder_layers", type=int, default=8)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--output_dir",  type=str, default=str(ROOT / "Visuals" / "plots"))
    parser.add_argument("--n_proj",      type=int, default=4)
    parser.add_argument("--device",      type=str, default=None,
                        help="cuda / mps / cpu (auto-detected if None)")
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Dataset:    {args.dataset}  ({args.cls_dir})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader, n_classes, _ = _build_loader(args.cls_dir, args.dataset, args.batch_size)
    print(f"  {len(loader.dataset)} samples, {n_classes} classes")

    embs, sample_labels, patch_to_sample = _extract_lejepa_patch_level(
        Path(args.ckpt), loader, args.encoder_layers, device
    )

    out_path = out_dir / f"lejepa_gaussianity_{args.dataset}.png"
    stats = make_figure(
        embs, sample_labels, patch_to_sample,
        output_path=out_path,
        dataset_name=args.dataset,
        n_proj=args.n_proj,
    )
    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k:>12s} : {v}")


if __name__ == "__main__":
    main()
