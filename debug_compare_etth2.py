#!/usr/bin/env python3
"""
debug_compare_etth2.py
======================
Runs JEPA (simple) and PatchTST (random) on ETTh2 pred_len=720,
then writes EVERY test window's context / target / prediction to CSV.

Output files (in the Models root):
  debug_etth2_context.csv     -- context window stats per (window, var)
  debug_etth2_predictions.csv -- full target + prediction series per (window, var)

Usage:
    python debug_compare_etth2.py
    python debug_compare_etth2.py --n_windows 0   # 0 = all windows
    python debug_compare_etth2.py --pred_len 96
    python debug_compare_etth2.py --jepa_ckpt output_model/JEPA/_epoch120best_model.pt
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── path setup ────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.resolve()
DJEPA_DIR = ROOT / "Discrete_JEPA"
JEPA_DIR  = ROOT / "JEPA"
NPT_DIR   = ROOT / "NPT"
PTST_DIR  = ROOT / "PatchTST_self_supervised"

for _p in [str(ROOT), str(DJEPA_DIR), str(JEPA_DIR), str(NPT_DIR), str(PTST_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset_registry import get_dataset_info


# ── data loaders ─────────────────────────────────────────────────────────────

def make_loader(csv_path, seq_len, pred_len, patch_size, split, batch_size,
                shuffle=False, jepa_config=None):
    """
    Both loaders use the same ETT-specific splits and StandardScaler.
    Returns (dataset, dataloader).
    """
    from data_loaders.data_puller import ForcastingDataPullerDescrete

    n_patches_ctx = seq_len  // patch_size
    n_patches_tgt = pred_len // patch_size

    cfg = dict(jepa_config)
    cfg["ratio_patches"]              = n_patches_ctx
    cfg["patch_size_forcasting"]      = patch_size
    cfg["horizon_t"]                  = n_patches_tgt
    cfg["val_prec_forcasting"]        = 0.1
    cfg["test_prec_forcasting"]       = 0.1
    cfg["window_step_forecasting"]    = 1
    cfg["path_data_forcasting"]       = [csv_path]
    cfg["timestampcols_forcasting"]   = ["date"]
    cfg["input_variables_forcasting"] = [
        ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    ]

    ds = ForcastingDataPullerDescrete(cfg, which=split)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                     shuffle=shuffle, drop_last=False)
    return ds, dl


# ── PatchTST (random init, RevIN via DINO backbone) ──────────────────────────

def run_patchtst(train_dl, test_dl, pred_len, patch_size, seq_len,
                 n_vars, epochs, device):
    """Linear probe: frozen random backbone + trained head.  RevIN is internal."""
    sys.path.insert(0, str(ROOT / "TSDiNO"))
    from models.patchTST import PatchTST

    num_patch = seq_len // patch_size
    model = PatchTST(
        c_in=n_vars, target_dim=pred_len,
        patch_len=patch_size, num_patch=num_patch,
        n_layers=3, n_heads=16, d_model=128,
        shared_embedding=True, d_ff=512,
        dropout=0.1, head_dropout=0.1,
        act='gelu', head_type='prediction', res_attention=False,
    ).to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(model.head.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-4,
        total_steps=epochs * len(train_dl),
        pct_start=0.3, anneal_strategy='cos')

    print(f"  [PatchTST-random] training head for {epochs} epochs …")
    for ep in range(epochs):
        model.backbone.eval(); model.head.train()
        for ctx, tgt in train_dl:
            B, P, PL, nv = ctx.shape
            x = ctx.reshape(B, P * PL, nv).float().to(device)  # [B, seq_len, n_vars]
            y = tgt.reshape(B, -1, nv).float().to(device)       # [B, pred_len, n_vars]
            pred = model(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    epoch {ep+1}/{epochs}")

    # collect
    model.eval()
    all_ctx, all_tgt, all_pred = [], [], []
    with torch.no_grad():
        for ctx, tgt in test_dl:
            B, P, PL, nv = ctx.shape
            x    = ctx.reshape(B, P * PL, nv).float().to(device)
            y    = tgt.reshape(B, -1, nv).float()
            pred = model(x).cpu()
            all_ctx.append(ctx.float()); all_tgt.append(y); all_pred.append(pred)

    return (torch.cat(all_ctx), torch.cat(all_tgt), torch.cat(all_pred))


# ── JEPA simple ──────────────────────────────────────────────────────────────

def run_jepa(train_dl, test_dl, ckpt_path, pred_len, patch_size, seq_len,
             n_vars, epochs, device, jepa_config):
    """Frozen pretrained JEPA encoder + trained PredictionHead.  No RevIN."""
    sys.path.insert(0, str(JEPA_DIR / "JEPA"))
    from JEPA.Encoder import Encoder
    from JEPA.Decoder import PredictionHead

    cfg         = jepa_config
    embed_dim   = cfg["encoder_embed_dim"]
    num_patches = seq_len // patch_size

    encoder = Encoder(
        num_patches=num_patches,
        dim_in=patch_size,
        embed_dim=embed_dim,
        nhead=cfg["nhead"],
        num_layers=cfg["num_encoder_layers"],
        mlp_ratio=cfg["mlp_ratio"],
        drop_rate=cfg["drop_rate"],
        attn_drop_rate=cfg["attn_drop_rate"],
    ).to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        key  = "target_encoder" if "target_encoder" in ckpt else "encoder"
        missing, _ = encoder.load_state_dict(ckpt[key], strict=False)
        print(f"  [JEPA] loaded from {ckpt_path}  (missing={len(missing)})")
    else:
        print(f"  [JEPA] WARNING: checkpoint not found at {ckpt_path}, random encoder")

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    head = PredictionHead(
        individual=False, n_vars=n_vars,
        d_model=embed_dim, num_patch=num_patches, forecast_len=pred_len,
    ).to(device)

    opt = torch.optim.Adam(head.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-4,
        total_steps=epochs * len(train_dl),
        pct_start=0.3, anneal_strategy='cos')

    def _inst_norm(x, eps=1e-6):
        mean = x.mean(dim=(1, 2), keepdim=True)
        std  = x.std(dim=(1, 2),  keepdim=True) + eps
        return (x - mean) / std, mean, std

    def _inst_denorm(x, mean, std):
        shape = [mean.shape[0]] + [1] * (x.ndim - 2) + [mean.shape[-1]]
        return x * std.reshape(shape) + mean.reshape(shape)

    print(f"  [JEPA] training head for {epochs} epochs …")
    for ep in range(epochs):
        head.train()
        for ctx, tgt in train_dl:
            if ctx.dim() == 3:
                ctx = ctx.unsqueeze(-1)
            ctx = ctx.float().to(device); tgt = tgt.float().to(device)
            B, h, PL, nv = tgt.shape
            target_flat = tgt.reshape(B, h * PL, nv)
            ctx_norm, ctx_mean, ctx_std = _inst_norm(ctx)
            with torch.no_grad():
                enc_out = encoder(ctx_norm)
                enc_p   = enc_out["data_patches"]                              # [B*nv, P, D]
            enc_p = enc_p.reshape(B, nv, num_patches, embed_dim).permute(0, 1, 3, 2)
            pred  = _inst_denorm(head(enc_p), ctx_mean, ctx_std)
            loss  = F.mse_loss(pred, target_flat)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    epoch {ep+1}/{epochs}")

    # collect
    head.eval()
    all_ctx, all_tgt, all_pred = [], [], []
    with torch.no_grad():
        for ctx, tgt in test_dl:
            if ctx.dim() == 3:
                ctx = ctx.unsqueeze(-1)
            ctx_d = ctx.float().to(device); tgt_d = tgt.float().to(device)
            B, h, PL, nv = tgt_d.shape
            target_flat = tgt_d.reshape(B, h * PL, nv)
            ctx_norm, ctx_mean, ctx_std = _inst_norm(ctx_d)
            enc_out = encoder(ctx_norm)
            enc_p   = enc_out["data_patches"]
            enc_p   = enc_p.reshape(B, nv, num_patches, embed_dim).permute(0, 1, 3, 2)
            pred    = _inst_denorm(head(enc_p), ctx_mean, ctx_std)
            all_ctx.append(ctx.float()); all_tgt.append(target_flat.cpu()); all_pred.append(pred.cpu())

    return (torch.cat(all_ctx), torch.cat(all_tgt), torch.cat(all_pred))


# ── CSV writing ───────────────────────────────────────────────────────────────

def write_csvs(ptst_ctx, ptst_tgt, ptst_pred,
               jepa_ctx, jepa_tgt, jepa_pred,
               n_windows, pred_len, var_names, out_dir):
    """
    Writes two CSV files:

    1. debug_etth2_context.csv
       window_idx, var, loader,
       ctx_mean, ctx_std, ctx_min, ctx_max, ctx_last

    2. debug_etth2_predictions.csv
       window_idx, var, t,
       target (from JEPA loader — same data as PatchTST),
       pred_patchtst, pred_jepa
    """
    n_ptst = len(ptst_tgt) if n_windows == 0 else min(n_windows, len(ptst_tgt))
    n_jepa = len(jepa_tgt) if n_windows == 0 else min(n_windows, len(jepa_tgt))
    n_both = min(n_ptst, n_jepa)
    n_vars = ptst_tgt.shape[-1]

    # ── 1. context CSV ────────────────────────────────────────────────────────
    ctx_path = out_dir / "debug_etth2_context.csv"
    ctx_fields = ["window_idx", "var", "loader",
                  "ctx_mean", "ctx_std", "ctx_min", "ctx_max", "ctx_last"]

    def ctx_rows(tensor, loader_name, n_win):
        """tensor: [N, P, PL, nv]"""
        for wi in range(n_win):
            for vi in range(n_vars):
                flat = tensor[wi, :, :, vi].reshape(-1)
                yield {
                    "window_idx": wi, "var": var_names[vi], "loader": loader_name,
                    "ctx_mean": f"{flat.mean().item():.5f}",
                    "ctx_std":  f"{flat.std().item():.5f}",
                    "ctx_min":  f"{flat.min().item():.5f}",
                    "ctx_max":  f"{flat.max().item():.5f}",
                    "ctx_last": f"{flat[-1].item():.5f}",
                }

    with open(ctx_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ctx_fields)
        w.writeheader()
        for row in ctx_rows(ptst_ctx, "patchtst", n_both):
            w.writerow(row)
        for row in ctx_rows(jepa_ctx, "jepa", n_both):
            w.writerow(row)
    print(f"  context CSV → {ctx_path}  ({n_both * n_vars * 2} rows)")

    # ── 2. predictions CSV ────────────────────────────────────────────────────
    pred_path = out_dir / "debug_etth2_predictions.csv"
    pred_fields = ["window_idx", "var", "t", "target", "pred_patchtst", "pred_jepa"]

    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pred_fields)
        w.writeheader()
        for wi in range(n_both):
            for vi in range(n_vars):
                for t in range(pred_len):
                    w.writerow({
                        "window_idx":   wi,
                        "var":          var_names[vi],
                        "t":            t,
                        "target":       f"{jepa_tgt[wi, t, vi].item():.5f}",
                        "pred_patchtst":f"{ptst_pred[wi, t, vi].item():.5f}",
                        "pred_jepa":    f"{jepa_pred[wi, t, vi].item():.5f}",
                    })

    total_rows = n_both * n_vars * pred_len
    print(f"  predictions CSV → {pred_path}  ({total_rows} rows)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_len",   type=int, default=720)
    ap.add_argument("--seq_len",    type=int, default=512,
                    help="Context length in timesteps (default 512 = 32 × 16)")
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--epochs",     type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_windows",  type=int, default=0,
                    help="Windows to write (0 = all)")
    ap.add_argument("--jepa_ckpt",  type=str, default=None,
                    help="JEPA checkpoint path (auto-detected if omitted)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"pred_len={args.pred_len}  seq_len={args.seq_len}  "
          f"patch_size={args.patch_size}  epochs={args.epochs}")

    ds_info   = get_dataset_info("etth2")
    csv_path  = ds_info["csv_path"]
    var_names = ds_info["columns"]
    n_vars    = ds_info["c_in"]
    print(f"ETTh2 : {csv_path}  ({n_vars} vars)")

    # load JEPA config
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "config_jepa", ROOT / "JEPA" / "config_files" / "config_jepa.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    jepa_cfg = dict(mod.config)

    # auto-detect checkpoint
    jepa_ckpt = args.jepa_ckpt
    if jepa_ckpt is None:
        import re
        ckpt_dir = ROOT / "output_model" / "JEPA"
        found = sorted(
            (int(m.group(1)), p)
            for p in ckpt_dir.glob("_epoch*best_model.pt")
            for m in [re.search(r'_epoch(\d+)best_model', p.stem)]
            if m
        )
        if found:
            jepa_ckpt = str(found[-1][1])
            print(f"JEPA ckpt : {jepa_ckpt}")
        else:
            jepa_ckpt = "NOTFOUND"
            print("JEPA ckpt : NOT FOUND — will use random encoder")

    # make sure pred_len divisible by patch_size
    assert args.pred_len % args.patch_size == 0, \
        f"pred_len={args.pred_len} not divisible by patch_size={args.patch_size}"
    assert args.seq_len  % args.patch_size == 0, \
        f"seq_len={args.seq_len} not divisible by patch_size={args.patch_size}"

    print("\n── building data loaders ──")
    _, ptst_train_dl = make_loader(csv_path, args.seq_len, args.pred_len, args.patch_size,
                                   'train', args.batch_size, shuffle=True,  jepa_config=jepa_cfg)
    _, ptst_test_dl  = make_loader(csv_path, args.seq_len, args.pred_len, args.patch_size,
                                   'test',  args.batch_size, shuffle=False, jepa_config=jepa_cfg)
    _, jepa_train_dl = make_loader(csv_path, args.seq_len, args.pred_len, args.patch_size,
                                   'train', args.batch_size, shuffle=True,  jepa_config=jepa_cfg)
    _, jepa_test_dl  = make_loader(csv_path, args.seq_len, args.pred_len, args.patch_size,
                                   'test',  args.batch_size, shuffle=False, jepa_config=jepa_cfg)

    print(f"  train windows : {len(ptst_train_dl.dataset)}")
    print(f"  test  windows : {len(ptst_test_dl.dataset)}")

    print("\n── PatchTST (random init, RevIN) ──")
    ptst_ctx, ptst_tgt, ptst_pred = run_patchtst(
        ptst_train_dl, ptst_test_dl,
        args.pred_len, args.patch_size, args.seq_len,
        n_vars, args.epochs, device)

    print("\n── JEPA simple (pretrained encoder, no RevIN) ──")
    jepa_ctx, jepa_tgt, jepa_pred = run_jepa(
        jepa_train_dl, jepa_test_dl, jepa_ckpt,
        args.pred_len, args.patch_size, args.seq_len,
        n_vars, args.epochs, device, jepa_cfg)

    # overall MSE
    ptst_mse = F.mse_loss(ptst_pred, ptst_tgt).item()
    jepa_mse = F.mse_loss(jepa_pred, jepa_tgt).item()
    print(f"\n── overall test MSE ──")
    print(f"  PatchTST-random : {ptst_mse:.4f}")
    print(f"  JEPA            : {jepa_mse:.4f}")

    # sanity check: are context/target identical across both loaders?
    ctx_same = torch.allclose(ptst_ctx, jepa_ctx, atol=1e-4)
    tgt_same = torch.allclose(ptst_tgt, jepa_tgt, atol=1e-4)
    print(f"\n  Context identical (ptst vs jepa loader): {ctx_same}")
    print(f"  Target  identical (ptst vs jepa loader): {tgt_same}")

    print("\n── writing CSVs ──")
    write_csvs(ptst_ctx, ptst_tgt, ptst_pred,
               jepa_ctx, jepa_tgt, jepa_pred,
               args.n_windows, args.pred_len, var_names, ROOT)

    print("\nDone.")


if __name__ == "__main__":
    main()
