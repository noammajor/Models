"""
Forecasting for NTP (Next-Token Patch Prediction).

Pipeline
--------
1. Load the pretrained NTP backbone (PatchTSTEncoder; frozen by default,
   unfrozen if linear_probe=False).
2. Attach a PredictionHead trained on (context_patches → forecast).
3. Train the head (and optionally the backbone) on the forecasting dataset's train split.
4. Evaluate MSE / MAE on the test split and save plots.

Data format (PatchTSTForcastingAdapter)
---------------------------------------
  context_patches : [B x context_size x patch_size x n_vars]
  target_patch    : [B x horizon_t   x patch_size x n_vars]

PatchTST input: [B x num_patch x n_vars x patch_len]  (permute last two dims)
PatchTST output (prediction head): [B x forecast_len x n_vars]
  where forecast_len = horizon_t * patch_size
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
_NTP_DIR     = os.path.dirname(os.path.abspath(__file__))   # code + checkpoints dir
_ROOT_DIR    = os.path.dirname(_NTP_DIR)
_SHARED_DIR  = os.path.join(_ROOT_DIR, "shared")

for _p in [_NTP_DIR, _ROOT_DIR, _SHARED_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.patchTST import PatchTST
from data_loaders.data_puller import PatchTSTForcastingAdapter


def _instance_norm(x, eps=1e-6):
    """Per-instance, per-variable normalization over the patch/time axes.
    x: [B, n_patches, patch_size, n_vars]  →  returns (x_norm, mean, std)
    mean/std shape: [B, 1, 1, n_vars]
    """
    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2),  keepdim=True) + eps
    return (x - mean) / std, mean, std


def _instance_denorm(x, mean, std):
    """Reverse _instance_norm.  x: [B, *, n_vars],  mean/std: [B, 1, 1, n_vars]."""
    shape = [mean.shape[0]] + [1] * (x.ndim - 2) + [mean.shape[-1]]
    return x * std.reshape(shape) + mean.reshape(shape)


# ── model factory ─────────────────────────────────────────────────────────────

def _get_forecasting_model(config, c_in, forecast_len, device, mlp_head: bool = False):
    """
    Build a PatchTST with a PredictionHead.

    backbone architecture must match the pretrained checkpoint exactly
    (same patch_len, num_patch, n_layers, d_model, n_heads, d_ff, causal).
    """
    return PatchTST(
        c_in=c_in,
        target_dim=forecast_len,
        patch_len=config["patch_size"],
        stride=config["patch_size"],
        num_patch=config["ratio_patches"],  # already set to context_patches by caller
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_model=config["d_model"],
        shared_embedding=True,
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        head_dropout=config.get("head_dropout_forecasting", config.get("head_dropout", 0.2)),
        act=config["act"],
        head_type="prediction",
        causal=True,          # keep causal=True to match pretrained settings
        res_attention=False,
        mlp_head=mlp_head,
    ).to(device)


# ── main entry point ──────────────────────────────────────────────────────────

def forecasting(config, checkpoint_path, linear_probe=True, mlp_head: bool = False):
    """
    Forecasting with NTP backbone + trained PredictionHead.

    Parameters
    ----------
    config          : dict from config_ntp.py (with forecasting keys added)
    checkpoint_path : path to the pretrained .pth file
    linear_probe    : if True, freeze backbone and train only head. If False, fine-tune backbone + head.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  NTP — Forecasting")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  device     : {device}")
    print(f"{'='*60}")

    forecast_dset = config.get("forecast_dataset", "ettm1")

    # ── resolve dataset info ──────────────────────────────────────────────────
    from dataset_registry import get_dataset_info
    ds_fore = get_dataset_info(forecast_dset)
    n_vars  = ds_fore["c_in"]

    patch_size   = config["patch_size"]
    horizon_t    = config.get("horizon_t", 4)
    forecast_len = horizon_t * patch_size

    # context_patches is fixed regardless of pred_len so the backbone always
    # has the same num_patch as the pretrained checkpoint (W_pos shape is stable)
    context_patches = config.get("context_patches", config["ratio_patches"] - horizon_t)
    # Override num_patch on the model config (ratio_patches → context size).
    fore_cfg = dict(config, ratio_patches=context_patches)

    epochs_head    = config.get("epochs_forecasting", 20)
    batch_size     = config["batch_size"]

    print(f"\n  dataset={forecast_dset}  n_vars={n_vars}")
    print(f"  context={context_patches} patches × {patch_size} = "
          f"{context_patches * patch_size} steps")
    print(f"  horizon={horizon_t} patches × {patch_size} = {forecast_len} steps")
    print(f"  epochs={epochs_head}\n")

    # ── data loaders (PatchTST-identical splits/normalization) ───────────────
    _seq_len  = context_patches * patch_size   # = 21 * 16 = 336
    _pred_len = horizon_t * patch_size
    _csv_path = ds_fore["csv_path"]

    train_fc = PatchTSTForcastingAdapter(_csv_path, 'train', _seq_len, _pred_len, patch_size)
    val_fc   = PatchTSTForcastingAdapter(_csv_path, 'val',   _seq_len, _pred_len, patch_size)
    test_fc  = PatchTSTForcastingAdapter(_csv_path, 'test',  _seq_len, _pred_len, patch_size)

    train_loader = torch.utils.data.DataLoader(
        train_fc, batch_size=batch_size, shuffle=True,
        num_workers=config.get("num_workers", 0))
    val_loader   = torch.utils.data.DataLoader(
        val_fc,   batch_size=batch_size, shuffle=False,
        num_workers=config.get("num_workers", 0))
    test_loader  = torch.utils.data.DataLoader(
        test_fc,  batch_size=batch_size, shuffle=False,
        num_workers=config.get("num_workers", 0))

    print(f"  train={len(train_fc)}  val={len(val_fc)}  test={len(test_fc)}  windows")

    # ── build model ───────────────────────────────────────────────────────────
    model = _get_forecasting_model(fore_cfg, n_vars, forecast_len, device, mlp_head=mlp_head)

    # Load pretrained backbone weights from the "encoder" key in the checkpoint
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        encoder_state = ckpt["encoder"]
        missing, unexpected = model.backbone.load_state_dict(encoder_state, strict=True)
        print(f"  Loaded encoder from epoch {ckpt.get('epoch', '?')}  "
              f"(missing={len(missing)}  unexpected={len(unexpected)})")
    else:
        print(f"  WARNING: checkpoint not found at {checkpoint_path}, "
              f"using random backbone weights")

    # Freeze backbone (linear probe) or train full model (fine-tune)
    if linear_probe:
        for param in model.backbone.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  [NTP forecast] MODE: linear probe — encoder FROZEN")
        print(f"  Trainable: {trainable:,} / {total:,} params\n")
    else:
        trainable = sum(p.numel() for p in model.parameters())
        print(f"  [NTP forecast] MODE: full fine-tuning — encoder UNFROZEN")
        print(f"  Trainable: {trainable:,} / {trainable:,} params\n")

    # ── train prediction head (and optionally backbone) ───────────────────────
    # LR: config value if set, else hardcoded default.
    _cfg_head_lr = config.get("lr_forcasting")
    _cfg_enc_lr  = config.get("lr_forcasting_encoder")
    lr_head = float(_cfg_head_lr) if _cfg_head_lr is not None else 0.0001
    enc_lr  = float(_cfg_enc_lr)  if _cfg_enc_lr  is not None else lr_head
    if linear_probe:
        optimizer = torch.optim.Adam(model.head.parameters(),
                                     lr=lr_head, weight_decay=1e-4)
        _max_lrs  = lr_head
        print(f"  [NTP forecast] head_lr={lr_head}")
    else:
        optimizer = torch.optim.Adam([
            {"params": model.head.parameters(),     "lr": lr_head},
            {"params": model.backbone.parameters(), "lr": enc_lr},
        ], weight_decay=1e-4)
        _max_lrs = [lr_head, enc_lr]
        print(f"  [NTP forecast] head_lr={lr_head}  encoder_lr={enc_lr}")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=_max_lrs,
        total_steps=epochs_head * len(train_loader),
        pct_start=0.3, anneal_strategy='cos',
    )

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(epochs_head):
        if linear_probe:
            model.backbone.eval()
            model.head.train()
        else:
            model.train()

        train_losses = []
        for context_patches, target_patch in train_loader:
            # context_patches : [B, context_size, patch_size, n_vars]
            # target_patch    : [B, horizon_t,    patch_size, n_vars]
            context_patches = context_patches.float().to(device)
            target_patch    = target_patch.float().to(device)

            B, h, P_L, n_v = target_patch.shape
            target_flat = target_patch.reshape(B, h * P_L, n_v)  # [B, forecast_len, n_vars]

            ctx_norm, ctx_mean, ctx_std = _instance_norm(context_patches)
            # PatchTST needs [B, num_patch, n_vars, patch_len]
            x = ctx_norm.permute(0, 1, 3, 2)

            pred = _instance_denorm(model(x), ctx_mean, ctx_std)
            loss = F.mse_loss(pred, target_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for context_patches, target_patch in val_loader:
                context_patches = context_patches.float().to(device)
                target_patch    = target_patch.float().to(device)
                B, h, P_L, n_v = target_patch.shape
                target_flat = target_patch.reshape(B, h * P_L, n_v)
                ctx_norm, ctx_mean, ctx_std = _instance_norm(context_patches)
                x    = ctx_norm.permute(0, 1, 3, 2)
                pred = _instance_denorm(model(x), ctx_mean, ctx_std)
                val_losses.append(F.mse_loss(pred, target_flat).item())

        train_l = float(np.mean(train_losses))
        val_l   = float(np.mean(val_losses))

        if epoch % 5 == 0 or epoch == epochs_head - 1:
            print(f"  epoch {epoch+1:3d}/{epochs_head}  "
                  f"train={train_l:.4f}  val={val_l:.4f}")

        if val_l < best_val_loss:
            best_val_loss = val_l
            best_state    = copy.deepcopy(model.state_dict())

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  Best val loss: {best_val_loss:.4f}")

    # ── test evaluation ───────────────────────────────────────────────────────
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for context_patches, target_patch in test_loader:
            context_patches = context_patches.float().to(device)
            target_patch    = target_patch.float().to(device)
            B, h, P_L, n_v = target_patch.shape
            target_flat = target_patch.reshape(B, h * P_L, n_v)
            ctx_norm, ctx_mean, ctx_std = _instance_norm(context_patches)
            x    = ctx_norm.permute(0, 1, 3, 2)
            pred = _instance_denorm(model(x), ctx_mean, ctx_std)
            all_preds.append(pred.cpu())
            all_targets.append(target_flat.cpu())

    if not all_preds:
        print("WARNING: test set is empty — skipping evaluation.")
        return None, None

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mse = F.mse_loss(all_preds, all_targets).item()
    mae = F.l1_loss(all_preds,  all_targets).item()
    last_batch = (all_preds[-1:], all_targets[-1:])
    print(f"\n  [NTP Forecast — {forecast_dset}]")
    print(f"  MSE : {mse:.4f}")
    print(f"  MAE : {mae:.4f}")

    # ── plots ─────────────────────────────────────────────────────────────────
    pred_out, target_out = last_batch   # each [1, forecast_len, n_vars]
    pretrain_dset = config.get("pretrain_dataset", "monash")
    save_dir = os.path.join(
        _NTP_DIR, "saved_models", pretrain_dset, "ntp", "output_model")
    os.makedirs(save_dir, exist_ok=True)

    n_plot = min(n_v, 3)   # plot up to 3 variables
    for var_idx in range(n_plot):
        gt   = target_out[0, :, var_idx].numpy()
        pred = pred_out[0, :, var_idx].numpy()

        plt.figure(figsize=(15, 5))
        plt.plot(gt,   label="Ground Truth", color="black", alpha=0.7, linewidth=2)
        plt.plot(pred, label="NTP Forecast", color="blue",  linestyle="--", alpha=0.9)
        plt.title(f"NTP Forecast — {forecast_dset} — Variable {var_idx} "
                  f"({forecast_len} steps)")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"zeroshot_{forecast_dset}_var{var_idx}.png"))
        plt.close()

    print(f"  Plots saved to {save_dir}\n")
    return mse, mae
