"""
Zero-shot forecasting for NPT (Next-Token Patch Prediction).

Pipeline
--------
1. Load the pretrained NPT backbone (PatchTSTEncoder, frozen).
2. Attach a PredictionHead trained on (context_patches → forecast).
3. Train only the head on the forecasting dataset's train split.
4. Evaluate MSE / MAE on the test split and save plots.

Data format (ForcastingDataPullerDescrete)
------------------------------------------
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
_NPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_NPT_DIR)
_DJEPA_DIR = os.path.join(_ROOT_DIR, "Discrete_JEPA")

for _p in [_NPT_DIR, _ROOT_DIR, _DJEPA_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.patchTST import PatchTST
from data_loaders.data_puller import ForcastingDataPullerDescrete, PatchTSTForcastingAdapter


# ── model factory ─────────────────────────────────────────────────────────────

def _get_forecasting_model(config, c_in, forecast_len, device):
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
        head_dropout=config["head_dropout"],
        act=config["act"],
        head_type="prediction",
        causal=True,          # keep causal=True to match pretrained settings
        res_attention=False,
    ).to(device)


# ── main entry point ──────────────────────────────────────────────────────────

def zeroshot_forecasting(config, checkpoint_path):
    """
    Zero-shot forecasting: frozen NPT backbone + trained PredictionHead.

    Parameters
    ----------
    config          : dict from config_ntp.py (with forecasting keys added)
    checkpoint_path : path to the pretrained .pth file
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  NPT — Zero-Shot Forecasting")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  device     : {device}")
    print(f"{'='*60}")

    forecast_dset = config.get("forecast_dataset", "ettm1")

    # ── resolve dataset info ──────────────────────────────────────────────────
    from dataset_registry import get_dataset_info
    ds_fore = get_dataset_info(forecast_dset)
    n_vars  = ds_fore["c_in"]

    # Build a config compatible with ForcastingDataPullerDescrete
    patch_size   = config["patch_size"]
    horizon_t    = config.get("horizon_t", 4)
    forecast_len = horizon_t * patch_size

    # context_patches is fixed regardless of pred_len so the backbone always
    # has the same num_patch as the pretrained checkpoint (W_pos shape is stable)
    context_patches = config.get("context_patches", config["ratio_patches"] - horizon_t)
    fore_cfg = dict(config)
    fore_cfg["ratio_patches"]               = context_patches  # data loader uses this as context size
    fore_cfg["patch_size_forcasting"]       = patch_size
    fore_cfg["horizon_t"]                   = horizon_t
    fore_cfg["val_prec_forcasting"]         = config.get("val_prec_forcasting",  config.get("val_prec",  0.1))
    fore_cfg["test_prec_forcasting"]        = config.get("test_prec_forcasting", config.get("test_prec", 0.1))
    fore_cfg["window_step_forecasting"]     = config.get("window_step_forecasting", 1)
    fore_cfg["path_data_forcasting"]        = [ds_fore["csv_path"]]
    fore_cfg["timestampcols_forcasting"]    = [ds_fore["timestamp_col"]]
    fore_cfg["input_variables_forcasting"]  = [ds_fore["columns"]]

    lr_head        = config.get("lr_forcasting",     1e-4)
    epochs_head    = config.get("epochs_forecasting", 20)
    batch_size     = config["batch_size"]

    print(f"\n  dataset={forecast_dset}  n_vars={n_vars}")
    print(f"  context={context_patches} patches × {patch_size} = "
          f"{context_patches * patch_size} steps")
    print(f"  horizon={horizon_t} patches × {patch_size} = {forecast_len} steps")
    print(f"  head lr={lr_head}  epochs={epochs_head}\n")

    # ── data loaders (PatchTST-identical splits/normalization) ───────────────
    _seq_len  = context_patches * patch_size   # = 21 * 16 = 336
    _pred_len = horizon_t * patch_size
    _csv_path = fore_cfg["path_data_forcasting"][0]

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
    model = _get_forecasting_model(fore_cfg, n_vars, forecast_len, device)

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

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone frozen. Head trainable params: {trainable:,}\n")

    # ── train prediction head ─────────────────────────────────────────────────
    lr_head = 1e-4
    optimizer = torch.optim.Adam(model.head.parameters(), lr=lr_head, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_head,
        total_steps=epochs_head * len(train_loader),
        pct_start=0.3, anneal_strategy='cos',
    )

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(epochs_head):
        # backbone stays in eval; only head is trained
        model.backbone.eval()
        model.head.train()

        train_losses = []
        for context_patches, target_patch in train_loader:
            # context_patches : [B, context_size, patch_size, n_vars]
            # target_patch    : [B, horizon_t,    patch_size, n_vars]
            context_patches = context_patches.float().to(device)
            target_patch    = target_patch.float().to(device)

            B, h, P_L, n_v = target_patch.shape
            target_flat = target_patch.reshape(B, h * P_L, n_v)  # [B, forecast_len, n_vars]

            # PatchTST needs [B, num_patch, n_vars, patch_len]
            x = context_patches.permute(0, 1, 3, 2)

            pred = model(x)                              # [B, forecast_len, n_vars]
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
                x    = context_patches.permute(0, 1, 3, 2)
                pred = model(x)
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
    mse_list, mae_list = [], []
    last_batch = None

    with torch.no_grad():
        for context_patches, target_patch in test_loader:
            context_patches = context_patches.float().to(device)
            target_patch    = target_patch.float().to(device)
            B, h, P_L, n_v = target_patch.shape
            target_flat = target_patch.reshape(B, h * P_L, n_v)
            x    = context_patches.permute(0, 1, 3, 2)
            pred = model(x)
            mse_list.append(F.mse_loss(pred.cpu(), target_flat.cpu()).item())
            mae_list.append(F.l1_loss(pred.cpu(),  target_flat.cpu()).item())
            last_batch = (pred.cpu(), target_flat.cpu())

    if not mse_list:
        print("WARNING: test set is empty — skipping evaluation.")
        return None, None

    mse = float(np.mean(mse_list))
    mae = float(np.mean(mae_list))
    print(f"\n  [NPT Zero-Shot — {forecast_dset}]")
    print(f"  MSE : {mse:.4f}")
    print(f"  MAE : {mae:.4f}")

    # ── plots ─────────────────────────────────────────────────────────────────
    pred_out, target_out = last_batch
    pretrain_dset = config.get("pretrain_dataset", "monash")
    save_dir = os.path.join(
        _NPT_DIR, "saved_models", pretrain_dset, "ntp", "output_model")
    os.makedirs(save_dir, exist_ok=True)

    sample   = 0
    n_plot   = min(n_v, 3)   # plot up to 3 variables
    for var_idx in range(n_plot):
        gt   = target_out[sample, :, var_idx].numpy()
        pred = pred_out[sample, :, var_idx].numpy()

        plt.figure(figsize=(15, 5))
        plt.plot(gt,   label="Ground Truth", color="black", alpha=0.7, linewidth=2)
        plt.plot(pred, label="NPT Forecast", color="blue",  linestyle="--", alpha=0.9)
        plt.title(f"NPT Zero-Shot — {forecast_dset} — Variable {var_idx} "
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
