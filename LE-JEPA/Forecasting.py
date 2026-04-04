"""
LE-JEPA zero-shot forecasting (linear probe).

Frozen pretrained encoder + linear PredictionHead trained on the forecasting split.
Identical pattern to JEPA's forcasting_zeroshot but simpler:
  - single encoder (no encoder_for copy)
  - load from LE-JEPA checkpoint format: {"epoch": N, "encoder": state_dict}
"""

import os
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
_JEPA_DIR = str(Path(__file__).parent.parent / "JEPA" / "JEPA")
if _JEPA_DIR not in sys.path:
    sys.path.insert(0, _JEPA_DIR)

from Decoder import PredictionHead


def forcasting_zeroshot(self, path):
    """
    Linear-probe forecasting with frozen LE-JEPA encoder.

    path : epoch tag string, e.g. "_epoch5"
           → loads {path_save}{path}best_model.pt
    """
    config           = self.config
    checkpoint_path  = f"{self.path_save}{path}best_model.pt"

    print(f"\n=== LE-JEPA Zero-Shot Forecasting ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    self.encoder.load_state_dict(ckpt["encoder"])
    self.encoder.to(self.device)
    self.encoder.eval()
    for p in self.encoder.parameters():
        p.requires_grad = False

    embed_dim   = config["encoder_embed_dim"]
    num_patches = config.get("context_t", config["ratio_patches"])
    h_t         = config["horizon_t"]
    P_L         = config["patch_size_forcasting"]
    n_v         = len(config["input_variables_forcasting"][0])

    forecast_head = PredictionHead(
        individual   = False,
        n_vars       = n_v,
        d_model      = embed_dim,
        num_patch    = num_patches,
        forecast_len = h_t * P_L,
    ).to(self.device)

    optimizer = torch.optim.SGD(
        forecast_head.parameters(),
        lr           = config.get("lr_forcasting", 1e-3),
        momentum     = 0.9,
        weight_decay = 1e-4,
    )

    # ── Train linear head ─────────────────────────────────────────────────────
    for epoch in range(self.epoch_t):
        forecast_head.train()
        total_loss = 0.0
        for context_patches, target_patch in self.forcast_train:
            if context_patches.dim() == 3:
                context_patches = context_patches.unsqueeze(-1)
            context_patches = context_patches.to(self.device)
            target_patch    = target_patch.to(self.device)

            B, h, PL, n_v_b = target_patch.shape

            optimizer.zero_grad()
            with torch.no_grad():
                enc_out = self.encoder(context_patches)
                enc_patches = enc_out["data_patches"]          # [B*n_v, num_patches, D]

            enc_p = enc_patches.reshape(B, n_v_b, num_patches, embed_dim).permute(0, 1, 3, 2)
            pred  = forecast_head(enc_p)                       # [B, h_t*P_L, n_v]
            target_flat = target_patch.reshape(B, h * PL, n_v_b)

            loss = F.mse_loss(pred, target_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"  [Forecast] Epoch {epoch} — loss: {total_loss / max(len(self.forcast_train), 1):.4f}")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    forecast_head.eval()
    mse_list, mae_list = [], []

    with torch.no_grad():
        for context_patches, target_patch in self.forcast_test:
            if context_patches.dim() == 3:
                context_patches = context_patches.unsqueeze(-1)
            context_patches = context_patches.to(self.device)
            target_patch    = target_patch.to(self.device)

            B, h, PL, n_v_b = target_patch.shape
            enc_out     = self.encoder(context_patches)
            enc_patches = enc_out["data_patches"]
            enc_p = enc_patches.reshape(B, n_v_b, num_patches, embed_dim).permute(0, 1, 3, 2)
            pred  = forecast_head(enc_p)
            target_flat = target_patch.reshape(B, h * PL, n_v_b)

            mse_list.append(F.mse_loss(pred.cpu(), target_flat.cpu()).item())
            mae_list.append(F.l1_loss(pred.cpu(),  target_flat.cpu()).item())

    if not mse_list:
        print("WARNING: forcast_test is empty.")
        return None, None

    mse = sum(mse_list) / len(mse_list)
    mae = sum(mae_list) / len(mae_list)
    print(f"  MSE: {mse:.4f}  MAE: {mae:.4f}")

    # Re-enable gradients for encoder (in case pretraining continues)
    for p in self.encoder.parameters():
        p.requires_grad = True

    return mse
