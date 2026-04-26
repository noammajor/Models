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


def _instance_norm(x, eps=1e-6):
    """Per-instance, per-variable normalization. x: [B, P, PL, n_vars]"""
    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2),  keepdim=True) + eps
    return (x - mean) / std, mean, std


def _instance_denorm(x, mean, std):
    """Reverse _instance_norm. x: [B, *, n_vars], mean/std: [B, 1, 1, n_vars]."""
    shape = [mean.shape[0]] + [1] * (x.ndim - 2) + [mean.shape[-1]]
    return x * std.reshape(shape) + mean.reshape(shape)


def forcasting_zeroshot(self, path, linear_probe=True):
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
    if linear_probe:
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        print(f"  [LE-JEPA forecast] MODE: linear probe — encoder FROZEN")
    else:
        print(f"  [LE-JEPA forecast] MODE: full fine-tuning — encoder UNFROZEN")

    embed_dim   = config["encoder_embed_dim"]
    num_patches = config.get("forecasting_context_patches", config["ratio_patches"])
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

    _all_params = list(self.encoder.parameters()) + list(forecast_head.parameters())
    # LR priority: config (user-set) > env var (script default).
    _cfg_head_lr = config.get("lr_forcasting")
    _cfg_enc_lr  = config.get("lr_forcasting_encoder")
    head_lr = float(_cfg_head_lr if _cfg_head_lr is not None
                    else os.environ["TS_FINETUNE_HEAD_LR"])
    enc_lr  = float(_cfg_enc_lr  if _cfg_enc_lr  is not None
                    else os.environ.get("TS_PRETRAIN_LR", head_lr))
    if linear_probe:
        optimizer = torch.optim.Adam(forecast_head.parameters(),
                                     lr=head_lr, weight_decay=1e-4)
        _max_lrs  = head_lr
        _trainable = sum(p.numel() for p in forecast_head.parameters())
    else:
        optimizer = torch.optim.Adam([
            {"params": forecast_head.parameters(), "lr": head_lr},
            {"params": self.encoder.parameters(),  "lr": enc_lr},
        ], weight_decay=1e-4)
        _max_lrs   = [head_lr, enc_lr]
        _trainable = sum(p.numel() for p in _all_params)
    _total = sum(p.numel() for p in _all_params)
    print(f"  Trainable: {_trainable:,} / {_total:,} params  (head_lr={head_lr}{'' if linear_probe else f', encoder_lr={enc_lr}'})")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=_max_lrs,
        total_steps=self.epoch_t * len(self.forcast_train),
        pct_start=0.3, anneal_strategy='cos',
    )

    _val_loader   = getattr(self, "forcast_val", None)
    best_val_loss = float("inf")
    best_state    = None

    # ── Train head (and optionally encoder) ──────────────────────────────────
    for epoch in range(self.epoch_t):
        forecast_head.train()
        if not linear_probe:
            self.encoder.train()
        total_loss = 0.0
        for context_patches, target_patch in self.forcast_train:
            if context_patches.dim() == 3:
                context_patches = context_patches.unsqueeze(-1)
            context_patches = context_patches.to(self.device)
            target_patch    = target_patch.to(self.device)

            B, h, PL, n_v_b = target_patch.shape

            optimizer.zero_grad()
            ctx_norm, ctx_mean, ctx_std = _instance_norm(context_patches)
            with torch.set_grad_enabled(not linear_probe):
                enc_out = self.encoder(ctx_norm)
                enc_patches = enc_out["data_patches"]          # [B*n_v, num_patches, D]

            enc_p = enc_patches.reshape(B, n_v_b, num_patches, embed_dim).permute(0, 1, 3, 2)
            pred  = _instance_denorm(forecast_head(enc_p), ctx_mean, ctx_std)
            target_flat = target_patch.reshape(B, h * PL, n_v_b)

            loss = F.mse_loss(pred, target_flat)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # ── val MSE + keep-best ──
        val_l = None
        if _val_loader is not None and len(_val_loader) > 0:
            forecast_head.eval()
            self.encoder.eval()
            _vl = 0.0; _vc = 0
            with torch.no_grad():
                for ctx_v, tgt_v in _val_loader:
                    if ctx_v.dim() == 3: ctx_v = ctx_v.unsqueeze(-1)
                    ctx_v = ctx_v.to(self.device); tgt_v = tgt_v.to(self.device)
                    Bv, hv, PLv, nv = tgt_v.shape
                    cn, cm, cs = _instance_norm(ctx_v)
                    eo = self.encoder(cn)["data_patches"]
                    ep = eo.reshape(Bv, nv, num_patches, embed_dim).permute(0, 1, 3, 2)
                    pv = _instance_denorm(forecast_head(ep), cm, cs)
                    tv_flat = tgt_v.reshape(Bv, hv * PLv, nv)
                    _vl += F.mse_loss(pv, tv_flat).item() * Bv
                    _vc += Bv
            val_l = _vl / max(_vc, 1)
            if val_l < best_val_loss:
                best_val_loss = val_l
                best_state = {
                    "head":    {k: v.detach().clone() for k, v in forecast_head.state_dict().items()},
                    "encoder": {k: v.detach().clone() for k, v in self.encoder.state_dict().items()} if not linear_probe else None,
                }

        if epoch % 10 == 0:
            _vmsg = f"  val={val_l:.4f}" if val_l is not None else ""
            print(f"  [Forecast] Epoch {epoch} — loss: {total_loss / max(len(self.forcast_train), 1):.4f}{_vmsg}")

    # ── Restore best checkpoint for test eval ──
    if best_state is not None:
        forecast_head.load_state_dict(best_state["head"])
        if best_state["encoder"] is not None:
            self.encoder.load_state_dict(best_state["encoder"])
        print(f"  [Forecast] Restored best checkpoint  (best val MSE={best_val_loss:.4f})")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    forecast_head.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for context_patches, target_patch in self.forcast_test:
            if context_patches.dim() == 3:
                context_patches = context_patches.unsqueeze(-1)
            context_patches = context_patches.to(self.device)
            target_patch    = target_patch.to(self.device)

            B, h, PL, n_v_b = target_patch.shape
            ctx_norm, ctx_mean, ctx_std = _instance_norm(context_patches)
            enc_out     = self.encoder(ctx_norm)
            enc_patches = enc_out["data_patches"]
            enc_p = enc_patches.reshape(B, n_v_b, num_patches, embed_dim).permute(0, 1, 3, 2)
            pred  = _instance_denorm(forecast_head(enc_p), ctx_mean, ctx_std)
            target_flat = target_patch.reshape(B, h * PL, n_v_b)

            all_preds.append(pred.cpu())
            all_targets.append(target_flat.cpu())

    if not all_preds:
        print("WARNING: forcast_test is empty.")
        return None, None

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mse = F.mse_loss(all_preds, all_targets).item()
    mae = F.l1_loss(all_preds,  all_targets).item()
    print(f"  MSE: {mse:.4f}  MAE: {mae:.4f}")

    # Re-enable gradients for encoder (in case pretraining continues)
    for p in self.encoder.parameters():
        p.requires_grad = True

    return mse
