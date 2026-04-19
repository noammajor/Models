"""
NPT anomaly detection (reconstruction-based).

NPT shares the PatchTST backbone — this is a thin wrapper that loads the
NPT checkpoint and delegates to the same reconstruction pipeline.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(_DIR)
_PTST  = os.path.join(_ROOT, "PatchTST_self_supervised")

for _p in [_DIR, _ROOT, _PTST]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models.patchTST import PatchTST


class _LinearReconDecoder(nn.Module):
    """[B, n_vars, d_model, n_patches] → [B, T, n_vars]"""
    def __init__(self, d_model: int, patch_size: int, n_vars: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_vars = n_vars
        self.proj = nn.Linear(d_model, patch_size)

    def forward(self, z):
        B, C, D, P = z.shape
        z   = z.permute(0, 1, 3, 2)                    # [B, C, P, D]
        out = self.proj(z)                              # [B, C, P, patch_size]
        out = out.reshape(B, C, P * self.patch_size)
        return out.permute(0, 2, 1)                     # [B, T, C]


def _adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0: break
                if pred[j] == 0: pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0: break
                if pred[j] == 0: pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def anomaly_zeroshot(config, checkpoint_path, anomaly_train, anomaly_test,
                     anomaly_ratio: float = 1.0):
    """
    Reconstruction-based anomaly detection with frozen NPT backbone.

    Args:
        config         : dict from config_ntp.py
        checkpoint_path: path to pretrained .pt checkpoint
        anomaly_train  : DataLoader — batches of patches [B, P, patch_size, n_vars]
        anomaly_test   : DataLoader — batches of (patches, labels [B, T])
        anomaly_ratio  : top-X% of combined energy flagged as anomaly
    Returns:
        dict with f1, precision, recall, accuracy, threshold
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== NPT Anomaly Detection ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    sample = next(iter(anomaly_train))
    patches_sample = sample[0] if isinstance(sample, (list, tuple)) else sample
    patch_len = patches_sample.shape[2]
    n_vars    = patches_sample.shape[3]

    # Load checkpoint first — infer num_patch from W_pos so size matches checkpoint
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    num_patch = state["backbone.W_pos"].shape[0]   # e.g. 21 for cw=336

    backbone = PatchTST(
        c_in         = n_vars,
        target_dim   = patch_len,
        patch_len    = patch_len,
        stride       = patch_len,
        num_patch    = num_patch,
        n_layers     = config.get("n_layers",  3),
        n_heads      = config.get("n_heads",   16),
        d_model      = config.get("d_model",   128),
        shared_embedding = True,
        d_ff         = config.get("d_ff",      512),
        dropout      = config.get("dropout",   0.2),
        head_dropout = config.get("head_dropout", 0.2),
        act          = "gelu",
        head_type    = "pretrain",
        res_attention = False,
    ).to(device)

    backbone.load_state_dict(state, strict=False)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    d_model  = config.get("d_model", 128)
    decoder  = _LinearReconDecoder(d_model, patch_len, n_vars).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(),
                                 lr=config.get("lr_anomaly", 1e-3))
    n_epochs  = config.get("epoch_anomaly", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    def _encode(patches):
        x = patches.permute(0, 1, 3, 2).to(device)
        return backbone.backbone(x)

    # ── (1) train decoder (with validation + early stopping) ─────────────────
    all_batches   = list(anomaly_train)
    n_val_b       = max(1, len(all_batches) // 5)
    train_batches = all_batches[:-n_val_b]
    val_batches   = all_batches[-n_val_b:]

    patience   = config.get("anomaly_patience", 3)
    best_val   = float('inf')
    best_state = None
    no_improve = 0

    print(f"  Training decoder ({n_epochs} epochs, patience={patience}) …")
    for epoch in range(n_epochs):
        # train
        decoder.train()
        total_loss = 0.0
        for batch in train_batches:
            patches = batch[0] if isinstance(batch, (list, tuple)) else batch
            patches = patches.to(device)
            B, P, PL, C = patches.shape
            with torch.no_grad():
                z = _encode(patches)
            recon  = decoder(z)
            target = patches.reshape(B, P * PL, C)
            loss   = F.mse_loss(recon, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        # validate
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_batches:
                patches = batch[0] if isinstance(batch, (list, tuple)) else batch
                patches = patches.to(device)
                B, P, PL, C = patches.shape
                z        = _encode(patches)
                recon    = decoder(z)
                target   = patches.reshape(B, P * PL, C)
                val_loss += F.mse_loss(recon, target).item()
        val_loss /= len(val_batches)
        # early stopping
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        scheduler.step()
        if epoch % max(1, n_epochs // 5) == 0:
            print(f"    epoch {epoch+1:3d} | train {total_loss/len(train_batches):.6f} | val {val_loss:.6f}")
    if best_state is not None:
        decoder.load_state_dict(best_state)

    # ── (2) train energy ─────────────────────────────────────────────────────
    decoder.eval()
    train_energy = []
    with torch.no_grad():
        for batch in anomaly_train:
            patches = batch[0] if isinstance(batch, (list, tuple)) else batch
            patches = patches.to(device)
            B, P, PL, C = patches.shape
            z      = _encode(patches)
            recon  = decoder(z)
            target = patches.reshape(B, P * PL, C)
            score  = F.mse_loss(recon, target, reduction="none").mean(dim=-1)
            train_energy.append(score.cpu().numpy())
    train_energy = np.concatenate(train_energy).reshape(-1)

    # ── (3) test energy ──────────────────────────────────────────────────────
    test_energy, all_labels = [], []
    with torch.no_grad():
        for patches, labels in anomaly_test:
            patches = patches.to(device)
            B, P, PL, C = patches.shape
            z      = _encode(patches)
            recon  = decoder(z)
            target = patches.reshape(B, P * PL, C)
            score  = F.mse_loss(recon, target, reduction="none").mean(dim=-1)
            test_energy.append(score.cpu().numpy())
            all_labels.append(labels.numpy())
    test_energy = np.concatenate(test_energy).reshape(-1)
    gt          = np.concatenate(all_labels).reshape(-1).astype(int)

    # ── (4) threshold & predict ───────────────────────────────────────────────
    threshold = np.percentile(np.concatenate([train_energy, test_energy]),
                              100 - anomaly_ratio)
    pred = (test_energy > threshold).astype(int)

    # ── (5) adjustment + metrics ──────────────────────────────────────────────
    gt, pred = _adjustment(gt.copy(), pred.copy())
    accuracy  = accuracy_score(gt, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt, pred, average="binary", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  [NPT] Anomaly Detection")
    print(f"  Threshold: {threshold:.6f}  (top {anomaly_ratio}%)")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"{'='*60}\n")

    return dict(f1=f1, precision=precision, recall=recall,
                accuracy=accuracy, threshold=threshold)
