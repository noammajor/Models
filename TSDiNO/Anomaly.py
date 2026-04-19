"""
TSDiNO anomaly detection (reconstruction-based).

Frozen pretrained DINO teacher encoder + linear reconstruction decoder trained
on normal data only.  Anomaly score = per-timestep reconstruction MSE.
Threshold = percentile of combined train+test energy.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models.patchTST import PatchTST


class _LinearReconDecoder(nn.Module):
    """[B, P, nvars, d_model] → [B, T, nvars]"""
    def __init__(self, d_model: int, patch_len: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(d_model, patch_len)

    def forward(self, z):
        # z: [B, P, nvars, d_model]
        out = self.proj(z)                        # [B, P, nvars, patch_len]
        B, P, C, PL = out.shape
        out = out.permute(0, 1, 3, 2)            # [B, P, patch_len, nvars]
        return out.reshape(B, P * PL, C)          # [B, T, nvars]


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


def anomaly_zeroshot(args, path_num, anomaly_train, anomaly_test,
                     anomaly_ratio: float = 1.0,
                     checkpoint_path: str = None):
    """
    Reconstruction-based anomaly detection with frozen DINO teacher encoder.

    Args:
        args             : argparse Namespace (same config used for training)
        path_num         : checkpoint number used to build path if checkpoint_path
                           is not given (e.g. 100 → checkpoint0100.pth)
        anomaly_train    : DataLoader — batches of patches [B, P, patch_len, n_vars]
        anomaly_test     : DataLoader — batches of (patches, labels [B, T])
        anomaly_ratio    : top-X% of combined energy flagged as anomaly
        checkpoint_path  : optional explicit path to a .pth checkpoint file.
                           If provided, path_num and args.output_dir are ignored
                           for checkpoint loading.
    Returns:
        dict with f1, precision, recall, accuracy, threshold
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build backbone (PatchTSTEncoder only, no DINOHead) ───────────────────
    backbone = PatchTST(
        c_in=args.c_in,
        target_dim=args.pred_len,
        patch_len=args.patch_len,
        num_patch=args.num_patches,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.embed_dim,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=0.0,
        head_dropout=0.0,
        act='gelu',
        head_type='Dino',
        res_attention=False,
        drop_path_rate=0.0,
        step_size=args.step_size,
    ).backbone  # PatchTSTEncoder

    # ── Resolve checkpoint path ───────────────────────────────────────────────
    if checkpoint_path is None:
        if isinstance(path_num, int):
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint{path_num:04d}.pth')
        else:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pth')

    print(f"\n=== TSDiNO Anomaly Detection ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        ckpt   = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        raw_sd = ckpt['teacher']
        # TSMultiCropWrapper wraps as backbone.* → strip prefix
        new_sd = {k[len('backbone.'):]: v
                  for k, v in raw_sd.items() if k.startswith('backbone.')}
        missing, unexpected = backbone.load_state_dict(new_sd, strict=False)
        print(f"  Loaded {len(new_sd)} weights | missing: {len(missing)} | unexpected: {len(unexpected)}")
    else:
        print(f"  WARNING: checkpoint not found — using random init.")

    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Infer shape from first batch
    sample         = next(iter(anomaly_train))
    patches_sample = sample[0] if isinstance(sample, (list, tuple)) else sample
    if patches_sample.dim() == 3:
        patches_sample = patches_sample.unsqueeze(-1)
    patch_len = patches_sample.shape[2]
    d_model   = args.embed_dim

    decoder   = _LinearReconDecoder(d_model, patch_len).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    n_epochs  = getattr(args, 'epoch_anomaly', 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    def _encode(patches):
        """
        patches: [B, P, patch_len, C]
        → backbone input: [B, P, C, patch_len]  (num_patch, nvars, patch_len)
        → backbone output: [B, P+1, C, d_model] (includes CLS at pos 0)
        → drop CLS → [B, P, C, d_model]
        """
        if patches.dim() == 3:
            patches = patches.unsqueeze(-1)
        patches = patches.to(device)
        x = patches.permute(0, 1, 3, 2)   # [B, P, C, patch_len]
        z = backbone(x)                    # [B, P+1, C, d_model]
        return z[:, 1:, :, :]             # drop CLS → [B, P, C, d_model]

    # ── (1) train decoder (with validation + early stopping) ─────────────────
    all_batches   = list(anomaly_train)
    n_val_b       = max(1, len(all_batches) // 5)
    train_batches = all_batches[:-n_val_b]
    val_batches   = all_batches[-n_val_b:]

    patience   = getattr(args, 'anomaly_patience', 3)
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
            if patches.dim() == 3: patches = patches.unsqueeze(-1)
            raw = patches.to(device)
            B, P, PL, C = raw.shape
            with torch.no_grad():
                z = _encode(raw)
            recon  = decoder(z)
            target = raw.reshape(B, P * PL, C)
            loss   = F.mse_loss(recon, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        # validate
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_batches:
                patches = batch[0] if isinstance(batch, (list, tuple)) else batch
                if patches.dim() == 3: patches = patches.unsqueeze(-1)
                raw = patches.to(device)
                B, P, PL, C = raw.shape
                z      = _encode(raw)
                recon  = decoder(z)
                target = raw.reshape(B, P * PL, C)
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
            if patches.dim() == 3: patches = patches.unsqueeze(-1)
            raw = patches.to(device)
            B, P, PL, C = raw.shape
            z      = _encode(raw)
            recon  = decoder(z)
            target = raw.reshape(B, P * PL, C)
            score  = F.mse_loss(recon, target, reduction="none").mean(dim=-1)  # [B, T]
            train_energy.append(score.cpu().numpy())
    train_energy = np.concatenate(train_energy).reshape(-1)

    # ── (3) test energy ──────────────────────────────────────────────────────
    test_energy, all_labels = [], []
    with torch.no_grad():
        for patches, labels in anomaly_test:
            if patches.dim() == 3: patches = patches.unsqueeze(-1)
            raw = patches.to(device)
            B, P, PL, C = raw.shape
            z      = _encode(raw)
            recon  = decoder(z)
            target = raw.reshape(B, P * PL, C)
            score  = F.mse_loss(recon, target, reduction="none").mean(dim=-1)  # [B, T]
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
    print(f"  [TSDiNO] Anomaly Detection")
    print(f"  Threshold: {threshold:.6f}  (top {anomaly_ratio}%)")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"{'='*60}\n")

    return dict(f1=f1, precision=precision, recall=recall,
                accuracy=accuracy, threshold=threshold)
