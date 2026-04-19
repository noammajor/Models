"""
LE-JEPA anomaly detection (reconstruction-based).

Frozen pretrained encoder + linear reconstruction decoder trained on
normal data only. Anomaly score = per-timestep reconstruction MSE.
Threshold = percentile of combined train+test energy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def _instance_norm(x, eps=1e-6):
    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2),  keepdim=True) + eps
    return (x - mean) / std, mean, std


class _LinearReconDecoder(nn.Module):
    """[B, P, embed_dim] → [B, T, n_vars]"""
    def __init__(self, embed_dim: int, patch_size: int, n_vars: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_vars = n_vars
        self.proj = nn.Linear(embed_dim, patch_size * n_vars)

    def forward(self, z):
        B, P, _ = z.shape
        out = self.proj(z)
        return out.reshape(B, P * self.patch_size, self.n_vars)


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


def anomaly_zeroshot(self, path, anomaly_train, anomaly_test,
                     anomaly_ratio: float = 1.0):
    """
    Reconstruction-based anomaly detection with frozen LE-JEPA encoder.

    Args:
        path          : checkpoint tag, e.g. "" or "_epoch50"
        anomaly_train : DataLoader — batches of patches [B, P, patch_size, n_vars]
        anomaly_test  : DataLoader — batches of (patches, labels [B, T])
        anomaly_ratio : top-X% of combined energy flagged as anomaly
    Returns:
        dict with f1, precision, recall, accuracy, threshold
    """
    config          = self.config
    checkpoint_path = f"{self.path_save}{path}best_model.pt"

    print(f"\n=== LE-JEPA Anomaly Detection ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    self.encoder.load_state_dict(ckpt["encoder"])
    self.encoder.to(self.device)
    self.encoder.eval()
    for p in self.encoder.parameters():
        p.requires_grad = False

    embed_dim = config["encoder_embed_dim"]

    # Infer shape from first batch
    sample = next(iter(anomaly_train))
    patches_sample = sample[0] if isinstance(sample, (list, tuple)) else sample
    if patches_sample.dim() == 3:
        patches_sample = patches_sample.unsqueeze(-1)
    patch_size = patches_sample.shape[2]
    n_vars     = patches_sample.shape[3]

    decoder   = _LinearReconDecoder(embed_dim, patch_size, n_vars).to(self.device)
    optimizer = torch.optim.Adam(decoder.parameters(),
                                 lr=config.get("lr_anomaly", 1e-3))
    n_epochs  = config.get("epoch_anomaly", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    def _encode(patches):
        """patches [B, P, PL, C] → z [B, P, embed_dim]"""
        if patches.dim() == 3:
            patches = patches.unsqueeze(-1)
        patches = patches.to(self.device)
        B, P, PL, C = patches.shape
        # reshape to [B*C, P, PL] for encoder
        x = patches.permute(0, 3, 1, 2).reshape(B * C, P, PL)
        norm, _, _ = _instance_norm(x)
        enc = self.encoder(norm)                              # [B*C, P, embed_dim]
        return enc.reshape(B, C, P, embed_dim).mean(dim=1)   # [B, P, embed_dim]

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
            if patches.dim() == 3: patches = patches.unsqueeze(-1)
            raw = patches.to(self.device)
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
                raw = patches.to(self.device)
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
            raw = patches.to(self.device)
            B, P, PL, C = raw.shape
            z      = _encode(raw)
            recon  = decoder(z)
            target = raw.reshape(B, P * PL, C)
            score  = F.mse_loss(recon, target, reduction="none").mean(dim=-1)
            train_energy.append(score.cpu().numpy())
    train_energy = np.concatenate(train_energy).reshape(-1)

    # ── (3) test energy ──────────────────────────────────────────────────────
    test_energy, all_labels = [], []
    with torch.no_grad():
        for patches, labels in anomaly_test:
            if patches.dim() == 3: patches = patches.unsqueeze(-1)
            raw = patches.to(self.device)
            B, P, PL, C = raw.shape
            z      = _encode(raw)
            recon  = decoder(z)
            target = raw.reshape(B, P * PL, C)
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
    print(f"  [LE-JEPA] Anomaly Detection")
    print(f"  Threshold: {threshold:.6f}  (top {anomaly_ratio}%)")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"{'='*60}\n")

    return dict(f1=f1, precision=precision, recall=recall,
                accuracy=accuracy, threshold=threshold)
