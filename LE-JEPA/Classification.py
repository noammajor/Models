"""
LE-JEPA linear-probe classification.

Frozen pretrained encoder + ClassificationHead trained on the classification split.
Head: last patch → flatten(n_vars * embed_dim) → dropout → linear → n_classes.
Identical pattern to PatchTST's ClassificationHead.
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Import ClassificationHead from JEPA (shared)
_JEPA_DIR = str(Path(__file__).parent.parent / "JEPA" / "JEPA")
if _JEPA_DIR not in sys.path:
    sys.path.insert(0, _JEPA_DIR)

from Decoder import ClassificationHead


def _instance_norm(x, eps=1e-6):
    """Per-instance, per-variable normalization. x: [B, P, PL, n_vars]"""
    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2),  keepdim=True) + eps
    return (x - mean) / std, mean, std


def classification_zeroshot(self, path, classification_train, classification_val,
                             classification_test, n_classes):
    """
    Linear-probe classification with frozen LE-JEPA encoder.

    Args:
        path                  : checkpoint tag, e.g. "" → loads {path_save}{path}best_model.pt
        classification_train/val/test : DataLoaders from ClassificationDataPuller
                                        each batch: (patches [B, P, PL, n_vars], labels [B])
        n_classes             : total number of target classes
    Returns:
        test accuracy (float)
    """
    config          = self.config
    checkpoint_path = f"{self.path_save}{path}best_model.pt"

    print(f"\n=== LE-JEPA Classification (linear probe) ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    self.encoder.load_state_dict(ckpt["encoder"])
    self.encoder.to(self.device)
    self.encoder.eval()
    for p in self.encoder.parameters():
        p.requires_grad = False

    embed_dim = config["encoder_embed_dim"]

    # Infer n_vars and num_patches from first batch
    patch_len = config.get("patch_size_forcasting", 16)
    sample_patches, _ = next(iter(classification_train))
    if sample_patches.dim() == 3:
        # Raw (B, T, C) from UEADataset — patch to (B, P, patch_len, C)
        B0, T0, C0 = sample_patches.shape
        T_pad = ((T0 + patch_len - 1) // patch_len) * patch_len
        sample_patches = F.pad(sample_patches, (0, 0, 0, T_pad - T0))
        sample_patches = sample_patches.reshape(B0, T_pad // patch_len, patch_len, C0)
    num_patches = sample_patches.shape[1]
    n_v         = sample_patches.shape[-1]

    cls_head = ClassificationHead(
        n_vars       = n_v,
        d_model      = embed_dim,
        n_classes    = n_classes,
        head_dropout = config.get("head_dropout", 0.1),
    ).to(self.device)

    n_epochs  = config.get("epoch_classification", 20)
    optimizer = torch.optim.Adam(cls_head.parameters(),
                                 lr=config.get("lr_classification", 1e-3),
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.get("lr_classification", 1e-3),
        total_steps=n_epochs * len(classification_train),
        pct_start=0.3, anneal_strategy='cos',
    )

    def _to_patches(x):
        """Convert raw (B, T, C) → (B, P, patch_len, C) if needed."""
        if x.dim() == 3:
            B_, T_, C_ = x.shape
            T_pad_ = ((T_ + patch_len - 1) // patch_len) * patch_len
            x = F.pad(x, (0, 0, 0, T_pad_ - T_))
            x = x.reshape(B_, T_pad_ // patch_len, patch_len, C_)
        return x

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(n_epochs):
        cls_head.train()
        correct, total = 0, 0
        for patches, labels in classification_train:
            patches = _to_patches(patches).to(self.device)
            labels  = labels.to(self.device)
            B, P, PL, n_v_ = patches.shape

            optimizer.zero_grad()
            ctx_norm, _, _ = _instance_norm(patches)
            with torch.no_grad():
                enc_out = self.encoder(ctx_norm)
                enc     = enc_out["data_patches"]          # [B*n_v, P, embed_dim]
            enc_p  = enc.reshape(B, n_v_, num_patches, embed_dim).permute(0, 1, 3, 2)
            logits = cls_head(enc_p)                       # [B, n_classes]
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += len(labels)
        train_acc = correct / total

        # Validation
        cls_head.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for patches, labels in classification_val:
                patches = _to_patches(patches).to(self.device)
                labels  = labels.to(self.device)
                B, P, PL, n_v_ = patches.shape
                ctx_norm, _, _ = _instance_norm(patches)
                enc_out = self.encoder(ctx_norm)
                enc     = enc_out["data_patches"]
                enc_p   = enc.reshape(B, n_v_, num_patches, embed_dim).permute(0, 1, 3, 2)
                logits  = cls_head(enc_p)
                vc += (logits.argmax(1) == labels).sum().item()
                vt += len(labels)
        val_acc = vc / vt
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in cls_head.state_dict().items()}
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")

    # Test with best checkpoint
    if best_state is not None:
        cls_head.load_state_dict(best_state)
    cls_head.eval()
    tc, tt = 0, 0
    with torch.no_grad():
        for patches, labels in classification_test:
            patches = _to_patches(patches).to(self.device)
            labels  = labels.to(self.device)
            B, P, PL, n_v_ = patches.shape
            ctx_norm, _, _ = _instance_norm(patches)
            enc_out = self.encoder(ctx_norm)
            enc     = enc_out["data_patches"]
            enc_p   = enc.reshape(B, n_v_, num_patches, embed_dim).permute(0, 1, 3, 2)
            logits  = cls_head(enc_p)
            tc += (logits.argmax(1) == labels).sum().item()
            tt += len(labels)
    test_acc = tc / tt
    print(f"[LE-JEPA] Test Accuracy: {test_acc:.4f}  (best val: {best_val_acc:.4f})")
    return test_acc
