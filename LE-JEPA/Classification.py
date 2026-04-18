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

    patch_len = config.get("patch_size_forcasting", 16)

    # Infer from first batch — may already be pre-patched (B, P, PL, C) or raw (B, T, C)
    sample_patches, _ = next(iter(classification_train))
    if sample_patches.dim() == 4:
        num_patches = sample_patches.shape[1]
        target_T    = num_patches * patch_len
        n_v         = sample_patches.shape[-1]
    else:
        ds = classification_train.dataset
        max_T = max(s.shape[0] for s in ds._samples) if hasattr(ds, '_samples') else patch_len
        target_T    = ((max_T + patch_len - 1) // patch_len) * patch_len
        num_patches = target_T // patch_len
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
        """Resample (B, T, C) to fixed target_T then reshape to (B, num_patches, patch_len, C)."""
        if x.dim() == 3:
            B_, T_, C_ = x.shape
            if T_ != target_T:
                idx = torch.linspace(0, T_ - 1, target_T, device=x.device).long()
                x = x[:, idx, :]
            x = x.reshape(B_, num_patches, patch_len, C_)
        return x

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
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | train acc {correct/total:.4f}")
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
    print(f"[LE-JEPA] Test Accuracy: {test_acc:.4f}")
    return test_acc
