"""
LE-JEPA linear-probe classification — TSLib style.

Frozen encoder → [B, n_vars, embed_dim, n_patches] → masked mean pool → linear head.
Training: RAdam + gradient clipping (max_norm=4.0) + early stopping on -val_accuracy.
"""

import sys
import torch
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from classification_utils import train_cls_tslib

_JEPA_DIR = str(Path(__file__).parent.parent / "JEPA" / "JEPA")
if _JEPA_DIR not in sys.path:
    sys.path.insert(0, _JEPA_DIR)

from Decoder import ClassificationHead  # kept for compat; not used directly


def _instance_norm(x, eps=1e-6):
    """Per-instance, per-variable normalization. x: [B, P, PL, n_vars]"""
    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2),  keepdim=True) + eps
    return (x - mean) / std, mean, std


def classification_zeroshot(self, path, classification_train, classification_val,
                             classification_test, n_classes):
    """
    Linear-probe classification with frozen LE-JEPA encoder (TSLib style).

    Batches from loader: (patches [B,P,PL,n_vars], labels [B]) or 3-tuple with mask.
    """
    config          = self.config
    checkpoint_path = f"{self.path_save}{path}best_model.pt"

    print(f"\n=== LE-JEPA Classification (TSLib linear probe) ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    self.encoder.load_state_dict(ckpt["encoder"])
    self.encoder.to(self.device)
    self.encoder.eval()
    for p in self.encoder.parameters():
        p.requires_grad_(False)

    embed_dim = config["encoder_embed_dim"]
    encoder   = self.encoder
    device    = self.device

    # Infer n_vars and num_patches from first batch
    sample = next(iter(classification_train))[0]
    if sample.dim() == 3:
        sample = sample.unsqueeze(-1)
    num_patches = sample.shape[1]

    def encode_fn(patches, key_padding_mask=None):
        # patches: [B, P, PL, n_vars]
        if patches.dim() == 3:
            patches = patches.unsqueeze(-1)
        B, P, PL, n_v = patches.shape
        ctx_norm, _, _ = _instance_norm(patches)
        enc_out = encoder(ctx_norm, key_padding_mask=key_padding_mask, method_classification=True)
        enc     = enc_out["data_patches"]           # [B*n_v, P, embed_dim]
        # reshape to [B, n_vars, embed_dim, n_patches]
        return enc.reshape(B, n_v, P, embed_dim).permute(0, 1, 3, 2)

    return train_cls_tslib(
        encode_fn          = encode_fn,
        n_classes          = n_classes,
        train_loader       = classification_train,
        val_loader         = classification_val,
        test_loader        = classification_test,
        n_epochs           = config.get("epoch_classification",   50),
        lr                 = config.get("lr_classification",      1e-4),
        patience           = config.get("patience_classification", 10),
        head_dropout       = config.get("head_dropout",           0.1),
        device             = device,
    )
