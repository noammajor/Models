"""
PatchTST linear-probe classification — TSLib style.

Frozen PatchTST backbone → [B, n_vars, d_model, n_patches] → masked mean pool → linear head.
Training: RAdam + gradient clipping (max_norm=4.0) + early stopping on -val_accuracy.
"""

import os
import sys
import torch

_DIR     = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_DIR)
_DJEPA   = os.path.join(_ROOT, "Discrete_JEPA")

for _p in [_DIR, _ROOT, _DJEPA]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models.patchTST import PatchTST
from classification_utils import train_cls_tslib


def classification_zeroshot(config, checkpoint_path, classification_train,
                             classification_val, classification_test, n_classes):
    """
    Linear-probe classification with frozen PatchTST backbone (TSLib style).

    Batches: (patches [B,P,PL,n_vars], labels [B]) or 3-tuple  (x, labels, mask).
    - Pre-patched 4D input  [B, P, PL, n_vars] comes from ClassificationDataPuller.
    - Raw 3D input          [B, T, C]           comes from make_uea_dataloaders.
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_len = config.get("patch_len", 16)

    print(f"\n=== PatchTST Classification (TSLib linear probe) ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    # Infer input shape from first batch
    sample = next(iter(classification_train))[0]
    if sample.dim() == 4:                       # pre-patched [B, P, PL, n_vars]
        num_patch = sample.shape[1]
        n_v       = sample.shape[-1]
    else:                                        # raw [B, T, C]
        T         = sample.shape[1]
        n_v       = sample.shape[-1]
        num_patch = T // patch_len

    model = PatchTST(
        c_in         = n_v,
        target_dim   = n_classes,
        patch_len    = patch_len,
        stride       = patch_len,
        num_patch    = num_patch,
        n_layers     = config.get("n_layers",     3),
        n_heads      = config.get("n_heads",      16),
        d_model      = config.get("d_model",      128),
        shared_embedding = True,
        d_ff         = config.get("d_ff",         512),
        dropout      = config.get("dropout",      0.2),
        head_dropout = config.get("head_dropout", 0.2),
        act          = "gelu",
        head_type    = "pretrain",
        res_attention= False,
    ).to(device)

    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Loaded backbone — missing: {len(missing)}, unexpected: {len(unexpected)}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def encode_fn(x, key_padding_mask=None):
        if x.dim() == 4:                        # [B, P, PL, n_vars]
            x = x.permute(0, 1, 3, 2)          # [B, P, n_vars, PL]
        else:                                    # [B, T, C]
            B, T, C = x.shape
            n_p = T // patch_len
            x = x.reshape(B, n_p, patch_len, C).permute(0, 1, 3, 2)  # [B, n_p, C, PL]
        return model.model(x, key_padding_mask=key_padding_mask,
                           method_classification=True)                  # [B, n_vars, d_model, n_patches]

    return train_cls_tslib(
        encode_fn    = encode_fn,
        n_classes    = n_classes,
        train_loader = classification_train,
        val_loader   = classification_val,
        test_loader  = classification_test,
        n_epochs     = config.get("epoch_classification",   50),
        lr           = config.get("lr_classification",      1e-4),
        patience     = config.get("patience_classification", 10),
        head_dropout = config.get("head_dropout",           0.2),
        device       = device,
    )
