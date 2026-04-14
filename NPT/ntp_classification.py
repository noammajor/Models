"""
NPT linear-probe classification — TSLib style.

Frozen PatchTST backbone → [B, n_vars, d_model, n_patches] → masked mean pool → linear head.
Training: RAdam + gradient clipping (max_norm=4.0) + early stopping on -val_accuracy.
"""

import os
import sys
import torch

_NPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_NPT_DIR)
_DJEPA_DIR = os.path.join(_ROOT_DIR, "Discrete_JEPA")

for _p in [_NPT_DIR, _ROOT_DIR, _DJEPA_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.patchTST import PatchTST
from classification_utils import train_cls_tslib


def _build_backbone(config, c_in, num_patch, device):
    return PatchTST(
        c_in         = c_in,
        target_dim   = config["patch_size"],
        patch_len    = config["patch_size"],
        stride       = config["patch_size"],
        num_patch    = num_patch,
        n_layers     = config["n_layers"],
        n_heads      = config["n_heads"],
        d_model      = config["d_model"],
        shared_embedding = True,
        d_ff         = config["d_ff"],
        dropout      = config["dropout"],
        head_dropout = config["head_dropout"],
        act          = config["act"],
        head_type    = "pretrain",
        causal       = True,
        res_attention= False,
    ).to(device)


def classification_zeroshot(config, checkpoint_path, classification_train,
                             classification_val, classification_test, n_classes):
    """
    Linear-probe classification with frozen NPT backbone (TSLib style).

    Batches: (patches [B,P,PL,n_vars], labels [B]) or 3-tuple  (x, labels, mask).
    - Pre-patched 4D input  [B, P, PL, n_vars] comes from ClassificationDataPuller.
    - Raw 3D input          [B, T, C]           comes from make_uea_dataloaders.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== NPT Classification (TSLib linear probe) ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    patch_len = config["patch_size"]

    # Infer input shape from first batch
    sample = next(iter(classification_train))[0]
    if sample.dim() == 4:                       # pre-patched [B, P, PL, n_vars]
        num_patch = sample.shape[1]
        n_v       = sample.shape[-1]
    else:                                        # raw [B, T, C]
        T         = sample.shape[1]
        n_v       = sample.shape[-1]
        num_patch = T // patch_len

    backbone = _build_backbone(config, c_in=n_v, num_patch=num_patch, device=device)
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    backbone.load_state_dict(state, strict=False)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    def encode_fn(x, key_padding_mask=None):
        if x.dim() == 4:                        # [B, P, PL, n_vars]
            x = x.permute(0, 1, 3, 2)          # [B, P, n_vars, PL]
        else:                                    # [B, T, C]
            B, T, C = x.shape
            n_p = T // patch_len
            x = x.reshape(B, n_p, patch_len, C).permute(0, 1, 3, 2)  # [B, n_p, C, PL]
        return backbone.backbone(x, key_padding_mask=key_padding_mask,
                                 method_classification=True)           # [B, n_vars, d_model, n_patches]

    return train_cls_tslib(
        encode_fn    = encode_fn,
        n_classes    = n_classes,
        train_loader = classification_train,
        val_loader   = classification_val,
        test_loader  = classification_test,
        n_epochs     = config.get("epoch_classification",   50),
        lr           = config.get("lr_classification",      1e-4),
        patience     = config.get("patience_classification", 10),
        head_dropout = config.get("head_dropout",           0.1),
        device       = device,
    )
