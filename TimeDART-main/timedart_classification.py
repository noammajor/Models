"""
TimeDaRT classification.

Pretrained TimeDaRT (Model) backbone (frozen by default; unfrozen if
linear_probe=False) + ClassificationHead.
Identical pattern to JEPA / LE-JEPA / NTP / PatchTST:
  - Channel-independent PatchEmbedding (same as pretrain)
  - CausalTransformer encoder
  - ClassificationHead: last patch → flatten(n_vars * d_model) → dropout → linear → n_classes
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace

_DIR = Path(__file__).parent
_ROOT = _DIR.parent

for _p in [str(_DIR), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.TimeDART import Model
from utils.tools import transfer_weights


class ClassificationHead(nn.Module):
    """Last patch → flatten(n_vars * d_model) → dropout → linear → n_classes."""

    def __init__(self, n_vars, d_model, n_classes, head_dropout, mlp_head: bool = False, hidden_dim: int = 512):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        if mlp_head:
            self.linear = nn.Sequential(
                nn.Linear(n_vars * d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            self.linear  = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        """
        x: [B, n_vars, d_model, n_patches]
        returns: [B, n_classes]
        """
        x = x[:, :, :, -1]        # last patch: [B, n_vars, d_model]
        x = self.flatten(x)        # [B, n_vars * d_model]
        x = self.dropout(x)
        return self.linear(x)


def _build_model_args(config, n_vars, device):
    patch_len = config.get("patch_len", 16)
    stride    = config.get("stride",    patch_len)
    input_len = 72 * stride          # matches exactly 72 patches from our collate
    return SimpleNamespace(
        input_len    = input_len,
        d_model      = config.get("d_model",   256),
        n_heads      = config.get("n_heads",   8),
        d_ff         = config.get("d_ff",      512),
        dropout      = config.get("dropout",   0.1),
        head_dropout = config.get("head_dropout", 0.1),
        device       = device,
        task_name    = "pretrain",   # builds encoder + pretrain head (head unused)
        pred_len     = 0,
        use_norm     = config.get("use_norm", True),
        patch_len    = patch_len,
        stride       = stride,
        time_steps   = config.get("time_steps", 1000),
        scheduler    = config.get("scheduler",  "cosine"),
        mask_ratio   = config.get("mask_ratio", 1.0),
        e_layers     = config.get("e_layers",   3),
        d_layers     = config.get("d_layers",   1),
        enc_in       = n_vars,
        dec_in       = n_vars,
        c_out        = n_vars,
    )


def _encode(model, x, padding_mask=None):
    """Run encoder-only path of Model (no diffusion, no decoder, no head).

    x:            [B, T, C]
    padding_mask: [B, P] bool, True=real data (optional)
    returns:      [B, C, d_model, P]  — matches ClassificationHead interface
    """
    B, T, C = x.shape
    if model.use_norm:
        means  = x.mean(dim=1, keepdim=True).detach()
        x      = x - means
        stdevs = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x      = x / stdevs
    x = model.channel_independence(x)   # [B*C, T, 1]
    x = model.patch(x)                  # [B*C, P, patch_len]
    P = x.shape[1]
    x = model.enc_embedding(x)          # [B*C, P, d_model]
    x = model.positional_encoding(x)    # [B*C, P, d_model]

    # Expand padding mask from [B, P] to [B*C, P] and invert for PyTorch convention
    kpm = None
    if padding_mask is not None:
        pm  = padding_mask.to(x.device).bool()              # [B, P], True=real
        kpm = (~pm).unsqueeze(1).expand(-1, C, -1).reshape(B * C, P)  # [B*C, P], True=ignore

    x = model.encoder(x, is_mask=False, key_padding_mask=kpm)  # [B*C, P, d_model]
    x = x.reshape(B, C, P, model.d_model)   # [B, C, P, d_model]
    return x.permute(0, 1, 3, 2)            # [B, C, d_model, P]


def classification(config, checkpoint_path,
                   classification_train, classification_val,
                   classification_test, n_classes,
                   linear_probe=True, mlp_head: bool = False):
    """
    Classification with TimeDaRT encoder.

    Args:
        config               : dict from config_timedart.py
        checkpoint_path      : path to pretrained ckpt_best.pth
        classification_train/val/test : DataLoaders (3-tuple batches)
                               each batch: (patches [B, P, PL, n_vars], labels [B],
                                            padding_mask [B, P])
        n_classes            : total number of target classes
        linear_probe         : if True, freeze backbone and train only head. If False, fine-tune backbone + head.
    Returns:
        test accuracy (float)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== TimeDaRT Classification ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    sample_patches, _, _ = next(iter(classification_train))
    n_v = sample_patches.shape[-1]   # [B, P, PL, n_vars]

    args  = _build_model_args(config, n_v, device)
    model = Model(args).float().to(device)

    # Load encoder + enc_embedding (PatchEmbedding — same arch as pretrain)
    if os.path.exists(checkpoint_path):
        model = transfer_weights(checkpoint_path, model, exclude_head=True, device=str(device))
    else:
        print(f"  WARNING: checkpoint not found at {checkpoint_path}, using random init")

    if linear_probe:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print(f"  [TimeDaRT classify] MODE: linear probe — encoder FROZEN")
    else:
        model.train()
        print(f"  [TimeDaRT classify] MODE: full fine-tuning — encoder UNFROZEN")

    d_model = config.get("d_model", 256)
    cls_head = ClassificationHead(
        n_vars=n_v, d_model=d_model,
        n_classes=n_classes,
        head_dropout=config.get("head_dropout", 0.1),
        mlp_head=mlp_head,
    ).to(device)

    n_epochs    = config.get("epoch_classification", 20)
    _all_params = list(model.parameters()) + list(cls_head.parameters())
    # LR priority: config (user-set) > script default.
    _cfg_head_lr = config.get("lr_classification")
    _cfg_enc_lr  = config.get("lr_classification_encoder")
    head_lr = float(_cfg_head_lr) if _cfg_head_lr is not None else 1e-3
    enc_lr  = float(_cfg_enc_lr)  if _cfg_enc_lr  is not None else head_lr
    if linear_probe:
        optimizer = torch.optim.Adam(cls_head.parameters(),
                                     lr=head_lr, weight_decay=1e-4)
        _max_lrs   = head_lr
        _trainable = sum(p.numel() for p in cls_head.parameters())
    else:
        optimizer = torch.optim.Adam([
            {"params": cls_head.parameters(), "lr": head_lr},
            {"params": model.parameters(),    "lr": enc_lr},
        ], weight_decay=1e-4)
        _max_lrs   = [head_lr, enc_lr]
        _trainable = sum(p.numel() for p in _all_params)
        print(f"  [TimeDart classify] head_lr={head_lr}  encoder_lr={enc_lr}")
    _total = sum(p.numel() for p in _all_params)
    print(f"  Trainable: {_trainable:,} / {_total:,} params")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=_max_lrs,
        total_steps=n_epochs * len(classification_train),
        pct_start=0.3, anneal_strategy='cos',
    )

    def _to_flat(patches):
        B, P, PL, C = patches.shape
        return patches.reshape(B, P * PL, C)

    for epoch in range(n_epochs):
        cls_head.train()
        if not linear_probe:
            model.train()
        correct, total = 0, 0
        for patches, labels, padding_mask in classification_train:
            x      = _to_flat(patches).float().to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(not linear_probe):
                enc = _encode(model, x, padding_mask)  # [B, C, d_model, P]
            logits = cls_head(enc)         # [B, n_classes]
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
        for patches, labels, padding_mask in classification_test:
            x      = _to_flat(patches).float().to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)
            enc    = _encode(model, x, padding_mask)
            logits = cls_head(enc)
            tc += (logits.argmax(1) == labels).sum().item()
            tt += len(labels)
    test_acc = tc / tt
    print(f"[TimeDaRT] Test Accuracy: {test_acc:.4f}")
    return test_acc
