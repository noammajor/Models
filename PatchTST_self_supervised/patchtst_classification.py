"""
PatchTST classification.

Pretrained PatchTST backbone (frozen by default; unfrozen if linear_probe=False)
+ ClassificationHead trained on the classification split.
Head: last patch → flatten(n_vars * d_model) → dropout → linear → n_classes.
Identical pattern to DINO/NTP ClassificationHead.
"""

import os
import sys
import torch
import torch.nn.functional as F

_DIR     = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_DIR)
_SHARED   = os.path.join(_ROOT, "shared")

for _p in [_DIR, _ROOT, _SHARED]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models.patchTST import PatchTST


def classification(config, checkpoint_path, classification_train,
                   classification_val, classification_test, n_classes,
                   linear_probe=True):
    """
    Classification with PatchTST backbone.

    Args:
        config                : dict from config_patchtst.py
        checkpoint_path       : path to pretrained .pth file
        classification_train/val/test : DataLoaders from ClassificationDataPuller
                                        each batch: (patches [B, P, PL, n_vars], labels [B])
        n_classes             : total number of target classes
        linear_probe          : if True, freeze backbone and train only head. If False, fine-tune backbone + head.
    Returns:
        test accuracy (float)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== PatchTST Classification ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    # Infer shape from first batch
    sample_patches, _, _ = next(iter(classification_train))
    num_patch = sample_patches.shape[1]   # [B, P, PL, n_vars]
    n_v       = sample_patches.shape[-1]
    patch_len = sample_patches.shape[2]

    # Build model with classification head
    model = PatchTST(
        c_in         = n_v,
        target_dim   = n_classes,
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
        head_type    = "classification",
        res_attention = False,
    ).to(device)

    # Load pretrained backbone weights (skip head and shape mismatches)
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_dict and model_dict[k].shape == v.shape
                    and "head" not in k}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"  Loaded {len(filtered)}/{len(model_dict)} parameters from checkpoint")
    else:
        print("  Random encoder — skipping checkpoint load")

    # Freeze backbone (linear probe) or fine-tune full model
    if linear_probe:
        for name, p in model.named_parameters():
            p.requires_grad = ("head" in name)
        print(f"  [PatchTST classify] MODE: linear probe — encoder FROZEN")
    else:
        for p in model.parameters():
            p.requires_grad = True
        print(f"  [PatchTST classify] MODE: full fine-tuning — encoder UNFROZEN")
    _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {_trainable:,} / {_total:,} params")

    n_epochs    = config.get("epoch_classification", 20)
    # LR priority: config (user-set) > script default.
    _cfg_head_lr = config.get("lr_classification")
    _cfg_enc_lr  = config.get("lr_classification_encoder")
    head_lr = float(_cfg_head_lr) if _cfg_head_lr is not None else 1e-3
    enc_lr  = float(_cfg_enc_lr)  if _cfg_enc_lr  is not None else head_lr
    if linear_probe:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=head_lr, weight_decay=1e-4,
        )
        _max_lrs = head_lr
    else:
        head_params = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]
        enc_params  = [p for n, p in model.named_parameters() if "head" not in n and p.requires_grad]
        optimizer = torch.optim.Adam([
            {"params": head_params, "lr": head_lr},
            {"params": enc_params,  "lr": enc_lr},
        ], weight_decay=1e-4)
        _max_lrs = [head_lr, enc_lr]
        print(f"  [PatchTST classify] head_lr={head_lr}  encoder_lr={enc_lr}")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=_max_lrs,
        total_steps=n_epochs * len(classification_train),
        pct_start=0.3, anneal_strategy='cos',
    )

    for epoch in range(n_epochs):
        model.train()
        correct, total = 0, 0
        for patches, labels, padding_mask in classification_train:
            # PatchTST expects [B, P, n_vars, PL]
            x            = patches.permute(0, 1, 3, 2).to(device)
            labels       = labels.to(device)
            padding_mask = padding_mask.to(device)   # [B, P] bool
            optimizer.zero_grad()
            logits = model(x, padding_mask=padding_mask)
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += len(labels)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | train acc {correct/total:.4f}")

    model.eval()
    tc, tt = 0, 0
    with torch.no_grad():
        for patches, labels, padding_mask in classification_test:
            x            = patches.permute(0, 1, 3, 2).to(device)
            labels       = labels.to(device)
            padding_mask = padding_mask.to(device)
            logits = model(x, padding_mask=padding_mask)
            tc += (logits.argmax(1) == labels).sum().item()
            tt += len(labels)
    test_acc = tc / tt
    print(f"[PatchTST] Test Accuracy: {test_acc:.4f}")
    return test_acc
