"""
NPT linear-probe classification.

Frozen pretrained NPT (PatchTST) backbone + ClassificationHead trained on the
classification split. Head: last patch → flatten(n_vars * d_model) → dropout
→ linear → n_classes. Identical pattern to PatchTST's ClassificationHead.
"""

import os
import sys
import torch
import torch.nn.functional as F

_NPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_NPT_DIR)
_DJEPA_DIR = os.path.join(_ROOT_DIR, "Discrete_JEPA")

for _p in [_NPT_DIR, _ROOT_DIR, _DJEPA_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.patchTST import PatchTST, ClassificationHead


def _build_backbone(config, c_in, num_patch, device):
    """Build PatchTST backbone with pretrain head (weights will be loaded)."""
    return PatchTST(
        c_in=c_in,
        target_dim=config["patch_size"],   # pretrain head dim — overwritten after load
        patch_len=config["patch_size"],
        stride=config["patch_size"],
        num_patch=num_patch,
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_model=config["d_model"],
        shared_embedding=True,
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        head_dropout=config["head_dropout"],
        act=config["act"],
        head_type="pretrain",
        causal=True,
        res_attention=False,
    ).to(device)


def classification_zeroshot(config, checkpoint_path, classification_train,
                             classification_val, classification_test, n_classes):
    """
    Linear-probe classification with frozen NPT backbone.

    Args:
        config                : dict from config_ntp.py
        checkpoint_path       : path to pretrained .pt file
        classification_train/val/test : DataLoaders from ClassificationDataPuller
                                        each batch: (patches [B, P, PL, n_vars], labels [B])
        n_classes             : total number of target classes
    Returns:
        test accuracy (float)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== NPT Classification (linear probe) ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    # Infer n_vars and num_patches from first batch
    sample_patches, _ = next(iter(classification_train))
    num_patch = sample_patches.shape[1]
    n_v       = sample_patches.shape[-1]   # [B, P, PL, n_vars]

    # Build backbone and load pretrained weights
    backbone = _build_backbone(config, c_in=n_v, num_patch=num_patch, device=device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    backbone.load_state_dict(state, strict=False)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    d_model = config["d_model"]
    cls_head = ClassificationHead(
        n_vars       = n_v,
        d_model      = d_model,
        n_classes    = n_classes,
        head_dropout = config.get("head_dropout", 0.1),
    ).to(device)

    n_epochs  = config.get("epoch_classification", 20)
    optimizer = torch.optim.Adam(cls_head.parameters(),
                                 lr=config.get("lr_classification", 1e-3),
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.get("lr_classification", 1e-3),
        total_steps=n_epochs * len(classification_train),
        pct_start=0.3, anneal_strategy='cos',
    )

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(n_epochs):
        cls_head.train()
        correct, total = 0, 0
        for patches, labels in classification_train:
            patches = patches.to(device)   # [B, P, PL, n_vars]
            labels  = labels.to(device)
            # PatchTST expects [B, P, n_vars, PL]
            x = patches.permute(0, 1, 3, 2)
            optimizer.zero_grad()
            with torch.no_grad():
                # backbone returns [B, n_vars, d_model, P] with pretrain head replaced
                enc = backbone.backbone(x)    # [B, n_vars, d_model, P]
            logits = cls_head(enc)            # [B, n_classes]
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
                patches = patches.to(device)
                labels  = labels.to(device)
                x   = patches.permute(0, 1, 3, 2)
                enc = backbone.backbone(x)
                logits = cls_head(enc)
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
            patches = patches.to(device)
            labels  = labels.to(device)
            x   = patches.permute(0, 1, 3, 2)
            enc = backbone.backbone(x)
            logits = cls_head(enc)
            tc += (logits.argmax(1) == labels).sum().item()
            tt += len(labels)
    test_acc = tc / tt
    print(f"[NPT] Test Accuracy: {test_acc:.4f}  (best val: {best_val_acc:.4f})")
    return test_acc
