"""
TimeDaRT linear-probe classification.

Uses TimeDaRT's own ClsModel (task_name="finetune") with:
  - Pretrained encoder loaded from pretrain checkpoint via transfer_weights
  - Encoder frozen (linear probe)
  - ClsEmbedding + OldClsHead trained on the classification split

ClsEmbedding is a Conv1d on the raw multivariate sequence, so its weights
are always randomly initialised (different arch from the pretrain PatchEmbedding).
Only the CausalTransformer encoder is transferred from the pretrain checkpoint.
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace

_DIR = Path(__file__).parent
_ROOT = _DIR.parent

for _p in [str(_DIR), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.TimeDART import ClsModel
from utils.tools import transfer_weights


def _build_args(config, n_vars, n_classes, device):
    patch_len = config.get("patch_len", 16)
    stride    = config.get("stride",    patch_len)
    # seq_len inside ClsModel = int((input_len - patch_len) / stride) + 2
    # With 72 patches: (input_len - patch_len) / stride + 2 = 72
    #   → input_len = (72 - 2) * stride + patch_len = 70*stride + patch_len
    input_len = 70 * stride + patch_len
    return SimpleNamespace(
        input_len    = input_len,
        d_model      = config.get("d_model",     256),
        n_heads      = config.get("n_heads",     8),
        d_ff         = config.get("d_ff",        512),
        dropout      = config.get("dropout",     0.1),
        head_dropout = config.get("head_dropout", 0.1),
        device       = device,
        task_name    = "finetune",   # builds ClsEmbedding + OldClsHead
        pred_len     = 0,            # unused for classification
        use_norm     = config.get("use_norm", False),
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
        num_classes  = n_classes,
    )


def classification_zeroshot(config, checkpoint_path,
                             classification_train, classification_val,
                             classification_test, n_classes):
    """
    Linear-probe classification with frozen TimeDaRT encoder.

    Args:
        config               : dict from config_timedart.py
        checkpoint_path      : path to pretrained ckpt_best.pth
        classification_train/val/test : DataLoaders (3-tuple batches)
                               each batch: (patches [B, P, PL, n_vars], labels [B],
                                            padding_mask [B, P])
        n_classes            : total number of target classes
    Returns:
        test accuracy (float)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== TimeDaRT Classification (linear probe) ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    # Infer n_vars from first batch
    sample_patches, _, _ = next(iter(classification_train))
    n_v = sample_patches.shape[-1]   # [B, P, PL, n_vars]

    args  = _build_args(config, n_v, n_classes, device)
    model = ClsModel(args).float().to(device)

    # Load pretrained encoder weights (enc_embedding will show as unmatched — different arch)
    if os.path.exists(checkpoint_path):
        model = transfer_weights(checkpoint_path, model, exclude_head=True, device=str(device))
    else:
        print(f"  WARNING: checkpoint not found at {checkpoint_path}, using random init")

    # Freeze the encoder (pretrained); train ClsEmbedding + head
    for name, param in model.named_parameters():
        param.requires_grad = ("encoder" not in name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} parameters")

    n_epochs  = config.get("epoch_classification", 20)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.get("lr_classification", 1e-3),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.get("lr_classification", 1e-3),
        total_steps=n_epochs * len(classification_train),
        pct_start=0.3, anneal_strategy='cos',
    )

    def _to_flat(patches):
        """[B, P, PL, C] → [B, P*PL, C]"""
        B, P, PL, C = patches.shape
        return patches.reshape(B, P * PL, C)

    for epoch in range(n_epochs):
        model.train()
        # keep encoder frozen even in train mode
        for name, module in model.named_modules():
            if "encoder" in name:
                module.eval()
        correct, total_n = 0, 0
        for patches, labels, padding_mask in classification_train:
            x      = _to_flat(patches).float().to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(x)          # ClsModel.forward → forecast → OldClsHead
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total_n += len(labels)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | train acc {correct/total_n:.4f}")

    model.eval()
    tc, tt = 0, 0
    with torch.no_grad():
        for patches, labels, padding_mask in classification_test:
            x      = _to_flat(patches).float().to(device)
            labels = labels.to(device)
            logits = model(x)
            tc += (logits.argmax(1) == labels).sum().item()
            tt += len(labels)
    test_acc = tc / tt
    print(f"[TimeDaRT] Test Accuracy: {test_acc:.4f}")
    return test_acc
