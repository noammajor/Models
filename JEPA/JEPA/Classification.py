"""
JEPA linear-probe classification.

Frozen pretrained encoder + ClassificationHead trained on the classification split.
Head: last patch → flatten(n_vars * embed_dim) → dropout → linear → n_classes.
Identical pattern to PatchTST's ClassificationHead.
"""

import torch
import torch.nn.functional as F

from JEPA.Decoder import ClassificationHead
from JEPA.Training import _instance_norm


def classification_zeroshot(self, path, classification_train, classification_val,
                             classification_test, n_classes,
                             checkpoint_path_override=None,
                             random_encoder=False,
                             linear_probe=True):
    """
    Linear-probe classification with frozen JEPA encoder.

    Args:
        path                  : checkpoint tag, e.g. "" → loads {path_save}{path}best_model.pt
        classification_train/val/test : DataLoaders from ClassificationDataPuller
                                        each batch: (patches [B, P, PL, n_vars], labels [B])
        n_classes             : total number of target classes
    Returns:
        test accuracy (float)
    """
    config          = self.config
    checkpoint_path = checkpoint_path_override if checkpoint_path_override is not None \
                      else f"{self.path_save}{path}best_model.pt"

    print(f"\n=== JEPA Classification (linear probe) ===")
    if random_encoder:
        print("Random encoder — skipping checkpoint load")
    else:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.encoder_for.load_state_dict(ckpt["target_encoder"])
    self.encoder_for.to(self.device)
    if linear_probe:
        self.encoder_for.eval()
        for p in self.encoder_for.parameters():
            p.requires_grad = False
        print(f"  [JEPA classify] MODE: linear probe — encoder FROZEN")
    else:
        print(f"  [JEPA classify] MODE: full fine-tuning — encoder UNFROZEN")

    embed_dim = config["encoder_embed_dim"]
    encoder_num_patches = self.encoder_for.W_pos.shape[0]

    patch_len = config.get("patch_size", 16)

    # Infer from first batch — may already be pre-patched (B, P, PL, C) or raw (B, T, C)
    sample_patches, _, _ = next(iter(classification_train))
    if sample_patches.dim() == 4:
        # Pre-patched by _patch_collate
        num_patches = sample_patches.shape[1]
        target_T    = num_patches * patch_len
        n_v         = sample_patches.shape[-1]
    else:
        # Raw (B, T, C) — fix target_T from dataset max to be consistent across batches
        ds = classification_train.dataset
        max_T = max(s.shape[0] for s in ds._samples) if hasattr(ds, '_samples') else patch_len
        target_T    = ((max_T + patch_len - 1) // patch_len) * patch_len
        num_patches = target_T // patch_len
        n_v         = sample_patches.shape[-1]

    def _to_patches(x, padding_mask=None):
        """Resample (B, T, C) to fixed target_T then reshape to (B, num_patches, patch_len, C).
        Always subsamples patch dimension to encoder_num_patches to match encoder PE.
        Returns (x, padding_mask) where padding_mask is (B, P) bool, True = real data."""
        if x.dim() == 3:
            B_, T_, C_ = x.shape
            if T_ != target_T:
                idx = torch.linspace(0, T_ - 1, target_T, device=x.device).long()
                x = x[:, idx, :]
                if padding_mask is not None:
                    real_count = padding_mask.to(x.device).sum(dim=1)        # (B,)
                    patch_idx  = idx[torch.arange(num_patches, device=x.device) * patch_len]
                    padding_mask = patch_idx.unsqueeze(0) < real_count.unsqueeze(1)
            else:
                if padding_mask is not None and padding_mask.shape[1] != num_patches:
                    real_count   = padding_mask.to(x.device).sum(dim=1)
                    patch_starts = torch.arange(num_patches, device=x.device) * patch_len
                    padding_mask = patch_starts.unsqueeze(0) < real_count.unsqueeze(1)
            x = x.reshape(B_, num_patches, patch_len, C_)
        if x.shape[1] != encoder_num_patches:
            idx = torch.linspace(0, x.shape[1] - 1, encoder_num_patches, device=x.device).long()
            x = x[:, idx, :]
            if padding_mask is not None:
                padding_mask = padding_mask.to(x.device)[:, idx]
        return x, padding_mask

    cls_head = ClassificationHead(
        n_vars       = n_v,
        d_model      = embed_dim,
        n_classes    = n_classes,
        head_dropout = config.get("head_dropout", 0.1),
    ).to(self.device)

    n_epochs    = config.get("epoch_classification", 20)
    _all_params = list(self.encoder_for.parameters()) + list(cls_head.parameters())
    _params     = list(cls_head.parameters()) if linear_probe else _all_params
    _trainable  = sum(p.numel() for p in _params)
    _total      = sum(p.numel() for p in _all_params)
    print(f"  Trainable: {_trainable:,} / {_total:,} params")
    optimizer = torch.optim.Adam(_params,
                                 lr=config.get("lr_classification", 1e-3),
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.get("lr_classification", 1e-3),
        total_steps=n_epochs * len(classification_train),
        pct_start=0.3, anneal_strategy='cos',
    )

    for epoch in range(n_epochs):
        cls_head.train()
        if not linear_probe:
            self.encoder_for.train()
        correct, total = 0, 0
        for patches, labels, padding_mask in classification_train:
            patches, padding_mask = _to_patches(patches, padding_mask)
            patches      = patches.to(self.device)
            padding_mask = padding_mask.to(self.device)
            labels       = labels.to(self.device)
            B, P, PL, n_v_ = patches.shape

            optimizer.zero_grad()
            ctx_norm, _, _ = _instance_norm(patches)
            with torch.set_grad_enabled(not linear_probe):
                enc_out = self.encoder_for(ctx_norm, padding_mask=padding_mask)
                enc     = enc_out["data_patches"]          # [B*n_v, P, embed_dim]
            enc_p  = enc.reshape(B, n_v_, encoder_num_patches, embed_dim).permute(0, 1, 3, 2)
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
        for patches, labels, padding_mask in classification_test:
            patches, padding_mask = _to_patches(patches, padding_mask)
            patches      = patches.to(self.device)
            padding_mask = padding_mask.to(self.device)
            labels       = labels.to(self.device)
            B, P, PL, n_v_ = patches.shape
            ctx_norm, _, _ = _instance_norm(patches)
            enc_out = self.encoder_for(ctx_norm, padding_mask=padding_mask)
            enc     = enc_out["data_patches"]
            enc_p   = enc.reshape(B, n_v_, encoder_num_patches, embed_dim).permute(0, 1, 3, 2)
            logits  = cls_head(enc_p)
            tc += (logits.argmax(1) == labels).sum().item()
            tt += len(labels)
    test_acc = tc / tt
    print(f"[JEPA] Test Accuracy: {test_acc:.4f}")
    return test_acc
