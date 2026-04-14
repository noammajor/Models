"""
TSLib-style classification: frozen encoder + linear probe.

Training loop matches TSLib's exp_classification.py:
  - RAdam optimizer
  - Gradient clipping max_norm=4.0
  - Early stopping on negated validation accuracy (patience-based)
  - Full flatten over patches: [B, n_vars, d_model, n_patches] → [B, n_vars*d_model*n_patches]

Each dataset gets its own head sized to its fixed n_patches — same pattern as forecasting heads.
All classification functions in this project call train_cls_tslib().
Batches are 2-tuples (input, labels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ───────────────────────────────────────────────────────────────────

class _CLS_Head(nn.Module):
    def __init__(self, d_in: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(d_in, n_classes)

    def forward(self, x):
        return self.fc(self.drop(x))


def _eval_acc(encode_fn, head, loader, device):
    head.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            x      = batch[0].float().to(device)
            labels = batch[1].long().to(device).squeeze()
            mask   = batch[2].to(device) if len(batch) > 2 else None

            enc    = encode_fn(x, key_padding_mask=mask)  # [B, n_vars, d_model, n_patches]
            z      = enc.flatten(1)                        # [B, n_vars * d_model * n_patches]
            logits = head(z)
            preds  = torch.argmax(F.softmax(logits, dim=-1), dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.shape[0]
    head.train()
    return correct / total if total > 0 else 0.0


# ── main trainer ─────────────────────────────────────────────────────────────

def train_cls_tslib(encode_fn, n_classes,
                    train_loader, val_loader, test_loader,
                    n_epochs    = 50,
                    lr          = 1e-4,
                    patience    = 10,
                    head_dropout= 0.1,
                    device      = None):
    """TSLib-style linear-probe classification training.

    Pull data first, then build a head sized to that dataset's n_patches — same
    pattern as forecasting heads.

    Args:
        encode_fn   : callable (x) -> enc [B, n_vars, d_model, n_patches]
                      x may be raw [B, T, C] or pre-patched [B, P, PL, n_vars].
                      Frozen — called inside torch.no_grad().
        n_classes   : number of target classes
        train/val/test_loader : yield (input, labels); pad_T fixed per dataset
        n_epochs    : max training epochs
        lr          : RAdam learning rate
        patience    : early stopping patience (epochs without val acc improvement)
        head_dropout: dropout before linear head
        device      : torch device (default: cuda if available)

    Returns:
        test accuracy (float)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pull one batch to infer head input size: n_vars * d_model * n_patches
    sample_batch = next(iter(train_loader))
    sample_x    = sample_batch[0][:2].float().to(device)
    sample_mask = sample_batch[2][:2].to(device) if len(sample_batch) > 2 else None
    with torch.no_grad():
        sample_enc = encode_fn(sample_x, key_padding_mask=sample_mask)  # [2, n_vars, d_model, n_patches]
    d_in = sample_enc.flatten(1).shape[1]  # n_vars * d_model * n_patches

    head      = _CLS_Head(d_in, n_classes, head_dropout).to(device)
    optimizer = torch.optim.RAdam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val  = 0.0
    best_sd   = None
    no_imp    = 0

    for epoch in range(n_epochs):
        head.train()
        correct = total = 0

        for batch in train_loader:
            x      = batch[0].float().to(device)
            labels = batch[1].long().to(device).squeeze()
            mask   = batch[2].to(device) if len(batch) > 2 else None

            with torch.no_grad():
                enc = encode_fn(x, key_padding_mask=mask)  # [B, n_vars, d_model, n_patches]

            z      = enc.flatten(1)        # [B, n_vars * d_model * n_patches]
            logits = head(z)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=4.0)
            optimizer.step()

            preds    = torch.argmax(F.softmax(logits, dim=-1), dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.shape[0]

        train_acc = correct / total if total > 0 else 0.0
        val_acc   = _eval_acc(encode_fn, head, val_loader, device)

        improved = val_acc > best_val
        if improved:
            best_val = val_acc
            best_sd  = {k: v.clone() for k, v in head.state_dict().items()}
            no_imp   = 0
        else:
            no_imp  += 1

        if epoch % 5 == 0 or improved:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                  f"train {train_acc:.4f} | val {val_acc:.4f}"
                  + (" ★" if improved else ""))

        if no_imp >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_sd is not None:
        head.load_state_dict(best_sd)

    test_acc = _eval_acc(encode_fn, head, test_loader, device)
    print(f"  Test accuracy : {test_acc:.4f}  (best val: {best_val:.4f})")
    return test_acc
