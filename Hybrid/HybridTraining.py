import os
import copy

import torch
import torch.nn.functional as F

from losses import _sigreg_loss


def _instance_norm(x, eps=1e-6):
    """Per-sample normalization over patch and patch-length dims. x: [B, P, P_L, F]"""
    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2),  keepdim=True) + eps
    return (x - mean) / std, mean, std


def _embedding_stats(z: torch.Tensor) -> dict:
    z_flat = z.reshape(-1, z.size(-1)).float().detach()
    mean = z_flat.mean(0)
    std  = z_flat.std(0)
    try:
        s = torch.linalg.svdvals(z_flat - z_flat.mean(0))
        s = s / s.sum()
        eff_rank = torch.exp(-(s * torch.log(s + 1e-8)).sum()).item()
    except Exception:
        eff_rank = float('nan')
    return {"emb_mean": mean.mean().item(), "emb_std": std.mean().item(), "eff_rank": eff_rank}


def compute_hybrid_loss(self, patches, global_step, batch_idx=0, epoch=0):
    """
    loss = φ · lejepa_loss + (1−φ) · ntp_loss

    patches: [B, P, P_L, F]
    """
    patches = patches.float()
    B, P, P_L, C = patches.shape
    context_P = self.config['context_patches']
    phi = self.phi

    if self.config.get('revin', True):
        patches, _, _ = _instance_norm(patches)

    total_loss = torch.tensor(0.0, device=self.device)
    metrics = {}

    # ── LEJEPA branch ─────────────────────────────────────────────────────────
    if phi > 0:
        T = P * P_L
        x = patches.reshape(B, T, C)
        v1 = self.augment_v1(x)                                           # [B, T, C]
        v2 = self.augment_v2(x)

        z1 = self.encoder(v1.reshape(B, P, P_L, C))['data_patches']      # [B*C, P, D]
        z2 = self.encoder(v2.reshape(B, P, P_L, C))['data_patches']

        pred_loss = F.mse_loss(z1, z2)
        n_slices  = self.config.get('sigreg_num_slices', 512)
        sigreg    = 0.5 * (_sigreg_loss(z1, global_step, n_slices) +
                           _sigreg_loss(z2, global_step, n_slices))
        lambd      = self.config.get('lambda_sigreg', 0.008)
        lejepa_loss = (1.0 - lambd) * pred_loss + lambd * sigreg

        total_loss = total_loss + phi * lejepa_loss
        metrics.update({
            'lejepa_loss': lejepa_loss.item(),
            'pred_loss':   pred_loss.item(),
            'sigreg':      sigreg.item(),
        })

    # ── NTP branch ────────────────────────────────────────────────────────────
    if phi < 1:
        ctx = patches[:, :context_P]               # [B, context_P, P_L, F]
        tgt = patches[:, context_P:]               # [B, horizon_P, P_L, F]

        enc_out = self.encoder(ctx)
        z_ctx   = enc_out['data_patches']          # [B*C, context_P, D]
        D       = z_ctx.shape[-1]
        # reshape to [B, C, D, context_P] — the layout NTPHead expects
        z_ctx = z_ctx.reshape(B, C, context_P, D).permute(0, 1, 3, 2)

        pred     = self.ntp_head(z_ctx)            # [B, horizon_P, F, P_L]
        tgt_perm = tgt.permute(0, 1, 3, 2)        # [B, horizon_P, F, P_L]
        ntp_loss = F.mse_loss(pred, tgt_perm)

        total_loss = total_loss + (1.0 - phi) * ntp_loss
        metrics['ntp_loss'] = ntp_loss.item()

    metrics['total_loss'] = total_loss.item()

    if batch_idx % 5 == 0:
        line = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        if phi > 0:
            s = _embedding_stats(z1)
            line += (f"  emb_mean: {s['emb_mean']:.3f}"
                     f"  emb_std: {s['emb_std']:.3f}"
                     f"  rank: {s['eff_rank']:.1f}")
        print(f"Epoch {epoch}, Batch {batch_idx} — {line}")

    return total_loss, metrics


def evaluate(self, val_loader, global_step, epoch):
    self.encoder.eval()
    self.ntp_head.eval()
    val_loss = 0.0
    val_metrics: dict = {}

    with torch.no_grad():
        for batch in val_loader:
            patches = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            loss, metrics = self.compute_hybrid_loss(patches, global_step, epoch=epoch)
            val_loss += loss.item()
            for k, v in metrics.items():
                val_metrics[k] = val_metrics.get(k, 0.0) + v

    n = max(len(val_loader), 1)
    self.encoder.train()
    self.ntp_head.train()
    return val_loss / n, {k: v / n for k, v in val_metrics.items()}


def save_model(self, encoder, optimizer, epoch, path_save):
    os.makedirs(os.path.dirname(path_save) or ".", exist_ok=True)
    path = f"{path_save}best_model.pt"
    torch.save({"epoch": epoch, "encoder": encoder.state_dict()}, path)
    print(f"Saved checkpoint: {path}")


def train_and_evaluate(self):
    self.encoder  = self.encoder.to(self.device)
    self.ntp_head = self.ntp_head.to(self.device)
    best_val_loss = float("inf")
    global_step   = 0

    self.save_model(self.encoder, self.optimizer, 0, f"{self.path_save}_INITIAL")

    for epoch in range(self.config["num_epochs"]):
        print(f"Starting Epoch {epoch}/{self.config['num_epochs']}")
        self.encoder.train()
        self.ntp_head.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            patches = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)

            self.optimizer.zero_grad()
            loss, _ = self.compute_hybrid_loss(
                patches, global_step, batch_idx=batch_idx, epoch=epoch
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.ntp_head.parameters()),
                self.config["clip_grad"],
            )
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()
            global_step  += 1

        epoch_loss = running_loss / max(len(self.train_loader), 1)
        print(f"Epoch {epoch} — train loss: {epoch_loss:.4f}  lr: {self.optimizer.param_groups[0]['lr']:.3g}")

        val_loss, val_dict = self.evaluate(self.val_loader, global_step, epoch)
        print(f"Epoch {epoch} — val loss: {val_loss:.4f} | {val_dict}")

        if val_loss < best_val_loss and epoch >= self.warmup_epochs:
            best_val_loss = val_loss
            self.save_model(self.encoder, self.optimizer, epoch, self.path_save)
            self.best_encoder = copy.deepcopy(self.encoder.state_dict())
            print("New best — model saved.")

        self.save_model(self.encoder, self.optimizer, epoch, f"{self.path_save}_epoch{epoch}")

    print("Training complete.")
    test_loss, test_dict = self.evaluate(self.test_loader, global_step, epoch=-1)
    print(f"TEST | loss: {test_loss:.4f} | {test_dict}")
    return test_loss, test_dict, self.best_encoder
