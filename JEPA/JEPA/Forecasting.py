import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from JEPA.Decoder import LinearDecoder, PredictionHead, ClassificationHead
from JEPA.Training import _instance_norm


def _instance_denorm(x, mean, std):
    """Reverse _instance_norm. x: [B, *, n_vars], mean/std: [B, 1, 1, n_vars]."""
    shape = [mean.shape[0]] + [1] * (x.ndim - 2) + [mean.shape[-1]]
    return x * std.reshape(shape) + mean.reshape(shape)


def forecasting(self, path, linear_probe=True, mlp_head: bool = False):
    """Non-autoregressive forecasting: slides real context forward, no prediction feedback.
    Runs TWICE: first with random encoder (baseline), then with trained encoder.

    linear_probe=True : encoder frozen, only forecast head trained.
    linear_probe=False: encoder unfrozen, head + encoder trained jointly (fine-tune).
    mlp_head=True     : use 1-hidden-layer MLP head (Linear→GELU→Linear) instead of single Linear.
    """
    config    = self.config   # always use instance config, not module-level
    epoch_tag = path
    checkpoint_path = f"{self.path_save}{path}best_model.pt"

    name_loader = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    for run_type in ['TRAINED']:
        print(f"\n=== Zero-Shot Forecasting ({run_type}) ===")

        # Move models to device
        self.encoder_for.to(self.device)
        self.predictor_for.to(self.device)

        if run_type == 'TRAINED':
            print(f"Loading checkpoint: {checkpoint_path}")
            self.encoder_for.load_state_dict(name_loader["target_encoder"])
        else:
            for m in self.encoder_for.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            for m in self.predictor_for.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()

        embed_dim   = config["encoder_embed_dim"]
        num_patches = config.get("forecasting_context_patches", config["ratio_patches"])
        h_t         = config["horizon_t"]
        P_L         = config["patch_size_forcasting"]
        # n_vars inferred from the first sample of the forecast loader
        n_v_for = self.forcast_train.dataset[0][0].shape[-1]
        self.forecast_head_patch = PredictionHead(individual=False, n_vars=n_v_for, d_model=embed_dim, num_patch=num_patches, forecast_len=h_t * P_L, mlp_head=mlp_head).to(self.device)
        # Train decoders
        '''
        optimizer = torch.optim.AdamW([
            {"params": self.forecast_head_patch.parameters(), "lr": 5e-4,   "weight_decay": 1e-4}
        ])
        '''
        
        # LR: config value if set, else hardcoded default.
        _cfg_head_lr = config.get("lr_forcasting")
        _cfg_enc_lr  = config.get("lr_forcasting_encoder")
        head_lr = float(_cfg_head_lr) if _cfg_head_lr is not None else 0.0001
        enc_lr  = float(_cfg_enc_lr)  if _cfg_enc_lr  is not None else head_lr
        if linear_probe:
            optimizer = torch.optim.Adam(
                self.forecast_head_patch.parameters(),
                lr=head_lr, weight_decay=1e-4,
            )
            _max_lrs = head_lr
        else:
            optimizer = torch.optim.Adam([
                {"params": self.forecast_head_patch.parameters(), "lr": head_lr},
                {"params": self.encoder_for.parameters(),         "lr": enc_lr},
            ], weight_decay=1e-4)
            _max_lrs = [head_lr, enc_lr]
        print(f"  [JEPA forecast] MODE: {'linear probe — encoder FROZEN' if linear_probe else f'full fine-tune — head_lr={head_lr} encoder_lr={enc_lr}'}")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=_max_lrs,
            total_steps=self.epoch_t * len(self.forcast_train),
            pct_start=0.3, anneal_strategy='cos',
        )

        _val_loader = getattr(self, "forcast_val", None)
        best_val_loss = float("inf")
        best_state    = None

        for epoch in range(self.epoch_t):
            if linear_probe:
                self.encoder_for.eval()
            else:
                self.encoder_for.train()
            self.predictor_for.eval()
            self.forecast_head_patch.train()
            total_loss = 0.0
            for context_patches, target_patch in self.forcast_train:
                if context_patches.dim() == 3:
                    context_patches = context_patches.unsqueeze(-1)
                context_patches = context_patches.to(self.device)
                target_patch = target_patch.to(self.device)
                B, h_t, P_L, n_v = target_patch.shape  # [B, h, P_L, n_v]
                optimizer.zero_grad()
                ctx_norm, ctx_mean, ctx_std = _instance_norm(context_patches)
                with torch.set_grad_enabled(not linear_probe):
                    encoder_out = self.encoder_for(ctx_norm)
                    encoder_patches  = encoder_out["data_patches"]         # [B*n_v, ctx, embed_dim]
                # reshape to [B, n_v, embed_dim, num_patch/S]
                enc_p = encoder_patches.reshape(B, n_v, num_patches, embed_dim).permute(0, 1, 3, 2)
                pred_patch = _instance_denorm(self.forecast_head_patch(enc_p), ctx_mean, ctx_std)
                target_flat = target_patch.reshape(B, h_t * P_L, n_v)
                loss = F.mse_loss(pred_patch, target_flat)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # ── val MSE + keep-best ──
            val_l = None
            if _val_loader is not None and len(_val_loader) > 0:
                self.encoder_for.eval()
                self.forecast_head_patch.eval()
                _vl = 0.0; _vc = 0
                with torch.no_grad():
                    for ctx_v, tgt_v in _val_loader:
                        if ctx_v.dim() == 3: ctx_v = ctx_v.unsqueeze(-1)
                        ctx_v = ctx_v.to(self.device); tgt_v = tgt_v.to(self.device)
                        Bv, hv, PLv, nv = tgt_v.shape
                        cn, cm, cs = _instance_norm(ctx_v)
                        eo = self.encoder_for(cn)["data_patches"]
                        ep = eo.reshape(Bv, nv, num_patches, embed_dim).permute(0, 1, 3, 2)
                        pv = _instance_denorm(self.forecast_head_patch(ep), cm, cs)
                        tv_flat = tgt_v.reshape(Bv, hv * PLv, nv)
                        _vl += F.mse_loss(pv, tv_flat).item() * Bv
                        _vc += Bv
                val_l = _vl / max(_vc, 1)
                if val_l < best_val_loss:
                    best_val_loss = val_l
                    best_state = {
                        "head":    {k: v.detach().clone() for k, v in self.forecast_head_patch.state_dict().items()},
                        "encoder": {k: v.detach().clone() for k, v in self.encoder_for.state_dict().items()} if not linear_probe else None,
                    }
            if epoch % 10 == 0:
                _vmsg = f"  val={val_l:.4f}" if val_l is not None else ""
                print(f"[{run_type}] Epoch: {epoch} - Loss: {total_loss/len(self.forcast_train):.4f}{_vmsg}")

        # ── Restore best checkpoint for test eval ──
        if best_state is not None:
            self.forecast_head_patch.load_state_dict(best_state["head"])
            if best_state["encoder"] is not None:
                self.encoder_for.load_state_dict(best_state["encoder"])
            print(f"[{run_type}] Restored best checkpoint  (best val MSE={best_val_loss:.4f})")

        # --- Evaluation on full test set ---
        self.encoder_for.eval()
        self.predictor_for.eval()
        self.forecast_head_patch.eval()

        all_preds, all_targets = [], []
        last_batch = None  # keep last batch for plotting

        with torch.no_grad():
            for context_patches, target_patch in self.forcast_test:
                target_patch = target_patch.to(self.device)
                context_patches = context_patches.to(self.device)
                if context_patches.dim() == 3:
                    context_patches = context_patches.unsqueeze(-1)
                B, h_t, P_L, n_v = target_patch.shape
                target_flat_orig = target_patch.reshape(B, h_t * P_L, n_v)

                ctx_norm, ctx_mean, ctx_std = _instance_norm(context_patches)
                encoder_out = self.encoder_for(ctx_norm)
                encoder_patches  = encoder_out["data_patches"]
                enc_p = encoder_patches.reshape(B, n_v, num_patches, embed_dim).permute(0, 1, 3, 2)
                pred_p2p = _instance_denorm(self.forecast_head_patch(enc_p), ctx_mean, ctx_std)

                all_preds.append(pred_p2p.cpu())
                all_targets.append(target_flat_orig.cpu())
                last_batch = (pred_p2p, target_flat_orig)

        if not all_preds:
            print("WARNING: forcast_test is empty, skipping evaluation.")
            return

        all_preds   = torch.cat(all_preds,   dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        norm_lossP2P = F.mse_loss(all_preds, all_targets).item()
        mae_lossP2P  = F.l1_loss(all_preds,  all_targets).item()
        print(f"[{run_type}] MSE  — P2P: {norm_lossP2P:.4f}")
        print(f"[{run_type}] MAE  — P2P: {mae_lossP2P:.4f}")

        # Plot from last batch
        pred_p2p, target_flat = last_batch
        sample = 0
        path_s = os.path.join(self.path_save, "output_model")
        os.makedirs(path_s, exist_ok=True)
        for var_idx in range(n_v):
            gt    = target_flat[sample, :, var_idx].cpu().numpy()
            p2p   = pred_p2p[sample, :, var_idx].cpu().numpy()

            plt.figure(figsize=(15, 5))
            plt.plot(gt,    label='Ground Truth', color='black',  alpha=0.7, linewidth=2)
            plt.plot(p2p,   label='P2P',          color='blue',   linestyle='--', alpha=0.9)
            plt.title(f"Zero-Shot {run_type} — Variable {var_idx} ({h_t * P_L} steps)")
            plt.xlabel("Time Steps")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)

            save_name = f"{run_type.lower()}_var{var_idx}{epoch_tag}.png"
            plt.savefig(os.path.join(path_s, save_name))
            plt.close()

        print(f"[{run_type}] Plots saved to {os.path.join(self.path_save, 'output_model')}")
        return norm_lossP2P
