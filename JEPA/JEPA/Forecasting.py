import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from data_loaders.data_puller import ForcastingDataPullerDescrete
from JEPA.Decoder import LinearDecoder, PredictionHead
from JEPA.Training import _instance_norm


def forcasting_zeroshot(self, path):
    """Non-autoregressive forecasting: slides real context forward, no prediction feedback.
    Runs TWICE: first with random encoder (baseline), then with trained encoder."""
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
        n_v_for = len(config["input_variables_forcasting"][0])
        self.forecast_head_patch = PredictionHead(individual=False, n_vars=n_v_for, d_model=embed_dim, num_patch=num_patches, forecast_len=h_t * P_L).to(self.device)
        # Train decoders
        '''
        optimizer = torch.optim.AdamW([
            {"params": self.forecast_head_patch.parameters(), "lr": 5e-4,   "weight_decay": 1e-4}
        ])
        '''
        
        optimizer = torch.optim.Adam(
            self.forecast_head_patch.parameters(),
            lr=config.get("lr_forcasting", 1e-4),
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.get("lr_forcasting", 1e-4),
            total_steps=self.epoch_t * len(self.forcast_train),
            pct_start=0.3, anneal_strategy='cos',
        )

        for epoch in range(self.epoch_t):
            self.encoder_for.eval()
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
                with torch.no_grad():
                    encoder_out = self.encoder_for(context_patches)
                    encoder_patches  = encoder_out["data_patches"]         # [B*n_v, ctx, embed_dim]
                # reshape to [B, n_v, embed_dim, num_patch/S]
                enc_p = encoder_patches.reshape(B, n_v, num_patches, embed_dim).permute(0, 1, 3, 2)
                pred_patch = self.forecast_head_patch(enc_p)  # [B, h_t*P_L, n_v]
                target_flat = target_patch.reshape(B, h_t * P_L, n_v)
                loss = F.mse_loss(pred_patch, target_flat)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
            if epoch % 10 == 0:
                print(f"[{run_type}] Epoch: {epoch} - Loss: {total_loss/len(self.forcast_train):.4f}")

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

                encoder_out = self.encoder_for(context_patches)
                encoder_patches  = encoder_out["data_patches"]
                enc_p = encoder_patches.reshape(B, n_v, num_patches, embed_dim).permute(0, 1, 3, 2)
                pred_p2p = self.forecast_head_patch(enc_p)  # [B, h_t*P_L, n_v]

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
