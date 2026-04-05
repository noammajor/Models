import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from data_loaders.data_puller import ForcastingDataPullerDescrete
from Discrete_JEPA.Decoder import LinearDecoder, PredictionHead



def _instance_norm(x, eps=1e-6):
    """Per-instance, per-variable RevIN normalization over the time axis.

    x: [B, n_patches, patch_size, n_vars]
    returns: (x_norm, mean, std)  — mean/std shape [B, 1, 1, n_vars]
    """
    mean = x.mean(dim=(1, 2), keepdim=True)
    std  = x.std(dim=(1, 2), keepdim=True) + eps
    return (x - mean) / std, mean, std


def _instance_denorm(x, mean, std):
    """Reverse _instance_norm. x: [B, *, n_vars], mean/std: [B, 1, 1, n_vars]."""
    shape = [mean.shape[0]] + [1] * (x.ndim - 2) + [mean.shape[-1]]
    return x * std.reshape(shape) + mean.reshape(shape)



def predictor_forecasting(self, path, num_epochs=200):
    """
    Predictor-based forecasting evaluation.

    Pipeline (encoder + predictor frozen, only LinearDecoder trained):
      1. Encode context patches 0-23 with the target (EMA) encoder
      2. Run P2P predictor with target_mask=[24..29] -> predicted embeddings
      3. Decode predicted embeddings -> raw patch values via LinearDecoder

    Uses in-window targets (indices 24-29) so the predictor's PE covers
    every position without extrapolation.

    Compares TRAINED (pretrained encoder + predictor) vs RANDOM init.
    """
    checkpoint_path = f"{self.path_save}{path}best_model.pt"
    name_loader = torch.load(checkpoint_path, map_location="cpu")
    config      = self.config

    ctx_len   = config["ratio_patches"] - config["horizon_t"]   # 24
    horizon   = config["horizon_t"]                              # 6
    embed_dim = config["encoder_embed_dim"]
    P_L       = config["patch_size_forcasting"]

    # In-window dataset: context = first ctx_len patches, target = last horizon patches
    inwindow_cfg = {**config, "ratio_patches": ctx_len}
    train_ds     = ForcastingDataPullerDescrete(inwindow_cfg, which='train')
    test_ds      = ForcastingDataPullerDescrete(inwindow_cfg, which='test')
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,  drop_last=False)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=config["batch_size"], shuffle=False, drop_last=False)

    # Fixed target indices [24..29] — within predictor PE range [0, 29]
    target_indices = torch.arange(ctx_len, ctx_len + horizon).unsqueeze(0)  # [1, horizon]

    for run_type in ['RANDOM', 'TRAINED']:
        print(f"\n=== Predictor Forecasting ({run_type}) ===")

        # Reset to fresh random weights
        for m in self.encoder_for.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        nn.init.trunc_normal_(self.encoder_for.semantic_tokens, std=0.02)
        for m in self.predictor_for.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        nn.init.trunc_normal_(self.predictor_for.mask_token,     std=0.02)
        nn.init.trunc_normal_(self.predictor_for.semantic_query, std=0.02)

        if run_type == 'TRAINED':
            self.encoder_for.load_state_dict(name_loader["target_encoder"])
            self.predictor_for.load_state_dict(name_loader["predictor"])

        self.encoder_for.to(self.device).eval()
        self.predictor_for.to(self.device).eval()

        # Only the decoder is trained — encoder + predictor are frozen
        decoder   = LinearDecoder(emb_dim=embed_dim, patch_size=P_L).to(self.device)
        optimizer = torch.optim.AdamW(
            decoder.parameters(), lr=config["lr_forcasting"], weight_decay=1e-4)

        for epoch in range(num_epochs):
            decoder.train()
            total_loss = 0.0

            for context_patches, target_patch in train_loader:
                if context_patches.dim() == 3:
                    context_patches = context_patches.unsqueeze(-1)
                context_patches = context_patches.to(self.device)
                target_patch    = target_patch.to(self.device)
                B, h, P_L_b, n_v = target_patch.shape

                t_mask = target_indices.expand(B, -1).to(self.device)  # [B, horizon]

                optimizer.zero_grad()
                with torch.no_grad():
                    enc_out  = self.encoder_for(context_patches)
                    ctx_emb  = enc_out["data_patches"]                  # [B*n_v, ctx, D]
                    pred_emb = self.predictor_for(
                        ctx_emb, task="P2P", target_mask=t_mask)        # [B*n_v, h, D]

                # [B*n_v, h, D] -> [B*n_v, h, P_L] -> [B, h, P_L, n_v]
                pred_raw = decoder(pred_emb).view(B, n_v, h, P_L_b).permute(0, 2, 3, 1)

                loss = F.mse_loss(pred_raw, target_patch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 50 == 0:
                print(f"[{run_type}] Epoch {epoch:3d} - Loss: {total_loss/len(train_loader):.4f}")

        # Evaluate on test set
        decoder.eval()
        mse_list, mae_list = [], []
        with torch.no_grad():
            for context_patches, target_patch in test_loader:
                if context_patches.dim() == 3:
                    context_patches = context_patches.unsqueeze(-1)
                context_patches = context_patches.to(self.device)
                target_patch    = target_patch.to(self.device)
                B, h, P_L_b, n_v = target_patch.shape

                t_mask = target_indices.expand(B, -1).to(self.device)

                enc_out  = self.encoder_for(context_patches)
                ctx_emb  = enc_out["data_patches"]
                pred_emb = self.predictor_for(ctx_emb, task="P2P", target_mask=t_mask)
                pred_raw = decoder(pred_emb).view(B, n_v, h, P_L_b).permute(0, 2, 3, 1)

                mse_list.append(F.mse_loss(pred_raw.cpu(), target_patch.cpu()).item())
                mae_list.append(F.l1_loss( pred_raw.cpu(), target_patch.cpu()).item())

        mse = sum(mse_list) / len(mse_list)
        mae = sum(mae_list) / len(mae_list)
        print(f"[{run_type}] Test MSE: {mse:.4f},  MAE: {mae:.4f}")


def finetuning_forecasting(self, path, num_epochs=300):
    """
    Full fine-tuning evaluation.

    Both encoder and head are fine-tuned jointly at the same LR on the
    full forecasting training set.  The only difference between runs is
    the starting point: pretrained target encoder vs random init.

    This tests whether pretraining provides a better initialisation that
    leads to faster convergence or a lower final loss.

    Head: flatten all patch embeddings -> Linear(P*D, h*P_L)
    """
    checkpoint_path = f"{self.path_save}{path}best_model.pt"
    name_loader     = torch.load(checkpoint_path, map_location="cpu")
    config          = self.config

    embed_dim  = config["encoder_embed_dim"]
    num_patches = config["ratio_patches"]
    h_t        = config["horizon_t"]
    P_L        = config["patch_size_forcasting"]
    num_sem    = config["num_semantic_tokens"]
    lr         = config["lr_forcasting"]

    for run_type in ['RANDOM', 'TRAINED']:
        print(f"\n=== Full Fine-Tuning ({run_type}) ===")

        # Reset encoder to fresh random weights every run
        for m in self.encoder_for.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        nn.init.trunc_normal_(self.encoder_for.semantic_tokens, std=0.02)
        for m in self.vector_quantizer.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        if run_type == 'TRAINED':
            self.encoder_for.load_state_dict(name_loader["target_encoder"])
            if "vector_quantizer" in name_loader:
                self.vector_quantizer.load_state_dict(name_loader["vector_quantizer"])

        self.encoder_for.to(self.device)
        self.vector_quantizer.to(self.device)

        # Head: flatten -> linear
        head_patch = nn.Linear(num_patches * embed_dim, h_t * P_L).to(self.device)
        head_sem   = nn.Linear(num_sem * embed_dim,     h_t * P_L).to(self.device)

        # Differential LRs: encoder at 10% of head LR to prevent overfitting
        optimizer = torch.optim.AdamW([
            {"params": self.encoder_for.parameters(), "lr": lr * 0.1, "weight_decay": 1e-2},
            {"params": list(head_patch.parameters()) + list(head_sem.parameters()), "lr": lr, "weight_decay": 1e-3},
        ])

        for epoch in range(num_epochs):
            self.encoder_for.train()
            head_patch.train()
            head_sem.train()
            total_loss = 0.0

            for context_patches, target_patch in self.forcast_train:
                if context_patches.dim() == 3:
                    context_patches = context_patches.unsqueeze(-1)
                context_patches = context_patches.to(self.device)
                target_patch    = target_patch.to(self.device)
                B, h, P_L_b, n_v = target_patch.shape

                optimizer.zero_grad()
                ctx_norm, ctx_mean, ctx_std = _instance_norm(context_patches)  # RevIN
                enc_out = self.encoder_for(ctx_norm)
                patch_flat = enc_out["data_patches"].flatten(1)       # [B*n_v, P*D]
                sem_flat   = enc_out["quantized_semantic"].flatten(1) # [B*n_v, S*D]

                # predict in normalised space, then denorm before loss
                pred_patch = head_patch(patch_flat).view(B, n_v, h, P_L_b).permute(0, 2, 3, 1) * ctx_std + ctx_mean
                pred_sem   = head_sem(sem_flat).view(B, n_v, h, P_L_b).permute(0, 2, 3, 1)     * ctx_std + ctx_mean

                loss = F.mse_loss(pred_patch, target_patch) + F.mse_loss(pred_sem, target_patch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder_for.parameters()) +
                    list(head_patch.parameters()) +
                    list(head_sem.parameters()),
                    1.0,
                )
                optimizer.step()
                total_loss += loss.item()

            if epoch % 25 == 0:
                print(f"[{run_type}] Epoch {epoch:3d} - Loss: {total_loss/len(self.forcast_train):.4f}")

        # Evaluate on test set
        self.encoder_for.eval()
        head_patch.eval()
        head_sem.eval()

        mse_p_list, mse_s_list = [], []
        with torch.no_grad():
            for context_patches, target_patch in self.forcast_test:
                if context_patches.dim() == 3:
                    context_patches = context_patches.unsqueeze(-1)
                context_patches = context_patches.to(self.device)
                target_patch    = target_patch.to(self.device)
                B, h, P_L_b, n_v = target_patch.shape

                ctx_norm, ctx_mean, ctx_std = _instance_norm(context_patches)  # RevIN
                enc_out    = self.encoder_for(ctx_norm)
                patch_flat = enc_out["data_patches"].flatten(1)
                sem_flat   = enc_out["quantized_semantic"].flatten(1)

                pred_patch = head_patch(patch_flat).view(B, n_v, h, P_L_b).permute(0, 2, 3, 1) * ctx_std + ctx_mean
                pred_sem   = head_sem(sem_flat).view(B, n_v, h, P_L_b).permute(0, 2, 3, 1)     * ctx_std + ctx_mean

                mse_p_list.append(F.mse_loss(pred_patch.cpu(), target_patch.cpu()).item())
                mse_s_list.append(F.mse_loss(pred_sem.cpu(),   target_patch.cpu()).item())

        mse_p = sum(mse_p_list) / len(mse_p_list)
        mse_s = sum(mse_s_list) / len(mse_s_list)
        print(f"[{run_type}] Test MSE - Patch: {mse_p:.4f}, Sem: {mse_s:.4f}, Mix: {(mse_p+mse_s)/2:.4f}")


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
        self.vector_quantizer.to(self.device)

        if run_type == 'TRAINED':
            print(f"Loading checkpoint: {checkpoint_path}")
            self.encoder_for.load_state_dict(name_loader["target_encoder"])
            if "vector_quantizer" in name_loader:
                self.vector_quantizer.load_state_dict(name_loader["vector_quantizer"])
        else:
            for m in self.encoder_for.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            for m in self.predictor_for.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            for m in self.vector_quantizer.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            torch.nn.init.trunc_normal_(self.encoder_for.semantic_tokens, std=0.02)

        embed_dim   = config["encoder_embed_dim"]
        num_patches = config["ratio_patches"]
        h_t         = config["horizon_t"]
        P_L         = config["patch_size_forcasting"]
        num_sem     = config["num_semantic_tokens"]
        n_v_for = len(config["input_variables_forcasting"][0])
        self.forecast_head_patch = PredictionHead(individual=False, n_vars=n_v_for, d_model=embed_dim, num_patch=num_patches, forecast_len=h_t * P_L).to(self.device)
        self.forecast_head_sem   = PredictionHead(individual=False, n_vars=n_v_for, d_model=embed_dim, num_patch=num_sem, forecast_len=h_t * P_L).to(self.device)

        lr_fc = config["lr_forcasting"]
        optimizer = torch.optim.AdamW(
            list(self.forecast_head_patch.parameters()) + list(self.forecast_head_sem.parameters()),
            lr=lr_fc, weight_decay=1e-4,
        )
        for epoch in range(self.epoch_t):
            self.encoder_for.eval()
            self.predictor_for.eval()
            self.vector_quantizer.eval()
            self.forecast_head_sem.train()
            self.forecast_head_patch.train()
            total_loss = 0.0
            for context_patches, target_patch in self.forcast_train:
                if context_patches.dim() == 3:
                    context_patches = context_patches.unsqueeze(-1)
                context_patches = context_patches.to(self.device)
                target_patch = target_patch.to(self.device)
                B, h_t, P_L, n_v = target_patch.shape  # [B, h, P_L, n_v]
                optimizer.zero_grad()

                target_flat = target_patch.reshape(B, h_t * P_L, n_v)

                ctx_norm, ctx_mean, ctx_std = _instance_norm(context_patches)
                with torch.no_grad():
                    encoder_out = self.encoder_for(ctx_norm)
                    encoder_patches  = encoder_out["data_patches"]         # [B*n_v, ctx, embed_dim]
                    encoder_semantic = encoder_out["quantized_semantic"]   # [B*n_v, S,   embed_dim]
                    _, encoder_semantic, _, _, _, _ = self.vector_quantizer(encoder_semantic)
                # reshape to [B, n_v, embed_dim, num_patch/S]
                enc_p = encoder_patches.reshape(B, n_v, num_patches, embed_dim).permute(0, 1, 3, 2)
                pred_patch = _instance_denorm(self.forecast_head_patch(enc_p), ctx_mean, ctx_std)
                enc_s = encoder_semantic.reshape(B, n_v, num_sem, embed_dim).permute(0, 1, 3, 2)
                pred_s2p = _instance_denorm(self.forecast_head_sem(enc_s), ctx_mean, ctx_std)
                loss = F.mse_loss(pred_patch, target_flat) + F.mse_loss(pred_s2p, target_flat)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print(f"[{run_type}] Epoch: {epoch} - Loss: {total_loss/len(self.forcast_train):.4f}")

        # --- Evaluation on full test set ---
        self.encoder_for.eval()
        self.predictor_for.eval()
        self.forecast_head_patch.eval()
        self.forecast_head_sem.eval()

        mse_p2p_list, mse_s2p_list, mae_p2p_list, mae_s2p_list = [], [], [], []
        last_batch = None  # keep last batch for plotting

        with torch.no_grad():
            for context_patches, target_patch in self.forcast_test:
                target_patch = target_patch.to(self.device)
                context_patches = context_patches.to(self.device)
                if context_patches.dim() == 3:
                    context_patches = context_patches.unsqueeze(-1)
                B, h_t, P_L, n_v = target_patch.shape

                ctx_norm, ctx_mean, ctx_std = _instance_norm(context_patches)
                encoder_out = self.encoder_for(ctx_norm)
                encoder_patches  = encoder_out["data_patches"]
                encoder_semantic = encoder_out["quantized_semantic"]
                _, encoder_semantic, _, _, _, _ = self.vector_quantizer(encoder_semantic)
                enc_p    = encoder_patches.reshape(B, n_v, num_patches, embed_dim).permute(0, 1, 3, 2)
                pred_p2p = _instance_denorm(self.forecast_head_patch(enc_p), ctx_mean, ctx_std)
                enc_s    = encoder_semantic.reshape(B, n_v, num_sem, embed_dim).permute(0, 1, 3, 2)
                pred_s2p = _instance_denorm(self.forecast_head_sem(enc_s), ctx_mean, ctx_std)
                target_flat = target_patch.reshape(B, h_t * P_L, n_v)

                mse_p2p_list.append(F.mse_loss(pred_p2p.cpu(), target_flat.cpu()).item())
                mse_s2p_list.append(F.mse_loss(pred_s2p.cpu(), target_flat.cpu()).item())
                mae_p2p_list.append(F.l1_loss(pred_p2p.cpu(),  target_flat.cpu()).item())
                mae_s2p_list.append(F.l1_loss(pred_s2p.cpu(),  target_flat.cpu()).item())
                last_batch = (pred_p2p, pred_s2p, target_flat)

        if not mse_p2p_list:
            print("WARNING: forcast_test is empty, skipping evaluation.")
            return

        norm_lossP2P = sum(mse_p2p_list) / len(mse_p2p_list)
        norm_lossS2P = sum(mse_s2p_list) / len(mse_s2p_list)
        mae_lossP2P  = sum(mae_p2p_list) / len(mae_p2p_list)
        mae_lossS2P  = sum(mae_s2p_list) / len(mae_s2p_list)
        norm_mixloss = (norm_lossP2P + norm_lossS2P) / 2.0
        mae_mixloss  = (mae_lossP2P  + mae_lossS2P)  / 2.0
        print(f"[{run_type}] MSE  — P2P: {norm_lossP2P:.4f}, S2P: {norm_lossS2P:.4f}, mix: {norm_mixloss:.4f}")
        print(f"[{run_type}] MAE  — P2P: {mae_lossP2P:.4f}, S2P: {mae_lossS2P:.4f}, mix: {mae_mixloss:.4f}")

        # Plot from last batch
        pred_p2p, pred_s2p, target_flat = last_batch
        sample = 0
        path_s = os.path.join(self.path_save, "output_model")
        os.makedirs(path_s, exist_ok=True)
        for var_idx in range(n_v):
            gt    = target_flat[sample, :, var_idx].cpu().numpy()
            p2p   = pred_p2p[sample, :, var_idx].cpu().numpy()
            s2p   = pred_s2p[sample, :, var_idx].cpu().numpy()
            mixed = (p2p + s2p) / 2.0

            plt.figure(figsize=(15, 5))
            plt.plot(gt,    label='Ground Truth', color='black',  alpha=0.7, linewidth=2)
            plt.plot(p2p,   label='P2P',          color='blue',   linestyle='--', alpha=0.9)
            plt.plot(s2p,   label='S2P',          color='orange', linestyle='--', alpha=0.9)
            plt.plot(mixed, label='Mixed',         color='red',    linestyle='--', alpha=0.9)
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
    

def predictor_s2p_p2p_forecasting(self, path, num_epochs=200):
    """
    Predictor-based forecasting using both P2P and S2P paths.

    Pipeline (encoder + predictor + VQ frozen, only decoders trained):
      P2P: encoder(context) → predictor(P2P) → LinearDecoder_p2p → raw patches
      S2P: encoder(context) → VQ(semantic)   → predictor(S2P)    → LinearDecoder_s2p → raw patches

    Each path has its own separate LinearDecoder.
    """
    checkpoint_path = f"{self.path_save}{path}best_model.pt"
    name_loader     = torch.load(checkpoint_path, map_location="cpu")
    config          = self.config

    ctx_len   = config["ratio_patches"] - config["horizon_t"]
    horizon   = config["horizon_t"]
    embed_dim = config["encoder_embed_dim"]
    P_L       = config["patch_size_forcasting"]

    inwindow_cfg = {**config, "ratio_patches": ctx_len}
    train_ds     = ForcastingDataPullerDescrete(inwindow_cfg, which='train')
    test_ds      = ForcastingDataPullerDescrete(inwindow_cfg, which='test')
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,  drop_last=False)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=config["batch_size"], shuffle=False, drop_last=False)

    target_indices = torch.arange(ctx_len, ctx_len + horizon).unsqueeze(0)  # [1, horizon]

    for run_type in ['TRAINED']:
        print(f"\n=== Predictor S2P+P2P Forecasting ({run_type}) ===")

        for m in self.encoder_for.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        nn.init.trunc_normal_(self.encoder_for.semantic_tokens, std=0.02)
        for m in self.predictor_for.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        nn.init.trunc_normal_(self.predictor_for.mask_token,     std=0.02)
        nn.init.trunc_normal_(self.predictor_for.semantic_query, std=0.02)
        for m in self.vector_quantizer.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        if run_type == 'TRAINED':
            self.encoder_for.load_state_dict(name_loader["target_encoder"])
            self.predictor_for.load_state_dict(name_loader["predictor"])
            if "vector_quantizer" in name_loader:
                self.vector_quantizer.load_state_dict(name_loader["vector_quantizer"])

        self.encoder_for.to(self.device).eval()
        self.predictor_for.to(self.device).eval()
        self.vector_quantizer.to(self.device).eval()

        decoder_p2p = LinearDecoder(emb_dim=embed_dim, patch_size=P_L).to(self.device)
        decoder_s2p = LinearDecoder(emb_dim=embed_dim, patch_size=P_L).to(self.device)

        optimizer = torch.optim.AdamW(
            list(decoder_p2p.parameters()) + list(decoder_s2p.parameters()),
            lr=config["lr_forcasting"], weight_decay=1e-4,
        )

        for epoch in range(num_epochs):
            decoder_p2p.train()
            decoder_s2p.train()
            total_loss = 0.0

            for context_patches, target_patch in train_loader:
                if context_patches.dim() == 3:
                    context_patches = context_patches.unsqueeze(-1)
                context_patches = context_patches.to(self.device)
                target_patch    = target_patch.to(self.device)
                B, h, P_L_b, n_v = target_patch.shape

                t_mask = target_indices.expand(B, -1).to(self.device)

                optimizer.zero_grad()
                with torch.no_grad():
                    enc_out              = self.encoder_for(context_patches)
                    ctx_emb              = enc_out["data_patches"]       # [B*n_v, ctx, D]
                    sem_emb              = enc_out["quantized_semantic"] # [B*n_v, S, D]
                    _, sem_emb, _, _, _, _ = self.vector_quantizer(sem_emb)
                    pred_p2p_emb         = self.predictor_for(ctx_emb, task="P2P", target_mask=t_mask)  # [B*n_v, h, D]
                    pred_s2p_emb         = self.predictor_for(sem_emb, task="S2P", target_mask=t_mask)  # [B*n_v, h, D]

                pred_p2p_raw = decoder_p2p(pred_p2p_emb).view(B, n_v, h, P_L_b).permute(0, 2, 3, 1)
                pred_s2p_raw = decoder_s2p(pred_s2p_emb).view(B, n_v, h, P_L_b).permute(0, 2, 3, 1)

                loss = F.mse_loss(pred_p2p_raw, target_patch) + F.mse_loss(pred_s2p_raw, target_patch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 50 == 0:
                print(f"[{run_type}] Epoch {epoch:3d} - Loss: {total_loss/len(train_loader):.4f}")

        decoder_p2p.eval()
        decoder_s2p.eval()
        mse_p2p_list, mse_s2p_list, mae_p2p_list, mae_s2p_list = [], [], [], []

        with torch.no_grad():
            for context_patches, target_patch in test_loader:
                if context_patches.dim() == 3:
                    context_patches = context_patches.unsqueeze(-1)
                context_patches = context_patches.to(self.device)
                target_patch    = target_patch.to(self.device)
                B, h, P_L_b, n_v = target_patch.shape

                t_mask           = target_indices.expand(B, -1).to(self.device)
                enc_out          = self.encoder_for(context_patches)
                ctx_emb          = enc_out["data_patches"]
                sem_emb          = enc_out["quantized_semantic"]
                _, sem_emb, _, _, _, _ = self.vector_quantizer(sem_emb)
                pred_p2p_emb     = self.predictor_for(ctx_emb, task="P2P", target_mask=t_mask)
                pred_s2p_emb     = self.predictor_for(sem_emb, task="S2P", target_mask=t_mask)

                pred_p2p_raw = decoder_p2p(pred_p2p_emb).view(B, n_v, h, P_L_b).permute(0, 2, 3, 1)
                pred_s2p_raw = decoder_s2p(pred_s2p_emb).view(B, n_v, h, P_L_b).permute(0, 2, 3, 1)

                mse_p2p_list.append(F.mse_loss(pred_p2p_raw.cpu(), target_patch.cpu()).item())
                mse_s2p_list.append(F.mse_loss(pred_s2p_raw.cpu(), target_patch.cpu()).item())
                mae_p2p_list.append(F.l1_loss( pred_p2p_raw.cpu(), target_patch.cpu()).item())
                mae_s2p_list.append(F.l1_loss( pred_s2p_raw.cpu(), target_patch.cpu()).item())

        mse_p2p = sum(mse_p2p_list) / len(mse_p2p_list)
        mse_s2p = sum(mse_s2p_list) / len(mse_s2p_list)
        mae_p2p = sum(mae_p2p_list) / len(mae_p2p_list)
        mae_s2p = sum(mae_s2p_list) / len(mae_s2p_list)
        print(f"[{run_type}] MSE — P2P: {mse_p2p:.4f}, S2P: {mse_s2p:.4f}, mix: {(mse_p2p+mse_s2p)/2:.4f}")
        print(f"[{run_type}] MAE — P2P: {mae_p2p:.4f}, S2P: {mae_s2p:.4f}, mix: {(mae_p2p+mae_s2p)/2:.4f}")

