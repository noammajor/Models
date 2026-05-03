config = {
    "path_save": "./output_model/LE-JEPA/",
    "seed": 42,
    "num_epochs": 20,
    "batch_size": 128,
    "num_workers": 4,
    "clip_grad": 1.0,
    "warmup_ratio": 0.05,
    # Optimizer: "adamw" or "sgd"
    "optimizer": "adamw",
    "lr_adamw": 5e-4,
    "lr_sgd": 3e-3,           # same as JEPA
    "weight_decay": 1e-2,
    "momentum": 0.9,          # SGD only

    # ── Loss ──────────────────────────────────────────────────────────────────
    # LeJEPA objective: (1-λ)*pred_loss + λ*sigreg
    "lambda_sigreg": 0.008,
    "sigreg_num_slices": 1024,

    # ── Encoder (same architecture as JEPA) ──────────────────────────────────
    "encoder_embed_dim": 256,
    "nhead": 8,
    "num_encoder_layers": 5,
    "mlp_ratio": 4.0,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "head_dropout_forecasting": 0.2,
    "patch_size": 16,
    "ratio_patches": 32,   # 21×16 = 336 timesteps — aligned with forecasting context
    # Required by MonashDataPullerJEPA — mask indices are returned but ignored by LE-JEPA
    "mask_ratio": 0.25,
    "masking_type": "multi_block",
    "num_blocks": 2,

    # ── Augmentations ─────────────────────────────────────────────────────────
    "aug_noise_std": 0.05,              # Gaussian jitter std
    "aug_amplitude_range": (0.8, 1.2),  # amplitude scaling range
    "aug_channel_drop_p": 0.2,          # per-channel zero-out probability
    "aug_freq_mask_ratio": 0.3,         # fraction of FFT frequencies to mask
    # DWT augmentation — two views use different modes, same as DINO:
    #   view 1 (smooth/global): soft_threshold — strips noise, preserves structure
    #   view 2 (perturbed/local): high_perturb — adds Gaussian noise to detail coeffs
    # Set either to None to disable DWT for that view.
    # Options: 'low_pass', 'soft_threshold', 'zero_out_detail', 'high_perturb', 'band_scale'
    "view1_dwt_mode": "soft_threshold",
    "view2_dwt_mode": "high_perturb",
    # Shared DWT parameters (same as DINO defaults)
    "dwt_wavelet":                  "db4",
    "dwt_level":                    3,
    "dwt_soft_threshold_sigma":     0.3,
    "dwt_zero_out_ratio":           0.4,
    "dwt_finest_levels":            2,
    "dwt_high_perturb_noise_range": (0.1, 0.3),
    "dwt_band_scale_approx_range":  (0.80, 1.20),
    "dwt_band_scale_detail_range":  (0.40, 1.60),

    # ── Pretrain Dataset ──────────────────────────────────────────────────────
    "pretrain_dataset": "etth1",
    "forecast_dataset": "etth1",
    "timestampcols": ["date"],
    "input_variables": [
        ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
    ],
    "path_data": ["./data/ETTh1.csv"],
    "val_prec": 0.1,
    "test_prec": 0.1,

    # ── Pretraining data source ───────────────────────────────────────────────
    # pretrain_source: "monash" | "synthetic" | "monash+synthetic"
    "pretrain_source":    "monash",

    # ── Forecasting downstream ────────────────────────────────────────────────
    "epoch_t": 20,
    "context_t": 21,
    "horizon_t": 6,
    "patch_size_forcasting": 16,
    "patches_to_forcast": 6,
    "patches_size_forecasting": 16,
    "lr_forcasting": 2e-4,
    "batch_size_forecast": 128,
    "affine_revin": True,
    "forecasting_modes": ["zeroshot"],
}
