config = {
    # ── Save ──────────────────────────────────────────────────────────────────
    "path_save": "./output_model/Hybrid/",
    "seed": 42,

    # ── Hybrid mixing ─────────────────────────────────────────────────────────
    # φ ∈ [0, 1]: loss = φ·lejepa_loss + (1−φ)·ntp_loss
    # φ=1 → pure LEJEPA,  φ=0 → pure NTP
    "phi": 0.1,

    # ── Training ──────────────────────────────────────────────────────────────
    "num_epochs":    20,
    "batch_size":    128,
    "num_workers":   4,
    "clip_grad":     1.0,
    "warmup_ratio":  0.05,
    "optimizer":     "adamw",
    "lr_adamw":      5e-4,
    "weight_decay":  1e-2,
    "revin":         True,   # instance-normalize patches before both tasks

    # ── LEJEPA loss ───────────────────────────────────────────────────────────
    "lambda_sigreg":    0.008,
    "sigreg_num_slices": 1024,

    # ── Shared encoder (LEJEPA architecture) ──────────────────────────────────
    "encoder_embed_dim":  256,
    "nhead":              8,
    "num_encoder_layers": 5,
    "mlp_ratio":          4.0,
    "drop_rate":          0.0,
    "attn_drop_rate":     0.0,

    # ── Patching ──────────────────────────────────────────────────────────────
    # Total window = ratio_patches patches; NTP splits into context + horizon.
    # LEJEPA branch uses the full ratio_patches window for contrastive task.
    "patch_size":      16,
    "ratio_patches":   27,    # 21 context + 6 horizon patches
    "context_patches": 21,
    "horizon_t":       6,
    "mask_ratio":      0.25,
    "masking_type":    "multi_block",
    "num_blocks":      2,

    # ── NTP prediction head ───────────────────────────────────────────────────
    # ntp_head_d_ff defaults to 4 × encoder_embed_dim if not set
    "head_dropout": 0.2,
    "act":          "gelu",

    # ── Augmentations — LE-JEPA set ───────────────────────────────────────────
    "aug_noise_std":         0.05,
    "aug_amplitude_range":   (0.8, 1.2),
    "aug_channel_drop_p":    0.2,
    "aug_freq_mask_ratio":   0.3,
    # DWT (shared with DINO; view modes below)
    "view1_dwt_mode":        "soft_threshold",
    "view2_dwt_mode":        "high_perturb",
    "dwt_wavelet":                  "db4",
    "dwt_level":                    3,
    "dwt_soft_threshold_sigma":     0.3,
    "dwt_zero_out_ratio":           0.4,
    "dwt_finest_levels":            2,
    "dwt_high_perturb_noise_range": (0.1, 0.3),
    "dwt_band_scale_approx_range":  (0.80, 1.20),
    "dwt_band_scale_detail_range":  (0.40, 1.60),

    # ── Augmentations — TSDiNO-exclusive set ──────────────────────────────────
    "aug_polar_warp_range":           (0.7, 1.3),
    "aug_rotation_angle_range":       (0, 0.3927),   # (0, π/8)
    "aug_boost_b_range":              (0.01, 0.3),
    "aug_lorentz_v_range":            (0.2, 0.6),
    "aug_hyperbolic_warp_range":      (0.5, 1.5),
    "aug_hyperbolic_shift_magnitude": 0.3,

    # ── Pretraining data source ───────────────────────────────────────────────
    # "monash" | "synthetic" | "monash+synthetic"
    "pretrain_source": "monash",
    "val_prec":  0.1,
    "test_prec": 0.1,

    # ── Downstream forecasting ────────────────────────────────────────────────
    "head_dropout_forecasting": 0.2,
    "epoch_t":               20,
    "context_t":             21,
    "patch_size_forcasting": 16,
    "patches_to_forcast":    6,
    "patches_size_forecasting": 16,
    "lr_forcasting":         2e-4,
    "batch_size_forecast":   128,
    "affine_revin":          True,
    "forecasting_modes":     ["zeroshot"],
}
