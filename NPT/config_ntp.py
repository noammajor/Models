config = {
    # ── Datasets ──────────────────────────────────────────────────────────────
    "pretrain_dataset":  "monash",
    "forecast_dataset":  "ettm1",

    # ── Pretraining data source ───────────────────────────────────────────────
    # pretrain_source: "monash" | "synthetic" | "monash+synthetic"
    "pretrain_source":    "monash",
    "monash_data_dir":    "/home/shared/datasets/Monash",
    "monash_min_len":     512,
    "synthetic_data_dir": "/home/shared/datasets/synthetic",  # dir containing .arrow files

    # ── Patching (JEPA-style data loader) ─────────────────────────────────────
    # patch_size  = length of each patch (≡ patch_len in PatchTST)
    # ratio_patches = number of patches per window (context = patch_size * ratio_patches)
    "patch_size":     16,
    "ratio_patches":  27,   # 16 * 27 = 432 total window (21 context + 6 horizon)
    "context_patches": 21,  # fixed context size for forecasting (independent of pred_len)
    "masking_type":   "causal",   # forecasting-style: context = first patches, target = last horizon patches
    "mask_ratio":     0.4,
    "num_blocks":     1,
    "val_prec":       0.1,
    "test_prec":      0.1,

    # ── Model ─────────────────────────────────────────────────────────────────
    "n_layers":     3,
    "n_heads":      16,
    "d_model":      128,
    "d_ff":         512,
    "dropout":      0.2,
    "head_dropout": 0.2,
    "act":          "gelu",

    # ── Training ──────────────────────────────────────────────────────────────
    "num_epochs":    20,
    "batch_size":    64,
    "lr":            1e-4,
    "revin":         True,
    "num_workers":   0,

    # ── Zero-Shot Forecasting ─────────────────────────────────────────────────
    # horizon_t          = number of future patches to forecast
    # patch_size_forcasting is automatically set to patch_size at runtime
    "horizon_t":              6,     # 6 patches × 16 = 96-step horizon
    "val_prec_forcasting":    0.1,
    "test_prec_forcasting":   0.1,
    "window_step_forecasting": 1,    # stride between forecasting windows
    "lr_forcasting":          1e-4,
    "epochs_forecasting":     20,

    # ── Misc ──────────────────────────────────────────────────────────────────
    "pretrained_model_id": 1,
}
