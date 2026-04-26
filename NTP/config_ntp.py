# Configuration dictionary for NTP pretraining and downstream tasks.
config = {
    # ── Datasets ──────────────────────────────────────────────────────────────
    "pretrain_dataset":  "monash",
    "forecast_dataset":  "ettm1",

    # ── Pretraining data source ───────────────────────────────────────────────
    # pretrain_source: "monash" | "synthetic" | "monash+synthetic"
    "pretrain_source":    "monash",
    #put in your paths for the datasets here if different from the defaults

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
    "batch_size_forecast": 256,
    "lr":            1e-4,
    "revin":         True,
    "num_workers":   4,

    # ── Forecasting ───────────────────────────────────────────────────────────
    # horizon_t          = number of future patches to forecast
    # patch_size_forcasting is automatically set to patch_size at runtime
    "horizon_t":              6,     # 6 patches × 16 = 96-step horizon
    "lr_forcasting":          1e-4,  # head LR for forecasting
    "lr_forcasting_encoder":  None,  # encoder LR when fine-tuning (linear_probe=False); None → env var
    "epochs_forecasting":     20,

    # ── Classification ────────────────────────────────────────────────────────
    "epoch_classification":      20,
    "lr_classification":         None,  # head LR; None → script default (1e-3)
    "lr_classification_encoder": None,  # encoder LR when fine-tuning; None → head_lr

    # ── Anomaly Detection ─────────────────────────────────────────────────────
    "epoch_anomaly":         10,
    "lr_anomaly":            None,  # head LR; None → script default (1e-3)
    "lr_anomaly_encoder":    None,  # encoder LR when fine-tuning; None → head_lr

    # ── Misc ──────────────────────────────────────────────────────────────────
    "pretrained_model_id": 1,

    # ── Save path ─────────────────────────────────────────────────────────────
    # If None, defaults to: NTP/saved_models/{pretrain_source}/ntp/layers{n_layers}/
    # Set to a directory string to override.
    "path_save":      None,
}
