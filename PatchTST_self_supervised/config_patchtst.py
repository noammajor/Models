config = {
    # ── Datasets ──────────────────────────────────────────────────────────────
    "pretrain_dataset":   "monash",
    "forecast_dataset":   "ettm1",

    # ── Pretraining data source ───────────────────────────────────────────────
    # pretrain_source: "monash" | "synthetic" | "monash+synthetic"
    "pretrain_source":    "monash",
    "monash_data_dir":    "/home/shared/datasets/Monash",
    "monash_min_len":     512,
    "synthetic_data_dir": "/home/shared/datasets/synthetic_data_TS",  # dir containing .arrow files

    # ── Patching ──────────────────────────────────────────────────────────────
    "context_points":  336,
    "target_points":   96,
    "patch_len":       16,
    "stride":          16,

    # ── Model ─────────────────────────────────────────────────────────────────
    "n_layers":     3,
    "n_heads":      16,
    "d_model":      128,
    "d_ff":         512,
    "dropout":      0.2,
    "head_dropout": 0.2,
    "revin":        True,

    # ── Pretraining ───────────────────────────────────────────────────────────
    "mask_ratio":          0.4,
    "n_epochs_pretrain":   20,
    "batch_size":          64,
    "num_workers":         4,
    "batch_size_forecast": 32,
    "finetune_lr":         4e-4,
    "pretrained_model_id": 1,
    "model_type":          "based_model",
}
