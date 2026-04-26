config = {
    # ── Datasets ──────────────────────────────────────────────────────────────
    "pretrain_dataset":   "monash",
    "forecast_dataset":   "ettm1",

    # ── Pretraining data source ───────────────────────────────────────────────
    # pretrain_source: "monash" | "synthetic" | "monash+synthetic"
    "pretrain_source":    "monash",
    # add your paths for the datasets here if different from the defaults

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
    "batch_size":          128,
    "num_workers":         4,
    "batch_size_forecast": 32,
    "finetune_lr":         4e-4,    # forecasting fine-tune LR (subprocess flag)
    "pretrained_model_id": 1,
    "model_type":          "based_model",

    # ── Classification ────────────────────────────────────────────────────────
    "epoch_classification":      20,
    "lr_classification":         None,  # head LR; None → 1e-3 default
    "lr_classification_encoder": None,  # encoder LR when fine-tuning; None → head_lr

    # ── Anomaly Detection ─────────────────────────────────────────────────────
    "epoch_anomaly":         10,
    "lr_anomaly":            None,  # head LR; None → 1e-3 default
    "lr_anomaly_encoder":    None,  # encoder LR when fine-tuning; None → head_lr
}
