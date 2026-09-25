# Defaults reproduce the per-model Monash protocol of the paper: the encoder is
# 8 layers, d_model 128, 16 heads, d_ff 512, patch length 16, over a 336-step
# context (21 patches), pre-trained on Monash for 20 epochs.
#   learning rate  1e-4
#   batch size     64
# Non-overlapping patches (stride = patch_len = 16).
# Equivalent to: ./scripts/run_per_model.sh mae <task> <seed>
# Classification uses the same settings with a 1152-step context (72 patches).
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
    "stride":          16,   # =patch_len → non-overlapping, 336/16 = 21 patches (overlap experiment: 8)

    # ── Model ─────────────────────────────────────────────────────────────────
    "n_layers":     8,
    "n_heads":      16,
    "d_model":      128,
    "d_ff":         512,
    "dropout":      0.2,
    "head_dropout": 0.2,
    "head_dropout_forecasting": 0.2,
    "revin":        True,

    # ── Pretraining ───────────────────────────────────────────────────────────
    "mask_ratio":          0.4,
    "n_epochs_pretrain":   20,
    "batch_size":          64,
    "lr":                  1e-4,   # pre-training LR; overridden by --lr
    "num_workers":         4,
    "batch_size_forecast": 128,
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
