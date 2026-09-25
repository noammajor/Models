# Defaults reproduce the per-model Monash protocol of the paper: the encoder is
# 8 layers, d_model 128, 16 heads, d_ff 512, patch length 16, over a 336-step
# context (21 patches), pre-trained on Monash for 20 epochs.
#   learning rate  2e-4
#   batch size     128
# Equivalent to: ./scripts/run_per_model.sh softclt <task> <seed>
# Classification uses the same settings with a 1152-step context (72 patches).
config = {
    # ── Architecture (DINO-aligned) ───────────────────────────────────────────
    "patch_len":    16,
    "num_patches":  21,      # 21 × 16 = 336 timesteps (same context window as DINO)
    # SoftCLT pretrains with NON-overlapping patches (unfold step == patch_len). Without
    # this key the forecaster inherits step_size=8 from DINO/config.py and unfolds with
    # 50% overlap — a train/test mismatch, and it also changes num_patch (41 vs 21) so the
    # pretrained W_pos no longer fits. Keep it equal to patch_len.
    "step_size":    16,
    "embed_dim":    128,
    "n_layers":     8,
    "n_heads":      16,
    "d_ff":         512,
    "dropout":      0.1,

    # ── Soft-CL hypers ────────────────────────────────────────────────────────
    # tau_temp > 0  → soft temporal labels (sigmoid timelag weighting)
    # tau_inst = 0  → hard instance labels (DTW sim-matrix not computed)
    "tau_temp":  2.0,
    "tau_inst":  0.0,
    "lambda_":   0.5,     # balance instance vs temporal loss

    # ── Pretraining ──────────────────────────────────────────────────────────
    "lr":         2e-4,
    "batch_size": 128,
    "epochs":     20,

    # pretrain_source: "monash" | "synthetic" | "monash+synthetic" | None (CSV)
    "pretrain_source": "monash",
    "monash_min_len":  512,

    "output_dir": "./checkpoints_softclt",

    # ── Downstream: Forecasting ───────────────────────────────────────────────
    "pred_len":               96,
    "epochs_forecasting":     20,
    "lr_forecasting":         2e-4,
    "min_lr_forecasting":     2e-5,
    "batch_size_forecast":    128,
    "head_dropout":           0.1,
    "head_dropout_forecasting": 0.2,
    "linear_probe":           True,

    # ── Downstream: Classification ────────────────────────────────────────────
    "epochs_classification":     20,
    "min_lr_classification":     1e-6,

    # ── Downstream protocol (shared across every objective and the controls) ──
    # Classification: LR 1e-3, batch 64, 20 epochs.
    # Anomaly detection: Adam, LR 1e-3, cosine annealing, 10 epochs,
    #   early stopping at patience 3.
    # MLP probe (when --head mlp): one hidden layer of width 512 with GELU,
    #   built in the model code. Dropout 0.2 for classification and
    #   forecasting; the anomaly-detection probe uses no dropout.
    "lr_classification":        1e-3,
    "batch_size_classification": 64,
    "epoch_classification":     20,
    "lr_anomaly":               1e-3,
    "epoch_anomaly":            10,
    "anomaly_patience":         3,
}
