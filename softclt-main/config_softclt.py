config = {
    # ── Architecture (DINO-aligned) ───────────────────────────────────────────
    "patch_len":    16,
    "num_patches":  21,      # 21 × 16 = 336 timesteps (same context window as DINO)
    # SoftCLT pretrains with NON-overlapping patches (unfold step == patch_len). Without
    # this key the forecaster inherits step_size=8 from TSDiNO/config.py and unfolds with
    # 50% overlap — a train/test mismatch, and it also changes num_patch (41 vs 21) so the
    # pretrained W_pos no longer fits. Keep it equal to patch_len.
    "step_size":    16,
    "embed_dim":    128,
    "n_layers":     5,
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
    "lr":         1e-3,
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
    "lr_classification":         1e-3,
    "min_lr_classification":     1e-6,
    "batch_size_classification": 64,
}
