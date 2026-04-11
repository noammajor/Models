config = {

    # ── Pretraining data source ───────────────────────────────────────────────
    # "monash" | "synthetic" | "monash+synthetic"
    "pretrain_source":    "monash",
    "monash_data_dir":    "/home/shared/datasets/Monash",
    "monash_min_len":     512,
    "synthetic_data_dir": "/home/shared/datasets/synthetic_data_TS",

    # ── Model architecture ────────────────────────────────────────────────────
    # e_layers is the encoder depth — swept over [2, 4, 8, 12, 24]
    # model: "PatchTST" uses a bidirectional encoder; "TimeDART" uses CausalTransformer
    "model":       "PatchTST",
    "e_layers":    3,
    "d_model":     256,    # matches embed_dim used across JEPA / LE-JEPA / DINO
    "n_heads":     8,
    "d_ff":        512,    # matches d_ff used across other models
    "patch_len":   16,     # matches patch_size used across JEPA / LE-JEPA
    "stride":      16,
    "seq_len":     336,    # context window length
    "label_len":   0,
    "dropout":     0.1,
    "head_dropout": 0.1,

    # ── Diffusion ─────────────────────────────────────────────────────────────
    "time_steps":  1000,
    "scheduler":   "cosine",
    "mask_ratio":  1.0,

    # ── Pretraining optimisation ──────────────────────────────────────────────
    "train_epochs":    20,
    "batch_size":      128,
    "learning_rate":   1e-4,
    "lr_decay":        0.5,
    "num_workers":     4,

    # ── Forecasting fine-tune ─────────────────────────────────────────────────
    "epochs_forecasting":  10,
    "lr_forecasting":      1e-4,
    "batch_size_forecast": 128,
    "patience":            3,
    "lradj":               "decay",
    "pct_start":           0.3,
    "features":            "M",    # multivariate → multivariate
}
