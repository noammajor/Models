# Defaults reproduce the per-model Monash protocol of the paper: the encoder is
# 8 layers, d_model 128, 16 heads, d_ff 512, patch length 16, over a 336-step
# context (21 patches), pre-trained on Monash for 20 epochs.
#   learning rate  1e-4
#   batch size     64
# Equivalent to: ./scripts/run_per_model.sh diffusion <task> <seed>
# Classification uses the same settings with a 1152-step context (72 patches).
config = {

    # ── Pretraining data source ───────────────────────────────────────────────
    # "monash" | "synthetic" | "monash+synthetic"
    "pretrain_source":    "monash",

    # ── Model architecture ────────────────────────────────────────────────────
    # e_layers is the encoder depth — swept over [2, 4, 8, 12, 24]
    # model: "PatchTST" uses a bidirectional encoder; "TimeDART" uses CausalTransformer
    "model":       "TimeDART",
    "e_layers":    8,
    "d_model":     128,    # matches embed_dim used across JEPA / LE-JEPA / DINO
    "n_heads":     16,
    "d_ff":        512,    # matches d_ff used across other models
    "patch_len":   16,     # matches patch_size used across JEPA / LE-JEPA
    "stride":      16,
    "seq_len":     336,    # context window length
    "label_len":   0,
    "dropout":     0.1,
    "head_dropout": 0.1,
    "head_dropout_forecasting": 0.2,

    # ── Diffusion ─────────────────────────────────────────────────────────────
    "time_steps":  1000,
    "scheduler":   "cosine",
    "mask_ratio":  1.0,

    # ── Pretraining optimisation ──────────────────────────────────────────────
    "train_epochs":    20,
    "batch_size":      64,
    "learning_rate":   1e-4,
    "lr_decay":        0.5,
    "num_workers":     4,

    # ── Forecasting fine-tune ─────────────────────────────────────────────────
    "epochs_forecasting":  20,
    "lr_forecasting":      1e-4,
    "batch_size_forecast": 128,
    "patience":            7,
    "lradj":               "decay",
    "pct_start":           0.3,
    "features":            "M",    # multivariate → multivariate
}
