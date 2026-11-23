# Defaults reproduce the per-model Monash protocol of the paper: the encoder is
# 8 layers, d_model 128, 16 heads, d_ff 512, patch length 16, over a 336-step
# context (21 patches), pre-trained on Monash for 20 epochs.
#   learning rate  1e-4 encoder, 1.2e-2 augmentation net (AutoTCL repo defaults)
#   batch size     128
# Equivalent to: ./scripts/run_per_model.sh autotcl <task> <seed>
# Classification uses the same settings with a 1152-step context (72 patches).
#
# AutoTCL (Zheng et al., ICLR 2024, "Parametric Augmentation for Time Series
# Contrastive Learning") is TS2Vec-style contrastive learning whose positive
# view is produced by a learned augmentation network instead of a fixed
# transform: a small dilated-conv net reads the raw window and emits a
# per-timestep keep gate h(x) (Gumbel-relaxed, optionally binarised) and a
# per-timestep scale g(x); the view is  x' = g(x) * h(x) * x.  The encoder is
# trained with InfoNCE on max-pooled embeddings plus a local InfoNCE over
# sub-window crops; the augmentation net is trained in alternation to keep the
# two views' embedding distributions close (MMD) while gating enough of the
# input (the "PRI" objective), with a smoothness prior on the gate.
config = {
    # ── Architecture (DINO-aligned) ───────────────────────────────────────────
    "patch_len":    16,
    "num_patches":  21,      # 21 × 16 = 336 timesteps (same context window as DINO)
    # Non-overlapping patches at pre-training time; keep step_size == patch_len so
    # the downstream forecaster unfolds the same way and W_pos still fits.
    "step_size":    16,
    "embed_dim":    128,
    "n_layers":     8,
    "n_heads":      16,
    "d_ff":         512,
    "dropout":      0.1,

    # ── Augmentation network h(x), g(x) ──────────────────────────────────────
    "aug_hidden_dims": 64,      # width of the dilated-conv augmentation net
    "aug_depth":       2,       # residual dilated-conv blocks in it
    "aug_dim":         16,      # its output width, read by the two 1-unit heads
    "augmask_mode":    "binomial",   # input masking inside the augmentation net
    "gamma_zeta":      0.05,    # stretch of the relaxed Bernoulli gate
    "gumbel_bias":     0.05,    # noise floor of the Gumbel sample
    "hard_mask":       True,    # straight-through binarised gate
    "meta_lr":         1.2e-2,  # AdamW rate of the augmentation net (AutoTCL repo default --meta_lr)
    "ratio_step":      1,       # augmentation-net update every k encoder steps
    "reg_thres":       0.4,     # minimum mean gate mass before the penalty applies
    "reg_weight":      0.1,     # weight of that gate-mass penalty
    "regular_weight":  0.0,     # weight of the temporal-smoothness prior on h(x)

    # ── Contrastive loss on the encoder ───────────────────────────────────────
    "temperature":   1.0,       # InfoNCE temperature (AutoTCL default)
    "local_weight":  0.1,       # weight of the local (crop-level) InfoNCE
    "local_k":       16,        # crops per window in the local term (TS2Vec/InfoTS)
    # Input masking of the encoder view. SoftCLT's transformer wrapper applies no
    # input mask, and AutoTCL's learned augmentation already gates timesteps, so
    # the encoder sees the un-masked window by default.
    "mask_mode":     "all_true",

    # ── Pretraining ──────────────────────────────────────────────────────────
    "lr":         1e-4,     # AutoTCL repo default --lr (encoder)
    "batch_size": 128,
    "epochs":     20,

    # pretrain_source: "monash" | "synthetic" | "monash+synthetic" | None (CSV)
    "pretrain_source": "monash",
    "monash_min_len":  512,

    "output_dir": "./checkpoints_autotcl",

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
