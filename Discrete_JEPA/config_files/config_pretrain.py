# ETTm1 configuration for Discrete JEPA
# Data paths use "./" relative to Discrete_JEPA/ directory;
# Train_and_downstream.py resolves these to absolute paths before passing to the model.

config = {
    "path_save": "./output_model/DiscreteJEPA/",
    "lr": 2e-3,
    "num_epochs": 20,
    "ema_momentum": 0.996,
    "codebook_lr": 1e-3,
    "weight_decay": 3e-3,
    "perplexity_loss_weight": 0.5,
    "lr_pred": 3e-3,
    "weight_decay_pred": 2e-3,

    # masking
    "mask_ratio": 0.25,
    "masking_type": "multi_block",
    "num_blocks": 2,

    # encoder
    "num_semantic_tokens": 16,
    "encoder_embed_dim": 256,
    "nhead": 8,
    "num_encoder_layers": 5,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "qk_scale": None,
    "drop_rate": 0.00,
    "attn_drop_rate": 0.00,
    "kernel_size": 6,
    "encoder_kernel_size": 6,
    "embed_bias": True,
    "encoder_embed_bias": True,
    "codebook_size": 256,
    "commitment_cost": 0.25,
    "vq_ema_decay": 0.99,
    "patch_size": 16,
    "patch_size_forcasting": 16,

    # predictor
    "predictor_embed_dim": 128,
    "predictor_nhead": 4,
    "predictor_num_layers": 3,

    # ── Datasets ──────────────────────────────────────────────────────────────
    # Names must match keys in dataset_registry.py.
    # forecast_dataset defaults to pretrain_dataset when left as None.
    "pretrain_dataset":  "etth2",
    "forecast_dataset":  "etth2",

    # data
    "checkpoint_save": 5000,
    "checkpoint_print": 30,
    "ratio_patches": 32,
    "batch_size": 64,

    # loader
    "clip_grad": 1.0,
    "warmup_ratio": 0.05,
    "ipe_scale": 1.25,

    # loss weights
    "lambda_weights": {
        "P2P":1.0,
        "S2P": 1.0,
        "P2S": 0.80,
    },
    "preplexity_coeff": 0.04,
    "token_diversity": 0.15,
    "vigreg_patches": 0.00,
    "decorr_coeff": 0.0,
    "vigreg_var": 1.0,
    "vigreg_covar": 0.04,
    "grounding_coeff": 0.10,
    "beta_vq": 1.0,
    "vq_warmup": 0.05,
    "val_prec": 0.2,
    "test_prec": 0.1,

    # ── ETTm1: 7 variables split into 2 groups of 4 (last group repeats HUFL) ──
    "timestampcols": ["date"],
    "input_variables": [
        "HUFL", "HULL", "MUFL", "MULL","LUFL", "LULL", "OT",   # HUFL repeated to fill 4th slot
    ],
    "path_data": [
        "./data/ETThe.csv"
    ],
    "chunk_size": 128,

    # forecasting downstream
    "epoch_t": 100,
    "context_t": 21,
    "horizon_t": 6,
    "input_variables_forcasting": [
        ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
    ],
    "val_prec_forcasting": 0.1,
    "test_prec_forcasting": 0.1,
    "window_step_forecasting": 1,  # raw-timestep stride between windows — matches PatchTST/DINO (non-overlapping patches, max training samples)
    "timestampcols_forcasting": ["date"],
    "path_data_forcasting": ["./data/ETTm1.csv"],
    "patches_to_forcast": 6,
    "patches_size_forecasting": 32,
    "lr_forcasting": 1e-3,
    "affine_revin": True,

    # ── Forecasting modes ─────────────────────────────────────────────────────
    # List any combination of: "zeroshot", "finetuning", "predictor"
    #   "zeroshot"   — frozen encoder, linear probe trained on top (forcasting_zeroshot)
    #   "finetuning" — encoder + head fine-tuned jointly (finetuning_forecasting)
    #   "predictor"  — frozen encoder + predictor, only decoder trained (predictor_forecasting)
    "forecasting_modes": ["zeroshot"],

    # ── Pretraining data source ───────────────────────────────────────────────
    # pretrain_source: "monash" | "synthetic" | "monash+synthetic"
    "pretrain_source":    "monash",
    "monash_data_dir":    "/home/shared/datasets/Monash",
    "monash_min_len":     512,
    "synthetic_data_dir": "/home/shared/datasets/synthetic_data_TS",  # dir containing .arrow files
}
