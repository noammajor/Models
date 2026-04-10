# ETTm1 configuration for JEPA (P2P only — no VQ / semantic tokens)
# Data paths use "./" relative to the JEPA/ directory.

config = {
    "path_save": "./output_model/JEPA/",
    "lr": 3e-3,
    "num_epochs": 20,
    "ema_momentum": 0.996,
    "weight_decay": 3e-3,
    "lr_pred": 5e-3,
    "weight_decay_pred": 1e-4,

    # masking
    "mask_ratio": 0.25,
    "masking_type": "multi_block",
    "num_blocks": 2,

    # encoder 
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
    "patch_size": 16,
    "patch_size_forcasting": 16,

    # predictor
    "predictor_embed_dim": 128,
    "predictor_nhead": 4,
    "predictor_num_layers":3,

    # ── Datasets ──────────────────────────────────────────────────────────────
    "pretrain_dataset": "etth1",
    "forecast_dataset": "etth1",

    # data
    "checkpoint_save": 5000,
    "checkpoint_print": 30,
    "ratio_patches": 32,
    "batch_size": 64,
    "num_workers": 4,

    # loader
    "clip_grad": 1.0,
    "warmup_ratio": 0.05,
    "ipe_scale": 1.25,

    # loss weights
    "invar": 1.0,
    "vigreg_var": 1.0,   # VICReg variance term coefficient
    "vigreg_cov": 0.04,   # VICReg covariance term coefficient
    "val_prec": 0.1,
    "test_prec": 0.1,

    # ── ETTm1: 7 variables ────────────────────────────────────────────────────
    "timestampcols": ["date"],
    "input_variables": [
        "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT",
    ],
    "path_data": [
        "./data/ETTh1.csv"
    ],
    "chunk_size": 128,

    # forecasting downstream
    "epoch_t": 20,
    "context_t": 21,
    "horizon_t":6,
    "input_variables_forcasting": [
        ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
    ],
    "val_prec_forcasting": 0.1,
    "test_prec_forcasting": 0.1,
    "window_step_forecasting": 1,
    "timestampcols_forcasting": ["date"],
    "path_data_forcasting": ["./data/ETTh1.csv"],
    "patches_to_forcast": 8,
    "patches_size_forecasting": 16,
    "lr_forcasting": 4e-4,
    "batch_size_forecast": 128,
    "affine_revin": True,
    

    # ── Forecasting modes ─────────────────────────────────────────────────────
    "forecasting_modes": ["zeroshot"],

    # ── Pretraining data source ───────────────────────────────────────────────
    # pretrain_source: "monash" | "synthetic" | "monash+synthetic"
    "pretrain_source":    "monash",
    "monash_data_dir":    "/home/shared/datasets/Monash",
    "monash_min_len":     512,
    "synthetic_data_dir": "/home/shared/datasets/synthetic_data_TS",  # dir containing .arrow files
}
