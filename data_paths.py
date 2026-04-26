"""
Centralised data paths shared across all models.

Edit values here when running on a different machine. Per-model configs may
override individual keys if a particular model needs a different location.
"""

DATA_PATHS = {
    # ── Pretraining ──────────────────────────────────────────────────────────
    "monash_data_dir":         "/home/shared/datasets/Monash",
    "monash_min_len":          512,
    "synthetic_data_dir":      "/home/shared/datasets/synthetic_data_TS",     # .arrow files
    "synthetic_mix_data_dir":  "/home/shared/datasets/synthetic_TS_Mix",      # smaller curated mix
    # ── Forecasting CSVs (consumed by dataset_registry) ──────────────────────
    "forecasting_data_dir":    "/home/shared/datasets/forecasting_data",
    # ── Downstream task datasets ─────────────────────────────────────────────
    "classification_data_dir": "/home/shared/datasets/Classification_TS",
    "anomaly_data_dir":        "/home/shared/datasets/Anomaly_TS",
}
