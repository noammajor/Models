"""
SoftCLT pretrain helper.

Wraps TS2Vec + PatchTransformerWrapper and handles:
  - data loading from Monash, synthetic, or CSV sources
  - pretraining
  - saving/loading the PatchTSTEncoder backbone
"""

import sys
import os
from pathlib import Path

import numpy as np
import torch

# ── path wiring ───────────────────────────────────────────────────────────────
_SOFTCLT_DIR = Path(__file__).parent.resolve()
_TS2VEC_DIR  = _SOFTCLT_DIR / "softclt_ts2vec"
for _p in [str(_TS2VEC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from soft_ts2vec import TS2Vec


# ── data loading helpers ──────────────────────────────────────────────────────

def load_csv_train_data(csv_path: str, timestamp_col: str = "date") -> np.ndarray:
    """
    Load a CSV forecasting dataset, scale it, and return the training split as
    (1, T_train, C) float32.  TS2Vec.fit() will slice it into
    max_train_length=336 windows internally.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df   = pd.read_csv(csv_path)
    df   = df.drop(columns=[timestamp_col], errors="ignore")
    data = df.values.astype(np.float32)

    T, C      = data.shape
    train_len = int(T * 0.6)
    scaler    = StandardScaler()
    train     = scaler.fit_transform(data[:train_len])
    return train[np.newaxis]   # (1, T_train, C)


def load_monash_windows(monash_dir: str, seq_len: int, min_len: int = 512) -> np.ndarray:
    """Collect all Monash training windows as (N, seq_len, 1) float32."""
    from data_loaders.data_puller import MonashWindowDatasetTimeDart
    ds      = MonashWindowDatasetTimeDart(monash_dir, seq_len=seq_len, which="train", min_len=min_len)
    windows = [ds[i][0].numpy() for i in range(len(ds))]   # each (seq_len, 1)
    arr     = np.stack(windows)                              # (N, seq_len, 1)
    print(f"[SoftCLT] Monash windows: {arr.shape}")
    return arr


def load_synthetic_windows(synth_dir: str, seq_len: int, min_len: int = 512) -> np.ndarray:
    """Collect all synthetic training windows as (N, seq_len, C) float32."""
    from data_loaders.data_puller import SyntheticWindowDatasetTimeDart
    ds      = SyntheticWindowDatasetTimeDart(synth_dir, seq_len=seq_len, which="train", min_len=min_len)
    windows = [ds[i][0].numpy() for i in range(len(ds))]
    arr     = np.stack(windows)
    print(f"[SoftCLT] Synthetic windows: {arr.shape}")
    return arr


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_backbone(model: TS2Vec, path: str, epoch: int, cfg: dict):
    """Save only the PatchTSTEncoder backbone state dict."""
    torch.save({
        "backbone": model._net.backbone.state_dict(),
        "epoch":    epoch,
        "config":   cfg,
    }, path)
    print(f"[SoftCLT] Backbone saved → {path}")


def load_backbone_into_patchtst(ckpt_path: str, patchtst_model) -> int:
    """
    Load a SoftCLT backbone checkpoint into any PatchTST model.
    Returns the epoch the checkpoint was saved at.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone_sd  = ckpt["backbone"]
    model_sd     = patchtst_model.state_dict()
    loaded, skip = 0, 0
    new_sd = {}
    for k, v in backbone_sd.items():
        full_key = "backbone." + k
        if full_key in model_sd and model_sd[full_key].shape == v.shape:
            new_sd[full_key] = v
            loaded += 1
        else:
            skip += 1
    patchtst_model.load_state_dict(new_sd, strict=False)
    print(f"[SoftCLT] Loaded {loaded} backbone params into PatchTST ({skip} skipped / head)")
    return ckpt.get("epoch", 0)


# ── model builder ─────────────────────────────────────────────────────────────

def build_model(cfg: dict, c_in: int, device: str = "cuda") -> TS2Vec:
    """Instantiate TS2Vec with the PatchTransformerWrapper backend."""
    return TS2Vec(
        input_dims       = c_in,
        output_dims      = cfg["embed_dim"],
        patch_len        = cfg["patch_len"],
        patch_n_layers   = cfg["n_layers"],
        patch_n_heads    = cfg["n_heads"],
        patch_d_ff       = cfg["d_ff"],
        lr               = cfg["lr"],
        batch_size       = cfg["batch_size"],
        max_train_length = cfg["patch_len"] * cfg["num_patches"],   # 336
        soft_temporal    = (cfg.get("tau_temp", 0) > 0),
        soft_instance    = (cfg.get("tau_inst",  0) > 0),
        tau_temp         = cfg.get("tau_temp", 0),
        lambda_          = cfg.get("lambda_", 0.5),
        device           = device,
    )
