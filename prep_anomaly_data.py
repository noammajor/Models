#!/usr/bin/env python3
"""
prep_anomaly_data.py — Preprocess anomaly detection datasets into standardised format.

Reads each dataset from its native format (2D .npy, CSV, or Excel),
applies a sliding window, and saves as:
    {out_dir}/{DATASET}/train.npy       [N_train, window_size, C]
    {out_dir}/{DATASET}/test.npy        [N_test,  window_size, C]
    {out_dir}/{DATASET}/test_labels.npy [N_test,  window_size]   int

Usage:
    python prep_anomaly_data.py
    python prep_anomaly_data.py --window_size 100 --stride 100
    python prep_anomaly_data.py --in_dir /path/to/raw --out_dir /path/to/Anomaly_TS
"""

import argparse
import numpy as np
import os
from pathlib import Path


def sliding_windows(x, window_size, stride):
    """x: [T, C] → [N, window_size, C]"""
    T = x.shape[0]
    starts = range(0, T - window_size + 1, stride)
    return np.stack([x[s:s + window_size] for s in starts])


def sliding_windows_1d(x, window_size, stride):
    """x: [T] → [N, window_size]"""
    T = x.shape[0]
    starts = range(0, T - window_size + 1, stride)
    return np.stack([x[s:s + window_size] for s in starts])


def load_npy(path):
    return np.load(path, allow_pickle=True).astype(np.float32)


def load_npy_labels(path):
    return np.load(path, allow_pickle=True).astype(np.int64)


def process_smd(in_dir, out_dir, window_size, stride):
    d = Path(in_dir) / "SMD"
    o = Path(out_dir) / "SMD"
    o.mkdir(parents=True, exist_ok=True)

    X_tr = load_npy(d / "SMD_train.npy")           # [T, 38]
    X_te = load_npy(d / "SMD_test.npy")             # [T, 38]
    labels = load_npy_labels(d / "SMD_test_label.npy")  # [T]

    # Normalize using train stats
    mean = X_tr.mean(axis=0, keepdims=True)
    std  = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    train_w  = sliding_windows(X_tr, window_size, stride)
    test_w   = sliding_windows(X_te, window_size, stride)
    labels_w = sliding_windows_1d(labels, window_size, stride)

    np.save(o / "train.npy",       train_w.astype(np.float32))
    np.save(o / "test.npy",        test_w.astype(np.float32))
    np.save(o / "test_labels.npy", labels_w.astype(np.int64))
    print(f"SMD  → train {train_w.shape}  test {test_w.shape}  labels {labels_w.shape}")


def process_msl(in_dir, out_dir, window_size, stride):
    d = Path(in_dir) / "MSL"
    o = Path(out_dir) / "MSL"
    o.mkdir(parents=True, exist_ok=True)

    X_tr = load_npy(d / "MSL_train.npy")                # [T, 55]
    X_te = load_npy(d / "MSL_test.npy")                 # [T, 55]
    labels = np.load(d / "MSL_test_label.npy").astype(np.int64)  # [T]

    mean = X_tr.mean(axis=0, keepdims=True)
    std  = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    train_w  = sliding_windows(X_tr, window_size, stride)
    test_w   = sliding_windows(X_te, window_size, stride)
    labels_w = sliding_windows_1d(labels, window_size, stride)

    np.save(o / "train.npy",       train_w.astype(np.float32))
    np.save(o / "test.npy",        test_w.astype(np.float32))
    np.save(o / "test_labels.npy", labels_w.astype(np.int64))
    print(f"MSL  → train {train_w.shape}  test {test_w.shape}  labels {labels_w.shape}")


def process_smap(in_dir, out_dir, window_size, stride):
    d = Path(in_dir) / "SMAP"
    o = Path(out_dir) / "SMAP"
    o.mkdir(parents=True, exist_ok=True)

    X_tr = load_npy(d / "SMAP_train.npy")               # [T, 25]
    X_te = load_npy(d / "SMAP_test.npy")                # [T, 25]
    labels = np.load(d / "SMAP_test_label.npy").astype(np.int64)  # [T]

    mean = X_tr.mean(axis=0, keepdims=True)
    std  = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    train_w  = sliding_windows(X_tr, window_size, stride)
    test_w   = sliding_windows(X_te, window_size, stride)
    labels_w = sliding_windows_1d(labels, window_size, stride)

    np.save(o / "train.npy",       train_w.astype(np.float32))
    np.save(o / "test.npy",        test_w.astype(np.float32))
    np.save(o / "test_labels.npy", labels_w.astype(np.int64))
    print(f"SMAP → train {train_w.shape}  test {test_w.shape}  labels {labels_w.shape}")


def process_psm(in_dir, out_dir, window_size, stride):
    import pandas as pd
    d = Path(in_dir) / "PSM"
    o = Path(out_dir) / "PSM"
    o.mkdir(parents=True, exist_ok=True)

    X_tr = pd.read_csv(d / "train.csv").drop(columns=["timestamp_(min)"], errors="ignore").values.astype(np.float32)
    X_te = pd.read_csv(d / "test.csv").drop(columns=["timestamp_(min)"], errors="ignore").values.astype(np.float32)
    label_df = pd.read_csv(d / "test_label.csv")
    label_col = "label" if "label" in label_df.columns else label_df.columns[-1]
    labels = label_df[label_col].values.flatten().astype(np.int64)

    # Replace NaNs
    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0)

    mean = X_tr.mean(axis=0, keepdims=True)
    std  = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    train_w  = sliding_windows(X_tr, window_size, stride)
    test_w   = sliding_windows(X_te, window_size, stride)
    labels_w = sliding_windows_1d(labels, window_size, stride)

    np.save(o / "train.npy",       train_w.astype(np.float32))
    np.save(o / "test.npy",        test_w.astype(np.float32))
    np.save(o / "test_labels.npy", labels_w.astype(np.int64))
    print(f"PSM  → train {train_w.shape}  test {test_w.shape}  labels {labels_w.shape}")


def process_swat(in_dir, out_dir, window_size, stride):
    import pandas as pd
    d = Path(in_dir) / "SWaT"
    o = Path(out_dir) / "SWaT"
    o.mkdir(parents=True, exist_ok=True)

    # Try CSV first, fall back to Excel
    if (d / "swat_train.csv").exists():
        train_df = pd.read_csv(d / "swat_train.csv", low_memory=False)
        train_df = train_df.apply(pd.to_numeric, errors='coerce')
        train_df = train_df.dropna(axis=1, how='all')
    elif (d / "SWaT_Dataset_Normal_v1.xlsx").exists():
        train_df = pd.read_excel(d / "SWaT_Dataset_Normal_v1.xlsx", header=1)
    else:
        print("SWaT: no train file found, skipping")
        return

    if (d / "SWaT_Dataset_Attack_v0.xlsx").exists():
        test_df = pd.read_excel(d / "SWaT_Dataset_Attack_v0.xlsx", header=1)
    else:
        print("SWaT: no attack file found, skipping")
        return

    # Drop timestamp/label columns, extract label
    label_col = "Normal/Attack" if "Normal/Attack" in test_df.columns else test_df.columns[-1]
    labels = (test_df[label_col].str.strip().str.lower() != "normal").astype(np.int64).values

    # Drop non-numeric and label columns by selecting only float64/float32 cols
    for df in [train_df, test_df]:
        for col in list(df.columns):
            if col.strip() in ("Timestamp", "Normal/Attack") or df[col].dtype == object:
                df.drop(columns=[col], inplace=True)

    X_tr = train_df.values.astype(np.float32)
    X_te = test_df.values.astype(np.float32)
    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0)

    mean = X_tr.mean(axis=0, keepdims=True)
    std  = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    train_w  = sliding_windows(X_tr, window_size, stride)
    test_w   = sliding_windows(X_te, window_size, stride)
    labels_w = sliding_windows_1d(labels, window_size, stride)

    np.save(o / "train.npy",       train_w.astype(np.float32))
    np.save(o / "test.npy",        test_w.astype(np.float32))
    np.save(o / "test_labels.npy", labels_w.astype(np.int64))
    print(f"SWaT → train {train_w.shape}  test {test_w.shape}  labels {labels_w.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",      default="anomoly detection data")
    parser.add_argument("--out_dir",     default="Anomaly_TS")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--stride",      type=int, default=100)
    parser.add_argument("--datasets",    nargs="+",
                        default=["SMD", "MSL", "SMAP", "PSM", "SWaT"])
    args = parser.parse_args()

    print(f"window_size={args.window_size}  stride={args.stride}")
    print(f"in:  {args.in_dir}")
    print(f"out: {args.out_dir}\n")

    processors = {
        "SMD":  process_smd,
        "MSL":  process_msl,
        "SMAP": process_smap,
        "PSM":  process_psm,
        "SWaT": process_swat,
    }

    for ds in args.datasets:
        if ds in processors:
            try:
                processors[ds](args.in_dir, args.out_dir, args.window_size, args.stride)
            except Exception as e:
                print(f"{ds}: ERROR — {e}")
        else:
            print(f"{ds}: unknown dataset, skipping")

    print(f"\nDone. Standardised data in: {args.out_dir}/")
    print("Upload with:")
    print(f"  rsync -av {args.out_dir}/ majorno@<server>:/home/shared/datasets/Anomaly_TS/")


if __name__ == "__main__":
    main()
