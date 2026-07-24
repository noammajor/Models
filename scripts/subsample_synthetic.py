#!/usr/bin/env python3
"""
subsample_synthetic.py — Draw a Monash-sized subset of the existing synthetic set.

Motivation: the synthetic set is much larger than Monash, so a "synthetic vs Monash"
comparison confounds scale with origin. This selects a random subset of the SAME
synthetic series (same generator, same distribution) whose total size matches Monash,
so only origin differs.

"Size" is measured exactly as scripts/count_dataset_sizes.py measures it: each arrow
row contributes len(target) timesteps, counting only rows with len >= min_len (the
same filter the pretraining loaders apply). Rows are drawn at random (seeded) until the
cumulative filtered timesteps reach --target_timesteps, then written to --out_dir as a
single .arrow file with the original {start, target} records preserved.

Usage:
    python scripts/subsample_synthetic.py \
        --synthetic_dir /home/shared/datasets/synthetic_data_TS \
        --out_dir       /home/shared/datasets/synthetic_data_TS_monashsize \
        --target_timesteps 445011429 \
        --min_len 512 --seed 42

Then verify parity:
    python scripts/count_dataset_sizes.py --no_synthetic --min_len 512 \
        --monash_dir /home/shared/datasets/synthetic_data_TS_monashsize   # counts .tsf only
    # or point count_arrow_dir at the new dir.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# Monash filtered (len >= 512) timesteps, from count_dataset_sizes.py.
DEFAULT_TARGET = 445_011_429
DEFAULT_SYNTHETIC = "/home/shared/datasets/synthetic_data_TS"


def _read_table(path: str):
    with pa.memory_map(path, "r") as src:
        try:
            reader = ipc.open_file(src)
        except Exception:
            src.seek(0)
            reader = ipc.open_stream(src)
        return reader.read_all()


def _target_col(table):
    for c in ("target", "values", "data"):
        if c in table.schema.names:
            return c
    return table.schema.names[0]


def main():
    ap = argparse.ArgumentParser(description="Subsample synthetic to Monash size")
    ap.add_argument("--synthetic_dir", default=DEFAULT_SYNTHETIC)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_timesteps", type=int, default=DEFAULT_TARGET,
                    help=f"Total filtered timesteps to match (default Monash = {DEFAULT_TARGET:,})")
    ap.add_argument("--min_len", type=int, default=512,
                    help="Only include/count series with length >= min_len (matches loaders)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true",
                    help="Report what would be selected without writing the arrow file")
    args = ap.parse_args()

    sdir = Path(args.synthetic_dir)
    files = sorted(f for f in os.listdir(sdir) if f.endswith(".arrow"))
    if not files:
        sys.exit(f"No .arrow files in {sdir}")

    # 1. Index every eligible row as (file_idx, row_idx, eff_timesteps, n_channels).
    #    A row's target is [T] (1 series) or [C, T] (C series — the loaders treat each
    #    channel as a separate univariate series). The min_len filter applies to T; the
    #    row's effective size is C * T timesteps and C series — exactly what pretraining
    #    sees after channel expansion. Matching Monash's univariate timesteps to this
    #    expanded total is the fair "same amount of data" comparison.
    print(f"Scanning {len(files)} arrow files in {sdir} ...", flush=True)
    index = []          # (fi, ri, eff_timesteps, n_channels)
    tables = []
    tcols = []
    shpcols = []        # gluonts '<target>._np_shape' sidecar per file (or None)
    n_multivariate = 0
    for fi, fname in enumerate(files):
        table = _read_table(str(sdir / fname))
        col = _target_col(table)
        shp_col = f"{col}._np_shape" if f"{col}._np_shape" in table.schema.names else None
        tables.append(table)
        tcols.append(col)
        shpcols.append(shp_col)
        for ri in range(len(table)):
            arr = table[col][ri].as_py()
            if arr is None:
                continue
            # gluonts stores target flattened + a '<target>._np_shape' sidecar, so the
            # true shape comes from that column, not from the flattened values.
            if shp_col is not None:
                shp = table[shp_col][ri].as_py()
                if not shp:
                    continue
                shp = [int(x) for x in shp]
                if len(shp) >= 2:                 # [C, T] — C channel-series of length T
                    C, T = shp[0], shp[-1]
                    n_multivariate += 1
                else:                             # [T] — one univariate series
                    C, T = 1, shp[0]
            else:
                a = np.asarray(arr)
                if a.ndim >= 2:
                    C, T = a.shape[0], a.shape[-1]
                    n_multivariate += 1
                else:
                    C, T = 1, len(a)
            if T >= args.min_len:                 # filter on the per-series length T
                index.append((fi, ri, C * T, C))
        print(f"  {fname}: {len(table):,} rows", flush=True)

    kind = (f"multivariate ({n_multivariate:,} of the rows have channels)"
            if n_multivariate else "univariate")
    eligible_T = sum(t for _, _, t, _ in index)
    eligible_series = sum(c for _, _, _, c in index)
    print(f"\nDetected {kind} data.", flush=True)
    print(f"Eligible rows (T >= {args.min_len}): {len(index):,}  "
          f"| eligible series (channel-expanded): {eligible_series:,}  "
          f"| eligible timesteps: {eligible_T:,}", flush=True)
    print(f"Target timesteps (Monash): {args.target_timesteps:,}  "
          f"({100*args.target_timesteps/max(eligible_T,1):.1f}% of eligible)", flush=True)

    if eligible_T < args.target_timesteps:
        sys.exit(f"ERROR: synthetic only has {eligible_T:,} eligible timesteps, "
                 f"less than target {args.target_timesteps:,}. Cannot match.")

    # 2. Shuffle deterministically and accumulate until the target is reached.
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(index))
    chosen = []
    cum = 0
    cum_series = 0
    for k in order:
        fi, ri, eff, C = index[k]
        chosen.append((fi, ri))
        cum += eff
        cum_series += C
        if cum >= args.target_timesteps:
            break

    print(f"\nSelected {len(chosen):,} rows  |  channel-expanded series: {cum_series:,}  "
          f"|  timesteps: {cum:,}  "
          f"(overshoot {cum - args.target_timesteps:,} = "
          f"{100*(cum-args.target_timesteps)/args.target_timesteps:.3f}%)", flush=True)

    if args.dry_run:
        print("[dry_run] not writing.", flush=True)
        return

    # 3. Rebuild the selected records ({start, target, ...}) and write one arrow file.
    from gluonts.dataset.arrow import ArrowWriter
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _records():
        # Emit ONLY clean {start, target} records. The target is reconstructed to its
        # real shape from the gluonts '<target>._np_shape' sidecar; ArrowWriter re-adds
        # that sidecar itself, so we must NOT pass '<target>._np_shape' back as a field
        # (doing so is what triggered the 'KeyError: target._np_shape' write failure).
        for fi, ri in chosen:
            table = tables[fi]
            col = tcols[fi]
            shp_col = shpcols[fi]
            tgt = np.asarray(table[col][ri].as_py())
            if shp_col is not None:
                shp = table[shp_col][ri].as_py()
                if shp:
                    tgt = tgt.reshape([int(x) for x in shp])
            rec = {"target": tgt}
            if "start" in table.schema.names:
                rec["start"] = table["start"][ri].as_py()
            else:
                rec["start"] = np.datetime64("2000-01-01 00:00", "s")
            yield rec

    out_path = out / "synthetic_monashsize.arrow"
    ArrowWriter(compression="lz4").write_to_file(_records(), out_path)
    print(f"\nWrote {len(chosen):,} rows ({cum_series:,} channel-expanded series) → {out_path}", flush=True)
    print(f"Total timesteps: {cum:,}  (Monash target: {args.target_timesteps:,})", flush=True)


if __name__ == "__main__":
    main()
