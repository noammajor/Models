#!/usr/bin/env python3
"""
count_dataset_sizes.py — Count exact data points in Monash and synthetic datasets.

Reports per-file and total:
  - number of series
  - number of series passing min_len filter
  - total timesteps (sum of all series lengths)
  - total timesteps after min_len filter

Usage:
    python count_dataset_sizes.py
    python count_dataset_sizes.py --monash_dir /path/to/Monash
    python count_dataset_sizes.py --synthetic_dir /path/to/synthetic
    python count_dataset_sizes.py --min_len 512
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from data_loaders.data_puller import _read_tsf_series

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MONASH    = "/Users/noammajor/Desktop/Monash"
DEFAULT_SYNTHETIC = "/home/shared/datasets/synthetic_data_TS"
DEFAULT_SYNTH_MIX = "/home/shared/datasets/synthetic_TS_Mix"
MIN_LEN           = 512


def count_tsf_dir(data_dir: str, min_len: int, label: str):
    """Count all series and timesteps in a directory of .tsf files."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"\n[{label}] Directory not found: {data_dir}")
        return

    tsf_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.tsf'))
    if not tsf_files:
        print(f"\n[{label}] No .tsf files found in {data_dir}")
        return

    print(f"\n{'='*70}")
    print(f"  {label}  —  {data_dir}")
    print(f"  min_len filter: {min_len}")
    print(f"{'='*70}")
    print(f"  {'File':<45} {'Series':>8} {'≥minlen':>8} {'Total T':>14} {'Filtered T':>14}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*14} {'-'*14}")

    total_series   = 0
    total_filtered = 0
    total_T        = 0
    filtered_T     = 0

    for fname in tsf_files:
        path = os.path.join(data_dir, fname)
        try:
            series_list = _read_tsf_series(path)
        except Exception as e:
            print(f"  {'  SKIP ' + fname:<45}  {str(e)}")
            continue

        n_all      = len(series_list)
        n_ok       = sum(1 for s in series_list if not np.isnan(s).any() and len(s) >= min_len)
        t_all      = sum(len(s) for s in series_list if not np.isnan(s).any())
        t_filtered = sum(len(s) for s in series_list if not np.isnan(s).any() and len(s) >= min_len)

        total_series   += n_all
        total_filtered += n_ok
        total_T        += t_all
        filtered_T     += t_filtered

        print(f"  {fname:<45} {n_all:>8,} {n_ok:>8,} {t_all:>14,} {t_filtered:>14,}")

    print(f"  {'─'*45} {'─'*8} {'─'*8} {'─'*14} {'─'*14}")
    print(f"  {'TOTAL':<45} {total_series:>8,} {total_filtered:>8,} {total_T:>14,} {filtered_T:>14,}")
    print()
    print(f"  Summary for {label}:")
    print(f"    All series       : {total_series:,}")
    print(f"    Filtered series  : {total_filtered:,}  (len >= {min_len})")
    print(f"    Total timesteps  : {total_T:,}")
    print(f"    Filtered timesteps: {filtered_T:,}")

    return dict(n_series=total_series, n_filtered=total_filtered,
                total_T=total_T, filtered_T=filtered_T)


def count_arrow_dir(data_dir: str, min_len: int, label: str):
    """Count all series and timesteps in a directory of .arrow files."""
    try:
        from gluonts.dataset.arrow import ArrowFile
    except ImportError:
        try:
            import pyarrow as pa
        except ImportError:
            print(f"\n[{label}] pyarrow/gluonts not available — cannot read .arrow files")
            return

    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"\n[{label}] Directory not found: {data_dir}")
        return

    arrow_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.arrow'))
    if not arrow_files:
        print(f"\n[{label}] No .arrow files found in {data_dir}")
        return

    print(f"\n{'='*70}")
    print(f"  {label}  —  {data_dir}")
    print(f"  min_len filter: {min_len}")
    print(f"{'='*70}")
    print(f"  {'File':<45} {'Series':>8} {'≥minlen':>8} {'Total T':>14} {'Filtered T':>14}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*14} {'-'*14}")

    total_series   = 0
    total_filtered = 0
    total_T        = 0
    filtered_T     = 0

    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError:
        print(f"\n[{label}] pyarrow not installed")
        return

    for fname in arrow_files:
        path = os.path.join(data_dir, fname)
        try:
            with pa.memory_map(path, 'r') as source:
                try:
                    reader = ipc.open_file(source)
                except Exception:
                    source.seek(0)
                    reader = ipc.open_stream(source)
                table = reader.read_all()

            n_all = len(table)
            lengths = []
            target_col = None
            for col in ("target", "values", "data"):
                if col in table.schema.names:
                    target_col = col
                    break

            if target_col is None:
                # try first column
                target_col = table.schema.names[0]

            for i in range(n_all):
                val = table[target_col][i]
                if hasattr(val, 'as_py'):
                    arr = val.as_py()
                    if arr is not None:
                        lengths.append(len(arr) if hasattr(arr, '__len__') else 1)

            if not lengths:
                print(f"  {fname:<45} {'?':>8}")
                continue

            lengths = np.array(lengths)
            n_ok      = int((lengths >= min_len).sum())
            t_all     = int(lengths.sum())
            t_filt    = int(lengths[lengths >= min_len].sum())

            total_series   += n_all
            total_filtered += n_ok
            total_T        += t_all
            filtered_T     += t_filt

            print(f"  {fname:<45} {n_all:>8,} {n_ok:>8,} {t_all:>14,} {t_filt:>14,}")

        except Exception as e:
            print(f"  {'  SKIP ' + fname:<45}  {e}")

    print(f"  {'─'*45} {'─'*8} {'─'*8} {'─'*14} {'─'*14}")
    print(f"  {'TOTAL':<45} {total_series:>8,} {total_filtered:>8,} {total_T:>14,} {filtered_T:>14,}")
    print()
    print(f"  Summary for {label}:")
    print(f"    All series       : {total_series:,}")
    print(f"    Filtered series  : {total_filtered:,}  (len >= {min_len})")
    print(f"    Total timesteps  : {total_T:,}")
    print(f"    Filtered timesteps: {filtered_T:,}")

    return dict(n_series=total_series, n_filtered=total_filtered,
                total_T=total_T, filtered_T=filtered_T)


def main():
    parser = argparse.ArgumentParser(description="Count dataset sizes")
    parser.add_argument("--monash_dir",    default=DEFAULT_MONASH)
    parser.add_argument("--synthetic_dir", default=DEFAULT_SYNTHETIC)
    parser.add_argument("--synth_mix_dir", default=DEFAULT_SYNTH_MIX)
    parser.add_argument("--min_len",       type=int, default=MIN_LEN)
    parser.add_argument("--no_synthetic",  action="store_true")
    args = parser.parse_args()

    results = {}

    results["monash"] = count_tsf_dir(args.monash_dir, args.min_len, "Monash")

    if not args.no_synthetic:
        results["synthetic"]  = count_arrow_dir(args.synthetic_dir, args.min_len, "Synthetic (full)")
        results["synth_mix"]  = count_arrow_dir(args.synth_mix_dir, args.min_len, "Synthetic Mix (monash+synthetic)")

    # ── combined summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  COMBINED SUMMARY")
    print(f"{'='*70}")
    for name, r in results.items():
        if r:
            print(f"  {name:<20}  series={r['n_filtered']:>8,}  timesteps={r['filtered_T']:>14,}")

    if results.get("monash") and results.get("synth_mix"):
        m = results["monash"]
        s = results["synth_mix"]
        combined_series = m["n_filtered"] + s["n_filtered"]
        combined_T      = m["filtered_T"] + s["filtered_T"]
        print(f"  {'monash+synthetic':<20}  series={combined_series:>8,}  timesteps={combined_T:>14,}")
    print()


if __name__ == "__main__":
    main()
