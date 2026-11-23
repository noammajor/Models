"""Hold Monash files out of pre-training: TS_MONASH_EXCLUDE.

Every Monash pre-training loader (shared/data_loaders/data_puller.py,
DINO/dataPuller.py, MAE/datautils.py) lists the .tsf files through
monash_tsf_files(), so one environment variable removes a file for every
objective:

    TS_MONASH_EXCLUDE=electricity_hourly          (comma-separated substrings)

A pattern that matches no file is an error rather than a silent no-op, so a
typo cannot leave the held-out set in the corpus. Train_and_downstream.py tags
checkpoint dirs with the excluded names (monash_exclude_tag), so a held-out
backbone never shares a path with a full-Monash one.
"""
import os


def monash_exclude_patterns():
    raw = os.environ.get("TS_MONASH_EXCLUDE", "")
    return [p.strip() for p in raw.split(",") if p.strip()]


def monash_exclude_tag():
    """'' when nothing is excluded, else '_no_<pattern>[_<pattern>...]'."""
    pats = monash_exclude_patterns()
    if not pats:
        return ""
    clean = ["".join(c if c.isalnum() else "_" for c in p).strip("_") for p in pats]
    return "_no_" + "_".join(c for c in clean if c)


def monash_tsf_files(data_dir):
    """Sorted .tsf file names in data_dir, minus any TS_MONASH_EXCLUDE match."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".tsf"))
    pats = monash_exclude_patterns()
    if not pats:
        return files
    kept, dropped = [], []
    for f in files:
        (dropped if any(p in f for p in pats) else kept).append(f)
    unmatched = [p for p in pats if not any(p in f for f in dropped)]
    if unmatched:
        raise ValueError(f"TS_MONASH_EXCLUDE pattern(s) {unmatched} match no .tsf file in {data_dir}")
    print(f"[monash] held out of pre-training (TS_MONASH_EXCLUDE): {', '.join(dropped)}")
    return kept
