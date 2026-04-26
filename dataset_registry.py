"""Central dataset registry.

Add a new entry to DATASETS to support a new CSV dataset.
The forecasting CSV root comes from data_paths.DATA_PATHS["forecasting_data_dir"]
— edit that file once when moving between machines.
"""

import os
from pathlib import Path

from data_paths import DATA_PATHS

_DATA_DIR = Path(DATA_PATHS["forecasting_data_dir"])

# JEPA groups variables into chunks of this many columns.
# Every group must be the same size; the last group is padded by repeating
# its first column if the total variable count isn't divisible by GROUP_SIZE.
_JEPA_GROUP_SIZE = 4


def _make_jepa_groups(columns: list, group_size: int = _JEPA_GROUP_SIZE) -> list:
    """Split *columns* into equal-length groups, padding the last one if needed."""
    groups = []
    for i in range(0, len(columns), group_size):
        group = list(columns[i : i + group_size])
        while len(group) < group_size:
            group.append(group[0])   # repeat first col of this group to pad
        groups.append(group)
    return groups


# ── Registry ─────────────────────────────────────────────────────────────────
# Keys must match the names passed to  --dset_pretrain / --dset_finetune  in
# the PatchTST scripts and to  run(dataset=...)  in Train_and_downstream.py.
#
# patchtst_cls  "ETT_minute" | "ETT_hour" | "Custom"
# columns       list of data-column names (everything except the timestamp).
#               Set to None to auto-detect from the CSV header at runtime.

DATASETS: dict = {
    "ettm1": {
        "csv_filename":    "ETTm1.csv",
        "patchtst_cls":    "ETT_minute",
        "timestamp_col":   "date",
        "columns":         ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        "jepa_group_size": 7,   # all 7 vars in one group
    },
    "etth1": {
        "csv_filename":    "ETTh1.csv",
        "patchtst_cls":    "ETT_hour",
        "timestamp_col":   "date",
        "columns":         ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        "jepa_group_size": 7,   # all 7 vars in one group
    },
    "etth2": {
        "csv_filename":    "ETTh2.csv",
        "patchtst_cls":    "ETT_hour",
        "timestamp_col":   "date",
        "columns":         ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        "jepa_group_size": 7,   # all 7 vars in one group
    },
    "ettm2": {
        "csv_filename":    "ETTm2.csv",
        "patchtst_cls":    "ETT_minute",
        "timestamp_col":   "date",
        "columns":         ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        "jepa_group_size": 7,   # all 7 vars in one group
    },
    "weather": {
        "csv_filename":    "weather.csv",
        "patchtst_cls":    "Custom",
        "timestamp_col":   "date",
        "columns":         None,  # auto-detected from CSV header
        "jepa_group_size": 7,
    },
    "electricity": {
        "csv_filename":  "electricity.csv",
        "patchtst_cls":  "Custom",
        "timestamp_col": "date",
        "columns": None,  # auto-detected from CSV header
    },
    "traffic": {
        "csv_filename":  "traffic.csv",
        "patchtst_cls":  "Custom",
        "timestamp_col": "date",
        "columns": None,  # auto-detected from CSV header
    },
}


def get_dataset_info(name: str) -> dict:
    """Return a fully-resolved info dict for *name*.

    Extra keys added at call time:
      csv_path      – absolute path to the CSV file
      data_dir      – directory containing the CSV (with trailing separator)
      c_in          – number of data columns
      jepa_groups   – list-of-lists ready for JEPA input_variables
    """
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASETS)}"
        )
    info = dict(DATASETS[name])
    info["name"]     = name
    info["csv_path"] = str(_DATA_DIR / info["csv_filename"])
    info["data_dir"] = str(_DATA_DIR) + os.sep

    # Auto-detect columns if not explicitly listed
    if info["columns"] is None:
        import pandas as pd
        df = pd.read_csv(info["csv_path"], nrows=0)
        info["columns"] = [c for c in df.columns if c != info["timestamp_col"]]

    info["c_in"]        = len(info["columns"])
    group_size          = info.pop("jepa_group_size", _JEPA_GROUP_SIZE)
    info["jepa_groups"] = _make_jepa_groups(info["columns"], group_size)
    return info
