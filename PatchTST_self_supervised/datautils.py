


import numpy as np
import pandas as pd
import torch
from torch import nn
import sys
import os as _os

# Root directory containing ETT CSVs — pulled from the central dataset_registry
_ROOT_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
from dataset_registry import _DATA_DIR as _ETT_DATA_DIR_PATH
_ETT_DATA_DIR = str(_ETT_DATA_DIR_PATH) + _os.sep

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange', 'monash',
         'synthetic', 'monash+synthetic',
        ]


# ── Monash .tsf reader ────────────────────────────────────────────────────────

def _read_tsf_series(path):
    """Read a Monash .tsf file. Returns a list of 1-D numpy float32 arrays."""
    from sklearn.preprocessing import StandardScaler as _SS  # noqa: imported for type-check only
    found_data = False
    series_list = []
    with open(path, 'r', encoding='cp1252') as f:
        for line in f:
            line = line.strip()
            if line == '@data':
                found_data = True
                continue
            if not found_data or not line:
                continue
            parts = line.split(':')
            vals = parts[-1].split(',')
            try:
                arr = np.array([float(v) for v in vals], dtype=np.float32)
                series_list.append(arr)
            except ValueError:
                continue
    return series_list


class MonashDatasetPatchTST:
    """Monash .tsf dataset for PatchTST self-supervised pretraining.

    Returns (x, y) with shapes [context_points, 1] and [target_points, 1].
    The PatchMaskCB callback replaces y with the unmasked patches at training
    time, so the y returned here is only used to infer dls.c.
    """

    def __init__(self, data_dir, context_points, target_points,
                 min_len=512, val_frac=0.1, test_frac=0.1,
                 split='train', scale=True, **kwargs):
        from sklearn.preprocessing import StandardScaler
        import glob

        window_size = context_points + target_points
        self._windows = []

        for fpath in sorted(glob.glob(_os.path.join(data_dir, '*.tsf'))):
            try:
                series_list = _read_tsf_series(fpath)
            except Exception as e:
                print(f"  MonashDatasetPatchTST: skipping {_os.path.basename(fpath)} — {e}")
                continue
            for series in series_list:
                if np.isnan(series).any() or len(series) < min_len:
                    continue
                T = len(series)
                val_len   = int(T * val_frac)
                test_len  = int(T * test_frac)
                train_len = T - val_len - test_len
                if scale:
                    scaler = StandardScaler()
                    scaler.fit(series[:train_len].reshape(-1, 1))
                    series = scaler.transform(series.reshape(-1, 1)).flatten().astype(np.float32)
                if split == 'train':
                    seg = series[:train_len]
                elif split == 'val':
                    seg = series[train_len:train_len + val_len]
                else:
                    seg = series[train_len + val_len:]
                if len(seg) < window_size:
                    continue
                # stride = context_points to avoid too many overlapping windows
                for start in range(0, len(seg) - window_size + 1, context_points):
                    self._windows.append(seg[start:start + window_size])

        self._context = context_points
        self._target  = target_points

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        w = self._windows[idx]
        x = torch.tensor(w[:self._context]).unsqueeze(-1)  # [context_points, 1]
        y = torch.tensor(w[self._context:]).unsqueeze(-1)  # [target_points,  1]
        return x, y


class SyntheticDatasetPatchTST:
    """Synthetic .arrow dataset for PatchTST self-supervised pretraining.

    Mirrors MonashDatasetPatchTST — returns (x, y) with shapes
    [context_points, 1] and [target_points, 1].
    Handles both univariate [T] and multivariate [C, T] targets (each channel
    becomes a separate series).
    """

    def __init__(self, data_dir, context_points, target_points,
                 min_len=512, val_frac=0.1, test_frac=0.1,
                 split='train', scale=True, **kwargs):
        import glob
        from sklearn.preprocessing import StandardScaler
        import pyarrow as pa

        window_size = context_points + target_points
        self._windows = []

        for fpath in sorted(glob.glob(_os.path.join(data_dir, '*.arrow'))):
            fname = _os.path.basename(fpath)
            try:
                with pa.memory_map(fpath, 'r') as src:
                    table = pa.ipc.open_file(src).read_all()
            except Exception:
                try:
                    with open(fpath, 'rb') as f:
                        reader = pa.ipc.open_stream(f)
                        table = reader.read_all()
                except Exception as e:
                    print(f"  SyntheticDatasetPatchTST: skipping {fname} — {e}")
                    continue

            for row in table.column('target'):
                arr = row.as_py()
                if arr is None:
                    continue
                arr = np.array(arr, dtype=np.float32)
                if arr.ndim == 1:
                    channels = [arr]
                elif arr.ndim == 2:
                    channels = [arr[c] for c in range(arr.shape[0])]
                else:
                    continue

                for series in channels:
                    if np.isnan(series).any() or len(series) < min_len:
                        continue
                    T = len(series)
                    val_len   = int(T * val_frac)
                    test_len  = int(T * test_frac)
                    train_len = T - val_len - test_len
                    if scale:
                        scaler = StandardScaler()
                        scaler.fit(series[:train_len].reshape(-1, 1))
                        series = scaler.transform(series.reshape(-1, 1)).flatten().astype(np.float32)
                    if split == 'train':
                        seg = series[:train_len]
                    elif split == 'val':
                        seg = series[train_len:train_len + val_len]
                    else:
                        seg = series[train_len + val_len:]
                    if len(seg) < window_size:
                        continue
                    for start in range(0, len(seg) - window_size + 1, context_points):
                        self._windows.append(seg[start:start + window_size])

        self._context = context_points
        self._target  = target_points

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        w = self._windows[idx]
        x = torch.tensor(w[:self._context]).unsqueeze(-1)
        y = torch.tensor(w[self._context:]).unsqueeze(-1)
        return x, y


def get_dls(params):

    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

    if params.dset == 'ettm1':
        root_path = _ETT_DATA_DIR
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'ettm2':
        root_path = _ETT_DATA_DIR
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_minute,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'etth1':
        root_path = _ETT_DATA_DIR
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'etth2':
        root_path = _ETT_DATA_DIR
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_ETT_hour,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )


    elif params.dset == 'electricity':
        root_path = _ETT_DATA_DIR
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'electricity.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'traffic':
        root_path = _ETT_DATA_DIR
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'traffic.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'weather':
        root_path = _ETT_DATA_DIR
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'illness':
        root_path = '/data/datasets/public/illness/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'national_illness.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'exchange':
        root_path = '/data/datasets/public/exchange_rate/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'exchange_rate.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'monash':
        data_dir = params.monash_data_dir
        min_len  = getattr(params, 'monash_min_len', 512)
        dls = DataLoaders(
            datasetCls=MonashDatasetPatchTST,
            dataset_kwargs={
                'data_dir':       data_dir,
                'context_points': params.context_points,
                'target_points':  params.target_points,
                'min_len':        min_len,
                'scale':          True,
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'synthetic':
        synth_dir = params.synthetic_data_dir
        min_len   = getattr(params, 'monash_min_len', 512)
        dls = DataLoaders(
            datasetCls=SyntheticDatasetPatchTST,
            dataset_kwargs={
                'data_dir':       synth_dir,
                'context_points': params.context_points,
                'target_points':  params.target_points,
                'min_len':        min_len,
                'scale':          True,
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'monash+synthetic':
        import torch.utils.data as _tud

        min_len   = getattr(params, 'monash_min_len', 512)
        _shared = dict(context_points=params.context_points,
                       target_points=params.target_points,
                       min_len=min_len, scale=True)

        class _CombinedDataset:
            """Wraps ConcatDataset of Monash + Synthetic, accepts split kwarg."""
            def __init__(self, monash_dir, synth_dir, split='train', **kw):
                m = MonashDatasetPatchTST(data_dir=monash_dir, split=split, **kw)
                s = SyntheticDatasetPatchTST(data_dir=synth_dir, split=split, **kw)
                self._ds = _tud.ConcatDataset([m, s])
            def __len__(self):  return len(self._ds)
            def __getitem__(self, i):  return self._ds[i]

        dls = DataLoaders(
            datasetCls=_CombinedDataset,
            dataset_kwargs=dict(monash_dir=params.monash_data_dir,
                                synth_dir=params.synthetic_data_dir,
                                **_shared),
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls



if __name__ == "__main__":
    class Params:
        dset= 'etth2'
        context_points= 384
        target_points= 96
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='M'
    params = Params
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
    breakpoint()
