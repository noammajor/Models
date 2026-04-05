import pandas as pd
import torch
import random
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.preprocessing import StandardScaler


# ── Shared Monash .tsf reader ─────────────────────────────────────────────────

def _read_tsf_series(path):
    """Read a Monash .tsf file. Returns a list of 1-D numpy float32 arrays."""
    found_data = False
    series_list = []
    with open(path, 'r', encoding='cp1252') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('@data'):
                found_data = True
            elif not line.startswith('@') and found_data:
                # Each data line: attr1:attr2:...:v1,v2,v3,...
                vals_str = line.split(':')[-1].split(',')
                vals = []
                for v in vals_str:
                    v = v.strip()
                    vals.append(np.nan if v == '?' else float(v))
                if vals:
                    series_list.append(np.array(vals, dtype=np.float32))
    return series_list


# ── DINO pretraining ──────────────────────────────────────────────────────────

class DataPuller(Dataset):
    def __init__(self, data_dir, split='train', transform=None,
                 batch_size=32, patch_size=12, step_size=12, c_in=7):
        self.data_dir    = data_dir
        self.which       = split
        self.transform   = transform
        self.patch_size  = patch_size
        self.num_patches = batch_size
        self.val_prec    = 0.1
        self.test_prec   = 0.1
        self.step_size   = step_size
        self.window_size = (self.num_patches - 1) * self.step_size + self.patch_size

        self.Train_Val_Test_splits = {'train': [], 'val': [], 'test': []}
        self.all_map               = {'train': [], 'val': [], 'test': []}

        self.get_data()

    def get_data(self):
        df = pd.read_csv(self.data_dir, parse_dates=['date'])
        fcols = df.select_dtypes('float').columns.tolist()
        df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
        icols = df.select_dtypes('integer').columns
        df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
        df.sort_values(by='date', inplace=True)

        input_vars = [col for col in df.columns if col != 'date']
        data       = df[input_vars].values.astype(np.float32)

        val_len    = int(len(data) * self.val_prec)
        test_len   = int(len(data) * self.test_prec)
        train_len  = len(data) - val_len - test_len

        train_np = data[:train_len]
        val_np   = data[train_len : train_len + val_len]
        test_np  = data[train_len + val_len :]

        # Fit StandardScaler on training data only
        self.scaler = StandardScaler()
        self.scaler.fit(train_np)

        train_np = self.scaler.transform(train_np)
        val_np   = self.scaler.transform(val_np)
        test_np  = self.scaler.transform(test_np)

        self.Train_Val_Test_splits['train'] = [torch.from_numpy(train_np)]
        self.Train_Val_Test_splits['val']   = [torch.from_numpy(val_np)]
        self.Train_Val_Test_splits['test']  = [torch.from_numpy(test_np)]

        for split_name in ['train', 'val', 'test']:
            tensor      = self.Train_Val_Test_splits[split_name][0]
            num_samples = (tensor.size(0) - self.window_size) // self.step_size
            print(f"Number of samples in {split_name} set: {num_samples}")
            for i in range(num_samples):
                self.all_map[split_name].append((0, i))

    def __len__(self):
        return len(self.all_map[self.which])

    def __getitem__(self, idx):
        file_idx, start_offset = self.all_map[self.which][idx]
        source_data = self.Train_Val_Test_splits[self.which][file_idx]
        start = start_offset * self.step_size
        end   = start + self.window_size
        chunk = source_data[start:end]   # [window_size, n_vars] — already normalised

        if self.transform:
            chunk = self.transform(chunk)
        return chunk


# ── forecasting training ──────────────────────────────────────────────────────

class DataPullerForecastingTrain():
    def __init__(self, data_dir, split='train', batch_size=16, patch_size=12,
                 step_size=1, pred_len=96,
                 var_list=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
                 c_in=7):
        self.data_dir    = data_dir
        self.which       = split
        self.patch_size  = patch_size
        self.num_patches = batch_size
        self.step_size   = step_size
        self.pred_len    = pred_len
        self.window_size = self.num_patches * self.patch_size
        self.var_names   = var_list
        self.var_indices = []

        print(f"num patches:{self.num_patches}")
        print(f"step size: {self.step_size}")
        print(f"patch size: {self.patch_size}")

        self.Train_Val_Test_splits = {'train': [], 'val': [], 'test': []}
        self.all_map               = {'train': [], 'val': [], 'test': []}

        self.get_data()

    def get_data(self):
        df = pd.read_csv(self.data_dir)
        if 'date' in df.columns:
            df = df.drop(columns=['date'])

        all_cols         = df.columns.tolist()
        self.var_indices = [all_cols.index(name) for name in self.var_names]

        data = df.values.astype(np.float32)

        n = len(data)
        fname = os.path.basename(self.data_dir).lower()
        if 'etth' in fname:
            train_len = 12 * 30 * 24
            val_len   = 4  * 30 * 24
            test_len  = 4  * 30 * 24
        elif 'ettm' in fname:
            train_len = 12 * 30 * 24 * 4
            val_len   = 4  * 30 * 24 * 4
            test_len  = 4  * 30 * 24 * 4
        else:
            train_len = int(n * 0.7)
            test_len  = int(n * 0.2)
            val_len   = n - train_len - test_len

        train_np = data[:train_len,  self.var_indices]
        val_np   = data[train_len : train_len + val_len, self.var_indices]
        test_np  = data[train_len + val_len :,           self.var_indices]

        # Fit StandardScaler on training columns only
        self.scaler = StandardScaler()
        self.scaler.fit(train_np)

        self.Train_Val_Test_splits = {
            'train': [torch.from_numpy(self.scaler.transform(train_np))],
            'val':   [torch.from_numpy(self.scaler.transform(val_np))],
            'test':  [torch.from_numpy(self.scaler.transform(test_np))],
        }

        for split_name in ['train', 'val', 'test']:
            tensor      = self.Train_Val_Test_splits[split_name][0]
            num_samples = (tensor.size(0) - self.window_size - self.pred_len) // self.step_size
            for i in range(num_samples):
                self.all_map[split_name].append((0, i))

    def __getitem__(self, idx):
        file_idx, start_offset = self.all_map[self.which][idx]
        source_data = self.Train_Val_Test_splits[self.which][file_idx]

        start = start_offset * self.step_size
        mid   = start + self.window_size
        end   = mid   + self.pred_len

        return source_data[start:mid], source_data[mid:end]

    def __len__(self):
        return len(self.all_map[self.which])


# ── forecasting testing ───────────────────────────────────────────────────────

class DataPullerForecastingTesting(Dataset):
    def __init__(self, data_dir, split='test', batch_size=16, patch_size=12,
                 step_size=1, pred_len=96,
                 var_list=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
                 c_in=7, scaler=None):
        self.data_dir    = data_dir
        self.which       = split
        self.patch_size  = patch_size
        self.num_patches = batch_size
        self.step_size   = step_size
        self.pred_len    = pred_len
        self.window_size = self.num_patches * self.patch_size
        self.var_names   = var_list
        self.var_indices = []
        self._ext_scaler = scaler  # optional pre-fitted scaler from training set

        self.Train_Val_Test_splits = {'train': [], 'val': [], 'test': []}
        self.all_map               = {'train': [], 'val': [], 'test': []}

        self.get_data()

    def get_data(self):
        df = pd.read_csv(self.data_dir)
        if 'date' in df.columns:
            df = df.drop(columns=['date'])

        all_cols         = df.columns.tolist()
        self.var_indices = [all_cols.index(name) for name in self.var_names]

        data = df.values.astype(np.float32)

        n = len(data)
        fname = os.path.basename(self.data_dir).lower()
        if 'etth' in fname:
            train_len = 12 * 30 * 24
            val_len   = 4  * 30 * 24
            test_len  = 4  * 30 * 24
        elif 'ettm' in fname:
            train_len = 12 * 30 * 24 * 4
            val_len   = 4  * 30 * 24 * 4
            test_len  = 4  * 30 * 24 * 4
        else:
            train_len = int(n * 0.7)
            test_len  = int(n * 0.2)
            val_len   = n - train_len - test_len

        train_np = data[:train_len,  self.var_indices]
        val_np   = data[train_len : train_len + val_len, self.var_indices]
        test_np  = data[train_len + val_len :,           self.var_indices]

        # Use the scaler from the training dataset if provided; otherwise fit a new one.
        # Always fit on train split so val/test are scaled with the same statistics.
        if self._ext_scaler is not None:
            self.scaler = self._ext_scaler
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(train_np)

        self.Train_Val_Test_splits = {
            'train': [torch.from_numpy(self.scaler.transform(train_np))],
            'val':   [torch.from_numpy(self.scaler.transform(val_np))],
            'test':  [torch.from_numpy(self.scaler.transform(test_np))],
        }

        for split_name in ['train', 'val', 'test']:
            tensor      = self.Train_Val_Test_splits[split_name][0]
            num_samples = (tensor.size(0) - self.window_size - self.pred_len) // self.step_size
            for i in range(num_samples):
                self.all_map[split_name].append((0, i))

    def __getitem__(self, idx):
        file_idx, start_offset = self.all_map[self.which][idx]
        source_data = self.Train_Val_Test_splits[self.which][file_idx]

        start = start_offset * self.step_size
        mid   = start + self.window_size
        end   = mid   + self.pred_len

        x_patches = source_data[start:mid]   # [window_size, n_vars]
        y_future  = source_data[mid:end]      # [pred_len,   n_vars]
        return x_patches, y_future

    def __len__(self):
        return len(self.all_map[self.which])


# ── classification ────────────────────────────────────────────────────────────

class DataPullerClassification(Dataset):
    """
    UCI HAR Dataset loader for time series classification.
    Loads 9 inertial sensor signals with 6 activity labels.
    StandardScaler is always fitted on the training split.
    """
    def __init__(self, data_dir, split='train', c_in=9, seq_len=128, normalize=True):
        self.data_dir  = data_dir
        self.split     = split
        self.c_in      = c_in
        self.seq_len   = seq_len
        self.normalize = normalize

        self.signal_files = [
            'body_acc_x',   'body_acc_y',   'body_acc_z',
            'body_gyro_x',  'body_gyro_y',  'body_gyro_z',
            'total_acc_x',  'total_acc_y',  'total_acc_z',
        ]

        self.data, self.labels = self._load_data()

    def _load_raw_signals(self, split):
        split_dir   = os.path.join(self.data_dir, split)
        signals_dir = os.path.join(split_dir, 'Inertial Signals')
        all_signals = []
        for signal_name in self.signal_files:
            file_path = os.path.join(signals_dir, f'{signal_name}_{split}.txt')
            if not os.path.exists(file_path):
                alt_split = 'test' if split == 'train' else 'train'
                alt_path  = os.path.join(signals_dir, f'{signal_name}_{alt_split}.txt')
                if os.path.exists(alt_path):
                    print(f"Warning: {file_path} not found, using {alt_path}")
                    file_path = alt_path
            all_signals.append(np.loadtxt(file_path))   # [n_samples, seq_len]
        data = np.stack(all_signals, axis=-1)            # [n_samples, seq_len, n_vars]
        return data.astype(np.float32)

    def _load_data(self):
        data = self._load_raw_signals(self.split)        # [N, T, n_vars]
        n_samples, seq_len, n_vars = data.shape

        if self.normalize:
            # Always fit scaler on training data
            train_data = self._load_raw_signals('train') if self.split != 'train' else data
            # Flatten to [N*T, n_vars] for fitting
            self.scaler = StandardScaler()
            self.scaler.fit(train_data.reshape(-1, n_vars))
            # Transform current split
            data = self.scaler.transform(data.reshape(-1, n_vars)).reshape(n_samples, seq_len, n_vars)

        data   = torch.from_numpy(data)

        label_path = os.path.join(self.data_dir, self.split, f'y_{self.split}.txt')
        labels = np.loadtxt(label_path, dtype=int)
        labels = torch.from_numpy(labels - 1).long()   # 1-6 → 0-5

        print(f"Loaded {self.split} set: {data.shape[0]} samples, "
              f"{data.shape[1]} timesteps, {data.shape[2]} variables")
        print(f"Labels shape: {labels.shape}, min: {labels.min()}, max: {labels.max()}")
        return data, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ── UCI HAR DINO pretraining ──────────────────────────────────────────────────

class DataPullerUCIDINO(Dataset):
    """
    UCI HAR Dataset loader for DINO pretraining.
    Returns only the time series data (no labels) with augmentations.
    StandardScaler is always fitted on the training split only.
    """
    def __init__(self, data_dir, split='train', transform=None,
                 batch_size=16, patch_size=16, step_size=16, c_in=9):
        self.data_dir    = data_dir
        self.split       = split
        self.transform   = transform
        self.patch_size  = patch_size
        self.num_patches = batch_size
        self.step_size   = step_size
        self.window_size = self.num_patches * self.patch_size

        self.signal_files = [
            'body_acc_x',   'body_acc_y',   'body_acc_z',
            'body_gyro_x',  'body_gyro_y',  'body_gyro_z',
            'total_acc_x',  'total_acc_y',  'total_acc_z',
        ]

        self.data    = self._load_data()
        self.windows = list(range(0, len(self.data) - self.window_size + 1, self.step_size))

    def _load_raw_signals(self, split):
        split_dir   = os.path.join(self.data_dir, split)
        signals_dir = os.path.join(split_dir, 'Inertial Signals')
        all_signals = []
        for signal_name in self.signal_files:
            file_path = os.path.join(signals_dir, f'{signal_name}_{split}.txt')
            if not os.path.exists(file_path):
                alt_split = 'test' if split == 'train' else 'train'
                alt_path  = os.path.join(signals_dir, f'{signal_name}_{alt_split}.txt')
                if os.path.exists(alt_path):
                    file_path = alt_path
            all_signals.append(np.loadtxt(file_path))   # [n_samples, seq_len=128]
        return np.stack(all_signals, axis=-1).astype(np.float32)  # [n_samples, 128, 9]

    def _load_data(self):
        data = self._load_raw_signals(self.split)  # [n_samples, 128, 9]
        n_samples, seq_len, n_vars = data.shape

        # Always fit StandardScaler on training data only
        train_data = self._load_raw_signals('train') if self.split != 'train' else data
        self.scaler = StandardScaler()
        self.scaler.fit(train_data.reshape(-1, n_vars))
        data = self.scaler.transform(data.reshape(-1, n_vars))    # [n_samples*128, 9]

        data = torch.from_numpy(data)
        print(f"Loaded {self.split} set for DINO: {data.shape[0]} timesteps, "
              f"{data.shape[1]} variables")
        return data

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start = self.windows[idx]
        end   = start + self.window_size
        chunk = self.data[start:end]   # [window_size, n_vars] — already normalised

        if self.transform:
            chunk = self.transform(chunk)
        return chunk


# ── Monash pretraining ────────────────────────────────────────────────────────

class MonashDataPuller(Dataset):
    """
    Loads all Monash .tsf files from a directory for DINO pretraining.

    Each univariate series is returned as [window_size, 1] — no variable-count
    restrictions. Used with a separate DataLoader from the main dataset so that
    different n_vars never need to be batched together.
    """

    def __init__(self, data_dir, split='train', transform=None,
                 batch_size=32, patch_size=12, step_size=12,
                 min_len=512, val_prec=0.1, test_prec=0.1):
        self.window_size = (batch_size - 1) * step_size + patch_size
        self.step_size   = step_size
        self.transform   = transform
        self.which       = split

        self._series = {'train': [], 'val': [], 'test': []}
        self._index  = {'train': [], 'val': [], 'test': []}

        self._load_all(data_dir, min_len, val_prec, test_prec)
        print(f"MonashDataPuller: {len(self._index['train'])} train  "
              f"| {len(self._index['val'])} val  "
              f"| {len(self._index['test'])} test  windows  "
              f"(window={self.window_size}, step={self.step_size})")

    def _load_all(self, data_dir, min_len, val_prec, test_prec):
        tsf_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.tsf'))

        for fname in tsf_files:
            path = os.path.join(data_dir, fname)
            try:
                series_list = _read_tsf_series(path)
            except Exception as e:
                print(f"  MonashDataPuller: skipping {fname} — {e}")
                continue

            loaded = 0
            for series in series_list:
                if np.isnan(series).any() or len(series) < min_len:
                    continue
                series = series.reshape(-1, 1)  # [T, 1]

                T         = len(series)
                val_len   = int(T * val_prec)
                test_len  = int(T * test_prec)
                train_len = T - val_len - test_len

                scaler = StandardScaler()
                scaler.fit(series[:train_len])
                scaled = scaler.transform(series).astype(np.float32)

                splits = {
                    'train': scaled[:train_len],
                    'val':   scaled[train_len : train_len + val_len],
                    'test':  scaled[train_len + val_len :],
                }
                for sname, arr in splits.items():
                    if len(arr) < self.window_size:
                        continue
                    t     = torch.from_numpy(arr)   # [T, 1]
                    s_idx = len(self._series[sname])
                    self._series[sname].append(t)
                    n_windows = (len(t) - self.window_size) // self.step_size + 1
                    for j in range(n_windows):
                        self._index[sname].append((s_idx, j))
                loaded += 1
            if loaded:
                print(f"  {fname}: {loaded} series")

    def __len__(self):
        return len(self._index[self.which])

    def __getitem__(self, idx):
        s_idx, j = self._index[self.which][idx]
        tensor = self._series[self.which][s_idx]
        start  = j * self.step_size
        chunk  = tensor[start : start + self.window_size]  # [window_size, 1]
        if self.transform:
            chunk = self.transform(chunk)
        return chunk


class SyntheticArrowDataPuller(Dataset):
    """
    Loads synthetic time series from GluonTS .arrow files for DINO pretraining.

    Mirrors the MonashDataPuller interface — returns [window_size, 1] tensors.
    Handles both univariate [T] and multivariate [C, T] targets (each channel
    is treated as a separate univariate series).
    """

    def __init__(self, data_dir, split='train', transform=None,
                 batch_size=32, patch_size=12, step_size=12,
                 min_len=512, val_prec=0.1, test_prec=0.1):
        self.window_size = (batch_size - 1) * step_size + patch_size
        self.step_size   = step_size
        self.transform   = transform
        self.which       = split

        self._series = {'train': [], 'val': [], 'test': []}
        self._index  = {'train': [], 'val': [], 'test': []}

        self._load_all(data_dir, min_len, val_prec, test_prec)
        print(f"SyntheticArrowDataPuller: {len(self._index['train'])} train  "
              f"| {len(self._index['val'])} val  "
              f"| {len(self._index['test'])} test  windows  "
              f"(window={self.window_size}, step={self.step_size})")

    def _load_all(self, data_dir, min_len, val_prec, test_prec):
        import pyarrow as pa

        arrow_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.arrow'))
        for fname in arrow_files:
            path = os.path.join(data_dir, fname)
            try:
                with pa.memory_map(path, 'r') as src:
                    table = pa.ipc.open_file(src).read_all()
            except Exception:
                try:
                    with open(path, 'rb') as f:
                        reader = pa.ipc.open_stream(f)
                        table = reader.read_all()
                except Exception as e:
                    print(f"  SyntheticArrowDataPuller: skipping {fname} — {e}")
                    continue

            loaded = 0
            for row in table.column('target'):
                arr = row.as_py()
                if arr is None:
                    continue
                arr = np.array(arr, dtype=np.float32)

                # Normalise to list of univariate [T] arrays
                if arr.ndim == 1:
                    channels = [arr]
                elif arr.ndim == 2:
                    channels = [arr[c] for c in range(arr.shape[0])]  # [C, T] → C series
                else:
                    continue

                for series in channels:
                    if np.isnan(series).any() or len(series) < min_len:
                        continue
                    series = series.reshape(-1, 1)  # [T, 1]

                    T         = len(series)
                    val_len   = int(T * val_prec)
                    test_len  = int(T * test_prec)
                    train_len = T - val_len - test_len

                    scaler = StandardScaler()
                    scaler.fit(series[:train_len])
                    scaled = scaler.transform(series).astype(np.float32)

                    splits = {
                        'train': scaled[:train_len],
                        'val':   scaled[train_len : train_len + val_len],
                        'test':  scaled[train_len + val_len :],
                    }
                    for sname, seg in splits.items():
                        if len(seg) < self.window_size:
                            continue
                        t     = torch.from_numpy(seg)
                        s_idx = len(self._series[sname])
                        self._series[sname].append(t)
                        n_windows = (len(t) - self.window_size) // self.step_size + 1
                        for j in range(n_windows):
                            self._index[sname].append((s_idx, j))
                    loaded += 1
            if loaded:
                print(f"  {fname}: {loaded} series")

    def __len__(self):
        return len(self._index[self.which])

    def __getitem__(self, idx):
        s_idx, j = self._index[self.which][idx]
        tensor = self._series[self.which][s_idx]
        start  = j * self.step_size
        chunk  = tensor[start : start + self.window_size]  # [window_size, 1]
        if self.transform:
            chunk = self.transform(chunk)
        return chunk
