import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from making_style import get_mask_style
import os
import sys
import pickle
import torch
import numpy as np
from pathlib import Path
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
                vals_str = line.split(':')[-1].split(',')
                vals = []
                for v in vals_str:
                    v = v.strip()
                    vals.append(np.nan if v == '?' else float(v))
                if vals:
                    series_list.append(np.array(vals, dtype=np.float32))
    return series_list

class DataPullerDJepa(Dataset):
    def __init__(self,
    data_paths,
    patch_size,
    batch_size,
    ratio_patches,
    mask_ratio,
    masking_type,
    num_semantic_tokens,
    input_variables,
    timestamp_cols,
    type_data,
    val_prec = 0.1,
    test_prec = 0.25,
    epochs = 5000,
    stride = None,
    num_blocks = 1):
        self.batch_size = batch_size
        self.ratio_patches = ratio_patches
        self.mask_ratio = mask_ratio
        self.masking_type = masking_type
        self.num_blocks = num_blocks
        self.num_semantic_tokens = num_semantic_tokens
        self.input_variables = input_variables
        self.timestamp_cols = timestamp_cols
        self.data_paths = data_paths
        self.val_prec = val_prec
        self.test_prec = test_prec
        self.which = type_data  # 'train', 'val', 'test'
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size  # default: non-overlapping
        self.chunk_size = self.patch_size + (self.ratio_patches - 1) * self.stride
        self.all_map = {'train': [], 'val': [], 'test': []}
        self.scaler = StandardScaler()
        self.epochs_completed = 0 
        self.epochs = epochs

        processed_dfs = []
        self.Train_Val_Test_splits = {
            'train': [],
            'val': [],
            'test': []
        }
        sizee = 0
        for path, t_col, input_vars in zip(data_paths, timestamp_cols, input_variables):
            df = pd.read_csv(path, parse_dates=[t_col], low_memory=False, sep=',')          
            fcols = df.select_dtypes("float").columns.tolist()
            df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
            processed_dfs.append(df)
            icols = df.select_dtypes("integer").columns
            df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")
            df.sort_values(by=[t_col], inplace=True)
            val_len = int(len(df) * self.val_prec)
            test_len = int(len(df) * self.test_prec)
            train_len = len(df) - val_len - test_len
            # Fit scaler on training portion only, transform all splits
            train_portion = df.iloc[:train_len][input_vars].values
            self.scaler.fit(train_portion)
            df_scaled = self.scaler.transform(df[input_vars].values)
            df_tensor = torch.tensor(df_scaled).float()
            print(f"--- Normalization Check ({self.which}) ---")
            print(f"Mean (should be ~0): {df_tensor.mean().item():.6f}")
            print(f"Std  (should be ~1): {df_tensor.std().item():.6f}")
            train_df, val_df, test_df = torch.split(df_tensor, [train_len, val_len, test_len])
            self.Train_Val_Test_splits['train'].append(train_df)
            self.Train_Val_Test_splits['val'].append(val_df)
            self.Train_Val_Test_splits['test'].append(test_df)
            # Sliding window with step=1 for maximum coverage on small datasets
        window_step = 1
        for split_name in ['train', 'val', 'test']:
            for file_idx, tensor in enumerate(self.Train_Val_Test_splits[split_name]):
                n = tensor.size(0)
                max_start = n - self.chunk_size
                if max_start < 0:
                    continue
                for start in range(0, max_start + 1, window_step):
                    self.all_map[split_name].append((file_idx, start))

    def __len__(self):
        return len(self.all_map[self.which])

    def __getitem__(self, idx):
        file_idx, start = self.all_map[self.which][idx]
        source_data = self.Train_Val_Test_splits[self.which][file_idx]
        start = min(start, max(0, source_data.size(0) - self.chunk_size))
        end = start + self.chunk_size
        chunk = source_data[start:end]
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(-1)
        patches = [chunk[i * self.stride : i * self.stride + self.patch_size] for i in range(self.ratio_patches)]
        patches_tensor = torch.stack(patches)  # [ratio_patches, patch_size, n_vars]
        masking_avg = self.mask_ratio + 0.3*(self.epochs_completed / 5000)
        context_idx, target_idx = get_mask_style(
            B=1,
            num_patches=self.ratio_patches,
            type=self.masking_type,
            p=self.mask_ratio,
            num_blocks=self.num_blocks,
        )
        self.epochs_completed += 1
        return patches_tensor, context_idx.squeeze(0), target_idx.squeeze(0)


# ── PatchTST-identical forecasting adapter ────────────────────────────────────

class PatchTSTForcastingAdapter(Dataset):
    """
    Wraps PatchTST's Dataset_ETT_hour / Dataset_ETT_minute / Dataset_Custom
    and reshapes (seq_x, seq_y) → (context_patches, target_patches).

    All split borders, normalization, and sliding windows are 100% identical
    to PatchTST's linear-probe setup.

    seq_x  [seq_len, n_vars]   →  context_patches [context_size, patch_size, n_vars]
    seq_y  [pred_len, n_vars]  →  target_patches  [h, patch_size, n_vars]

    Requires seq_len % patch_size == 0 and pred_len % patch_size == 0.
    """

    def __init__(self, csv_path: str, split: str, seq_len: int, pred_len: int, patch_size: int):
        from pathlib import Path as _Path
        _patchtst_dir = str(_Path(__file__).parent.parent.parent / "PatchTST_self_supervised")
        if _patchtst_dir not in sys.path:
            sys.path.insert(0, _patchtst_dir)
        from src.data.pred_dataset import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom

        assert seq_len  % patch_size == 0, f"seq_len={seq_len} not divisible by patch_size={patch_size}"
        assert pred_len % patch_size == 0, f"pred_len={pred_len} not divisible by patch_size={patch_size}"

        self.patch_size   = patch_size
        self.context_size = seq_len  // patch_size
        self.h            = pred_len // patch_size

        root      = os.path.dirname(os.path.abspath(csv_path))
        fname     = os.path.basename(csv_path)
        fname_low = fname.lower()
        size      = [seq_len, 0, pred_len]  # label_len=0: target immediately follows context

        if 'etth' in fname_low:
            self._ds = Dataset_ETT_hour(root, split=split, size=size, features='M', data_path=fname)
        elif 'ettm' in fname_low:
            self._ds = Dataset_ETT_minute(root, split=split, size=size, features='M', data_path=fname)
        else:
            self._ds = Dataset_Custom(root, split=split, size=size, features='M', data_path=fname)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        seq_x, seq_y = self._ds[idx]                                 # [seq_len, n_vars], [pred_len, n_vars]
        ctx = seq_x.reshape(self.context_size, self.patch_size, -1)  # [context_size, patch_size, n_vars]
        tgt = seq_y.reshape(self.h,            self.patch_size, -1)  # [h, patch_size, n_vars]
        return ctx, tgt


class ForcastingDataPuller(Dataset):
    def __init__(self,config, which='train'):
        self.patch_size = config["patch_size_forcasting"]
        self.context_size = config["ratio_patches"]
        self.input_variables_forcasting = config["input_variables_forcasting"]
        self.timestamp_cols = config["timestampcols"]
        self.val_prec = config["val_prec_forcasting"]
        self.test_prec = config["test_prec_forcasting"]
        self.data_paths = config["data_paths"]
        self.Train_Val_Test_splits = {'train': [], 'val': [], 'test': []}
        self.scaler = StandardScaler()
        self.which = which
        processed_dfs = []
       

        for path, t_col, input_vars in zip(self.data_paths, self.timestamp_cols, self.input_variables_forcasting):
            print(f"DEBUG: Processing dataset {path}, timestamp column: {t_col}, input variables: {input_vars}")
            df = pd.read_csv(path, parse_dates=[t_col], low_memory=False, sep=',')          
            fcols = df.select_dtypes("float").columns.tolist()
            df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
            processed_dfs.append(df)
            icols = df.select_dtypes("integer").columns
            df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")
            df.sort_values(by=[t_col], inplace=True)
            val_len = int(len(df) * self.val_prec)
            test_len = int(len(df) * self.test_prec)
            train_len = len(df) - val_len - test_len
            # --- NORMALIZATION START ---
            # Fit scaler only on training portion
            train_portion = df.iloc[:train_len][input_vars].values
            self.scaler.fit(train_portion)
            print("hereeee -jjjjjjj")
            # Transform the whole dataframe
            df_scaled = self.scaler.transform(df[input_vars].values)
            df_tensor = torch.tensor(df_scaled).float()
            # --- NORMALIZATION END ---
            print(f"--- Normalization Check ({self.which}) ---")
            print(f"Mean (should be ~0): {df_tensor.mean().item():.6f}")
            print(f"Std  (should be ~1): {df_tensor.std().item():.6f}")
            print(f"Min: {df_tensor.min().item():.6f}")
            print(f"Max: {df_tensor.max().item():.6f}")
            train_df, val_df, test_df = torch.split(df_tensor, [train_len, val_len, test_len])
            self.Train_Val_Test_splits['train'].append(train_df)
            self.Train_Val_Test_splits['val'].append(val_df)
            self.Train_Val_Test_splits['test'].append(test_df)
        self.series = torch.cat(self.Train_Val_Test_splits[self.which], dim=0)
        # Split the entire time series into patches
        self.patches_tensor = self.split_into_patches(self.series, self.patch_size)

    def split_into_patches(self, series, patch_size):
        num_patches = len(series) // patch_size
        patches = [series[i * patch_size:(i + 1) * patch_size] for i in range(num_patches)]
        return torch.stack(patches)  # Shape will be (num_patches, patch_size)

    def __len__(self):
        # Number of available samples based on the context size
        return len(self.patches_tensor) - self.context_size

    def __getitem__(self, idx):
        # Here we ensure that each time we return a context window of 10 patches
        if idx + self.context_size + 1 > len(self.patches_tensor):
            raise IndexError("Index out of range for context window")

        # Get context patches (previous 10) and the target patch (next one)
        context_patches = self.patches_tensor[idx:idx + self.context_size]
        target_patch = self.patches_tensor[idx + self.context_size]

        return context_patches.squeeze(-1), target_patch.squeeze(-1)



class DataPullerVQVAE(Dataset):
    def __init__(self, data_paths, flag='train', chunk_size=128,
                 input_variables=None, timestamp_cols=None,
                 val_prec=0.1, test_prec=0.25):

        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.chunk_size = chunk_size # Total window length for the Conv1D layers
        self.input_variables = input_variables
        self.timestamp_cols = timestamp_cols
        self.data_paths = data_paths

        self.val_prec = val_prec
        self.test_prec = test_prec

        self.scaler = StandardScaler()
        self.all_map = []
        self.data_splits = []

        self.__read_data__()

    def __read_data__(self):
        for path, t_col in zip(self.data_paths, self.timestamp_cols):
            df_raw = pd.read_csv(path, parse_dates=[t_col])
            df_raw.sort_values(by=[t_col], inplace=True)
            df_data = df_raw[self.input_variables]

            # Split Borders
            val_len = int(len(df_raw) * self.val_prec)
            test_len = int(len(df_raw) * self.test_prec)
            train_len = len(df_raw) - val_len - test_len

            # Scale based on training part only
            train_portion = df_data.iloc[:train_len]
            self.scaler.fit(train_portion.values)
            data = self.scaler.transform(df_data.values)

            tensor_data = torch.tensor(data).float()
            
            # Split data [cite: 64]
            train_part, val_part, test_part = torch.split(tensor_data, [train_len, val_len, test_len])
            
            split_map = {'train': train_part, 'val': val_part, 'test': test_part}
            active_tensor = split_map[self.flag]
            
            file_idx = len(self.data_splits)
            self.data_splits.append(active_tensor)
            
            # Create sliding window or non-overlapping chunk map
            num_chunks = (active_tensor.size(0) - self.chunk_size) + 1
            for start_idx in range(0, num_chunks, self.chunk_size): # Non-overlapping
                self.all_map.append((file_idx, start_idx))

    def __len__(self):
        return len(self.all_map)

    def __getitem__(self, index):
        file_idx, start = self.all_map[index]
        source_data = self.data_splits[file_idx]
        
        end = start + self.chunk_size
        chunk = source_data[start:end]

        return chunk.permute(1, 0) 

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class ForcastingDataPullerDescrete(Dataset):
    def __init__(self, config, which='train'):
        self.patch_size   = config["patch_size_forcasting"]
        self.context_size = config["ratio_patches"]   # number of non-overlapping patches in context
        self.h            = config["horizon_t"]       # number of non-overlapping patches to forecast
        # window_step_forecasting: raw-timestep stride between consecutive windows.
        # Default 1 matches PatchTST / DINO (maximum overlap, most training samples).
        self.window_step  = config.get("window_step_forecasting", 1)

        self.context_raw_len = self.context_size * self.patch_size
        self.target_raw_len  = self.h            * self.patch_size

        self.input_variables_forcasting = config["input_variables_forcasting"]
        self.timestamp_cols = config["timestampcols_forcasting"]
        self.val_prec  = config["val_prec_forcasting"]
        self.test_prec = config["test_prec_forcasting"]
        self.data_paths = config["path_data_forcasting"]
        self.Train_Val_Test_splits = {'train': [], 'val': [], 'test': []}
        self.which  = which
        self.scaler = StandardScaler()

        for run_idx, (path, t_col) in enumerate(zip(self.data_paths, self.timestamp_cols)):
            df = pd.read_csv(path, parse_dates=[t_col], low_memory=False, sep=',')
            fcols = df.select_dtypes("float").columns.tolist()
            df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
            icols = df.select_dtypes("integer").columns
            df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")
            df.sort_values(by=[t_col], inplace=True)
            n = len(df)
            fname = os.path.basename(path).lower()
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
            input_vars = self.input_variables_forcasting[run_idx]
            # Fit scaler on training portion only, transform all splits
            train_portion = df.iloc[:train_len][input_vars].values
            self.scaler.fit(train_portion)
            df_scaled = self.scaler.transform(df[input_vars].values)
            df_tensor = torch.tensor(df_scaled).float()
            seq_len = self.context_raw_len
            border1s = [0,
                        train_len - seq_len,
                        train_len + val_len - seq_len]
            border2s = [train_len,
                        train_len + val_len,
                        train_len + val_len + test_len]
            train_df = df_tensor[border1s[0] : border2s[0]]
            val_df   = df_tensor[border1s[1] : border2s[1]]
            test_df  = df_tensor[border1s[2] : border2s[2]]
            self.Train_Val_Test_splits['train'].append(train_df)
            self.Train_Val_Test_splits['val'].append(val_df)
            self.Train_Val_Test_splits['test'].append(test_df)
        self._rebuild()

    def rebuild(self):
        self._rebuild()

    def _rebuild(self):
        """Set self.series to the raw split tensor."""
        self.series = torch.cat(self.Train_Val_Test_splits[self.which], dim=0)  # [T, n_vars]

    def inverse_transform(self, tensor):
        """Inverse-normalize a tensor of shape [..., n_vars] back to original scale."""
        scale = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=tensor.device)
        mean  = torch.tensor(self.scaler.mean_,  dtype=torch.float32, device=tensor.device)
        return tensor * scale + mean

    def __len__(self):
        window_raw = self.context_raw_len + self.target_raw_len
        return max(0, (len(self.series) - window_raw) // self.window_step + 1)

    def __getitem__(self, idx):
        start = idx * self.window_step
        mid   = start + self.context_raw_len
        end   = mid   + self.target_raw_len
        # Slice raw timesteps then reshape into non-overlapping patches: [n_patches, patch_size, n_vars]
        context_patches = self.series[start:mid].reshape(self.context_size, self.patch_size, -1)
        target_patch    = self.series[mid:end].reshape(self.h,            self.patch_size, -1)
        return context_patches, target_patch



# ── Monash pretraining (JEPA) ─────────────────────────────────────────────────

class MonashDataPullerJEPA(Dataset):
    """
    Loads all Monash .tsf files for Discrete JEPA pretraining.

    Returns (patches_tensor, context_idx, target_idx) matching DataPullerDJepa.
    Each univariate series is returned as [ratio_patches, patch_size, 1] — no
    variable-count restrictions. Used with a separate DataLoader.

    Config keys used:
        monash_data_dir  – path to .tsf directory
        monash_min_len   – min raw series length (default 512)
        patch_size, ratio_patches, mask_ratio, masking_type, num_blocks,
        val_prec, test_prec
    """

    def __init__(self, config, which='train'):
        self.which          = which
        self.patch_size     = config['patch_size']
        self.ratio_patches  = config['ratio_patches']
        self.mask_ratio     = config['mask_ratio']
        self.masking_type   = config['masking_type']
        self.num_blocks     = config.get('num_blocks', 1)
        self.chunk_size     = self.ratio_patches * self.patch_size
        self.epochs_completed = 0

        val_prec  = config.get('val_prec', 0.1)
        test_prec = config.get('test_prec', 0.1)
        data_dir  = config['monash_data_dir']
        min_len   = config.get('monash_min_len', 512)

        self._series = {'train': [], 'val': [], 'test': []}
        self._index  = {'train': [], 'val': [], 'test': []}

        self._load_all(data_dir, min_len, val_prec, test_prec)
        print(f"MonashDataPullerJEPA: {len(self._index['train'])} train  "
              f"| {len(self._index['val'])} val  "
              f"| {len(self._index['test'])} test  chunks  "
              f"(chunk={self.chunk_size})")

    def _load_all(self, data_dir, min_len, val_prec, test_prec):
        tsf_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.tsf'))

        for fname in tsf_files:
            path = os.path.join(data_dir, fname)
            try:
                series_list = _read_tsf_series(path)
            except Exception as e:
                print(f"  MonashDataPullerJEPA: skipping {fname} — {e}")
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
                    if len(arr) < self.chunk_size:
                        continue
                    t     = torch.from_numpy(arr)   # [T, 1]
                    s_idx = len(self._series[sname])
                    self._series[sname].append(t)
                    n_chunks = len(t) // self.chunk_size
                    for ci in range(n_chunks):
                        self._index[sname].append((s_idx, ci))
                loaded += 1
            if loaded:
                print(f"  {fname}: {loaded} series")

    def __len__(self):
        return len(self._index[self.which])

    def __getitem__(self, idx):
        s_idx, ci = self._index[self.which][idx]
        tensor = self._series[self.which][s_idx]
        start  = ci * self.chunk_size
        chunk  = tensor[start : start + self.chunk_size]  # [chunk_size, 1]

        patches = [chunk[i * self.patch_size : (i + 1) * self.patch_size]
                   for i in range(self.ratio_patches)]
        patches_tensor = torch.stack(patches)  # [ratio_patches, patch_size, 1]

        context_idx, target_idx = get_mask_style(
            B=1,
            num_patches=self.ratio_patches,
            type=self.masking_type,
            p=self.mask_ratio,
            num_blocks=self.num_blocks,
        )
        self.epochs_completed += 1
        return patches_tensor, context_idx.squeeze(0), target_idx.squeeze(0)


class SyntheticArrowDataPullerJEPA(Dataset):
    """
    Loads synthetic time series from GluonTS .arrow files (produced by
    LMC_Synth.py and kernel-synth.py) for JEPA-style pretraining.

    Handles both univariate targets (shape [T]) and multivariate targets
    (shape [C, T]).  Each window is returned as [ratio_patches, patch_size, C],
    matching the shape produced by MonashDataPullerJEPA (with C=1 for univariate).

    Returns (patches_tensor, context_idx, target_idx) — identical contract to
    MonashDataPullerJEPA so ConcatDataset works transparently.

    Config keys used:
        synthetic_data_dir  – directory containing .arrow files
        patch_size, ratio_patches, mask_ratio, masking_type, num_blocks,
        val_prec, test_prec, monash_min_len (reused as min series length)
    """

    def __init__(self, config, which='train'):
        self.which          = which
        self.patch_size     = config['patch_size']
        self.ratio_patches  = config['ratio_patches']
        self.mask_ratio     = config['mask_ratio']
        self.masking_type   = config['masking_type']
        self.num_blocks     = config.get('num_blocks', 1)
        self.chunk_size     = self.ratio_patches * self.patch_size

        val_prec  = config.get('val_prec', 0.1)
        test_prec = config.get('test_prec', 0.1)
        data_dir  = config['synthetic_data_dir']
        min_len   = config.get('monash_min_len', 512)

        self._series = {'train': [], 'val': [], 'test': []}
        self._index  = {'train': [], 'val': [], 'test': []}

        self._load_all(data_dir, min_len, val_prec, test_prec)
        print(f"SyntheticArrowDataPullerJEPA: {len(self._index['train'])} train  "
              f"| {len(self._index['val'])} val  "
              f"| {len(self._index['test'])} test  chunks  "
              f"(chunk={self.chunk_size})")

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
                    print(f"  SyntheticArrowDataPullerJEPA: skipping {fname} — {e}")
                    continue

            targets = table.column('target')
            loaded = 0
            for row in targets:
                arr = row.as_py()
                if arr is None:
                    continue
                arr = np.array(arr, dtype=np.float32)

                # Normalise shape to [T, C]
                if arr.ndim == 1:
                    arr = arr[:, None]          # univariate [T] -> [T, 1]
                elif arr.ndim == 2:
                    arr = arr.T                 # multivariate [C, T] -> [T, C]
                else:
                    continue

                T = arr.shape[0]
                if T < min_len or np.isnan(arr).any():
                    continue

                val_len   = int(T * val_prec)
                test_len  = int(T * test_prec)
                train_len = T - val_len - test_len

                scaler = StandardScaler()
                scaler.fit(arr[:train_len])
                scaled = scaler.transform(arr).astype(np.float32)   # [T, C]

                splits = {
                    'train': scaled[:train_len],
                    'val':   scaled[train_len : train_len + val_len],
                    'test':  scaled[train_len + val_len :],
                }
                for sname, split_arr in splits.items():
                    if len(split_arr) < self.chunk_size:
                        continue
                    t     = torch.from_numpy(split_arr)   # [T, C]
                    s_idx = len(self._series[sname])
                    self._series[sname].append(t)
                    n_chunks = len(t) // self.chunk_size
                    for ci in range(n_chunks):
                        self._index[sname].append((s_idx, ci))
                loaded += 1
            if loaded:
                print(f"  {fname}: {loaded} series")

    def __len__(self):
        return len(self._index[self.which])

    def __getitem__(self, idx):
        s_idx, ci = self._index[self.which][idx]
        tensor = self._series[self.which][s_idx]
        start  = ci * self.chunk_size
        chunk  = tensor[start : start + self.chunk_size]   # [chunk_size, C]

        patches = [chunk[i * self.patch_size : (i + 1) * self.patch_size]
                   for i in range(self.ratio_patches)]
        patches_tensor = torch.stack(patches)   # [ratio_patches, patch_size, C]

        context_idx, target_idx = get_mask_style(
            B=1,
            num_patches=self.ratio_patches,
            type=self.masking_type,
            p=self.mask_ratio,
            num_blocks=self.num_blocks,
        )
        return patches_tensor, context_idx.squeeze(0), target_idx.squeeze(0)


# ── TimeDart pretraining window datasets ──────────────────────────────────────
#
# TimeDart's pretrain loop expects (batch_x, batch_y, batch_x_mark, batch_y_mark)
# where batch_x has shape (B, seq_len, C).  Only batch_x is used in the pretrain
# loss (diffusion reconstruction), so batch_y / marks are returned as zeros.

class MonashWindowDatasetTimeDart(Dataset):
    """
    Loads all Monash .tsf files for TimeDart pretraining.

    Each series is split train/val/test (same proportions used across the
    codebase: val_prec=0.1, test_prec=0.1), scaled with StandardScaler fitted
    on the training portion, then cut into non-overlapping windows of seq_len.

    Returns (batch_x, zeros_y, zeros_mark_x, zeros_mark_y) where
    batch_x is shape (seq_len, 1).  Only batch_x is consumed by TimeDart's
    pretrain loop; the remaining tensors satisfy the 4-tuple contract.

    Config keys used:
        monash_data_dir, monash_min_len, val_prec, test_prec
    """

    def __init__(self, data_dir, seq_len=336, which='train',
                 min_len=512, val_prec=0.1, test_prec=0.1):
        self.which   = which
        self.seq_len = seq_len

        self._series = {'train': [], 'val': [], 'test': []}
        self._index  = {'train': [], 'val': [], 'test': []}

        self._load_all(data_dir, min_len, val_prec, test_prec)
        print(f"MonashWindowDatasetTimeDart [{which}]: "
              f"{len(self._index['train'])} train | "
              f"{len(self._index['val'])} val | "
              f"{len(self._index['test'])} test windows (seq_len={seq_len})")

    def _load_all(self, data_dir, min_len, val_prec, test_prec):
        tsf_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.tsf'))
        for fname in tsf_files:
            path = os.path.join(data_dir, fname)
            try:
                series_list = _read_tsf_series(path)
            except Exception as e:
                print(f"  MonashWindowDatasetTimeDart: skipping {fname} — {e}")
                continue
            loaded = 0
            for series in series_list:
                if np.isnan(series).any() or len(series) < min_len:
                    continue
                series = series.reshape(-1, 1)   # [T, 1]
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
                    if len(arr) < self.seq_len:
                        continue
                    t     = torch.from_numpy(arr)
                    s_idx = len(self._series[sname])
                    self._series[sname].append(t)
                    n_windows = len(t) // self.seq_len
                    for wi in range(n_windows):
                        self._index[sname].append((s_idx, wi))
                loaded += 1
            if loaded:
                print(f"  {fname}: {loaded} series")

    def __len__(self):
        return len(self._index[self.which])

    def __getitem__(self, idx):
        s_idx, wi = self._index[self.which][idx]
        tensor    = self._series[self.which][s_idx]
        start     = wi * self.seq_len
        window    = tensor[start : start + self.seq_len]   # [seq_len, 1]
        zeros     = torch.zeros_like(window)
        return window, zeros, zeros, zeros


class SyntheticWindowDatasetTimeDart(Dataset):
    """
    Loads synthetic .arrow files for TimeDart pretraining.

    Same interface and return contract as MonashWindowDatasetTimeDart.
    Handles univariate [T] and multivariate [C, T] targets; each channel is
    treated as an independent univariate series (C=1 slice per window).

    Config keys used:
        synthetic_data_dir, monash_min_len (reused), val_prec, test_prec
    """

    def __init__(self, data_dir, seq_len=336, which='train',
                 min_len=512, val_prec=0.1, test_prec=0.1):
        self.which   = which
        self.seq_len = seq_len

        self._series = {'train': [], 'val': [], 'test': []}
        self._index  = {'train': [], 'val': [], 'test': []}

        self._load_all(data_dir, min_len, val_prec, test_prec)
        print(f"SyntheticWindowDatasetTimeDart [{which}]: "
              f"{len(self._index['train'])} train | "
              f"{len(self._index['val'])} val | "
              f"{len(self._index['test'])} test windows (seq_len={seq_len})")

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
                        table  = reader.read_all()
                except Exception as e:
                    print(f"  SyntheticWindowDatasetTimeDart: skipping {fname} — {e}")
                    continue

            loaded = 0
            for row in table.column('target'):
                arr = row.as_py()
                if arr is None:
                    continue
                arr = np.array(arr, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[:, None]          # [T] → [T, 1]
                elif arr.ndim == 2:
                    arr = arr.T                 # [C, T] → [T, C]
                else:
                    continue

                T = arr.shape[0]
                if T < min_len or np.isnan(arr).any():
                    continue

                val_len   = int(T * val_prec)
                test_len  = int(T * test_prec)
                train_len = T - val_len - test_len

                scaler = StandardScaler()
                scaler.fit(arr[:train_len])
                scaled = scaler.transform(arr).astype(np.float32)   # [T, C]

                splits = {
                    'train': scaled[:train_len],
                    'val':   scaled[train_len : train_len + val_len],
                    'test':  scaled[train_len + val_len :],
                }
                for sname, split_arr in splits.items():
                    if len(split_arr) < self.seq_len:
                        continue
                    t     = torch.from_numpy(split_arr)   # [T, C]
                    s_idx = len(self._series[sname])
                    self._series[sname].append(t)
                    n_windows = len(t) // self.seq_len
                    for wi in range(n_windows):
                        self._index[sname].append((s_idx, wi))
                loaded += 1
            if loaded:
                print(f"  {fname}: {loaded} series")

    def __len__(self):
        return len(self._index[self.which])

    def __getitem__(self, idx):
        s_idx, wi = self._index[self.which][idx]
        tensor    = self._series[self.which][s_idx]
        start     = wi * self.seq_len
        window    = tensor[start : start + self.seq_len]   # [seq_len, C]
        zeros     = torch.zeros_like(window)
        return window, zeros, zeros, zeros


# ── Classification ────────────────────────────────────────────────────────────

def _cls_load_dataset(root: Path, which: str, val_fraction: float):
    """
    Auto-detect dataset format and return (X [N,T,C] float32, y [N] int64).
    StandardScaler is fit on the train split (per variable) and applied to all splits.

    Supported formats
    -----------------
    1. Standard npy  : X_train.npy (N,T,C), y_train.npy (N,)
                       X_test.npy  (N,T,C), y_test.npy  (N,)
    2. Epilepsy npy  : train_d.npy (N,T,C), train_l.npy (N,)
                       test_d.npy  (N,T,C), test_l.npy  (N,)
    3. HAR pt        : train.pt, val.pt, test.pt
                       each a dict {'samples': Tensor (N,C,T), 'labels': Tensor (N,)}
    4. EEG pkl       : samples_train.pkl, samples_test.pkl
                       each a list of (str_label, ndarray (T,C), int_label)
    """
    root = Path(root)
    has_explicit_val = False

    # ── Format 3: HAR (.pt dicts) ────────────────────────────────────────────
    if (root / "train.pt").exists():
        has_explicit_val = True
        def _load_pt(p):
            d = torch.load(p, map_location="cpu")
            X = d["samples"].float().numpy()   # (N, C, T)
            X = X.transpose(0, 2, 1)           # → (N, T, C)
            y = d["labels"].long().numpy()
            return X, y
        X_tr, y_tr = _load_pt(root / "train.pt")
        X_va, y_va = _load_pt(root / "val.pt")
        X_te, y_te = _load_pt(root / "test.pt")

    # ── Format 4: EEG (.pkl lists) ───────────────────────────────────────────
    elif (root / "samples_train.pkl").exists():
        def _load_pkl(p):
            with open(p, "rb") as f:
                samples = pickle.load(f)
            X = np.stack([s[1] for s in samples], axis=0).astype(np.float32)  # (N,T,C)
            y = np.array([s[2] for s in samples], dtype=np.int64)
            return X, y
        X_tr, y_tr = _load_pkl(root / "samples_train.pkl")
        X_te, y_te = _load_pkl(root / "samples_test.pkl")

    # ── Format 2: Epilepsy npy ───────────────────────────────────────────────
    elif (root / "train_d.npy").exists():
        X_tr = np.load(root / "train_d.npy").astype(np.float32)
        y_tr = np.load(root / "train_l.npy").astype(np.int64)
        X_te = np.load(root / "test_d.npy").astype(np.float32)
        y_te = np.load(root / "test_l.npy").astype(np.int64)

    # ── Format 1: Standard npy ───────────────────────────────────────────────
    elif (root / "X_train.npy").exists():
        X_tr = np.load(root / "X_train.npy").astype(np.float32)
        y_tr = np.load(root / "y_train.npy").astype(np.int64)
        X_te = np.load(root / "X_test.npy").astype(np.float32)
        y_te = np.load(root / "y_test.npy").astype(np.int64)

    else:
        raise FileNotFoundError(
            f"Cannot detect dataset format in {root}. "
            "Expected one of: X_train.npy, train_d.npy, train.pt, samples_train.pkl")

    # StandardScaler fit on train X (per variable) — applied to all splits
    N, T, C = X_tr.shape
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, C))
    X_tr = scaler.transform(X_tr.reshape(-1, C)).reshape(N, T, C)
    Nt, Tt, _ = X_te.shape
    X_te = scaler.transform(X_te.reshape(-1, C)).reshape(Nt, Tt, C)

    if has_explicit_val:
        Nv, Tv, _ = X_va.shape
        X_va = scaler.transform(X_va.reshape(-1, C)).reshape(Nv, Tv, C)
    else:
        n_val   = max(1, int(len(X_tr) * val_fraction))
        n_train = len(X_tr) - n_val
        X_va, y_va = X_tr[n_train:], y_tr[n_train:]
        X_tr, y_tr = X_tr[:n_train], y_tr[:n_train]

    if which == "train":
        return X_tr, y_tr
    elif which == "val":
        return X_va, y_va
    else:
        return X_te, y_te


class ClassificationDataPuller(Dataset):
    """
    Elastic classification dataset loader. Supports multiple datasets (concatenated
    with re-mapped labels) and four file formats: standard npy, Epilepsy npy,
    HAR .pt dicts, EEG .pkl lists. All splits are StandardScaler-normalised using
    stats fit on the train split of each dataset.

    Args:
        data_dir      : root directory containing dataset sub-folders
                        (e.g. "/home/shared/datasets/Classification_TS")
        dataset_names : str or list of str — dataset sub-folder names
        patch_size    : patch length; seq_len is zero-padded to a multiple of this
        which         : "train" | "val" | "test"
        val_fraction  : fraction of train used as val for datasets without explicit val
    """

    def __init__(self, data_dir: str, dataset_names, patch_size: int,
                 which: str = "train", val_fraction: float = 0.1):
        assert which in ("train", "val", "test")
        self.patch_size = patch_size

        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        all_X, all_y = [], []
        label_offset = 0

        for dataset_name in dataset_names:
            root = Path(data_dir) / dataset_name
            X, y = _cls_load_dataset(root, which, val_fraction)

            # 0-index labels within this dataset, then offset by accumulated classes
            unique = np.unique(y)
            label_map = {v: i + label_offset for i, v in enumerate(sorted(unique))}
            y = np.array([label_map[v] for v in y], dtype=np.int64)
            label_offset += len(unique)

            all_X.append(X)
            all_y.append(y)

        self.n_classes = label_offset

        # Pad all arrays to the same seq_len (max across datasets, rounded to patch_size)
        max_T    = max(x.shape[1] for x in all_X)
        padded_T = int(np.ceil(max_T / patch_size)) * patch_size
        padded   = []
        for x in all_X:
            pad = padded_T - x.shape[1]
            if pad > 0:
                x = np.pad(x, ((0, 0), (0, pad), (0, 0)))
            padded.append(x)

        X_cat = np.concatenate(padded, axis=0)
        y_cat = np.concatenate(all_y,  axis=0)

        self.X         = torch.tensor(X_cat)
        self.y         = torch.tensor(y_cat)
        self.n_patches = padded_T // patch_size

        names_str = ", ".join(dataset_names)
        print(f"ClassificationDataPuller [{which}] ({names_str}): "
              f"{len(self.X)} samples, {padded_T} timesteps → "
              f"{self.n_patches} patches × {patch_size}, "
              f"{X_cat.shape[2]} vars, {self.n_classes} classes")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # [seq_len, n_vars] → [n_patches, patch_size, n_vars]
        x = self.X[idx]
        patches = x.reshape(self.n_patches, self.patch_size, -1)
        return patches, self.y[idx]


# ── Anomaly Detection ─────────────────────────────────────────────────────────

class AnomalyDataPuller(Dataset):
    """
    Loads a time-series anomaly detection dataset.

    Stand-in path: /home/shared/datasets/Anomaly_TS/{dataset_name}/
    Expected files (auto-detected):
        train.npy          — [N, T, C]  float,  no labels (normal data only)
        test.npy           — [N, T, C]  float
        test_labels.npy    — [N, T]     int (0=normal, 1=anomaly)
           or test_labels_processed.npy

    All splits are StandardScaler-normalised using stats fit on the train split.
    Windows are patched into [n_patches, patch_size, n_vars] on the fly.

    Args:
        data_dir   : root directory containing dataset sub-folders
        dataset    : sub-folder name, e.g. "MSL"
        patch_size : patch length; seq_len is zero-padded to a multiple of this
        which      : "train" | "test"
    """

    def __init__(self, data_dir: str, dataset: str, patch_size: int, which: str = "train"):
        assert which in ("train", "test")
        root = Path(data_dir) / dataset

        if not root.exists():
            raise FileNotFoundError(
                f"Anomaly dataset not found: {root}\n"
                f"Place your dataset files under {root}/ and re-run.")

        # ── load data ────────────────────────────────────────────────────────
        X_tr = np.load(root / "train.npy", allow_pickle=True).astype(np.float32)
        X_te = np.load(root / "test.npy",  allow_pickle=True).astype(np.float32)

        # auto-detect label file name
        for label_file in ("test_labels.npy", "test_labels_processed.npy",
                           "test_label.npy", "labels_test.npy"):
            if (root / label_file).exists():
                labels = np.load(root / label_file, allow_pickle=True).astype(np.int64)
                break
        else:
            raise FileNotFoundError(
                f"No label file found in {root}. "
                "Expected: test_labels.npy or test_labels_processed.npy")

        # ensure [N, T, C] — swap if [N, C, T]
        if X_tr.ndim == 2:
            X_tr = X_tr[:, :, None]   # [N, T] → [N, T, 1]
            X_te = X_te[:, :, None]
        if X_tr.shape[1] < X_tr.shape[2]:
            # likely [N, C, T] → transpose
            X_tr = X_tr.transpose(0, 2, 1)
            X_te = X_te.transpose(0, 2, 1)

        # ── StandardScaler fit on train ──────────────────────────────────────
        N, T, C = X_tr.shape
        scaler = StandardScaler()
        scaler.fit(X_tr.reshape(-1, C))
        X_tr = scaler.transform(X_tr.reshape(-1, C)).reshape(N, T, C)
        Nt, Tt, _ = X_te.shape
        X_te = scaler.transform(X_te.reshape(-1, C)).reshape(Nt, Tt, C)

        # ── pad to multiple of patch_size ────────────────────────────────────
        padded_T = int(np.ceil(T / patch_size)) * patch_size
        def _pad(x):
            p = padded_T - x.shape[1]
            return np.pad(x, ((0, 0), (0, p), (0, 0))) if p > 0 else x

        X_tr = _pad(X_tr)
        X_te = _pad(X_te)

        # labels: [N, T] or [N*T] — keep per-timestep for scoring
        # pad labels to same length
        if labels.ndim == 1:
            labels = labels.reshape(Nt, -1)
        lp = padded_T - labels.shape[1]
        if lp > 0:
            labels = np.pad(labels, ((0, 0), (0, lp)))

        self.patch_size = patch_size
        self.n_patches  = padded_T // patch_size
        self.n_vars     = C

        if which == "train":
            self.X      = torch.tensor(X_tr)
            self.labels = None
        else:
            self.X      = torch.tensor(X_te)
            self.labels = torch.tensor(labels)   # [N, T]

        print(f"AnomalyDataPuller [{which}] ({dataset}): "
              f"{len(self.X)} windows, {padded_T} timesteps → "
              f"{self.n_patches} patches × {patch_size}, {C} vars")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # [T, C] → [n_patches, patch_size, C]
        patches = self.X[idx].reshape(self.n_patches, self.patch_size, -1)
        if self.labels is not None:
            return patches, self.labels[idx]   # test: (patches, label_seq)
        return patches                          # train: patches only
