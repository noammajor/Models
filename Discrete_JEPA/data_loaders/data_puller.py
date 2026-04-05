import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from making_style import get_mask_style
import os
import torch
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
