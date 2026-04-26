"""
Unified training + forecasting runner for:
  - dino     (TSDINOA)
  - jepa     (JEPA )
  - lejepa   (LE-JEPA — two-view augmentation, SIGReg loss)
  - patchtst (PatchTST_self_supervised)
  - NTP
  - TimedART (diffussion)

Usage
-----
  python Train_and_downstream.py --model dino
  python Train_and_downstream.py --model jepa
  python Train_and_downstream.py --model lejepa
  python Train_and_downstream.py --model patchtst

Colab
-----
  !python Train_and_downstream.py --model dino
  or call run(model="dino", skip_train=False) directly after importing.
"""

import os, sys, copy, argparse, random
import subprocess
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

# Make sure the project root (where dataset_registry.py lives) is importable
_PROJECT_ROOT = str(Path(__file__).parent.resolve())
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset_registry import get_dataset_info

GLOBAL_SEED = 42
_SEED_TAG   = ''   # set by run() when seed is provided; used by runners to suffix checkpoint paths

# Per-dataset anomaly detection hyperparameters matching TSLib reference configs
_ANOMALY_RATIO = {
    "SMD":  0.5,   # TSLib uses 0.5 for SMD
    "MSL":  1.0,
    "SMAP": 1.0,
    "PSM":  1.0,
    "SWaT": 1.0,
}

def _get_anomaly_ratio(dataset: str, cfg: dict) -> float:
    """Return TSLib-matched anomaly ratio, falling back to config or 1.0."""
    return _ANOMALY_RATIO.get(dataset, cfg.get("anomaly_ratio", 1.0))

def _set_seed(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── helpers ──────────────────────────────────────────────────────────────────

def _add_path(p):
    """Prepend p to sys.path if not already present."""
    p = str(Path(p).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)

def _get_forecast_bs(config, default=256):
    """Return forecast batch size — env var TS_FORECAST_BS (set by run_layer_forecast.py) takes priority."""
    env = os.environ.get("TS_FORECAST_BS")
    if env is not None:
        return int(env)
    return config.get("batch_size_forecast", default)

def _get_cls_bs(config, key="batch_size", default=64):
    """Return classification batch size — env var TS_CLS_BS takes priority over config."""
    env = os.environ.get("TS_CLS_BS")
    if env is not None:
        return int(env)
    return config.get(key, default)

def _get_forecast_lr(config, key="lr_forecasting", default=2e-4):
    """Return forecast LR scaled by TS_FORECAST_LR_SCALE (set by run_layer_forecast.py)."""
    base = config.get(key, config.get("lr_forcasting", default))
    scale = float(os.environ.get("TS_FORECAST_LR_SCALE", 1.0))
    return base * scale

def _resolve_pretrain_source(config):
    """Return 'monash', 'synthetic', or 'monash+synthetic' (or None for CSV pretraining).

    Checks the explicit 'pretrain_source' key first (set this in your config).
    Falls back to the legacy pretrain_on_monash + synthetic_data_dir flags.
    """
    if 'pretrain_source' in config:
        return config['pretrain_source']   # None means CSV-only
    if config.get('pretrain_on_monash', False):
        return 'monash+synthetic' if config.get('synthetic_data_dir') else 'monash'
    return None

def _config_to_dino_args(cfg):
    """
    Convert the DINO config dict (TSDINOALT 4/config.py) into the
    SimpleNamespace that train_TS_DINO / test_run expect.
    """
    local_crops = cfg.get("local_crops", [])
    global_crops = cfg.get("global_crops", [])

    args = SimpleNamespace(
        # ── task ──────────────────────────────────────────────────────────
        task                        = cfg.get("task", "dino"),
        test_only                   = cfg.get("test_only", False),
        seed                        = cfg.get("seed", 0),
        output_dir                  = cfg.get("output_dir", "./checkpoints"),
        saveckp_freq                = cfg.get("saveckp_freq", 10),

        # ── data ──────────────────────────────────────────────────────────
        data_path                   = cfg.get("data_path", "UCI HAR Dataset"),
        data_path_forecast_training = cfg.get("data_path_forecast_training", ""),
        data_path_forecast_test     = cfg.get("data_path_forecast_test", ""),
        data_path_classification    = cfg.get("data_path_classification", "UCI HAR Dataset"),
        num_workers                 = cfg.get("num_workers", 4),
        batch_size_per_gpu          = cfg.get("batch_size_per_gpu", 64),
        batch_size_forecast         = _get_forecast_bs(cfg, 256),

        # ── model architecture ────────────────────────────────────────────
        c_in                        = cfg.get("c_in", 7),
        patch_len                   = cfg.get("patch_len", 12),
        step_size                   = cfg.get("step_size", 12),
        num_patches                 = cfg.get("num_patches", 32),
        n_layers                    = cfg.get("n_layers", 5),
        n_heads                     = cfg.get("n_heads", 16),
        embed_dim                   = cfg.get("embed_dim", 128),
        d_ff                        = cfg.get("d_ff", 512),
        dropout                     = cfg.get("dropout", 0.1),
        head_dropout                = cfg.get("head_dropout", 0.1),
        drop_path_rate              = cfg.get("drop_path_rate", 0.1),

        # ── DINO head ─────────────────────────────────────────────────────
        out_dim                     = cfg.get("out_dim", 20000),
        use_bn_in_head              = cfg.get("use_bn_in_head", False),
        norm_last_layer             = cfg.get("norm_last_layer", True),

        # ── DINO loss / temperatures ──────────────────────────────────────
        warmup_teacher_temp         = cfg.get("warmup_teacher_temp", 0.04),
        teacher_temp                = cfg.get("teacher_temp", 0.04),
        warmup_teacher_temp_epochs  = cfg.get("warmup_teacher_temp_epochs", 0),

        # ── EMA teacher ───────────────────────────────────────────────────
        momentum_teacher            = cfg.get("momentum_teacher", 0.9995),

        # ── optimizer ─────────────────────────────────────────────────────
        optimizer                   = cfg.get("optimizer", "adamw"),
        lr                          = cfg.get("lr", 0.0005),
        min_lr                      = cfg.get("min_lr", 1e-6),
        warmup_epochs               = cfg.get("warmup_epochs", 10),
        weight_decay                = cfg.get("weight_decay", 0.04),
        weight_decay_end            = cfg.get("weight_decay_end", 0.4),
        clip_grad                   = cfg.get("clip_grad", 3.0),
        use_fp16                    = cfg.get("use_fp16", False),
        freeze_last_layer           = cfg.get("freeze_last_layer", 1),

        # ── training schedule ─────────────────────────────────────────────
        epochs                      = cfg.get("epochs", 100),

        # ── augmentation (derived from crop specs) ────────────────────────
        # local_crops_number  = crop ratio of the first local crop
        # transformation_group_size = total number of local crops
        local_crops_number          = local_crops[0]["crop_ratio"] if local_crops else 0.5,
        transformation_group_size   = len(local_crops) if local_crops else 2,

        # ── distributed (defaults for single-GPU / CPU) ───────────────────
        dist_url                    = cfg.get("dist_url", "env://"),
        gpu                         = None,
        rank                        = 0,
        world_size                  = 1,
        dist_backend                = "nccl",

        # ── downstream: forecasting ───────────────────────────────────────
        pred_len                            = cfg.get("pred_len", 96),
        epochs_forecasting                  = cfg.get("epochs_forecasting", 10),
        lr_forecasting                      = _get_forecast_lr(cfg, "lr_forecasting", 0.001),
        min_lr_forecasting                  = cfg.get("min_lr_forecasting", 1e-5),
        parms_for_training_forecasting      = cfg.get("parms_for_training_forecasting", []),
        parms_for_testing_forecasting       = cfg.get("parms_for_testing_forecasting", []),
        path_num                            = cfg.get("path_num", 0),

        # ── downstream: classification ────────────────────────────────────
        n_classes                   = cfg.get("n_classes", 6),
        epochs_classification       = cfg.get("epochs_classification", 50),
        lr_classification           = cfg.get("lr_classification", 0.001),
        min_lr_classification       = cfg.get("min_lr_classification", 1e-6),
        batch_size_classification   = cfg.get("batch_size_classification", 64),
        seq_len_classification      = cfg.get("seq_len_classification", 128),
        c_in_classification         = cfg.get("c_in_classification", 9),
    )
    return args


# ── DINO ──────────────────────────────────────────────────────────────────────

def run_dino(skip_train: bool = False,
             pretrain_dataset: str = None,
             forecast_dataset: str = None,
             classification_dataset=None,
             anomaly_dataset: str = None,
             pred_lens=None,
             checkpoints=None,
             pretrain_only: bool = False,
             classification_only: bool = False,
             encoder_layers: int = None,
             predictor_layers: int = None,
             lr: float = None,
             pretrain_source: str = None,
             checkpoint: str = None,
             num_patches: int = None,
             seed: int = None,
             linear_probe: bool = True):
    dino_dir  = Path(__file__).parent / "TSDiNO"
    shared_dir = Path(__file__).parent / "shared"
    _add_path(dino_dir)
    _add_path(shared_dir)

    import sys as _sys, importlib.util as _ilu

    # Load TSDiNO config.py directly by file path — avoids any sys.path collision
    _cfg_spec = _ilu.spec_from_file_location("_dino_config", dino_dir / "config.py")
    _cfg_mod  = _ilu.module_from_spec(_cfg_spec)
    _cfg_spec.loader.exec_module(_cfg_mod)
    dino_cfg  = dict(_cfg_mod.config)

    # Inject under the bare name so that main.py's `from config import config as cfg` resolves
    # to our freshly loaded version, not whatever 'config' may already be cached in sys.modules.
    _sys.modules["config"] = _cfg_mod

    # Load TSDiNO main.py directly by file path
    _sys.modules.pop("main", None)                       # evict any stale 'main'
    _main_spec = _ilu.spec_from_file_location("_dino_main", dino_dir / "main.py")
    dino_main  = _ilu.module_from_spec(_main_spec)
    _sys.modules["main"] = dino_main                     # register before exec (handles internal refs)
    _main_spec.loader.exec_module(dino_main)

    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]

    dino_cfg = dict(dino_cfg)
    if pretrain_source is not None:
        dino_cfg['pretrain_source'] = pretrain_source
    if encoder_layers is not None:
        dino_cfg['n_layers'] = encoder_layers
        _src_tag = '' if dino_cfg.get('pretrain_source', 'monash') == 'monash' else f"_{dino_cfg['pretrain_source'].replace('+', '_')}"
        dino_cfg['output_dir'] = dino_cfg.get('output_dir', './checkpoints').rstrip('/') + f'{_src_tag}_layers{encoder_layers}' + _SEED_TAG
    if num_patches is not None:
        dino_cfg['num_patches'] = num_patches
        _cw = num_patches * dino_cfg.get('patch_len', 16)
        _base = dino_cfg.get('output_dir', './checkpoints').rstrip('/')
        dino_cfg['output_dir'] = str(Path(_base).parent / 'classification' / (Path(_base).name + f'_cw{_cw}'))
    if lr is not None:
        dino_cfg['lr'] = lr
    if seed is not None:
        dino_cfg['seed'] = seed
    pretrain_source = _resolve_pretrain_source(dino_cfg)
    use_global_data = pretrain_source is not None

    # Resolve forecast dataset (always needed for downstream)
    if not (anomaly_dataset is not None and forecast_dataset is None):
        forecast_dataset = forecast_dataset or dino_cfg.get("forecast_dataset")
    dino_cfg["lr_forecasting"] = _get_forecast_lr(dino_cfg, "lr_forecasting")
    if pretrain_only and use_global_data:
        dino_cfg['saveckp_freq'] = 1  # save every epoch

    if use_global_data:
        # No pretrain CSV needed; c_in = 1 (univariate global data)
        dino_cfg["c_in"] = 1
        if not pretrain_only and classification_dataset is None and anomaly_dataset is None:
            if forecast_dataset is None:
                raise ValueError("forecast_dataset must be set when pretraining on global data")
        if not pretrain_only and forecast_dataset is not None:
            ds_fore = get_dataset_info(forecast_dataset)
            dino_cfg["c_in"] = ds_fore["c_in"]
        if pretrain_source in ('monash', 'monash+synthetic'):
            monash_dir = dino_cfg.get('monash_data_dir', '../Monash')
            if not os.path.isabs(monash_dir):
                dino_cfg['monash_data_dir'] = str((dino_dir / monash_dir).resolve())
        if pretrain_source in ('synthetic', 'monash+synthetic'):
            _synth_key = 'synthetic_mix_data_dir' if pretrain_source == 'monash+synthetic' else 'synthetic_data_dir'
            synth_dir = dino_cfg.get(_synth_key, dino_cfg.get('synthetic_data_dir', '../Monash'))
            if not os.path.isabs(synth_dir):
                synth_dir = str((dino_dir / synth_dir).resolve())
            dino_cfg['synthetic_data_dir'] = synth_dir
        _src_label = {
            'monash':           f"Monash ({dino_cfg.get('monash_data_dir', '')})",
            'synthetic':        f"Synthetic ({dino_cfg.get('synthetic_data_dir', '')})",
            'monash+synthetic': "Monash + Synthetic",
        }.get(pretrain_source, pretrain_source)
        print("\n" + "="*60)
        print(f"  MODEL: DINO  (TSDiNO)")
        if pretrain_only:
            print(f"  pretrain: {_src_label}  [pretrain only]")
        else:
            print(f"  pretrain: {_src_label}   forecast: {forecast_dataset}")
        print("="*60)
    else:
        pretrain_dataset = pretrain_dataset or dino_cfg.get("pretrain_dataset")
        forecast_dataset = forecast_dataset or pretrain_dataset
        if pretrain_dataset is None:
            raise ValueError("pretrain_dataset not set — specify via run() or config.py")
        ds_pre  = get_dataset_info(pretrain_dataset)
        ds_fore = get_dataset_info(forecast_dataset)
        dino_cfg["data_path"] = ds_pre["csv_path"]
        dino_cfg["c_in"]      = ds_pre["c_in"]
        print("\n" + "="*60)
        print(f"  MODEL: DINO  (TSDiNO)")
        print(f"  pretrain: {pretrain_dataset}   forecast: {forecast_dataset}")
        print("="*60)
    if not pretrain_only and forecast_dataset is not None:
        dino_cfg["data_path_forecast_training"]    = ds_fore["csv_path"]
        dino_cfg["data_path_forecast_test"]        = ds_fore["csv_path"]
        dino_cfg["parms_for_training_forecasting"] = ds_fore["columns"]
        dino_cfg["parms_for_testing_forecasting"]  = ds_fore["columns"]

    # Propagate overrides into the module-level cfg dict that train_TS_DINO reads directly.
    # cfg in main.py is imported as `from config import config as cfg` — it's a reference
    # to the same dict object, so updating it in-place propagates everywhere.
    dino_main.cfg.update(dino_cfg)

    args = _config_to_dino_args(dino_cfg)

    # Resolve data paths relative to dino_dir so they work from any CWD
    # (skip if already absolute — e.g. injected from dataset_registry)
    for attr in ('data_path', 'data_path_forecast_training',
                 'data_path_forecast_test', 'data_path_classification'):
        val = getattr(args, attr, '')
        if val and not os.path.isabs(val):
            setattr(args, attr, str(dino_dir / val))

    args.linear_probe = linear_probe
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── pretraining ──────────────────────────────────────────────────────────
    if not skip_train:
        print("\n[DINO] Starting pretraining …")
        dino_main.train_TS_DINO(args)
    else:
        print("[DINO] Skipping pretraining.")

    if pretrain_only:
        print("\n[DINO] Pretrain-only mode — skipping forecasting.")
        return

    # If a direct checkpoint path is given, use it (TEMP: for local testing, remove after)
    best_ckpt = os.path.abspath(checkpoint) if checkpoint else "best"
    best_mse  = None

    if not classification_only and forecast_dataset is not None:
        # ── forecasting downstream ────────────────────────────────────────────
        print("\n[DINO] Running forecasting downstream task …")
        ckpts = checkpoints if checkpoints is not None else ["best"]
        best_ckpt = None
        best_mse  = float('inf')

        for pred_len in pred_lens:
            args.pred_len = pred_len
            is_search = (pred_len == pred_lens[0])
            ckpts_to_run = ckpts if is_search else [best_ckpt if best_ckpt is not None else ckpts[-1]]

            print(f"\n[DINO] pred_len={pred_len}"
                  + ("" if is_search else f"  [best ckpt={ckpts_to_run[0]}]"))
            for ckpt in ckpts_to_run:
                args.path_num = ckpt
                _ckpt_label = 'random init' if ckpt == 0 else ('best' if ckpt == 'best' else f'epoch {ckpt}')
                print(f"  → checkpoint {ckpt} ({_ckpt_label})")
                mse = dino_main.test_run(args)
                if is_search and mse is not None and mse < best_mse:
                    best_mse  = mse
                    best_ckpt = ckpt

            if is_search:
                print(f"\n[DINO] Best checkpoint at pred_len={pred_lens[0]}: "
                      f"epoch {best_ckpt} (MSE={best_mse:.6f})")

    # ── classification downstream ─────────────────────────────────────────────
    cls_acc = None
    if classification_dataset is not None:
        from data_loaders.data_puller import ClassificationDataPuller, make_uea_dataloaders
        cls_dir = dino_cfg.get("classification_data_dir", "/home/shared/datasets/Classification_TS")
        cls_bs  = dino_cfg.get("batch_size_classification", 64)
        p_s        = args.patch_len
        _n_patches = 72                         # classification encoder always uses 72 patches
        _target_T  = _n_patches * p_s
        import torch.nn.functional as _F
        def _dino_patch_collate(batch, _ps=p_s, _tT=_target_T, _nP=_n_patches):
            xs, ys, orig_lens = zip(*batch)
            orig_lens = torch.stack(orig_lens)                                   # (B,)
            max_t = max(x.shape[0] for x in xs)
            xs = torch.stack([_F.pad(x, (0, 0, 0, max_t - x.shape[0])) for x in xs])
            B_, T_, C_ = xs.shape
            if T_ != _tT:
                idx = torch.linspace(0, T_ - 1, _tT).long()
                xs  = xs[:, idx, :]
                patch_idx    = idx[torch.arange(_nP) * _ps]
                padding_mask = patch_idx.unsqueeze(0) < orig_lens.unsqueeze(1)  # (B, P)
            else:
                patch_starts = torch.arange(_nP) * _ps
                padding_mask = patch_starts.unsqueeze(0) < orig_lens.unsqueeze(1)
            xs = xs.reshape(B_, _nP, _ps, C_)
            return xs, torch.stack(ys), padding_mask
        if list(Path(os.path.join(cls_dir, classification_dataset)).glob("*_TRAIN.ts")):
            _raw_train, _, _raw_test, n_classes = make_uea_dataloaders(
                cls_dir, classification_dataset, batch_size=cls_bs)
            cls_train = torch.utils.data.DataLoader(
                _raw_train.dataset, batch_size=cls_bs, shuffle=True,
                collate_fn=_dino_patch_collate)
            cls_val   = None
            cls_test  = torch.utils.data.DataLoader(
                _raw_test.dataset, batch_size=cls_bs, shuffle=False,
                collate_fn=_dino_patch_collate)
        else:
            _mk = lambda split: torch.utils.data.DataLoader(
                ClassificationDataPuller(cls_dir, classification_dataset, p_s, which=split),
                batch_size=cls_bs, shuffle=(split == "train"))
            cls_train = _mk("train"); cls_val = _mk("val"); cls_test = _mk("test")
            n_classes = cls_train.dataset.n_classes
        args.path_num = best_ckpt if best_ckpt is not None else 0
        cls_acc = dino_main.train_classification(
            args, cls_train, cls_val, cls_test, n_classes)
        print(f"\n{'='*60}")
        print(f"  [DINO] Classification on {classification_dataset}")
        print(f"  Test Accuracy: {cls_acc:.4f}")
        print(f"{'='*60}")

    # ── anomaly detection downstream ──────────────────────────────────────────
    anom_result = None
    if anomaly_dataset is not None:
        import importlib.util as _ilu
        _anom_spec = _ilu.spec_from_file_location("tsdino_anomaly", dino_dir / "Anomaly.py")
        _anom_mod  = _ilu.module_from_spec(_anom_spec)
        _anom_spec.loader.exec_module(_anom_mod)

        from data_loaders.data_puller import AnomalyDataPuller
        anom_dir = dino_cfg.get("anomaly_data_dir", "/home/shared/datasets/Anomaly_TS")
        anom_bs  = dino_cfg.get("batch_size_anomaly", 64)
        p_s      = args.patch_len
        anom_train = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="train"),
            batch_size=anom_bs, shuffle=False)
        anom_test  = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="test"),
            batch_size=anom_bs, shuffle=False)

        args.path_num = best_ckpt if best_ckpt is not None else 0
        anom_result = _anom_mod.anomaly_zeroshot(
            args, args.path_num, anom_train, anom_test,
            anomaly_ratio=_get_anomaly_ratio(anomaly_dataset, dino_cfg),
            linear_probe=linear_probe)

    return best_ckpt, best_mse, cls_acc, anom_result


def _resolve_jepa_path(p: str, jepa_dir: Path) -> str:
    """Return *p* as-is if absolute, otherwise resolve relative to *jepa_dir*."""
    if os.path.isabs(p):
        return p
    return str((jepa_dir / p.lstrip('./').lstrip('/')).resolve())


# ── JEPA (P2P only, no VQ / semantic tokens) ─────────────────────────────────

def run_jepa(skip_train: bool = False,
                    pretrain_dataset: str = None,
                    forecast_dataset: str = None,
                    classification_dataset=None,
                    anomaly_dataset: str = None,
                    pred_lens=None,
                    checkpoints=None,
                    pretrain_only: bool = False,
                    encoder_layers: int = None,
                    predictor_layers: int = None,
                    lr: float = None,
                    pretrain_source: str = None,
                    checkpoint: str = None,
                    num_patches: int = None,
                    random_encoder: bool = False,
                    linear_probe: bool = True):
    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]

    jepa_dir  = Path(__file__).parent / "JEPA"
    shared_dir = Path(__file__).parent / "shared"
    _add_path(jepa_dir)
    _add_path(shared_dir)   # for shared data_loaders

    import importlib.util, torch

    # Load config by file path to avoid sys.modules cache conflicts
    _spec = importlib.util.spec_from_file_location(
        "config_jepa", jepa_dir / "config_files" / "config_jepa.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    config = dict(_mod.config)
    if pretrain_source is not None:
        config['pretrain_source'] = pretrain_source
        _src_tag = f"_{pretrain_source.replace('+', '_')}" if pretrain_source != 'monash' else ''
        base_path = './output_model/JEPA'
        config['path_save'] = base_path + _src_tag + '/'
    if encoder_layers is not None:
        config['num_encoder_layers'] = encoder_layers
        _src_tag = f"_{config['pretrain_source'].replace('+', '_')}" if config.get('pretrain_source', 'monash') != 'monash' else ''
        config['path_save'] = f'./output_model/JEPA{_src_tag}_layers{encoder_layers}{_SEED_TAG}/'
    if num_patches is not None:
        config['ratio_patches'] = num_patches
        _cw = num_patches * config.get('patch_size', 16)
        _p = Path(config['path_save'].rstrip('/'))
        config['path_save'] = str(_p.parent / 'classification' / (_p.name + f'_cw{_cw}')) + '/'
    if predictor_layers is not None:
        config['predictor_num_layers'] = predictor_layers
    if lr is not None:
        config['lr'] = lr

    from data_loaders.data_puller import (DataPullerDJepa, ForcastingDataPullerDescrete,
                                          MonashDataPullerJEPA, SyntheticArrowDataPullerJEPA,
                                          PatchTSTForcastingAdapter)
    from JEPA.Jepa import JEPA

    # Single-dataset pipeline: align splits so test never leaks into training.
    if pretrain_dataset is not None and pretrain_dataset == forecast_dataset:
        config['pretrain_source'] = None   # force CSV-only mode
        config['val_prec']  = config.get('val_prec_forcasting',  0.1)
        config['test_prec'] = config.get('test_prec_forcasting', 0.1)

    pretrain_source = _resolve_pretrain_source(config)
    use_global_data = pretrain_source is not None

    _caller_wants_forecast = forecast_dataset is not None  # preserve explicit None before config default
    forecast_dataset = forecast_dataset or config.get("forecast_dataset")
    if not _caller_wants_forecast:
        forecast_dataset = None   # caller passed None → don't default to config value
    config["lr_forcasting"] = _get_forecast_lr(config, "lr_forcasting")
    if use_global_data:
        if not pretrain_only and classification_dataset is None and forecast_dataset is None and anomaly_dataset is None:
            raise ValueError("forecast_dataset must be set when pretraining on global data")
        if pretrain_source in ('monash', 'monash+synthetic'):
            monash_dir = config.get('monash_data_dir', '../Monash')
            if not os.path.isabs(monash_dir):
                config['monash_data_dir'] = str((jepa_dir / monash_dir).resolve())
        if pretrain_source in ('synthetic', 'monash+synthetic'):
            _synth_key = 'synthetic_mix_data_dir' if pretrain_source == 'monash+synthetic' else 'synthetic_data_dir'
            synth_dir = config.get(_synth_key, config.get('synthetic_data_dir', '../Monash'))
            if not os.path.isabs(synth_dir):
                synth_dir = str((jepa_dir / synth_dir).resolve())
            config['synthetic_data_dir'] = synth_dir
        _src_label = {
            'monash':           f"Monash ({config.get('monash_data_dir', '')})",
            'synthetic':        f"Synthetic ({config.get('synthetic_data_dir', '')})",
            'monash+synthetic': "Monash + Synthetic (mix)",
        }.get(pretrain_source, pretrain_source)
        print("\n" + "="*60)
        print(f"  MODEL: JEPA (P2P)")
        if pretrain_only:
            print(f"  pretrain: {_src_label}  [pretrain only]")
        elif forecast_dataset:
            ds_fore = get_dataset_info(forecast_dataset)
            print(f"  pretrain: {_src_label}   forecast: {forecast_dataset}")
        else:
            print(f"  pretrain: {_src_label}   [classify only]")
        print("="*60)
    else:
        pretrain_dataset = pretrain_dataset or config.get("pretrain_dataset")
        forecast_dataset = forecast_dataset or pretrain_dataset
        if pretrain_dataset is None:
            raise ValueError("pretrain_dataset not set — specify via run() or config_jepa.py")
        ds_pre  = get_dataset_info(pretrain_dataset)
        ds_fore = get_dataset_info(forecast_dataset)
        n_groups = len(ds_pre["jepa_groups"])
        config["path_data"]       = [_resolve_jepa_path(ds_pre["csv_path"], jepa_dir)] * n_groups
        config["timestampcols"]   = [ds_pre["timestamp_col"]] * n_groups
        config["input_variables"] = ds_pre["jepa_groups"]
        print("\n" + "="*60)
        print(f"  MODEL: JEPA (P2P)")
        print(f"  pretrain: {pretrain_dataset}   forecast: {forecast_dataset}")
        print("="*60)

    if not pretrain_only and forecast_dataset is not None:
        config["path_data_forcasting"]       = [_resolve_jepa_path(ds_fore["csv_path"], jepa_dir)]
        config["timestampcols_forcasting"]   = [ds_fore["timestamp_col"]]
        config["input_variables_forcasting"] = [ds_fore["columns"]]

    # ── data ─────────────────────────────────────────────────────────────────
    if skip_train and use_global_data:
        # Skip loading pretrain data entirely — only need forecasting loaders
        print("\n[JEPA] skip_train=True + global pretrain: skipping pretrain data load.")
        input_dim       = config["patch_size"]  # univariate: input_dim = patch_size
        num_patches     = config["ratio_patches"]
        train_loader = val_loader = test_loader = None
    else:
        print("\n[JEPA] Loading datasets …")
        if use_global_data:
            import torch.utils.data as _tud
            if pretrain_source in ('monash', 'monash+synthetic'):
                train_dataset = MonashDataPullerJEPA(config, which='train')
                val_dataset   = MonashDataPullerJEPA(config, which='val')
                test_dataset  = MonashDataPullerJEPA(config, which='test')
            if pretrain_source in ('synthetic', 'monash+synthetic'):
                syn_train = SyntheticArrowDataPullerJEPA(config, which='train')
                syn_val   = SyntheticArrowDataPullerJEPA(config, which='val')
                syn_test  = SyntheticArrowDataPullerJEPA(config, which='test')
                if pretrain_source == 'monash+synthetic':
                    train_dataset = _tud.ConcatDataset([train_dataset, syn_train])
                    val_dataset   = _tud.ConcatDataset([val_dataset,   syn_val])
                    test_dataset  = _tud.ConcatDataset([test_dataset,  syn_test])
                else:
                    train_dataset, val_dataset, test_dataset = syn_train, syn_val, syn_test
        else:
            train_dataset = DataPullerDJepa(
                data_paths          = config["path_data"],
                patch_size          = config["patch_size"],
                batch_size          = config["batch_size"],
                ratio_patches       = config["ratio_patches"],
                mask_ratio          = config["mask_ratio"],
                masking_type        = config["masking_type"],
                num_semantic_tokens = config.get("num_semantic_tokens", 0),
                input_variables     = config["input_variables"],
                timestamp_cols      = config["timestampcols"],
                type_data           = "train",
                val_prec            = config["val_prec"],
                test_prec           = config["test_prec"],
                stride              = config.get("stride", None),
                num_blocks          = config.get("num_blocks", 1),
            )
            val_dataset  = copy.copy(train_dataset); val_dataset.which  = "val"
            test_dataset = copy.copy(train_dataset); test_dataset.which = "test"

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=config["batch_size"], shuffle=True)
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=config["batch_size"], shuffle=False)
        input_dim       = len(train_loader.dataset[0][0][0])
        num_patches     = len(train_loader.dataset[0][0])

    # Use PatchTST-identical data: seq_len=336 (21 patches × 16)
    _PATCHTST_SEQ_LEN = 336
    _ctx_patches = _PATCHTST_SEQ_LEN // config["patch_size_forcasting"]   # = 21
    config["forecasting_context_patches"] = _ctx_patches
    if pretrain_only or forecast_dataset is None:
        train_loader_fc = val_loader_fc = test_loader_fc = None
    else:
        _csv = config["path_data_forcasting"][0]
        _p_s = config["patch_size_forcasting"]
        _pl0 = pred_lens[0] if pred_lens else 96
        _fc_bs = _get_forecast_bs(config, 256)
        _fc_nw = config.get("num_workers", 4)
        train_loader_fc = torch.utils.data.DataLoader(
            PatchTSTForcastingAdapter(_csv, 'train', _PATCHTST_SEQ_LEN, _pl0, _p_s),
            batch_size=_fc_bs, shuffle=True,  num_workers=_fc_nw)
        val_loader_fc = torch.utils.data.DataLoader(
            PatchTSTForcastingAdapter(_csv, 'val',   _PATCHTST_SEQ_LEN, _pl0, _p_s),
            batch_size=_fc_bs, shuffle=False, num_workers=_fc_nw)
        test_loader_fc = torch.utils.data.DataLoader(
            PatchTSTForcastingAdapter(_csv, 'test',  _PATCHTST_SEQ_LEN, _pl0, _p_s),
            batch_size=_fc_bs, shuffle=False, num_workers=_fc_nw)

    # ── model ─────────────────────────────────────────────────────────────────
    model = JEPA(
        config          = config,
        input_dim       = input_dim,
        num_patches     = num_patches,
        steps_per_epoch = 1,
        train_loader    = train_loader,
        val_loader      = val_loader,
        test_loader     = test_loader,
        forcasting_train = train_loader_fc,
        forcasting_val   = val_loader_fc,
        forcasting_test  = test_loader_fc,
    )

    # ── pretraining ──────���────────────────────────────────────────────────────
    if not skip_train:
        print("\n[JEPA] Starting pretraining …")
        model.train_and_evaluate()
    else:
        print("[JEPA] Skipping pretraining.")

    if pretrain_only:
        print("\n[JEPA] Pretrain-only mode — skipping forecasting.")
        return

    # ── forecasting downstream ────────────────────────────────────────────────
    best_ckpt = None
    best_mse  = float('inf')
    if forecast_dataset is None:
        print("\n[JEPA] No forecast_dataset — skipping forecasting.")
    else:
        ckpts = checkpoints if checkpoints is not None else ["best"]
        p_s = config["patch_size_forcasting"]
        _csv = config["path_data_forcasting"][0]
        _fc_bs2 = _get_forecast_bs(config, 256)
        _fc_nw2 = config.get("num_workers", 4)

        for pred_len in pred_lens:
            h_t = pred_len // p_s
            model.forcast_train = torch.utils.data.DataLoader(
                PatchTSTForcastingAdapter(_csv, 'train', _PATCHTST_SEQ_LEN, pred_len, p_s),
                batch_size=_fc_bs2, shuffle=True,  num_workers=_fc_nw2)
            model.forcast_val = torch.utils.data.DataLoader(
                PatchTSTForcastingAdapter(_csv, 'val',   _PATCHTST_SEQ_LEN, pred_len, p_s),
                batch_size=_fc_bs2, shuffle=False, num_workers=_fc_nw2)
            model.forcast_test = torch.utils.data.DataLoader(
                PatchTSTForcastingAdapter(_csv, 'test',  _PATCHTST_SEQ_LEN, pred_len, p_s),
                batch_size=_fc_bs2, shuffle=False, num_workers=_fc_nw2)
            model.config["horizon_t"] = h_t

            is_search = (pred_len == pred_lens[0])
            ckpts_to_run = ckpts if is_search else [best_ckpt if best_ckpt is not None else ckpts[-1]]

            print(f"\n[JEPA] pred_len={pred_len} (horizon_t={h_t})"
                  + ("" if is_search else f"  [best ckpt={ckpts_to_run[0]}]"))
            for epoch in ckpts_to_run:
                print(f"  → checkpoint epoch {epoch}")
                ckpt_tag = "" if epoch == "best" else f"_epoch{epoch}"
                mse = model.forcasting_zeroshot(ckpt_tag, linear_probe=linear_probe)
                if is_search and mse is not None and mse < best_mse:
                    best_mse  = mse
                    best_ckpt = epoch

            if is_search:
                print(f"\n[JEPA] Best checkpoint at pred_len={pred_lens[0]}: "
                      f"epoch {best_ckpt} (mix MSE={best_mse:.4f})")

    # ── classification downstream ─────────────────────────────────────────────
    cls_acc = None
    if classification_dataset is not None:
        from data_loaders.data_puller import ClassificationDataPuller, make_uea_dataloaders
        cls_dir  = config.get("classification_data_dir", "/home/shared/datasets/Classification_TS")
        cls_bs   = config.get("batch_size", 64)
        p_s      = config["patch_size_forcasting"]
        if list(Path(os.path.join(cls_dir, classification_dataset)).glob("*_TRAIN.ts")):
            _uea_tr, _uea_va, _uea_te, n_classes = make_uea_dataloaders(
                cls_dir, classification_dataset, batch_size=cls_bs)
            # JEPA expects (B, P, PL, C) with P = config["ratio_patches"] (pretraining num_patches).
            # UEADataset returns (B, T, C) → subsample to ratio_patches * patch_size, then patch.
            _n_patches = 72                               # classification encoder always uses 72 patches
            _target_T  = _n_patches * p_s
            def _patch_collate(batch, _ps=p_s, _tT=_target_T, _nP=_n_patches):
                xs, ys, orig_lens = zip(*batch)
                orig_lens = torch.stack(orig_lens)                               # (B,)
                max_t = max(x.shape[0] for x in xs)
                xs = torch.stack([torch.nn.functional.pad(x, (0, 0, 0, max_t - x.shape[0])) for x in xs])  # (B, T, C)
                T  = xs.shape[1]
                if T != _tT:                              # uniform resample
                    idx = torch.linspace(0, T - 1, _tT).long()
                    xs  = xs[:, idx, :]
                    patch_idx    = idx[torch.arange(_nP) * _ps]
                    padding_mask = patch_idx.unsqueeze(0) < orig_lens.unsqueeze(1)  # (B, P)
                else:
                    patch_starts = torch.arange(_nP) * _ps
                    padding_mask = patch_starts.unsqueeze(0) < orig_lens.unsqueeze(1)
                xs = xs.reshape(len(xs), _nP, _ps, xs.shape[-1])   # (B, P, ps, C)
                return xs, torch.stack(ys), padding_mask
            _wrap = lambda ds, shuf: torch.utils.data.DataLoader(
                ds, batch_size=cls_bs, shuffle=shuf, collate_fn=_patch_collate)
            cls_train = _wrap(_uea_tr.dataset, True)
            cls_val   = None
            cls_test  = _wrap(_uea_te.dataset, False)
        else:
            _mk = lambda split: torch.utils.data.DataLoader(
                ClassificationDataPuller(cls_dir, classification_dataset, p_s, which=split),
                batch_size=cls_bs, shuffle=(split == "train"))
            cls_train = _mk("train"); cls_val = _mk("val"); cls_test = _mk("test")
            n_classes = cls_train.dataset.n_classes
        ckpt_tag  = f"_epoch{best_ckpt}" if best_ckpt is not None else ""
        _ckpt_override = os.path.abspath(checkpoint) if checkpoint else None
        cls_acc   = model.classification_zeroshot(ckpt_tag, cls_train, cls_val, cls_test, n_classes,
                                                  checkpoint_path_override=_ckpt_override,
                                                  random_encoder=random_encoder,
                                                  linear_probe=linear_probe)
        print(f"\n{'='*60}")
        print(f"  [JEPA] Classification on {classification_dataset}")
        print(f"  Test Accuracy: {cls_acc:.4f}")
        print(f"{'='*60}")

    # ── anomaly detection downstream ──────────────────────────────────────────
    anom_result = None
    if anomaly_dataset is not None:
        from data_loaders.data_puller import AnomalyDataPuller
        anom_dir = config.get("anomaly_data_dir", "/home/shared/datasets/Anomaly_TS")
        anom_bs  = config.get("batch_size", 64)
        p_s      = config["patch_size_forcasting"]
        anom_train = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="train"),
            batch_size=anom_bs, shuffle=False)
        anom_test  = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="test"),
            batch_size=anom_bs, shuffle=False)
        ckpt_tag   = f"_epoch{best_ckpt}" if best_ckpt is not None else ""
        anom_result = model.anomaly_detection(ckpt_tag, anom_train, anom_test,
                                              anomaly_ratio=_get_anomaly_ratio(anomaly_dataset, config),
                                              linear_probe=linear_probe)

    return best_ckpt, best_mse, cls_acc, anom_result


# ── PatchTST ──────────────────────────────────────────────────────────────────

def run_patchtst(skip_train: bool = False, pretrain_dataset: str = None, forecast_dataset: str = None,
                 classification_dataset=None, anomaly_dataset: str = None,
                 pretrain_only: bool = False, classification_only: bool = False, pred_lens=None,
                 checkpoints=None, random_encoder: bool = False, encoder_layers: int = None,
                 predictor_layers: int = None, lr: float = None, pretrain_source: str = None,
                 num_patches: int = None, seed: int = None, linear_probe: bool = True):
    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]
    patchtst_dir = Path(__file__).parent / "PatchTST_self_supervised"
    shared_dir    = Path(__file__).parent / "shared"
    _add_path(shared_dir)

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "config_patchtst", patchtst_dir / "config_patchtst.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    cfg = dict(_mod.config)
    if pretrain_source is not None:
        cfg['pretrain_source'] = pretrain_source
    if encoder_layers is not None:
        cfg['n_layers'] = encoder_layers
        cfg['pretrained_model_id'] = encoder_layers  # unique checkpoint per layer config
    if num_patches is not None:
        cfg['context_points'] = num_patches * cfg.get('patch_len', 16)

    pretrain_source = _resolve_pretrain_source(cfg)
    _pretrain_dset  = pretrain_dataset or cfg.get("pretrain_dataset", "ettm1")
    _forecast_dset  = None if (pretrain_only or classification_only or (anomaly_dataset is not None and forecast_dataset is None)) else (forecast_dataset or cfg.get("forecast_dataset") or _pretrain_dset)

    # pretrain_source overrides _pretrain_dset for global data
    if pretrain_source in ('monash', 'synthetic', 'monash+synthetic'):
        _pretrain_dset = pretrain_source

    monash_dir = synth_dir = None
    if _pretrain_dset in ('monash', 'monash+synthetic'):
        monash_dir = cfg.get("monash_data_dir", "../Monash")
        if not os.path.isabs(monash_dir):
            monash_dir = str((patchtst_dir / monash_dir).resolve())
    if _pretrain_dset in ('synthetic', 'monash+synthetic'):
        _synth_key = 'synthetic_mix_data_dir' if _pretrain_dset == 'monash+synthetic' else 'synthetic_data_dir'
        synth_dir = cfg.get(_synth_key, cfg.get("synthetic_data_dir", "../synthetic"))
        if not os.path.isabs(synth_dir):
            synth_dir = str((patchtst_dir / synth_dir).resolve())

    _src_label = {
        'monash':           f"Monash ({monash_dir})",
        'synthetic':        f"Synthetic ({synth_dir})",
        'monash+synthetic': "Monash + Synthetic",
    }.get(_pretrain_dset, _pretrain_dset)
    print("\n" + "="*60)
    print("  MODEL: PatchTST (self-supervised)")
    if pretrain_only:
        print(f"  pretrain: {_src_label}  [pretrain only]")
    else:
        print(f"  pretrain: {_src_label}   forecast: {_forecast_dset}")
    print("="*60)

    patchtst_dir = patchtst_dir.resolve()
    _add_path(patchtst_dir)

    # Build common pretrain args from config
    pretrain_cmd = [
        sys.executable, "patchtst_pretrain.py",
        "--dset_pretrain",       _pretrain_dset,
        "--context_points",      str(cfg.get("context_points",      512)),
        "--patch_len",           str(cfg.get("patch_len",           12)),
        "--stride",              str(cfg.get("stride",              12)),
        "--n_layers",            str(cfg.get("n_layers",            3)),
        "--n_heads",             str(cfg.get("n_heads",             16)),
        "--d_model",             str(cfg.get("d_model",             128)),
        "--d_ff",                str(cfg.get("d_ff",                512)),
        "--dropout",             str(cfg.get("dropout",             0.2)),
        "--head_dropout",        str(cfg.get("head_dropout",        0.2)),
        "--mask_ratio",          str(cfg.get("mask_ratio",          0.4)),
        "--n_epochs_pretrain",   str(cfg.get("n_epochs_pretrain",   10)),
        "--batch_size",          str(cfg.get("batch_size",          64)),
        "--revin",               str(int(cfg.get("revin",           True))),
        "--pretrained_model_id", str(cfg.get("pretrained_model_id", 1)),
        "--seed",                str(seed if seed is not None else GLOBAL_SEED),
    ]
    if monash_dir is not None:
        pretrain_cmd += ["--monash_data_dir",    monash_dir,
                         "--monash_min_len",      str(cfg.get("monash_min_len", 512))]
    if synth_dir is not None:
        pretrain_cmd += ["--synthetic_data_dir", synth_dir]
    if lr is not None:
        pretrain_cmd += ["--lr", str(lr)]
    _ptst_save_dir = None
    if num_patches is not None:
        _cw = num_patches * cfg.get('patch_len', 16)
        _ptst_save_dir = str(
            patchtst_dir / "saved_models" / "classification" /
            _pretrain_dset / "masked_patchtst" / cfg.get("model_type", "based_model") /
            f"layers{cfg.get('n_layers', 3)}_cw{_cw}{_SEED_TAG}"
        )
    elif _SEED_TAG:
        _ptst_save_dir = str(
            patchtst_dir / "saved_models" / _pretrain_dset /
            "masked_patchtst" / cfg.get("model_type", "based_model") /
            f"layers{cfg.get('n_layers', 3)}{_SEED_TAG}"
        )
    if _ptst_save_dir is not None:
        pretrain_cmd += ["--save_dir", _ptst_save_dir]

    # ── pretraining ───────────────────────────────────────────────────────────
    if not skip_train:
        print(f"\n[PatchTST] Starting pretraining on {_pretrain_dset} …")
        result = subprocess.run(pretrain_cmd, cwd=patchtst_dir, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("[PatchTST] Pretraining exited with errors.")
            print(result.stderr)
            return
    else:
        print("[PatchTST] Skipping pretraining.")

    # ── resolve checkpoint path (needed for both forecast and classify) ────────
    n_ep    = cfg.get("n_epochs_pretrain", 10)
    ctx     = cfg.get("context_points", 512)
    p_len   = cfg.get("patch_len", 12)
    stride  = cfg.get("stride", 12)
    m_ratio = cfg.get("mask_ratio", 0.4)
    m_id    = cfg.get("pretrained_model_id", 1)
    model_fname_base = (f"patchtst_pretrained_cw{ctx}_patch{p_len}_stride{stride}"
                        f"_epochs-pretrain{n_ep}_mask{m_ratio}_model{m_id}")
    _ckpt_epoch = checkpoints[0] if (checkpoints and checkpoints[0] is not None) else None
    model_fname = f"{model_fname_base}_{_ckpt_epoch}.pth" if _ckpt_epoch is not None else f"{model_fname_base}.pth"
    if _ptst_save_dir is not None:
        pretrained_model_path = os.path.join(_ptst_save_dir, model_fname)
    elif num_patches is not None:
        _cw = num_patches * cfg.get('patch_len', 16)
        pretrained_model_path = os.path.join(
            patchtst_dir, "saved_models", "classification", _pretrain_dset,
            "masked_patchtst", cfg.get("model_type", "based_model"),
            f"layers{cfg['n_layers']}_cw{_cw}", model_fname
        )
    else:
        pretrained_model_path = os.path.join(
            patchtst_dir, "saved_models", _pretrain_dset,
            "masked_patchtst", cfg.get("model_type", "based_model"),
            f"layers{cfg['n_layers']}", model_fname
        )
    if random_encoder:
        pretrained_model_path = None

    if pretrain_only:
        print("\n[PatchTST] Pretrain-only mode — skipping forecasting.")
        return

    # ── forecasting downstream ────────────────────────────────────────────────
    mse_val, mae_val = None, None
    if _forecast_dset is None:
        print("\n[PatchTST] No forecast_dataset — skipping forecasting.")
    else:
        import re as _re
        print(f"\n[PatchTST] Running forecasting fine-tuning on {_forecast_dset} …")
        for _pl in pred_lens:
            print(f"\n[PatchTST] pred_len={_pl}")
            result = subprocess.run(
                [sys.executable, "patchtst_finetune.py",
                 "--dset_finetune",      _forecast_dset,
                 "--is_finetune",        str(int(not linear_probe)),
                 "--is_linear_probe",    str(int(linear_probe)),
                 "--context_points",  str(cfg.get("context_points", 512)),
                 "--patch_len",       str(cfg.get("patch_len", 16)),
                 "--stride",          str(cfg.get("stride", 16)),
                 "--n_layers",        str(cfg.get("n_layers", 3)),
                 "--n_heads",         str(cfg.get("n_heads", 16)),
                 "--d_model",         str(cfg.get("d_model", 128)),
                 "--d_ff",            str(cfg.get("d_ff", 512)),
                 "--dropout",         str(cfg.get("dropout", 0.2)),
                 "--head_dropout",    str(cfg.get("head_dropout", 0.2)),
                 "--target_points",   str(_pl),
                 "--pretrained_model", str(pretrained_model_path) if pretrained_model_path is not None else "",
                 "--random_encoder",   str(int(random_encoder)),
                 "--batch_size",       str(_get_forecast_bs(cfg, 256)),
                 "--num_workers",      str(cfg.get("num_workers", 4)),
                 "--lr",               str(cfg.get("finetune_lr", 1e-4)),
                 "--seed",             str(seed if seed is not None else GLOBAL_SEED)],
                cwd=patchtst_dir, capture_output=True, text=True,
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"[PatchTST] pred_len={_pl} exited with errors.")
                print(result.stderr)
                continue

            _score_match = _re.search(r"score:\s*\[array\(([\d.]+)[^)]*\)[^,]*,\s*array\(([\d.]+)", result.stdout)
            if _score_match:
                _mse = float(_score_match.group(1))
                _mae = float(_score_match.group(2))
                print(f"[PatchTST] pred_len={_pl}  MSE={_mse:.4f}  MAE={_mae:.4f}")
                if mse_val is None or _mse < mse_val:
                    mse_val, mae_val = _mse, _mae

    # ── classification downstream ─────────────────────────────────────────────
    cls_acc = None
    if classification_dataset is not None:
        sys.path.insert(0, str(patchtst_dir))
        from patchtst_classification import classification_zeroshot as ptst_classify
        from data_loaders.data_puller import ClassificationDataPuller, make_uea_dataloaders
        cls_dir    = cfg.get("classification_data_dir", "/home/shared/datasets/Classification_TS")
        cls_bs     = _get_cls_bs(cfg, "batch_size", 64)
        p_s        = cfg.get("patch_len", 16)
        _n_patches = 72                               # classification encoder always uses 72 patches
        _target_T  = _n_patches * p_s
        import torch.nn.functional as _F
        def _ptst_patch_collate(batch, _ps=p_s, _tT=_target_T, _nP=_n_patches):
            xs, ys, orig_lens = zip(*batch)
            orig_lens = torch.stack(orig_lens)                                   # (B,)
            max_t = max(x.shape[0] for x in xs)
            xs = torch.stack([_F.pad(x, (0, 0, 0, max_t - x.shape[0])) for x in xs])
            B_, T_, C_ = xs.shape
            if T_ != _tT:
                idx = torch.linspace(0, T_ - 1, _tT).long()
                xs  = xs[:, idx, :]
                patch_idx    = idx[torch.arange(_nP) * _ps]
                padding_mask = patch_idx.unsqueeze(0) < orig_lens.unsqueeze(1)  # (B, P)
            else:
                patch_starts = torch.arange(_nP) * _ps
                padding_mask = patch_starts.unsqueeze(0) < orig_lens.unsqueeze(1)
            xs = xs.reshape(B_, _nP, _ps, C_)
            return xs, torch.stack(ys), padding_mask
        if list(Path(os.path.join(cls_dir, classification_dataset)).glob("*_TRAIN.ts")):
            _raw_tr, _, _raw_te, n_classes = make_uea_dataloaders(
                cls_dir, classification_dataset, batch_size=cls_bs)
            cls_train = torch.utils.data.DataLoader(
                _raw_tr.dataset, batch_size=cls_bs, shuffle=True,
                collate_fn=_ptst_patch_collate)
            cls_val   = None
            cls_test  = torch.utils.data.DataLoader(
                _raw_te.dataset, batch_size=cls_bs, shuffle=False,
                collate_fn=_ptst_patch_collate)
        else:
            _mk = lambda split: torch.utils.data.DataLoader(
                ClassificationDataPuller(cls_dir, classification_dataset, p_s, which=split),
                batch_size=cls_bs, shuffle=(split == "train"))
            cls_train = _mk("train"); cls_val = _mk("val"); cls_test = _mk("test")
            n_classes = cls_train.dataset.n_classes
        cls_acc = ptst_classify(cfg, pretrained_model_path, cls_train, cls_val, cls_test, n_classes,
                                linear_probe=linear_probe)
        print(f"\n{'='*60}")
        print(f"  [PatchTST] Classification on {classification_dataset}")
        print(f"  Test Accuracy: {cls_acc:.4f}")
        print(f"{'='*60}")

    # ── anomaly detection downstream ──────────────────────────────────────────
    anom_result = None
    if anomaly_dataset is not None:
        from patchtst_anomaly import anomaly_zeroshot as ptst_anomaly
        from data_loaders.data_puller import AnomalyDataPuller
        anom_dir = cfg.get("anomaly_data_dir", "/home/shared/datasets/Anomaly_TS")
        anom_bs  = cfg.get("batch_size", 64)
        p_s      = cfg.get("patch_len", 16)
        anom_train = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="train"),
            batch_size=anom_bs, shuffle=False)
        anom_test  = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="test"),
            batch_size=anom_bs, shuffle=False)
        anom_result = ptst_anomaly(cfg, pretrained_model_path, anom_train, anom_test,
                                   anomaly_ratio=_get_anomaly_ratio(anomaly_dataset, cfg),
                                   linear_probe=linear_probe)

    return mse_val, mae_val, cls_acc, anom_result


# ── NTP (Next-Token-Patch Prediction on PatchTST) ────────────────────────────

def run_ntp(skip_train: bool = False, pretrain_dataset: str = None, forecast_dataset: str = None,
            classification_dataset=None, anomaly_dataset: str = None,
            pretrain_only: bool = False, classification_only: bool = False,
            pred_lens=None,
            checkpoints=None, encoder_layers: int = None, predictor_layers: int = None,
            lr: float = None, pretrain_source: str = None, num_patches: int = None,
            linear_probe: bool = True):
    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]
    ntp_dir = Path(__file__).parent / "NTP"
    _add_path(ntp_dir)

    import importlib.util
    _spec = importlib.util.spec_from_file_location("config_ntp", ntp_dir / "config_ntp.py")
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    cfg = dict(_mod.config)
    if pretrain_source is not None:
        cfg['pretrain_source'] = pretrain_source
    if encoder_layers is not None:
        cfg['n_layers'] = encoder_layers
        cfg['pretrained_model_id'] = encoder_layers  # unique checkpoint per layer config
    if num_patches is not None:
        cfg['context_patches'] = num_patches
        # NPT computes context_patches = ratio_patches - horizon_t, so add horizon_t here
        # to ensure the encoder W_pos actually gets num_patches slots (not num_patches - 6)
        cfg['ratio_patches']   = num_patches + cfg.get('horizon_t', 6)
    if lr is not None:
        cfg['lr'] = lr

    _pretrain_source = _resolve_pretrain_source(cfg)
    _pretrain_dset   = pretrain_dataset or cfg.get("pretrain_dataset", "monash")
    if _pretrain_source in ('monash', 'synthetic', 'monash+synthetic'):
        _pretrain_dset = _pretrain_source
    _forecast_dset = None if (pretrain_only or classification_only or (anomaly_dataset is not None and forecast_dataset is None)) else (forecast_dataset or cfg.get("forecast_dataset") or _pretrain_dset)
    cfg["pretrain_dataset"] = _pretrain_dset
    cfg["forecast_dataset"] = _forecast_dset

    if _pretrain_dset in ('monash', 'monash+synthetic'):
        monash_dir = cfg.get("monash_data_dir", "../Monash")
        if not os.path.isabs(monash_dir):
            cfg["monash_data_dir"] = str((ntp_dir / monash_dir).resolve())
    if _pretrain_dset in ('synthetic', 'monash+synthetic'):
        _synth_key = 'synthetic_mix_data_dir' if _pretrain_dset == 'monash+synthetic' else 'synthetic_data_dir'
        synth_dir = cfg.get(_synth_key, cfg.get("synthetic_data_dir", "../synthetic"))
        if not os.path.isabs(synth_dir):
            synth_dir = str((ntp_dir / synth_dir).resolve())
        cfg["synthetic_data_dir"] = synth_dir

    _src_label = {
        'monash':           f"Monash ({cfg.get('monash_data_dir', '')})",
        'synthetic':        f"Synthetic ({cfg.get('synthetic_data_dir', '')})",
        'monash+synthetic': "Monash + Synthetic",
    }.get(_pretrain_dset, _pretrain_dset)
    print("\n" + "="*60)
    print("  MODEL: NPT (Next-Token-Patch Prediction)")
    if pretrain_only:
        print(f"  pretrain: {_src_label}  [pretrain only]")
    else:
        print(f"  pretrain: {_src_label}   forecast: {_forecast_dset}")
    print("="*60)

    from ntp_pretrain import pretrain_ntp, _model_fname
    from ntp_forecasting import zeroshot_forecasting

    # Resolve checkpoint path (used whether we train or skip)
    _npt_save_dir_override = None
    if num_patches is not None:
        _cw = num_patches * cfg.get('patch_size', 16)
        _npt_save_dir_override = str(
            ntp_dir / "saved_models" / "classification" / _pretrain_dset / "ntp" / f"layers{cfg['n_layers']}_cw{_cw}{_SEED_TAG}"
        )
    elif _SEED_TAG:
        _npt_save_dir_override = str(
            ntp_dir / "saved_models" / _pretrain_dset / "ntp" / f"layers{cfg['n_layers']}{_SEED_TAG}"
        )
    _save_dir = Path(_npt_save_dir_override) if _npt_save_dir_override else \
                ntp_dir / "saved_models" / _pretrain_dset / "ntp" / f"layers{cfg['n_layers']}"
    _base_name = _model_fname(cfg, _pretrain_dset)
    _ckpt_epoch = checkpoints[0] if (checkpoints and checkpoints[0] is not None) else None
    if _ckpt_epoch is not None:
        _ckpt_path = str(_save_dir / f"{_base_name}_epoch{_ckpt_epoch}.pt")
    else:
        _ckpt_path = str(_save_dir / f"{_base_name}.pt")

    if not skip_train:
        print(f"\n[NPT] Starting NTP pretraining on {_pretrain_dset} …")
        _ckpt_path = pretrain_ntp(cfg, save_dir_override=_npt_save_dir_override)   # returns best-model path
    else:
        print("[NPT] Skipping pretraining.")

    mse_trained = None
    mae_trained = None
    if _forecast_dset:
        print(f"\n[NPT] Running zero-shot forecasting on {_forecast_dset} …")
        for _pl in pred_lens:
            cfg["horizon_t"] = _pl // cfg["patch_size"]
            _mse, _mae = zeroshot_forecasting(cfg, _ckpt_path, linear_probe=linear_probe)
            if _mse is not None:
                print(f"\n{'='*60}")
                print(f"  Results on {_forecast_dset}  pred_len={_pl}")
                print(f"  {'':20s}  {'MSE':>8}  {'MAE':>8}")
                print(f"  {'NPT (pretrained)':20s}  {_mse:8.4f}  {_mae:8.4f}")
                print(f"{'='*60}")
                if mse_trained is None or _mse < mse_trained:
                    mse_trained, mae_trained = _mse, _mae
    else:
        print("[NPT] No forecast_dataset set — skipping forecasting.")

    # ── classification downstream ─────────────────────────────────────────────
    cls_acc = None
    if classification_dataset is not None:
        from ntp_classification import classification_zeroshot as npt_classify
        from data_loaders.data_puller import ClassificationDataPuller, make_uea_dataloaders
        cls_dir    = cfg.get("classification_data_dir", "/home/shared/datasets/Classification_TS")
        cls_bs     = cfg.get("batch_size", 64)
        p_s        = cfg["patch_size"]
        _n_patches = 72                               # classification encoder always uses 72 patches
        _target_T  = _n_patches * p_s
        import torch.nn.functional as _F
        def _npt_patch_collate(batch, _ps=p_s, _tT=_target_T, _nP=_n_patches):
            xs, ys, orig_lens = zip(*batch)
            orig_lens = torch.stack(orig_lens)                                   # (B,)
            max_t = max(x.shape[0] for x in xs)
            xs = torch.stack([_F.pad(x, (0, 0, 0, max_t - x.shape[0])) for x in xs])
            B_, T_, C_ = xs.shape
            if T_ != _tT:
                idx = torch.linspace(0, T_ - 1, _tT).long()
                xs  = xs[:, idx, :]
                patch_idx    = idx[torch.arange(_nP) * _ps]
                padding_mask = patch_idx.unsqueeze(0) < orig_lens.unsqueeze(1)  # (B, P)
            else:
                patch_starts = torch.arange(_nP) * _ps
                padding_mask = patch_starts.unsqueeze(0) < orig_lens.unsqueeze(1)
            xs = xs.reshape(B_, _nP, _ps, C_)
            return xs, torch.stack(ys), padding_mask
        if list(Path(os.path.join(cls_dir, classification_dataset)).glob("*_TRAIN.ts")):
            _raw_tr, _, _raw_te, n_classes = make_uea_dataloaders(
                cls_dir, classification_dataset, batch_size=cls_bs)
            cls_train = torch.utils.data.DataLoader(
                _raw_tr.dataset, batch_size=cls_bs, shuffle=True,
                collate_fn=_npt_patch_collate)
            cls_val   = None
            cls_test  = torch.utils.data.DataLoader(
                _raw_te.dataset, batch_size=cls_bs, shuffle=False,
                collate_fn=_npt_patch_collate)
        else:
            _mk = lambda split: torch.utils.data.DataLoader(
                ClassificationDataPuller(cls_dir, classification_dataset, p_s, which=split),
                batch_size=cls_bs, shuffle=(split == "train"))
            cls_train = _mk("train"); cls_val = _mk("val"); cls_test = _mk("test")
            n_classes = cls_train.dataset.n_classes
        cls_acc = npt_classify(cfg, _ckpt_path, cls_train, cls_val, cls_test, n_classes,
                               linear_probe=linear_probe)
        print(f"\n{'='*60}")
        print(f"  [NPT] Classification on {classification_dataset}")
        print(f"  Test Accuracy: {cls_acc:.4f}")
        print(f"{'='*60}")

    # ── anomaly detection downstream ──────────────────────────────────────────
    anom_result = None
    if anomaly_dataset is not None:
        from ntp_anomaly import anomaly_zeroshot as npt_anomaly
        from data_loaders.data_puller import AnomalyDataPuller
        anom_dir = cfg.get("anomaly_data_dir", "/home/shared/datasets/Anomaly_TS")
        anom_bs  = cfg.get("batch_size", 64)
        p_s      = cfg["patch_size"]
        anom_train = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="train"),
            batch_size=anom_bs, shuffle=False)
        anom_test  = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="test"),
            batch_size=anom_bs, shuffle=False)
        anom_result = npt_anomaly(cfg, _ckpt_path, anom_train, anom_test,
                                  anomaly_ratio=_get_anomaly_ratio(anomaly_dataset, cfg),
                                  linear_probe=linear_probe)

    return mse_trained, cls_acc, anom_result


# ── LE-JEPA ───────────────────────────────────────────────────────────────────

def run_lejepa(skip_train: bool = False,
               pretrain_dataset: str = None,
               forecast_dataset: str = None,
               classification_dataset=None,
               anomaly_dataset: str = None,
               pretrain_only: bool = False,
               classification_only: bool = False,
               pred_lens=None,
               checkpoints=None,
               encoder_layers: int = None,
               predictor_layers: int = None,
               lr: float = None,
               pretrain_source: str = None,
               num_patches: int = None,
               linear_probe: bool = True):
    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]

    lejepa_dir = Path(__file__).parent / "LE-JEPA"
    shared_dir  = Path(__file__).parent / "shared"
    _add_path(lejepa_dir)
    _add_path(shared_dir)   # shared data_loaders

    import importlib.util, torch

    _spec = importlib.util.spec_from_file_location(
        "config_lejepa", lejepa_dir / "config_lejepa.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    config = dict(_mod.config)
    if pretrain_source is not None:
        config['pretrain_source'] = pretrain_source
        _src_tag = f"_{pretrain_source.replace('+', '_')}" if pretrain_source != 'monash' else ''
        config['path_save'] = f'./output_model/LE-JEPA{_src_tag}/'
    if encoder_layers is not None:
        config['num_encoder_layers'] = encoder_layers
        _src_tag = f"_{config['pretrain_source'].replace('+', '_')}" if config.get('pretrain_source', 'monash') != 'monash' else ''
        config['path_save'] = f'./output_model/LE-JEPA{_src_tag}_layers{encoder_layers}{_SEED_TAG}/'
    if num_patches is not None:
        config['ratio_patches'] = num_patches
        _cw = num_patches * config.get('patch_size', 16)
        _p = Path(config['path_save'].rstrip('/'))
        config['path_save'] = str(_p.parent / 'classification' / (_p.name + f'_cw{_cw}')) + '/'
    if lr is not None:
        config['lr_sgd'] = lr

    from data_loaders.data_puller import (DataPullerDJepa, MonashDataPullerJEPA,
                                          SyntheticArrowDataPullerJEPA, PatchTSTForcastingAdapter)
    # Re-pin lejepa_dir to sys.path[0] — data_puller imports may have pushed JEPA/JEPA ahead of it,
    # causing `from Classification import` in LeJepa.py to grab the wrong Classification.py.
    _lj = str(lejepa_dir)
    if _lj in sys.path:
        sys.path.remove(_lj)
    sys.path.insert(0, _lj)
    from LeJepa import LeJEPA

    if pretrain_only or classification_only or (anomaly_dataset is not None and forecast_dataset is None):
        forecast_dataset = None
    elif forecast_dataset is None and not skip_train:
        # Fresh pretrain run — fall back to config default
        forecast_dataset = config.get('forecast_dataset')
    # else: forecast_dataset was explicitly passed — keep it
    if forecast_dataset is None:
        config.pop('path_data_forcasting', None)
        config.pop('forecast_dataset', None)
    config["lr_forcasting"] = _get_forecast_lr(config, "lr_forcasting")

    # Single-dataset mode: align splits so test never leaks into pretraining
    if pretrain_dataset and pretrain_dataset == forecast_dataset:
        config['pretrain_source'] = None   # force CSV-only mode
        config['val_prec']  = config.get('val_prec_forcasting',  0.1)
        config['test_prec'] = config.get('test_prec_forcasting', 0.1)

    pretrain_source = _resolve_pretrain_source(config)
    use_global_data = pretrain_source is not None

    if use_global_data:
        if not pretrain_only and classification_dataset is None and forecast_dataset is None and anomaly_dataset is None:
            raise ValueError("forecast_dataset must be set when pretraining on global data")
        if pretrain_source in ('monash', 'monash+synthetic'):
            monash_dir = config.get('monash_data_dir', '../Monash')
            if not os.path.isabs(monash_dir):
                config['monash_data_dir'] = str((lejepa_dir / monash_dir).resolve())
        if pretrain_source in ('synthetic', 'monash+synthetic'):
            _synth_key = 'synthetic_mix_data_dir' if pretrain_source == 'monash+synthetic' else 'synthetic_data_dir'
            synth_dir = config.get(_synth_key, config.get('synthetic_data_dir', '../Monash'))
            if not os.path.isabs(synth_dir):
                synth_dir = str((lejepa_dir / synth_dir).resolve())
            config['synthetic_data_dir'] = synth_dir
        _src_label = {
            'monash':           f"Monash ({config.get('monash_data_dir', '')})",
            'synthetic':        f"Synthetic ({config.get('synthetic_data_dir', '')})",
            'monash+synthetic': "Monash + Synthetic (mix)",
        }.get(pretrain_source, pretrain_source)
        print("\n" + "="*60)
        print(f"  MODEL: LE-JEPA")
        if pretrain_only:
            print(f"  pretrain: {_src_label}  [pretrain only]")
        elif classification_only or forecast_dataset is None:
            print(f"  pretrain: {_src_label}   [classify only]")
        else:
            ds_fore = get_dataset_info(forecast_dataset)
            print(f"  pretrain: {_src_label}   forecast: {forecast_dataset}")
        print("="*60)
    else:
        pretrain_dataset = pretrain_dataset or config.get('pretrain_dataset')
        forecast_dataset = forecast_dataset or pretrain_dataset
        if pretrain_dataset is None:
            raise ValueError("pretrain_dataset not set — specify via run() or config_lejepa.py")
        ds_pre  = get_dataset_info(pretrain_dataset)
        ds_fore = get_dataset_info(forecast_dataset)
        n_groups = len(ds_pre["jepa_groups"])

        def _resolve(p):
            return str((lejepa_dir / p).resolve()) if not os.path.isabs(p) else p

        config["path_data"]       = [_resolve(ds_pre["csv_path"])] * n_groups
        config["timestampcols"]   = [ds_pre["timestamp_col"]] * n_groups
        config["input_variables"] = ds_pre["jepa_groups"]
        print("\n" + "="*60)
        print(f"  MODEL: LE-JEPA")
        print(f"  pretrain: {pretrain_dataset}   forecast: {forecast_dataset}")
        print("="*60)

    if not pretrain_only and forecast_dataset is not None:
        ds_fore = get_dataset_info(forecast_dataset)
        config["path_data_forcasting"]       = [str((lejepa_dir / ds_fore["csv_path"]).resolve())]
        config["timestampcols_forcasting"]   = [ds_fore["timestamp_col"]]
        config["input_variables_forcasting"] = [ds_fore["columns"]]

    # ── data ──────────────────────────────────────────────────────────────────
    if skip_train and use_global_data:
        print("\n[LE-JEPA] skip_train=True + global pretrain: skipping pretrain data load.")
        input_dim   = config["patch_size"]
        num_patches = config["ratio_patches"]
        train_loader = val_loader = test_loader = None
    else:
        print("\n[LE-JEPA] Loading datasets …")
        if use_global_data:
            import torch.utils.data as _tud
            if pretrain_source in ('monash', 'monash+synthetic'):
                train_dataset = MonashDataPullerJEPA(config, which='train')
                val_dataset   = MonashDataPullerJEPA(config, which='val')
                test_dataset  = MonashDataPullerJEPA(config, which='test')
            if pretrain_source in ('synthetic', 'monash+synthetic'):
                syn_train = SyntheticArrowDataPullerJEPA(config, which='train')
                syn_val   = SyntheticArrowDataPullerJEPA(config, which='val')
                syn_test  = SyntheticArrowDataPullerJEPA(config, which='test')
                if pretrain_source == 'monash+synthetic':
                    train_dataset = _tud.ConcatDataset([train_dataset, syn_train])
                    val_dataset   = _tud.ConcatDataset([val_dataset,   syn_val])
                    test_dataset  = _tud.ConcatDataset([test_dataset,  syn_test])
                else:
                    train_dataset, val_dataset, test_dataset = syn_train, syn_val, syn_test
        else:
            train_dataset = DataPullerDJepa(
                data_paths          = config["path_data"],
                patch_size          = config["patch_size"],
                batch_size          = config["batch_size"],
                ratio_patches       = config["ratio_patches"],
                mask_ratio          = config.get("mask_ratio", 0.25),
                masking_type        = config.get("masking_type", "multi_block"),
                num_semantic_tokens = config.get("num_semantic_tokens", 0),
                input_variables     = config["input_variables"],
                timestamp_cols      = config["timestampcols"],
                type_data           = "train",
                val_prec            = config["val_prec"],
                test_prec           = config["test_prec"],
                num_blocks          = config.get("num_blocks", 1),
            )
            val_dataset  = copy.copy(train_dataset); val_dataset.which  = "val"
            test_dataset = copy.copy(train_dataset); test_dataset.which = "test"

        _nw = config.get("num_workers", 4)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,  num_workers=_nw)
        val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=config["batch_size"], shuffle=False, num_workers=_nw)
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=config["batch_size"], shuffle=False, num_workers=_nw)

        sample      = train_dataset[0]
        patch_sample = sample[0] if isinstance(sample, (list, tuple)) else sample
        num_patches  = patch_sample.shape[0]
        input_dim    = patch_sample.shape[1]

    # ── model ──────────────────────────────────────────────────────────────────
    model = LeJEPA(
        config      = config,
        input_dim   = input_dim,
        num_patches = num_patches,
        train_loader= train_loader,
        val_loader  = val_loader,
        test_loader = test_loader,
    )
    model.to(model.device)

    # ── pretraining ────────────────────────────────────────────────────────────
    if not skip_train:
        print("\n[LE-JEPA] Starting pretraining …")
        model.train_and_evaluate()
    else:
        print("[LE-JEPA] Skipping pretraining.")

    if pretrain_only:
        print("\n[LE-JEPA] Pretrain-only mode — skipping forecasting.")
        return

    # ── forecasting downstream ─────────────────────────────────────────────────
    best_mse = best_ckpt = None
    if forecast_dataset is None:
        print("\n[LE-JEPA] No forecast_dataset — skipping forecasting.")
    else:
        # Use PatchTST-identical data: seq_len=336 (21 patches × 16)
        _PATCHTST_SEQ_LEN = 336
        p_s  = config["patch_size_forcasting"]
        _ctx_patches = _PATCHTST_SEQ_LEN // p_s   # = 21
        config["forecasting_context_patches"] = _ctx_patches
        _csv = config["path_data_forcasting"][0]

        _best_mse  = float('inf')
        ckpts     = checkpoints if checkpoints is not None else ["best"]

        _fc_bs = _get_forecast_bs(config, 256)
        _fc_nw = config.get("num_workers", 4)
        for pred_len in pred_lens:
            h_t = pred_len // p_s
            model.forcast_train = torch.utils.data.DataLoader(
                PatchTSTForcastingAdapter(_csv, 'train', _PATCHTST_SEQ_LEN, pred_len, p_s),
                batch_size=_fc_bs, shuffle=True,  num_workers=_fc_nw)
            model.forcast_val   = torch.utils.data.DataLoader(
                PatchTSTForcastingAdapter(_csv, 'val',   _PATCHTST_SEQ_LEN, pred_len, p_s),
                batch_size=_fc_bs, shuffle=False, num_workers=_fc_nw)
            model.forcast_test  = torch.utils.data.DataLoader(
                PatchTSTForcastingAdapter(_csv, 'test',  _PATCHTST_SEQ_LEN, pred_len, p_s),
                batch_size=_fc_bs, shuffle=False, num_workers=_fc_nw)
            model.config["horizon_t"] = h_t

            is_search    = (pred_len == pred_lens[0])
            ckpts_to_run = ckpts if is_search else [best_ckpt if best_ckpt is not None else ckpts[-1]]

            print(f"\n[LE-JEPA] pred_len={pred_len} (horizon_t={h_t})"
                  + ("" if is_search else f"  [best ckpt={ckpts_to_run[0]}]"))

            for epoch in ckpts_to_run:
                print(f"  → checkpoint epoch {epoch}")
                ckpt_tag = "" if epoch == "best" else f"_epoch{epoch}"
                mse = model.forcasting_zeroshot(ckpt_tag, linear_probe=linear_probe)
                if is_search and mse is not None and mse < _best_mse:
                    _best_mse = mse
                    best_ckpt = epoch

            if is_search:
                print(f"\n[LE-JEPA] Best checkpoint at pred_len={pred_lens[0]}: "
                      f"epoch {best_ckpt} (MSE={_best_mse:.4f})")
        best_mse = _best_mse

    # ── classification downstream ─────────────────────────────────────────────
    cls_acc = None
    if classification_dataset is not None:
        from data_loaders.data_puller import ClassificationDataPuller, make_uea_dataloaders
        import torch.nn.functional as _F
        cls_dir    = config.get("classification_data_dir", "/home/shared/datasets/Classification_TS")
        cls_bs     = config.get("batch_size", 64)
        p_s        = config["patch_size_forcasting"]
        _n_patches = 72                               # classification encoder always uses 72 patches
        _target_T  = _n_patches * p_s
        def _lejepa_patch_collate(batch, _ps=p_s, _tT=_target_T, _nP=_n_patches):
            xs, ys, orig_lens = zip(*batch)
            orig_lens = torch.stack(orig_lens)                                   # (B,)
            max_t = max(x.shape[0] for x in xs)
            xs = torch.stack([_F.pad(x, (0, 0, 0, max_t - x.shape[0])) for x in xs])
            B_, T_, C_ = xs.shape
            if T_ != _tT:
                idx = torch.linspace(0, T_ - 1, _tT).long()
                xs  = xs[:, idx, :]
                patch_idx    = idx[torch.arange(_nP) * _ps]
                padding_mask = patch_idx.unsqueeze(0) < orig_lens.unsqueeze(1)  # (B, P)
            else:
                patch_starts = torch.arange(_nP) * _ps
                padding_mask = patch_starts.unsqueeze(0) < orig_lens.unsqueeze(1)
            xs = xs.reshape(B_, _nP, _ps, C_)
            return xs, torch.stack(ys), padding_mask
        if list(Path(os.path.join(cls_dir, classification_dataset)).glob("*_TRAIN.ts")):
            _raw_tr, _, _raw_te, n_classes = make_uea_dataloaders(
                cls_dir, classification_dataset, batch_size=cls_bs)
            _wrap = lambda ds, shuf: torch.utils.data.DataLoader(
                ds, batch_size=cls_bs, shuffle=shuf, collate_fn=_lejepa_patch_collate)
            cls_train = _wrap(_raw_tr.dataset, True)
            cls_val   = None
            cls_test  = _wrap(_raw_te.dataset, False)
        else:
            _mk = lambda split: torch.utils.data.DataLoader(
                ClassificationDataPuller(cls_dir, classification_dataset, p_s, which=split),
                batch_size=cls_bs, shuffle=(split == "train"))
            cls_train = _mk("train"); cls_val = _mk("val"); cls_test = _mk("test")
            n_classes = cls_train.dataset.n_classes
        ckpt_tag  = f"_epoch{best_ckpt}" if best_ckpt is not None else ""
        cls_acc   = model.classification_zeroshot(ckpt_tag, cls_train, cls_val, cls_test, n_classes,
                                                  linear_probe=linear_probe)
        print(f"\n{'='*60}")
        print(f"  [LE-JEPA] Classification on {classification_dataset}")
        print(f"  Test Accuracy: {cls_acc:.4f}")
        print(f"{'='*60}")

    # ── anomaly detection downstream ──────────────────────────────────────────
    anom_result = None
    if anomaly_dataset is not None:
        from data_loaders.data_puller import AnomalyDataPuller
        anom_dir = config.get("anomaly_data_dir", "/home/shared/datasets/Anomaly_TS")
        anom_bs  = config.get("batch_size", 64)
        p_s      = config["patch_size_forcasting"]
        anom_train = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="train"),
            batch_size=anom_bs, shuffle=False)
        anom_test  = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="test"),
            batch_size=anom_bs, shuffle=False)
        ckpt_tag    = f"_epoch{best_ckpt}" if best_ckpt is not None else ""
        anom_result = model.anomaly_zeroshot(ckpt_tag, anom_train, anom_test,
                                             anomaly_ratio=_get_anomaly_ratio(anomaly_dataset, config),
                                             linear_probe=linear_probe)

    return best_ckpt, best_mse, cls_acc, anom_result


# ── TimeDart ──────────────────────────────────────────────────────────────────

# Registry key → TimeDart data arg + csv filename + freq
_TIMEDART_DATASET_MAP = {
    "etth1":       ("ETTh1",       "ETTh1.csv",       "h"),
    "etth2":       ("ETTh2",       "ETTh2.csv",       "h"),
    "ettm1":       ("ETTm1",       "ETTm1.csv",       "t"),
    "ettm2":       ("ETTm2",       "ETTm2.csv",       "t"),
    "weather":     ("Weather",     "weather.csv",     "h"),
    "electricity": ("Electricity", "electricity.csv", "h"),
    "traffic":     ("Traffic",     "traffic.csv",     "h"),
}


def run_timedart(skip_train: bool = False,
                 pretrain_dataset: str = None,
                 forecast_dataset: str = None,
                 classification_dataset: str = None,
                 anomaly_dataset: str = None,
                 pretrain_only: bool = False,
                 pred_lens=None,
                 checkpoints=None,
                 encoder_layers: int = None,
                 lr: float = None,
                 pretrain_source: str = None,
                 gpu: int = None,
                 linear_probe: bool = True,
                 pretrain_cls_model: bool = False):
    """
    TimeDart: diffusion-based pretraining with Monash/Synthetic data,
    followed by forecasting fine-tune using our PatchTSTForcastingAdapter
    (same splits / normalisation as every other model).

    Backbone: PatchTST (bidirectional encoder, configured via config_timedart.py).
    Diffusion pretraining objective is unchanged.
    """
    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]

    timedart_dir = Path(__file__).parent / "TimeDART-main"
    shared_dir    = Path(__file__).parent / "shared"
    _add_path(timedart_dir)
    _add_path(shared_dir)

    import importlib.util as _ilu, torch
    from types import SimpleNamespace
    from collections import OrderedDict

    # ── load config ────────────────────────────────────────────────────────────
    _spec = _ilu.spec_from_file_location("config_timedart", timedart_dir / "config_timedart.py")
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    cfg = dict(_mod.config)

    if pretrain_source is not None:
        cfg['pretrain_source'] = pretrain_source
    if encoder_layers is not None:
        cfg['e_layers'] = encoder_layers
    if lr is not None:
        cfg['learning_rate'] = lr

    _src_tag  = f"_{cfg['pretrain_source'].replace('+', '_')}" if cfg.get('pretrain_source', 'monash') != 'monash' else ''
    ckpt_dir  = Path(__file__).parent / f"outputs/timedart_pretrain{_src_tag}_layers{cfg['e_layers']}"
    ckpt_file     = ckpt_dir / f"monash{_src_tag}" / "ckpt_best.pth"
    cls_ckpt_file = ckpt_dir / f"monash{_src_tag}_cls" / "ckpt_best.pth"

    _explicit_forecast = forecast_dataset   # None means caller didn't request forecasting
    forecast_dataset   = forecast_dataset or "etth1"
    pretrain_src       = _resolve_pretrain_source(cfg)

    # Always use cuda:0 — CUDA_VISIBLE_DEVICES is already set by the caller
    # to select the physical GPU, so the logical index is always 0.
    _gpu_idx = 0
    _device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*60)
    print(f"  MODEL: TimeDart  (backbone={cfg.get('model','PatchTST')})")
    print(f"  pretrain: {pretrain_src or pretrain_dataset}   forecast: {forecast_dataset}")
    print(f"  e_layers={cfg['e_layers']}  d_model={cfg['d_model']}  patch_len={cfg['patch_len']}")
    print("="*60)

    seq_len   = cfg['seq_len']
    patch_len = cfg['patch_len']
    stride    = cfg['stride']

    # Shared args namespace used for both pretrain and finetune
    base_args = SimpleNamespace(
        model              = cfg.get('model', 'PatchTST'),
        downstream_task    = "forecast",
        use_gpu            = torch.cuda.is_available(),
        gpu                = _gpu_idx,
        use_multi_gpu      = False,
        devices            = str(_gpu_idx),
        device             = _device,
        device_ids         = [_gpu_idx],
        input_len          = seq_len,
        seq_len            = seq_len,
        label_len          = cfg.get("label_len", 0),
        pred_len           = 96,
        test_pred_len      = 96,
        patch_len          = patch_len,
        stride             = stride,
        d_model            = cfg['d_model'],
        n_heads            = cfg['n_heads'],
        e_layers           = cfg['e_layers'],
        d_layers           = 1,
        d_ff               = cfg['d_ff'],
        dropout            = cfg['dropout'],
        head_dropout       = cfg.get('head_dropout', 0.1),
        fc_dropout         = 0.0,
        embed              = "timeF",
        activation         = "gelu",
        output_attention   = False,
        individual         = 0,
        factor             = 1,
        distil             = True,
        moving_avg         = 25,
        top_k              = 5,
        num_kernels        = 3,
        enc_in             = 1,
        dec_in             = 1,
        c_out              = 1,
        use_norm           = 1,
        time_steps         = cfg['time_steps'],
        scheduler          = cfg['scheduler'],
        mask_ratio         = cfg['mask_ratio'],
        lr_decay           = cfg['lr_decay'],
        pct_start          = cfg.get('pct_start', 0.3),
        lradj              = cfg.get('lradj', 'decay'),
        features           = cfg.get('features', 'M'),
        target             = "OT",
        freq               = "h",
        seasonal_patterns  = "Monthly",
        num_classes        = 6,
        num_workers        = cfg.get('num_workers', 4),
        train_epochs       = cfg.get('train_epochs', 20),
        batch_size         = cfg.get('batch_size', 128),
        learning_rate      = cfg['learning_rate'],
        patience           = cfg.get('patience', 3),
        accumulation_steps = 4,
        select_channels    = 1.0,
        use_amp            = False,
        lm                 = 3,
        positive_nums      = 3,
        rbtp               = 1,
        temperature        = 0.2,
        masked_rule        = "geometric",
        mask_rate          = 0.5,
        load_checkpoints   = None,
        pretrain_checkpoints = str(ckpt_dir),
        checkpoints        = str(Path(__file__).parent / "outputs" / "timedart_finetune"),
        transfer_checkpoints = "ckpt_best.pth",
        data               = "ETTh1",
        root_path          = "/tmp",
        data_path          = "ETTh1.csv",
        task_name          = "pretrain",
        llm_path           = "Qwen/Qwen2.5-0.5B",
        backbone           = "Qwen2.5-0.5B",
    )

    # ── pretraining ────────────────────────────────────────────────────────────
    if not skip_train:
        from data_loaders.data_puller import (MonashWindowDatasetTimeDart,
                                              SyntheticWindowDatasetTimeDart)

        monash_dir = cfg.get('monash_data_dir', '/home/shared/datasets/Monash')
        _synth_key = 'synthetic_mix_data_dir' if pretrain_src == 'monash+synthetic' else 'synthetic_data_dir'
        synth_dir  = cfg.get(_synth_key, cfg.get('synthetic_data_dir', '/home/shared/datasets/synthetic_data_TS'))
        min_len    = cfg.get('monash_min_len', 512)

        def _make_pretrain_loader(which):
            datasets = []
            if pretrain_src in ('monash', 'monash+synthetic'):
                datasets.append(MonashWindowDatasetTimeDart(
                    monash_dir, seq_len=seq_len, which=which, min_len=min_len))
            if pretrain_src in ('synthetic', 'monash+synthetic'):
                datasets.append(SyntheticWindowDatasetTimeDart(
                    synth_dir, seq_len=seq_len, which=which, min_len=min_len))
            ds = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
            return torch.utils.data.DataLoader(
                ds, batch_size=cfg['batch_size'], shuffle=(which == 'train'),
                num_workers=cfg.get('num_workers', 4), drop_last=True)

        train_loader = _make_pretrain_loader('train')
        val_loader   = _make_pretrain_loader('val')

        base_args.task_name = "pretrain"
        base_args.data      = "monash" + _src_tag

        # pretrain_cls_model=True → use ClsModel (Conv1d embedding) so the
        # Conv1d is pretrained and can be fully loaded for classification.
        if pretrain_cls_model:
            base_args.downstream_task = "classification"

        from exp.exp_timedart import Exp_TimeDART
        exp = Exp_TimeDART(base_args)

        # restore for forecasting fine-tune
        base_args.downstream_task = "forecast"

        _ckpt_subdir = (base_args.data + "_cls") if pretrain_cls_model else base_args.data
        ckpt_path = ckpt_dir / _ckpt_subdir
        ckpt_path.mkdir(parents=True, exist_ok=True)

        optimizer       = torch.optim.Adam(exp.model.parameters(), lr=cfg['learning_rate'])
        model_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg['lr_decay'])

        n_epochs = cfg.get('train_epochs', 20)
        min_vali = float('inf')

        print(f"\n[TimeDart] Pretraining ({n_epochs} epochs) …")
        for epoch in range(n_epochs):
            train_loss = exp.pretrain_one_epoch(train_loader, optimizer, model_scheduler)
            vali_loss  = exp.valid_one_epoch(val_loader)
            print(f"  Epoch {epoch+1}/{n_epochs} | train={train_loss:.4f} vali={vali_loss:.4f}")
            if vali_loss <= min_vali:
                min_vali = vali_loss
                enc_sd = OrderedDict(
                    (k, v) for k, v in exp.model.state_dict().items()
                    if "encoder" in k or "enc_embedding" in k)
                torch.save({"epoch": epoch, "model_state_dict": enc_sd},
                           ckpt_path / "ckpt_best.pth")
                print(f"    ✓ saved best (vali={vali_loss:.4f})")
        print(f"[TimeDart] Pretraining done. Checkpoint: {ckpt_path}/ckpt_best.pth")
    else:
        print("[TimeDart] Skipping pretraining.")

    if pretrain_only:
        return

    # ── forecasting downstream — our PatchTSTForcastingAdapter ────────────────
    # Wraps the exact same splits / normalisation used by all other models.
    # Returns (batch_x, batch_y, zeros_xmark, zeros_ymark) matching TimeDart's
    # train loop which only uses batch_x (input) and batch_y (target).

    if not ckpt_file.exists():
        print(f"[TimeDart] WARNING: checkpoint not found at {ckpt_file}")

    if _explicit_forecast is None and (classification_dataset is not None or anomaly_dataset is not None):
        # Anomaly-only or classification-only call — skip forecasting
        best_pred, best_mse, best_mae = None, float('inf'), float('inf')
    else:
        from data_loaders.data_puller import PatchTSTForcastingAdapter
        from exp.exp_timedart import Exp_TimeDART

        ds_info   = get_dataset_info(forecast_dataset)
        _csv      = ds_info["csv_path"]
        _c_in     = ds_info["c_in"]
        _fc_bs    = _get_forecast_bs(cfg, 128)
        _fc_nw    = cfg.get('num_workers', 4)
        _PATCHTST_SEQ_LEN = 336

        best_mse  = float('inf')
        best_mae  = float('inf')
        best_pred = None

        for pred_len in pred_lens:
            # Build our standard forecasting loaders
            def _fc_loader(split):
                ds = _FlatWindowAdapter(
                    PatchTSTForcastingAdapter(_csv, split, _PATCHTST_SEQ_LEN, pred_len, patch_len))
                return torch.utils.data.DataLoader(
                    ds, batch_size=_fc_bs, shuffle=(split == 'train'),
                    num_workers=_fc_nw, drop_last=True)

            ft_args = SimpleNamespace(**vars(base_args))
            ft_args.task_name      = "finetune"
            ft_args.pred_len       = pred_len
            ft_args.test_pred_len  = pred_len
            ft_args.train_epochs   = cfg.get('epochs_forecasting', 20)
            ft_args.learning_rate  = _get_forecast_lr(cfg, 'lr_forecasting', 1e-4)
            ft_args.batch_size     = _fc_bs
            ft_args.enc_in         = _c_in
            ft_args.dec_in         = _c_in
            ft_args.c_out          = _c_in
            ft_args.load_checkpoints = str(ckpt_file) if ckpt_file.exists() else None
            ft_args.checkpoints    = str(Path(__file__).parent / "outputs" / "timedart_finetune")
            # Use constant LR for forecasting — exponential decay kills LR by epoch 10
            ft_args.lradj          = "constant"
            ft_args.patience       = cfg.get('patience', 5)

            setting = f"timedart_{forecast_dataset}_pl{pred_len}_dm{cfg['d_model']}_el{cfg['e_layers']}"

            print(f"\n[TimeDart] {'Linear probing' if linear_probe else 'Fine-tuning'} pred_len={pred_len} on {forecast_dataset} …")
            exp = Exp_TimeDART(ft_args)

            if linear_probe:
                for name, param in exp.model.named_parameters():
                    if 'head' not in name:
                        param.requires_grad = False
                _trainable = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)
                _total     = sum(p.numel() for p in exp.model.parameters())
                print(f"  [TimeDart forecast] MODE: linear probe — encoder FROZEN")
                print(f"  Trainable: {_trainable:,} / {_total:,} params")
            else:
                _total = sum(p.numel() for p in exp.model.parameters())
                print(f"  [TimeDart forecast] MODE: full fine-tuning — encoder UNFROZEN")
                print(f"  Trainable: {_total:,} / {_total:,} params")

            # Inject our loaders directly — bypasses TimeDart's _get_data()
            exp._td_train_loader = _fc_loader('train')
            exp._td_val_loader   = _fc_loader('val')
            exp._td_test_loader  = _fc_loader('test')
            _patch_timedart_get_data(exp)

            exp.train(setting)

            # ── compute test MSE + MAE using full predictions ─────────────────────
            exp.model.eval()
            preds_list, trues_list = [], []
            with torch.no_grad():
                for batch_x, batch_y, _, _ in exp._td_test_loader:
                    batch_x = batch_x.float().to(_device)
                    batch_y = batch_y.float().to(_device)
                    pred = exp.model(batch_x)
                    f_dim = -1 if ft_args.features == "MS" else 0
                    pred = pred[:, -pred_len:, f_dim:]
                    batch_y = batch_y[:, -pred_len:, f_dim:]
                    preds_list.append(pred.detach().cpu().numpy())
                    trues_list.append(batch_y.detach().cpu().numpy())
            exp.model.train()
            import numpy as _np
            preds_arr = _np.concatenate(preds_list, axis=0)
            trues_arr = _np.concatenate(trues_list, axis=0)
            mse = float(_np.mean((preds_arr - trues_arr) ** 2))
            mae = float(_np.mean(_np.abs(preds_arr - trues_arr)))
            print(f"  [TimeDart] pred_len={pred_len} → test MSE={mse:.4f}  MAE={mae:.4f}")

            if pred_len == pred_lens[0] and mse < best_mse:
                best_mse  = mse
                best_mae  = mae
                best_pred = pred_len

    # ── classification downstream ─────────────────────────────────────────────
    cls_acc = None
    if classification_dataset is not None:
        from timedart_classification import classification_zeroshot as timedart_classify
        from data_loaders.data_puller import ClassificationDataPuller, make_uea_dataloaders
        cls_dir    = cfg.get("classification_data_dir", "/home/shared/datasets/Classification_TS")
        cls_bs     = _get_cls_bs(cfg, "batch_size_classification", 64)
        p_s        = cfg.get("patch_len", 16)
        _n_patches = 72                               # classification encoder always uses 72 patches
        _target_T  = _n_patches * p_s

        def _timedart_patch_collate(batch, _ps=p_s, _tT=_target_T, _nP=_n_patches):
            import torch.nn.functional as F
            xs, ys, orig_lens = zip(*batch)
            orig_lens = torch.stack(orig_lens)
            max_t = max(x.shape[0] for x in xs)
            xs = torch.stack([F.pad(x, (0, 0, 0, max_t - x.shape[0])) for x in xs])
            T = xs.shape[1]
            if T != _tT:
                idx = torch.linspace(0, T - 1, _tT).long()
                xs  = xs[:, idx, :]
                patch_idx    = idx[torch.arange(_nP) * _ps]
                padding_mask = patch_idx.unsqueeze(0) < orig_lens.unsqueeze(1)
            else:
                patch_starts = torch.arange(_nP) * _ps
                padding_mask = patch_starts.unsqueeze(0) < orig_lens.unsqueeze(1)
            xs = xs.reshape(len(xs), _nP, _ps, xs.shape[-1])
            return xs, torch.stack(ys), padding_mask

        if list(Path(os.path.join(cls_dir, classification_dataset)).glob("*_TRAIN.ts")):
            _raw_train, _, _raw_test, n_classes = make_uea_dataloaders(
                cls_dir, classification_dataset, batch_size=cls_bs)
            cls_train = torch.utils.data.DataLoader(
                _raw_train.dataset, batch_size=cls_bs, shuffle=True,
                collate_fn=_timedart_patch_collate)
            cls_val   = None
            cls_test  = torch.utils.data.DataLoader(
                _raw_test.dataset, batch_size=cls_bs, shuffle=False,
                collate_fn=_timedart_patch_collate)
        else:
            _mk = lambda split: torch.utils.data.DataLoader(
                ClassificationDataPuller(cls_dir, classification_dataset, p_s, which=split),
                batch_size=cls_bs, shuffle=(split == "train"))
            cls_train = _mk("train"); cls_val = _mk("val"); cls_test = _mk("test")
            n_classes = cls_train.dataset.n_classes

        _cls_ckpt = cls_ckpt_file if cls_ckpt_file.exists() else ckpt_file
        cls_acc = timedart_classify(cfg, str(_cls_ckpt), cls_train, cls_val, cls_test, n_classes,
                                    linear_probe=linear_probe)
        print(f"\n{'='*60}")
        print(f"  [TimeDaRT] Classification on {classification_dataset}")
        print(f"  Test Accuracy: {cls_acc:.4f}")
        print(f"{'='*60}")

    # ── anomaly detection downstream ──────────────────────────────────────────
    anom_result = None
    if anomaly_dataset is not None:
        from timedart_anomaly import anomaly_zeroshot as timedart_anomaly
        from data_loaders.data_puller import AnomalyDataPuller
        anom_dir = cfg.get("anomaly_data_dir", "/home/shared/datasets/Anomaly_TS")
        anom_bs  = cfg.get("batch_size", 64)
        p_s      = cfg.get("patch_len", 16)
        anom_train = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="train"),
            batch_size=anom_bs, shuffle=False)
        anom_test  = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="test"),
            batch_size=anom_bs, shuffle=False)
        anom_result = timedart_anomaly(cfg, str(ckpt_file), anom_train, anom_test,
                                       anomaly_ratio=_get_anomaly_ratio(anomaly_dataset, cfg),
                                       linear_probe=linear_probe)

    return best_pred, best_mse, best_mae, cls_acc, anom_result


class _FlatWindowAdapter(torch.utils.data.Dataset):
    """
    Wraps PatchTSTForcastingAdapter (which returns patched tensors) and
    flattens them back to (seq_x, seq_y, zeros_mark, zeros_mark) so that
    TimeDart's train/valid loops receive the (B, T, C) shape they expect.
    """
    def __init__(self, patched_ds):
        self._ds = patched_ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        ctx, tgt = self._ds[idx]
        # ctx: [n_patches, patch_size, C]  →  [seq_len, C]
        # tgt: [h,         patch_size, C]  →  [pred_len, C]
        seq_x = ctx.reshape(-1, ctx.shape[-1])
        seq_y = tgt.reshape(-1, tgt.shape[-1])
        zeros = torch.zeros_like(seq_x)
        return seq_x, seq_y, zeros, zeros


def _patch_timedart_get_data(exp):
    """Monkey-patch _get_data on a TimeDart Exp instance to return our loaders."""
    import types

    def _get_data(self, flag):
        loader = {'train': self._td_train_loader,
                  'val':   self._td_val_loader,
                  'test':  self._td_test_loader}[flag]
        return loader.dataset, loader

    exp._get_data = types.MethodType(_get_data, exp)


# ── Random baseline ───────────────────────────────────────────────────────────

def run_random(skip_train: bool = False, pretrain_dataset: str = None, forecast_dataset: str = None):
    random_dir = Path(__file__).parent / "random"
    ntp_dir    = Path(__file__).parent / "NTP"
    _add_path(random_dir)
    _add_path(ntp_dir)

    import importlib.util
    _spec = importlib.util.spec_from_file_location("config_ntp", ntp_dir / "config_ntp.py")
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    cfg = dict(_mod.config)

    _forecast_dset = forecast_dataset or cfg.get("forecast_dataset", "ettm1")
    cfg["forecast_dataset"] = _forecast_dset

    print("\n" + "="*60)
    print("  MODEL: Random Baseline (frozen random encoder)")
    print(f"  forecast: {_forecast_dset}")
    print("="*60)

    from random_forecasting import random_forecasting
    random_forecasting(cfg, _forecast_dset)


# ── entry point ───────────────────────────────────────────────────────────────

RUNNERS = {
    "dino":            run_dino,
    "jepa":            run_jepa,
    "lejepa":          run_lejepa,
    "patchtst":        run_patchtst,
    "patchtst_random": lambda skip_train=False, pretrain_dataset=None, forecast_dataset=None, classification_dataset=None, anomaly_dataset=None, pretrain_only=False, classification_only=False, pred_lens=None, checkpoints=None, encoder_layers=None, pretrain_source=None, num_patches=None, linear_probe=True: run_patchtst(skip_train=skip_train, pretrain_dataset=pretrain_dataset, forecast_dataset=forecast_dataset, classification_dataset=classification_dataset, anomaly_dataset=anomaly_dataset, pretrain_only=pretrain_only, classification_only=classification_only, pred_lens=pred_lens, checkpoints=checkpoints, random_encoder=True, encoder_layers=encoder_layers, pretrain_source=pretrain_source, num_patches=num_patches, linear_probe=linear_probe),
    "jepa_random": lambda skip_train=False, pretrain_dataset=None, forecast_dataset=None, classification_dataset=None, anomaly_dataset=None, pred_lens=None, checkpoints=None, pretrain_only=False, encoder_layers=None, predictor_layers=None, lr=None, pretrain_source=None, checkpoint=None, num_patches=None, linear_probe=True: run_jepa(skip_train=skip_train, pretrain_dataset=pretrain_dataset, forecast_dataset=forecast_dataset, classification_dataset=classification_dataset, anomaly_dataset=anomaly_dataset, pred_lens=pred_lens, checkpoints=checkpoints, pretrain_only=pretrain_only, encoder_layers=encoder_layers, predictor_layers=predictor_layers, lr=lr, pretrain_source=pretrain_source, checkpoint=checkpoint, num_patches=num_patches, random_encoder=True, linear_probe=linear_probe),
    "ntp":             run_ntp,
    "random":          run_random,
    "timedart":        run_timedart,
}

def run(model: str,
        task: str = None,
        skip_train: bool = False,
        dataset: str = None,
        pretrain_dataset: str = None,
        forecast_dataset: str = None,
        classification_dataset=None,
        anomaly_dataset: str = None,
        checkpoint: str = None,
        pred_lens=None,
        checkpoints=None,
        pretrain_only: bool = False,
        pred_len: int = None,
        encoder_layers: int = None,
        predictor_layers: int = None,
        lr: float = None,
        pretrain_source: str = None,
        gpu: int = None,
        num_patches: int = None,
        seed: int = None,
        pretrain_cls_model: bool = False,
        linear_probe: bool = True):
    """
    Unified entry point. Each run handles ONE task.

    task="pretrain"   — pretrain only (same as pretrain_only=True)
    task="forecast"   — skip pretraining, run forecasting only (same as skip_train=True)
    task="classify"   — skip pretraining, run classification only

    Backwards compatible — old flags (skip_train, pretrain_only) still work when task=None.

    Examples:
        run(model="jepa", task="pretrain")
        run(model="jepa", task="forecast", forecast_dataset="etth1", skip_train=True)
        run(model="jepa", task="classify",
            classification_dataset="EthanolConcentration", skip_train=True)

        # Old style still works:
        run(model="jepa", skip_train=False)
        run(model="jepa", pretrain_only=True)
    """
    # ── resolve task → old flags (backwards compat) ───────────────────────────
    classification_only = False
    if task is not None:
        task = task.lower()
        if task == "pretrain":
            pretrain_only = True
            skip_train    = False
        elif task == "forecast":
            skip_train    = True
            pretrain_only = False
            classification_dataset = None   # force no classification
        elif task == "classify":
            skip_train           = True
            pretrain_only        = False
            classification_only  = True
            forecast_dataset     = None     # force no forecasting
            anomaly_dataset      = None
        elif task == "anomaly":
            skip_train    = True
            pretrain_only = False
            forecast_dataset       = None
            classification_dataset = None
        else:
            raise ValueError(f"Unknown task '{task}'. Choose: pretrain | forecast | classify | anomaly")

    global _SEED_TAG
    if seed is not None:
        _set_seed(seed)
        _SEED_TAG = f'_seed{seed}'
    else:
        _set_seed()
        _SEED_TAG = ''
    model = model.lower()
    if model not in RUNNERS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(RUNNERS)}")
    # 'dataset' is shorthand for pretrain_dataset == forecast_dataset
    if dataset is not None:
        pretrain_dataset = pretrain_dataset or dataset
        forecast_dataset = forecast_dataset or dataset
    runner = RUNNERS[model]
    import inspect
    sig = inspect.signature(runner)
    kwargs = dict(skip_train=skip_train,
                  pretrain_dataset=pretrain_dataset,
                  forecast_dataset=forecast_dataset)
    if 'pretrain_only'          in sig.parameters: kwargs['pretrain_only']          = pretrain_only
    if 'classification_only'   in sig.parameters: kwargs['classification_only']   = classification_only
    if 'pred_lens'              in sig.parameters: kwargs['pred_lens']              = pred_lens
    if 'checkpoints'            in sig.parameters: kwargs['checkpoints']            = checkpoints
    if 'pred_len'               in sig.parameters: kwargs['pred_len']               = pred_len
    if 'encoder_layers'         in sig.parameters: kwargs['encoder_layers']         = encoder_layers
    if 'predictor_layers'       in sig.parameters: kwargs['predictor_layers']       = predictor_layers
    if 'lr'                     in sig.parameters: kwargs['lr']                     = lr
    if 'classification_dataset' in sig.parameters: kwargs['classification_dataset'] = classification_dataset
    if 'anomaly_dataset'        in sig.parameters: kwargs['anomaly_dataset']        = anomaly_dataset
    if 'checkpoint'             in sig.parameters: kwargs['checkpoint']             = checkpoint
    if 'pretrain_source'        in sig.parameters: kwargs['pretrain_source']        = pretrain_source
    if 'gpu'                    in sig.parameters: kwargs['gpu']                    = gpu
    if 'num_patches'            in sig.parameters: kwargs['num_patches']            = num_patches
    if 'seed'                   in sig.parameters: kwargs['seed']                   = seed
    if 'pretrain_cls_model'     in sig.parameters: kwargs['pretrain_cls_model']     = pretrain_cls_model
    if 'linear_probe'           in sig.parameters: kwargs['linear_probe']           = linear_probe
    return runner(**kwargs)


if __name__ == "__main__":
    from dataset_registry import DATASETS as _DATASETS
    parser = argparse.ArgumentParser(description="Unified training + forecasting runner")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(RUNNERS),
        help="Which model to run: dino | jepa | lejepa | patchtst | npt | random",
    )
    parser.add_argument(
        "--pretrain_dataset", type=str, default=None,
        choices=list(_DATASETS),
        help=f"Dataset for pretraining. Available: {list(_DATASETS)}",
    )
    parser.add_argument(
        "--forecast_dataset", type=str, default=None,
        choices=list(_DATASETS),
        help="Dataset for forecasting downstream (defaults to pretrain_dataset).",
    )
    parser.add_argument(
        "--skip_train", type=str, default="false",
        choices=["true", "false"],
        help="Skip pretraining and go straight to forecasting (true | false)",
    )
    parser.add_argument(
        "--pretrain_only", type=str, default="false",
        choices=["true", "false"],
        help="Run pretraining only, skip downstream evaluation (true | false)",
    )
    parser.add_argument("--task", type=str, default=None,
                        choices=["pretrain", "forecast", "classify", "anomaly"],
                        help="Task to run: pretrain | forecast | classify. "
                             "Overrides --skip_train / --pretrain_only when set.")
    parser.add_argument("--classification_dataset", type=str, default=None,
                        help="Dataset name for classification (subfolder under Classification_TS dir)")
    parser.add_argument("--anomaly_dataset",  type=str, default=None,
                        help="Dataset name for anomaly detection (subfolder under Anomaly_TS dir)")
    parser.add_argument("--checkpoint",       type=str, default=None,
                        help="Path to pretrained checkpoint to load for classification")
    parser.add_argument("--encoder_layers",   type=int,   default=None,
                        help="Override number of encoder transformer layers")
    parser.add_argument("--predictor_layers", type=int,   default=None,
                        help="Override number of predictor layers (JEPA models only)")
    parser.add_argument("--lr",               type=float, default=None,
                        help="Override pretraining learning rate")
    parser.add_argument("--pretrain_source",  type=str,   default=None,
                        choices=["monash", "synthetic", "monash+synthetic"],
                        help="Override pretrain data source (dino only)")
    parser.add_argument("--num_patches",      type=int,   default=None,
                        help="Override number of patches (context window = num_patches × patch_size)")
    parser.add_argument("--seed",             type=int,   default=None,
                        help="Random seed (also suffixes checkpoint paths with _seedN)")
    parser.add_argument("--pretrain_cls_model", type=str, default="false",
                        help="Use cls embedding model for pretraining (saves to separate _cls checkpoint)")
    args = parser.parse_args()
    run(model=args.model,
        task=args.task,
        skip_train=args.skip_train.lower() == "true",
        pretrain_dataset=args.pretrain_dataset,
        forecast_dataset=args.forecast_dataset,
        pretrain_only=args.pretrain_only.lower() == "true",
        classification_dataset=args.classification_dataset,
        anomaly_dataset=args.anomaly_dataset,
        checkpoint=args.checkpoint,
        encoder_layers=args.encoder_layers,
        predictor_layers=args.predictor_layers,
        lr=args.lr,
        pretrain_source=args.pretrain_source,
        num_patches=args.num_patches,
        seed=args.seed,
        pretrain_cls_model=args.pretrain_cls_model.lower() == "true")
