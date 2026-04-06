"""
Unified training + forecasting runner for:
  - dino        (TSDINOALT 4)
  - jepa        (Discrete_JEPA / DiscreteJEPA)
  - jepa_simple (JEPA — P2P only, no VQ / semantic tokens)
  - lejepa      (LE-JEPA — two-view augmentation, SIGReg loss)
  - patchtst    (PatchTST_self_supervised)

Usage
-----
  python Train_and_downstream.py --model dino
  python Train_and_downstream.py --model jepa  --skip_train true
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
        num_workers                 = cfg.get("num_workers", 0),
        batch_size_per_gpu          = cfg.get("batch_size_per_gpu", 64),
        batch_size_forecast         = cfg.get("batch_size_forecast", 256),

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
        lr_forecasting                      = cfg.get("lr_forecasting", 0.001),
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
             encoder_layers: int = None,
             predictor_layers: int = None,
             lr: float = None):
    dino_dir  = Path(__file__).parent / "TSDiNO"
    djepa_dir = Path(__file__).parent / "Discrete_JEPA"
    _add_path(dino_dir)
    _add_path(djepa_dir)

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
    if encoder_layers is not None:
        dino_cfg['n_layers'] = encoder_layers
        dino_cfg['output_dir'] = dino_cfg.get('output_dir', './checkpoints').rstrip('/') + f'_layers{encoder_layers}'
    if lr is not None:
        dino_cfg['lr'] = lr
    pretrain_source = _resolve_pretrain_source(dino_cfg)
    use_global_data = pretrain_source is not None

    # Resolve forecast dataset (always needed for downstream)
    forecast_dataset = forecast_dataset or dino_cfg.get("forecast_dataset")
    if pretrain_only and use_global_data:
        dino_cfg['saveckp_freq'] = 1  # save every epoch

    if use_global_data:
        # No pretrain CSV needed; c_in = 1 (univariate global data)
        dino_cfg["c_in"] = 1
        if not pretrain_only:
            if forecast_dataset is None:
                raise ValueError("forecast_dataset must be set when pretraining on global data")
            ds_fore = get_dataset_info(forecast_dataset)
            dino_cfg["c_in"] = ds_fore["c_in"]
        if pretrain_source in ('monash', 'monash+synthetic'):
            monash_dir = dino_cfg.get('monash_data_dir', '../Monash')
            if not os.path.isabs(monash_dir):
                dino_cfg['monash_data_dir'] = str((dino_dir / monash_dir).resolve())
        if pretrain_source in ('synthetic', 'monash+synthetic'):
            synth_dir = dino_cfg.get('synthetic_data_dir', '../Monash')
            if not os.path.isabs(synth_dir):
                dino_cfg['synthetic_data_dir'] = str((dino_dir / synth_dir).resolve())
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
    if not pretrain_only:
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

    # ── forecasting downstream ────────────────────────────────────────────────
    print("\n[DINO] Running forecasting downstream task …")
    ckpts = checkpoints if checkpoints is not None else [80, 120, 160, 200, 240, 300]
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
            print(f"  → checkpoint {ckpt} ({'random init' if ckpt == 0 else f'epoch {ckpt}'})")
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
        from data_loaders.data_puller import ClassificationDataPuller
        cls_dir = dino_cfg.get("classification_data_dir", "/home/shared/datasets/Classification_TS")
        cls_bs  = dino_cfg.get("batch_size_classification", 64)
        p_s     = args.patch_len
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
            anomaly_ratio=dino_cfg.get("anomaly_ratio", 1.0))

    return best_ckpt, best_mse, cls_acc, anom_result


# ── Discrete JEPA ─────────────────────────────────────────────────────────────

def _resolve_jepa_path(p: str, jepa_dir: Path) -> str:
    """Return *p* as-is if absolute, otherwise resolve relative to *jepa_dir*."""
    if os.path.isabs(p):
        return p
    return str((jepa_dir / p.lstrip('./').lstrip('/')).resolve())


def run_jepa(skip_train: bool = False,
             pretrain_dataset: str = None,
             forecast_dataset: str = None,
             pretrain_only: bool = False,
             pred_lens=None,
             checkpoints=None,
             encoder_layers: int = None,
             predictor_layers: int = None):
    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]

    jepa_dir = Path(__file__).parent / "Discrete_JEPA"
    _add_path(jepa_dir)

    import torch
    from config_files.config_pretrain import config
    from data_loaders.data_puller import (DataPullerDJepa, ForcastingDataPullerDescrete,
                                          MonashDataPullerJEPA, SyntheticArrowDataPullerJEPA)
    from Discrete_JEPA.Discrete_Jepa import DiscreteJEPA

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = dict(config)
    if encoder_layers is not None:
        config['num_encoder_layers'] = encoder_layers
        config['path_save'] = config.get('path_save', './output_model/DiscreteJEPA/').rstrip('/') + f'_layers{encoder_layers}/'
    if predictor_layers is not None:
        config['predictor_num_layers'] = predictor_layers

    # Single-dataset pipeline: force same dataset for pretrain and forecast,
    # disable Monash, and align split fractions so test never leaks into training.
    if pretrain_dataset is not None and pretrain_dataset == forecast_dataset:
        config['pretrain_source'] = None   # force CSV-only mode
        config['val_prec']  = config.get('val_prec_forcasting',  0.1)
        config['test_prec'] = config.get('test_prec_forcasting', 0.1)

    pretrain_source = _resolve_pretrain_source(config)
    use_global_data = pretrain_source is not None

    # Resolve forecast dataset (always needed for downstream, optional for pretrain_only)
    forecast_dataset = forecast_dataset or config.get("forecast_dataset")
    if use_global_data:
        if not pretrain_only and forecast_dataset is None:
            raise ValueError("forecast_dataset must be set when pretraining on global data")
        if pretrain_source in ('monash', 'monash+synthetic'):
            monash_dir = config.get('monash_data_dir', '../Monash')
            if not os.path.isabs(monash_dir):
                config['monash_data_dir'] = str((jepa_dir / monash_dir).resolve())
        if pretrain_source in ('synthetic', 'monash+synthetic'):
            synth_dir = config.get('synthetic_data_dir', '../Monash')
            if not os.path.isabs(synth_dir):
                config['synthetic_data_dir'] = str((jepa_dir / synth_dir).resolve())
        _src_label = {
            'monash':           f"Monash ({config.get('monash_data_dir', '')})",
            'synthetic':        f"Synthetic ({config.get('synthetic_data_dir', '')})",
            'monash+synthetic': "Monash + Synthetic",
        }.get(pretrain_source, pretrain_source)
        print("\n" + "="*60)
        print(f"  MODEL: Discrete JEPA")
        if pretrain_only:
            print(f"  pretrain: {_src_label}  [pretrain only]")
        else:
            ds_fore = get_dataset_info(forecast_dataset)
            print(f"  pretrain: {_src_label}   forecast: {forecast_dataset}")
        print("="*60)
    else:
        pretrain_dataset = pretrain_dataset or config.get("pretrain_dataset")
        forecast_dataset = forecast_dataset or pretrain_dataset
        if pretrain_dataset is None:
            raise ValueError("pretrain_dataset not set — specify via run() or config_pretrain.py")
        ds_pre  = get_dataset_info(pretrain_dataset)
        ds_fore = get_dataset_info(forecast_dataset)
        n_groups = len(ds_pre["jepa_groups"])
        config["path_data"]       = [_resolve_jepa_path(ds_pre["csv_path"], jepa_dir)] * n_groups
        config["timestampcols"]   = [ds_pre["timestamp_col"]] * n_groups
        config["input_variables"] = ds_pre["jepa_groups"]
        print("\n" + "="*60)
        print(f"  MODEL: Discrete JEPA")
        print(f"  pretrain: {pretrain_dataset}   forecast: {forecast_dataset}")
        print("="*60)

    if not pretrain_only:
        # Set forecasting paths from forecast dataset
        config["path_data_forcasting"]       = [_resolve_jepa_path(ds_fore["csv_path"], jepa_dir)]
        config["timestampcols_forcasting"]   = [ds_fore["timestamp_col"]]
        config["input_variables_forcasting"] = [ds_fore["columns"]]

    # ── data ─────────────────────────────────────────────────────────────────
    if skip_train and use_global_data:
        # Skip loading pretrain data entirely — only need forecasting loaders
        print("\n[JEPA] skip_train=True + global pretrain: skipping pretrain data load.")
        input_dim   = config["patch_size"]  # univariate: input_dim = patch_size
        num_patches = config["ratio_patches"]
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
                data_paths         = config["path_data"],
                patch_size         = config["patch_size"],
                batch_size         = config["batch_size"],
                ratio_patches      = config["ratio_patches"],
                mask_ratio         = config["mask_ratio"],
                masking_type       = config["masking_type"],
                num_semantic_tokens= config["num_semantic_tokens"],
                input_variables    = config["input_variables"],
                timestamp_cols     = config["timestampcols"],
                type_data          = "train",
                val_prec           = config["val_prec"],
                test_prec          = config["test_prec"],
                stride             = config.get("stride", None),
                num_blocks         = config.get("num_blocks", 1),
            )
            val_dataset  = copy.copy(train_dataset); val_dataset.which  = "val"
            test_dataset = copy.copy(train_dataset); test_dataset.which = "test"

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=config["batch_size"], shuffle=True)
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=config["batch_size"], shuffle=False)
        input_dim       = len(train_loader.dataset[0][0][0])
        num_patches     = len(train_loader.dataset[0][0])

    if pretrain_only:
        train_loader_fc = val_loader_fc = test_loader_fc = None
    else:
        forecasting_data = ForcastingDataPullerDescrete(config)
        val_fc   = copy.copy(forecasting_data); val_fc.which  = "val";  val_fc.rebuild()
        test_fc  = copy.copy(forecasting_data); test_fc.which = "test"; test_fc.rebuild()
        _fc_bs = config.get("batch_size_forecast", 256)
        train_loader_fc = torch.utils.data.DataLoader(forecasting_data, batch_size=_fc_bs, shuffle=True)
        val_loader_fc   = torch.utils.data.DataLoader(val_fc,           batch_size=_fc_bs, shuffle=True)
        test_loader_fc  = torch.utils.data.DataLoader(test_fc,          batch_size=_fc_bs, shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    model = DiscreteJEPA(
        config            = config,
        input_dim         = input_dim,
        num_patches       = num_patches,
        steps_per_epoch   = 1,
        train_loader      = train_loader,
        val_loader        = val_loader,
        test_loader       = test_loader,
        forcasting_train  = train_loader_fc,
        forcasting_val    = val_loader_fc,
        forcasting_test   = test_loader_fc,
    )

    # ── pretraining ───────────────────────────────────────────────────────────
    if not skip_train:
        print("\n[JEPA] Starting pretraining …")
        model.train_and_evaluate()
    else:
        print("[JEPA] Skipping pretraining.")

    if pretrain_only:
        print("\n[JEPA] Pretrain-only mode — skipping forecasting.")
        return

    # ── forecasting downstream ────────────────────────────────────────────────
    # CSV already loaded — update horizon on existing dataset objects per pred_len,
    # recreate DataLoaders (size changes), then loop checkpoints.  Test split is
    # never used during fine-tuning; only evaluated at the end of each run.
    modes = config.get("forecasting_modes", ["zeroshot"])
    _MODE_MAP = {
        "zeroshot":          "forcasting_zeroshot",
        "finetuning":        "finetuning_forecasting",
        "predictor":         "predictor_forecasting",
        "predictor_s2p_p2p": "predictor_s2p_p2p_forecasting",
    }
    ckpts = checkpoints if checkpoints is not None else [40, 60, 90, 120, 160]
    p_s = config["patch_size_forcasting"]
    best_ckpt = None
    best_mse  = float('inf')

    for pred_len in pred_lens:
        h_t = pred_len // p_s
        # Patch the already-loaded datasets in-place (no CSV re-read)
        for ds in [forecasting_data, val_fc, test_fc]:
            ds.h = h_t
            ds.target_raw_len = h_t * p_s
        _fc_bs = config.get("batch_size_forecast", 256)
        train_loader_fc = torch.utils.data.DataLoader(
            forecasting_data, batch_size=_fc_bs, shuffle=True)
        val_loader_fc   = torch.utils.data.DataLoader(
            val_fc,           batch_size=_fc_bs, shuffle=True)
        test_loader_fc  = torch.utils.data.DataLoader(
            test_fc,          batch_size=_fc_bs, shuffle=False)
        model.forcast_train = train_loader_fc
        model.forcast_val   = val_loader_fc
        model.forcast_test  = test_loader_fc
        model.config["horizon_t"] = h_t

        # pred_len=96 (first): sweep all checkpoints to find the best.
        # Remaining pred_lens: use only the best checkpoint found above.
        is_search = (pred_len == pred_lens[0])
        ckpts_to_run = ckpts if is_search else [best_ckpt if best_ckpt is not None else ckpts[-1]]

        print(f"\n[JEPA] pred_len={pred_len} (horizon_t={h_t})  modes={modes}"
              + (""  if is_search else f"  [best ckpt={ckpts_to_run[0]}]"))
        for epoch in ckpts_to_run:
            print(f"  → checkpoint epoch {epoch}")
            for mode in modes:
                method_name = _MODE_MAP.get(mode)
                if method_name is None:
                    print(f"  [JEPA] Unknown forecasting mode '{mode}', skipping.")
                    continue
                mse = getattr(model, method_name)(f"_epoch{epoch}")
                # During the pred_len=96 sweep, track the best checkpoint by MSE
                if is_search and mode == modes[0] and mse is not None and mse < best_mse:
                    best_mse  = mse
                    best_ckpt = epoch

        if is_search:
            print(f"\n[JEPA] Best checkpoint at pred_len={pred_lens[0]}: "
                  f"epoch {best_ckpt} (mix MSE={best_mse:.4f})")

    return best_ckpt, best_mse


# ── Discrete JEPA 2 (RevIN, no StandardScaler, denorm forecasting) ────────────

def run_jepa2(skip_train: bool = False,
              pretrain_dataset: str = None,
              forecast_dataset: str = None,
              pred_lens=None,
              checkpoints=None):
    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]

    jepa2_dir = Path(__file__).parent / "Discrete_JEPA_2"
    _add_path(jepa2_dir)

    import torch
    from config_files.config_pretrain import config
    from data_loaders.data_puller import (DataPullerDJepa, ForcastingDataPullerDescrete,
                                          MonashDataPullerJEPA, SyntheticArrowDataPullerJEPA)
    from Discrete_JEPA.Discrete_Jepa import DiscreteJEPA

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = dict(config)

    if pretrain_dataset is not None and pretrain_dataset == forecast_dataset:
        config['pretrain_source'] = None   # force CSV-only mode
        config['val_prec']  = config.get('val_prec_forcasting',  0.1)
        config['test_prec'] = config.get('test_prec_forcasting', 0.1)

    pretrain_source = _resolve_pretrain_source(config)
    use_global_data = pretrain_source is not None

    forecast_dataset = forecast_dataset or config.get("forecast_dataset")
    if use_global_data:
        if forecast_dataset is None:
            raise ValueError("forecast_dataset must be set when pretraining on global data")
        ds_fore = get_dataset_info(forecast_dataset)
        if pretrain_source in ('monash', 'monash+synthetic'):
            monash_dir = config.get('monash_data_dir', '../Monash')
            if not os.path.isabs(monash_dir):
                config['monash_data_dir'] = str((jepa2_dir / monash_dir).resolve())
        if pretrain_source in ('synthetic', 'monash+synthetic'):
            synth_dir = config.get('synthetic_data_dir', '../Monash')
            if not os.path.isabs(synth_dir):
                config['synthetic_data_dir'] = str((jepa2_dir / synth_dir).resolve())
        _src_label = {
            'monash':           f"Monash ({config.get('monash_data_dir', '')})",
            'synthetic':        f"Synthetic ({config.get('synthetic_data_dir', '')})",
            'monash+synthetic': "Monash + Synthetic",
        }.get(pretrain_source, pretrain_source)
        print("\n" + "="*60)
        print(f"  MODEL: Discrete JEPA 2  (RevIN)")
        print(f"  pretrain: {_src_label}   forecast: {forecast_dataset}")
        print("="*60)
    else:
        pretrain_dataset = pretrain_dataset or config.get("pretrain_dataset")
        forecast_dataset = forecast_dataset or pretrain_dataset
        if pretrain_dataset is None:
            raise ValueError("pretrain_dataset not set — specify via run() or config_pretrain.py")
        ds_pre  = get_dataset_info(pretrain_dataset)
        ds_fore = get_dataset_info(forecast_dataset)
        n_groups = len(ds_pre["jepa_groups"])
        config["path_data"]       = [_resolve_jepa_path(ds_pre["csv_path"], jepa2_dir)] * n_groups
        config["timestampcols"]   = [ds_pre["timestamp_col"]] * n_groups
        config["input_variables"] = ds_pre["jepa_groups"]
        print("\n" + "="*60)
        print(f"  MODEL: Discrete JEPA 2  (RevIN)")
        print(f"  pretrain: {pretrain_dataset}   forecast: {forecast_dataset}")
        print("="*60)

    config["path_data_forcasting"]       = [_resolve_jepa_path(ds_fore["csv_path"], jepa2_dir)]
    config["timestampcols_forcasting"]   = [ds_fore["timestamp_col"]]
    config["input_variables_forcasting"] = [ds_fore["columns"]]

    print("\n[JEPA2] Loading datasets …")
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
            data_paths         = config["path_data"],
            patch_size         = config["patch_size"],
            batch_size         = config["batch_size"],
            ratio_patches      = config["ratio_patches"],
            mask_ratio         = config["mask_ratio"],
            masking_type       = config["masking_type"],
            num_semantic_tokens= config["num_semantic_tokens"],
            input_variables    = config["input_variables"],
            timestamp_cols     = config["timestampcols"],
            type_data          = "train",
            val_prec           = config["val_prec"],
            test_prec          = config["test_prec"],
            stride             = config.get("stride", None),
            num_blocks         = config.get("num_blocks", 1),
        )
        val_dataset  = copy.copy(train_dataset); val_dataset.which  = "val"
        test_dataset = copy.copy(train_dataset); test_dataset.which = "test"

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=config["batch_size"], shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=config["batch_size"], shuffle=False)
    input_dim    = len(train_loader.dataset[0][0][0])

    forecasting_data = ForcastingDataPullerDescrete(config)
    val_fc   = copy.copy(forecasting_data); val_fc.which  = "val";  val_fc.rebuild()
    test_fc  = copy.copy(forecasting_data); test_fc.which = "test"; test_fc.rebuild()
    _fc_bs = config.get("batch_size_forecast", 256)
    train_loader_fc = torch.utils.data.DataLoader(forecasting_data, batch_size=_fc_bs, shuffle=True)
    val_loader_fc   = torch.utils.data.DataLoader(val_fc,           batch_size=_fc_bs, shuffle=True)
    test_loader_fc  = torch.utils.data.DataLoader(test_fc,          batch_size=_fc_bs, shuffle=False)

    model = DiscreteJEPA(
        config            = config,
        input_dim         = input_dim,
        num_patches       = len(train_loader.dataset[0][0]),
        steps_per_epoch   = len(train_loader),
        train_loader      = train_loader,
        val_loader        = val_loader,
        test_loader       = test_loader,
        forcasting_train  = train_loader_fc,
        forcasting_val    = val_loader_fc,
        forcasting_test   = test_loader_fc,
    )

    if not skip_train:
        print("\n[JEPA2] Starting pretraining …")
        model.train_and_evaluate()
    else:
        print("[JEPA2] Skipping pretraining.")

    modes = config.get("forecasting_modes", ["zeroshot"])
    _MODE_MAP = {
        "zeroshot":          "forcasting_zeroshot",
        "finetuning":        "finetuning_forecasting",
        "predictor":         "predictor_forecasting",
        "predictor_s2p_p2p": "predictor_s2p_p2p_forecasting",
    }
    ckpts = checkpoints if checkpoints is not None else [40, 60, 90, 120, 160]
    p_s = config["patch_size_forcasting"]
    best_ckpt = None
    best_mse  = float('inf')

    for pred_len in pred_lens:
        h_t = pred_len // p_s
        for ds in [forecasting_data, val_fc, test_fc]:
            ds.h = h_t
            ds.target_raw_len = h_t * p_s
        _fc_bs = config.get("batch_size_forecast", 256)
        train_loader_fc = torch.utils.data.DataLoader(
            forecasting_data, batch_size=_fc_bs, shuffle=True)
        val_loader_fc   = torch.utils.data.DataLoader(
            val_fc,           batch_size=_fc_bs, shuffle=True)
        test_loader_fc  = torch.utils.data.DataLoader(
            test_fc,          batch_size=_fc_bs, shuffle=False)
        model.forcast_train = train_loader_fc
        model.forcast_val   = val_loader_fc
        model.forcast_test  = test_loader_fc
        model.config["horizon_t"] = h_t

        is_search = (pred_len == pred_lens[0])
        ckpts_to_run = ckpts if is_search else [best_ckpt if best_ckpt is not None else ckpts[-1]]

        print(f"\n[JEPA2] pred_len={pred_len} (horizon_t={h_t})  modes={modes}"
              + ("" if is_search else f"  [best ckpt={ckpts_to_run[0]}]"))
        for epoch in ckpts_to_run:
            print(f"  → checkpoint epoch {epoch}")
            for mode in modes:
                method_name = _MODE_MAP.get(mode)
                if method_name is None:
                    print(f"  [JEPA2] Unknown forecasting mode '{mode}', skipping.")
                    continue
                mse = getattr(model, method_name)(f"_epoch{epoch}")
                if is_search and mode == modes[0] and mse is not None and mse < best_mse:
                    best_mse  = mse
                    best_ckpt = epoch

        if is_search:
            print(f"\n[JEPA2] Best checkpoint at pred_len={pred_lens[0]}: "
                  f"epoch {best_ckpt} (mix MSE={best_mse:.4f})")

    return best_ckpt, best_mse


# ── JEPA (P2P only, no VQ / semantic tokens) ─────────────────────────────────

def run_jepa_simple(skip_train: bool = False,
                    pretrain_dataset: str = None,
                    forecast_dataset: str = None,
                    classification_dataset=None,
                    anomaly_dataset: str = None,
                    pred_lens=None,
                    checkpoints=None,
                    pretrain_only: bool = False,
                    encoder_layers: int = None,
                    predictor_layers: int = None,
                    lr: float = None):
    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]

    jepa_dir  = Path(__file__).parent / "JEPA"
    djepa_dir = Path(__file__).parent / "Discrete_JEPA"
    _add_path(jepa_dir)
    _add_path(djepa_dir)   # for data_loaders (shared with Discrete JEPA)

    import importlib.util, torch

    # Load config by file path to avoid sys.modules cache conflicts
    _spec = importlib.util.spec_from_file_location(
        "config_jepa", jepa_dir / "config_files" / "config_jepa.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    config = dict(_mod.config)
    if encoder_layers is not None:
        config['num_encoder_layers'] = encoder_layers
        config['path_save'] = config.get('path_save', './output_model/JEPA/').rstrip('/') + f'_layers{encoder_layers}/'
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

    forecast_dataset = forecast_dataset or config.get("forecast_dataset")
    if use_global_data:
        if not pretrain_only and forecast_dataset is None:
            raise ValueError("forecast_dataset must be set when pretraining on global data")
        if pretrain_source in ('monash', 'monash+synthetic'):
            monash_dir = config.get('monash_data_dir', '../Monash')
            if not os.path.isabs(monash_dir):
                config['monash_data_dir'] = str((jepa_dir / monash_dir).resolve())
        if pretrain_source in ('synthetic', 'monash+synthetic'):
            synth_dir = config.get('synthetic_data_dir', '../Monash')
            if not os.path.isabs(synth_dir):
                config['synthetic_data_dir'] = str((jepa_dir / synth_dir).resolve())
        _src_label = {
            'monash':           f"Monash ({config.get('monash_data_dir', '')})",
            'synthetic':        f"Synthetic ({config.get('synthetic_data_dir', '')})",
            'monash+synthetic': "Monash + Synthetic",
        }.get(pretrain_source, pretrain_source)
        print("\n" + "="*60)
        print(f"  MODEL: JEPA (P2P)")
        if pretrain_only:
            print(f"  pretrain: {_src_label}  [pretrain only]")
        else:
            ds_fore = get_dataset_info(forecast_dataset)
            print(f"  pretrain: {_src_label}   forecast: {forecast_dataset}")
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

    if not pretrain_only:
        config["path_data_forcasting"]       = [_resolve_jepa_path(ds_fore["csv_path"], jepa_dir)]
        config["timestampcols_forcasting"]   = [ds_fore["timestamp_col"]]
        config["input_variables_forcasting"] = [ds_fore["columns"]]

    # ── data ─────────────────────────────────────────────────────────────────
    if skip_train and use_global_data:
        # Skip loading pretrain data entirely — only need forecasting loaders
        print("\n[JEPA simple] skip_train=True + global pretrain: skipping pretrain data load.")
        input_dim       = config["patch_size"]  # univariate: input_dim = patch_size
        num_patches     = config["ratio_patches"]
        train_loader = val_loader = test_loader = None
    else:
        print("\n[JEPA simple] Loading datasets …")
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
    if pretrain_only:
        train_loader_fc = val_loader_fc = test_loader_fc = None
    else:
        _csv = config["path_data_forcasting"][0]
        _p_s = config["patch_size_forcasting"]
        _pl0 = pred_lens[0] if pred_lens else 96
        _fc_bs = config.get("batch_size_forecast", 256)
        train_loader_fc = torch.utils.data.DataLoader(
            PatchTSTForcastingAdapter(_csv, 'train', _PATCHTST_SEQ_LEN, _pl0, _p_s),
            batch_size=_fc_bs, shuffle=True)
        val_loader_fc = torch.utils.data.DataLoader(
            PatchTSTForcastingAdapter(_csv, 'val',   _PATCHTST_SEQ_LEN, _pl0, _p_s),
            batch_size=_fc_bs, shuffle=False)
        test_loader_fc = torch.utils.data.DataLoader(
            PatchTSTForcastingAdapter(_csv, 'test',  _PATCHTST_SEQ_LEN, _pl0, _p_s),
            batch_size=_fc_bs, shuffle=False)

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
        print("\n[JEPA simple] Pretrain-only mode — skipping forecasting.")
        return

    # ── forecasting downstream ────────────────────────────────────────────────
    ckpts = checkpoints if checkpoints is not None else [80, 120, 160, 200, 240, 300]
    p_s = config["patch_size_forcasting"]
    _csv = config["path_data_forcasting"][0]
    best_ckpt = None
    best_mse  = float('inf')

    for pred_len in pred_lens:
        h_t = pred_len // p_s
        model.forcast_train = torch.utils.data.DataLoader(
            PatchTSTForcastingAdapter(_csv, 'train', _PATCHTST_SEQ_LEN, pred_len, p_s),
            batch_size=config["batch_size"], shuffle=True)
        model.forcast_val = torch.utils.data.DataLoader(
            PatchTSTForcastingAdapter(_csv, 'val',   _PATCHTST_SEQ_LEN, pred_len, p_s),
            batch_size=config["batch_size"], shuffle=False)
        model.forcast_test = torch.utils.data.DataLoader(
            PatchTSTForcastingAdapter(_csv, 'test',  _PATCHTST_SEQ_LEN, pred_len, p_s),
            batch_size=config["batch_size"], shuffle=False)
        model.config["horizon_t"] = h_t

        is_search = (pred_len == pred_lens[0])
        ckpts_to_run = ckpts if is_search else [best_ckpt if best_ckpt is not None else ckpts[-1]]

        print(f"\n[JEPA simple] pred_len={pred_len} (horizon_t={h_t})"
              + ("" if is_search else f"  [best ckpt={ckpts_to_run[0]}]"))
        for epoch in ckpts_to_run:
            print(f"  → checkpoint epoch {epoch}")
            mse = model.forcasting_zeroshot(f"_epoch{epoch}")
            if is_search and mse is not None and mse < best_mse:
                best_mse  = mse
                best_ckpt = epoch

        if is_search:
            print(f"\n[JEPA simple] Best checkpoint at pred_len={pred_lens[0]}: "
                  f"epoch {best_ckpt} (mix MSE={best_mse:.4f})")

    # ── classification downstream ─────────────────────────────────────────────
    cls_acc = None
    if classification_dataset is not None:
        from data_loaders.data_puller import ClassificationDataPuller
        cls_dir  = config.get("classification_data_dir", "/home/shared/datasets/Classification_TS")
        cls_bs   = config.get("batch_size", 64)
        p_s      = config["patch_size_forcasting"]
        _mk = lambda split: torch.utils.data.DataLoader(
            ClassificationDataPuller(cls_dir, classification_dataset, p_s, which=split),
            batch_size=cls_bs, shuffle=(split == "train"))
        cls_train = _mk("train"); cls_val = _mk("val"); cls_test = _mk("test")
        n_classes = cls_train.dataset.n_classes
        ckpt_tag  = f"_epoch{best_ckpt}" if best_ckpt is not None else ""
        cls_acc   = model.classification_zeroshot(ckpt_tag, cls_train, cls_val, cls_test, n_classes)
        print(f"\n{'='*60}")
        print(f"  [JEPA simple] Classification on {classification_dataset}")
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
        anom_result = model.anomaly_zeroshot(ckpt_tag, anom_train, anom_test,
                                             anomaly_ratio=config.get("anomaly_ratio", 1.0))

    return best_ckpt, best_mse, cls_acc, anom_result


# ── PatchTST ──────────────────────────────────────────────────────────────────

def run_patchtst(skip_train: bool = False, pretrain_dataset: str = None, forecast_dataset: str = None,
                 classification_dataset=None, anomaly_dataset: str = None,
                 pretrain_only: bool = False, pred_len: int = None,
                 checkpoints=None, random_encoder: bool = False, encoder_layers: int = None,
                 predictor_layers: int = None, lr: float = None):
    patchtst_dir = Path(__file__).parent / "PatchTST_self_supervised"
    djepa_dir    = Path(__file__).parent / "Discrete_JEPA"
    _add_path(djepa_dir)

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "config_patchtst", patchtst_dir / "config_patchtst.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    cfg = dict(_mod.config)
    if encoder_layers is not None:
        cfg['n_layers'] = encoder_layers
        cfg['pretrained_model_id'] = encoder_layers  # unique checkpoint per layer config

    pretrain_source = _resolve_pretrain_source(cfg)
    _pretrain_dset  = pretrain_dataset or cfg.get("pretrain_dataset", "ettm1")
    _forecast_dset  = forecast_dataset or cfg.get("forecast_dataset") or _pretrain_dset

    # pretrain_source overrides _pretrain_dset for global data
    if pretrain_source in ('monash', 'synthetic', 'monash+synthetic'):
        _pretrain_dset = pretrain_source

    monash_dir = synth_dir = None
    if _pretrain_dset in ('monash', 'monash+synthetic'):
        monash_dir = cfg.get("monash_data_dir", "../Monash")
        if not os.path.isabs(monash_dir):
            monash_dir = str((patchtst_dir / monash_dir).resolve())
    if _pretrain_dset in ('synthetic', 'monash+synthetic'):
        synth_dir = cfg.get("synthetic_data_dir", "../synthetic")
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

    patchtst_dir = str(patchtst_dir.resolve())

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
        "--seed",                str(GLOBAL_SEED),
    ]
    if monash_dir is not None:
        pretrain_cmd += ["--monash_data_dir",    monash_dir,
                         "--monash_min_len",      str(cfg.get("monash_min_len", 512))]
    if synth_dir is not None:
        pretrain_cmd += ["--synthetic_data_dir", synth_dir]
    if lr is not None:
        pretrain_cmd += ["--lr", str(lr)]

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
    pretrained_model_path = os.path.join(
        patchtst_dir, "saved_models", _pretrain_dset,
        "masked_patchtst", cfg.get("model_type", "based_model"),
        f"layers{cfg['n_layers']}", model_fname
    )

    if pretrain_only:
        print("\n[PatchTST] Pretrain-only mode — skipping forecasting.")
        return

    # ── forecasting downstream ────────────────────────────────────────────────
    _target_points = pred_len if pred_len is not None else cfg.get("target_points", 96)
    print(f"\n[PatchTST] Running forecasting fine-tuning on {_forecast_dset} (target_points={_target_points}) …")
    result = subprocess.run(
        [sys.executable, "patchtst_finetune.py",
         "--dset_finetune",      _forecast_dset,
         "--is_linear_probe",    "1",
         "--context_points",  str(cfg.get("context_points", 512)),
         "--patch_len",       str(cfg.get("patch_len", 16)),
         "--stride",          str(cfg.get("stride", 16)),
         "--n_layers",        str(cfg.get("n_layers", 3)),
         "--n_heads",         str(cfg.get("n_heads", 16)),
         "--d_model",         str(cfg.get("d_model", 128)),
         "--d_ff",            str(cfg.get("d_ff", 512)),
         "--dropout",         str(cfg.get("dropout", 0.2)),
         "--head_dropout",    str(cfg.get("head_dropout", 0.2)),
         "--target_points",   str(_target_points),
         "--pretrained_model", pretrained_model_path,
         "--random_encoder",   str(int(random_encoder)),
         "--batch_size",       str(cfg.get("batch_size_forecast", 256)),
         "--num_workers",      str(cfg.get("num_workers", 0)),
         "--lr",               str(cfg.get("finetune_lr", 1e-4)),
         "--seed",             "42"],
        cwd=patchtst_dir, capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[PatchTST] Forecasting fine-tuning exited with errors.")
        print(result.stderr)
        return None

    # Parse MSE from the saved _acc.csv — match on target_points to avoid reading wrong pred_len
    import glob as _glob
    acc_files = _glob.glob(os.path.join(
        patchtst_dir, "saved_models", _forecast_dset,
        "masked_patchtst", cfg.get("model_type", "based_model"),
        f"*_tw{_target_points}_*_acc.csv"
    ))
    mse_val = None
    if acc_files:
        import pandas as _pd
        acc = _pd.read_csv(sorted(acc_files)[-1])
        mse_val = float(acc["mse"].iloc[0])
        print(f"[PatchTST] MSE on {_forecast_dset}: {mse_val:.4f}")

    # ── classification downstream ─────────────────────────────────────────────
    cls_acc = None
    if classification_dataset is not None:
        sys.path.insert(0, patchtst_dir)
        from patchtst_classification import classification_zeroshot as ptst_classify
        from data_loaders.data_puller import ClassificationDataPuller
        cls_dir = cfg.get("classification_data_dir", "/home/shared/datasets/Classification_TS")
        cls_bs  = cfg.get("batch_size", 64)
        p_s     = cfg.get("patch_len", 16)
        _mk = lambda split: torch.utils.data.DataLoader(
            ClassificationDataPuller(cls_dir, classification_dataset, p_s, which=split),
            batch_size=cls_bs, shuffle=(split == "train"))
        cls_train = _mk("train"); cls_val = _mk("val"); cls_test = _mk("test")
        n_classes = cls_train.dataset.n_classes
        cls_acc = ptst_classify(cfg, pretrained_model_path, cls_train, cls_val, cls_test, n_classes)
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
                                   anomaly_ratio=cfg.get("anomaly_ratio", 1.0))

    return mse_val, cls_acc, anom_result


# ── NPT (NTP pretraining on PatchTST) ─────────────────────────────────────────

def run_ntp(skip_train: bool = False, pretrain_dataset: str = None, forecast_dataset: str = None,
            classification_dataset=None, anomaly_dataset: str = None,
            pretrain_only: bool = False, pred_len: int = None,
            checkpoints=None, encoder_layers: int = None, predictor_layers: int = None,
            lr: float = None):
    npt_dir = Path(__file__).parent / "NPT"
    _add_path(npt_dir)

    import importlib.util
    _spec = importlib.util.spec_from_file_location("config_ntp", npt_dir / "config_ntp.py")
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    cfg = dict(_mod.config)
    if encoder_layers is not None:
        cfg['n_layers'] = encoder_layers
        cfg['pretrained_model_id'] = encoder_layers  # unique checkpoint per layer config
    if lr is not None:
        cfg['lr'] = lr

    _pretrain_source = _resolve_pretrain_source(cfg)
    _pretrain_dset   = pretrain_dataset or cfg.get("pretrain_dataset", "monash")
    if _pretrain_source in ('monash', 'synthetic', 'monash+synthetic'):
        _pretrain_dset = _pretrain_source
    _forecast_dset = None if pretrain_only else (forecast_dataset or cfg.get("forecast_dataset") or _pretrain_dset)
    cfg["pretrain_dataset"] = _pretrain_dset
    cfg["forecast_dataset"] = _forecast_dset

    if _pretrain_dset in ('monash', 'monash+synthetic'):
        monash_dir = cfg.get("monash_data_dir", "../Monash")
        if not os.path.isabs(monash_dir):
            cfg["monash_data_dir"] = str((npt_dir / monash_dir).resolve())
    if _pretrain_dset in ('synthetic', 'monash+synthetic'):
        synth_dir = cfg.get("synthetic_data_dir", "../synthetic")
        if not os.path.isabs(synth_dir):
            cfg["synthetic_data_dir"] = str((npt_dir / synth_dir).resolve())

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
    _save_dir = npt_dir / "saved_models" / _pretrain_dset / "ntp" / f"layers{cfg['n_layers']}"
    _base_name = _model_fname(cfg, _pretrain_dset)
    _ckpt_epoch = checkpoints[0] if (checkpoints and checkpoints[0] is not None) else None
    if _ckpt_epoch is not None:
        _ckpt_path = str(_save_dir / f"{_base_name}_epoch{_ckpt_epoch}.pt")
    else:
        _ckpt_path = str(_save_dir / f"{_base_name}.pt")

    if not skip_train:
        print(f"\n[NPT] Starting NTP pretraining on {_pretrain_dset} …")
        _ckpt_path = pretrain_ntp(cfg)   # returns best-model path
    else:
        print("[NPT] Skipping pretraining.")

    mse_trained = None
    if _forecast_dset:
        if pred_len is not None:
            cfg["horizon_t"] = pred_len // cfg["patch_size"]

        print(f"\n[NPT] Running zero-shot forecasting on {_forecast_dset} …")
        mse_trained, mae_trained = zeroshot_forecasting(cfg, _ckpt_path)

        if mse_trained is not None:
            print(f"\n{'='*60}")
            print(f"  Results on {_forecast_dset}")
            print(f"  {'':20s}  {'MSE':>8}  {'MAE':>8}")
            print(f"  {'NPT (pretrained)':20s}  {mse_trained:8.4f}  {mae_trained:8.4f}")
            print(f"{'='*60}")
    else:
        print("[NPT] No forecast_dataset set — skipping forecasting.")

    # ── classification downstream ─────────────────────────────────────────────
    cls_acc = None
    if classification_dataset is not None:
        from ntp_classification import classification_zeroshot as npt_classify
        from data_loaders.data_puller import ClassificationDataPuller
        cls_dir = cfg.get("classification_data_dir", "/home/shared/datasets/Classification_TS")
        cls_bs  = cfg.get("batch_size", 64)
        p_s     = cfg["patch_size"]
        _mk = lambda split: torch.utils.data.DataLoader(
            ClassificationDataPuller(cls_dir, classification_dataset, p_s, which=split),
            batch_size=cls_bs, shuffle=(split == "train"))
        cls_train = _mk("train"); cls_val = _mk("val"); cls_test = _mk("test")
        n_classes = cls_train.dataset.n_classes
        cls_acc = npt_classify(cfg, _ckpt_path, cls_train, cls_val, cls_test, n_classes)
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
                                  anomaly_ratio=cfg.get("anomaly_ratio", 1.0))

    return mse_trained, cls_acc, anom_result


# ── LE-JEPA ───────────────────────────────────────────────────────────────────

def run_lejepa(skip_train: bool = False,
               pretrain_dataset: str = None,
               forecast_dataset: str = None,
               classification_dataset=None,
               anomaly_dataset: str = None,
               pretrain_only: bool = False,
               pred_lens=None,
               checkpoints=None,
               encoder_layers: int = None,
               predictor_layers: int = None,
               lr: float = None):
    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]

    lejepa_dir = Path(__file__).parent / "LE-JEPA"
    djepa_dir  = Path(__file__).parent / "Discrete_JEPA"
    _add_path(lejepa_dir)
    _add_path(djepa_dir)   # DataPullerDJepa + MonashDataPullerJEPA + ForcastingDataPullerDescrete

    import importlib.util, torch

    _spec = importlib.util.spec_from_file_location(
        "config_lejepa", lejepa_dir / "config_lejepa.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    config = dict(_mod.config)
    if encoder_layers is not None:
        config['num_encoder_layers'] = encoder_layers
        config['path_save'] = config.get('path_save', './output_model/LE-JEPA/').rstrip('/') + f'_layers{encoder_layers}/'
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

    forecast_dataset = forecast_dataset or config.get('forecast_dataset')

    # Single-dataset mode: align splits so test never leaks into pretraining
    if pretrain_dataset and pretrain_dataset == forecast_dataset:
        config['pretrain_source'] = None   # force CSV-only mode
        config['val_prec']  = config.get('val_prec_forcasting',  0.1)
        config['test_prec'] = config.get('test_prec_forcasting', 0.1)

    pretrain_source = _resolve_pretrain_source(config)
    use_global_data = pretrain_source is not None

    if use_global_data:
        if not pretrain_only and forecast_dataset is None:
            raise ValueError("forecast_dataset must be set when pretraining on global data")
        if pretrain_source in ('monash', 'monash+synthetic'):
            monash_dir = config.get('monash_data_dir', '../Monash')
            if not os.path.isabs(monash_dir):
                config['monash_data_dir'] = str((lejepa_dir / monash_dir).resolve())
        if pretrain_source in ('synthetic', 'monash+synthetic'):
            synth_dir = config.get('synthetic_data_dir', '../Monash')
            if not os.path.isabs(synth_dir):
                config['synthetic_data_dir'] = str((lejepa_dir / synth_dir).resolve())
        _src_label = {
            'monash':           f"Monash ({config.get('monash_data_dir', '')})",
            'synthetic':        f"Synthetic ({config.get('synthetic_data_dir', '')})",
            'monash+synthetic': "Monash + Synthetic",
        }.get(pretrain_source, pretrain_source)
        print("\n" + "="*60)
        print(f"  MODEL: LE-JEPA")
        if pretrain_only:
            print(f"  pretrain: {_src_label}  [pretrain only]")
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

    if not pretrain_only:
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

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,  num_workers=0)
        val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=config["batch_size"], shuffle=False, num_workers=0)
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=config["batch_size"], shuffle=False, num_workers=0)

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
    # Use PatchTST-identical data: seq_len=336 (21 patches × 16)
    _PATCHTST_SEQ_LEN = 336
    p_s  = config["patch_size_forcasting"]
    _ctx_patches = _PATCHTST_SEQ_LEN // p_s   # = 21
    config["forecasting_context_patches"] = _ctx_patches
    _csv = config["path_data_forcasting"][0]

    best_mse  = float('inf')
    best_ckpt = None
    ckpts     = checkpoints if checkpoints is not None else list(range(config["num_epochs"]))

    for pred_len in pred_lens:
        h_t = pred_len // p_s
        model.forcast_train = torch.utils.data.DataLoader(
            PatchTSTForcastingAdapter(_csv, 'train', _PATCHTST_SEQ_LEN, pred_len, p_s),
            batch_size=config["batch_size"], shuffle=True,  num_workers=0)
        model.forcast_val   = torch.utils.data.DataLoader(
            PatchTSTForcastingAdapter(_csv, 'val',   _PATCHTST_SEQ_LEN, pred_len, p_s),
            batch_size=config["batch_size"], shuffle=False, num_workers=0)
        model.forcast_test  = torch.utils.data.DataLoader(
            PatchTSTForcastingAdapter(_csv, 'test',  _PATCHTST_SEQ_LEN, pred_len, p_s),
            batch_size=config["batch_size"], shuffle=False, num_workers=0)
        model.config["horizon_t"] = h_t

        is_search    = (pred_len == pred_lens[0])
        ckpts_to_run = ckpts if is_search else [best_ckpt if best_ckpt is not None else ckpts[-1]]

        print(f"\n[LE-JEPA] pred_len={pred_len} (horizon_t={h_t})"
              + ("" if is_search else f"  [best ckpt={ckpts_to_run[0]}]"))

        for epoch in ckpts_to_run:
            print(f"  → checkpoint epoch {epoch}")
            mse = model.forcasting_zeroshot(f"_epoch{epoch}")
            if is_search and mse is not None and mse < best_mse:
                best_mse  = mse
                best_ckpt = epoch

        if is_search:
            print(f"\n[LE-JEPA] Best checkpoint at pred_len={pred_lens[0]}: "
                  f"epoch {best_ckpt} (MSE={best_mse:.4f})")

    # ── classification downstream ─────────────────────────────────────────────
    cls_acc = None
    if classification_dataset is not None:
        from data_loaders.data_puller import ClassificationDataPuller
        cls_dir = config.get("classification_data_dir", "/home/shared/datasets/Classification_TS")
        cls_bs  = config.get("batch_size", 64)
        p_s     = config["patch_size_forcasting"]
        _mk = lambda split: torch.utils.data.DataLoader(
            ClassificationDataPuller(cls_dir, classification_dataset, p_s, which=split),
            batch_size=cls_bs, shuffle=(split == "train"))
        cls_train = _mk("train"); cls_val = _mk("val"); cls_test = _mk("test")
        n_classes = cls_train.dataset.n_classes
        ckpt_tag  = f"_epoch{best_ckpt}" if best_ckpt is not None else ""
        cls_acc   = model.classification_zeroshot(ckpt_tag, cls_train, cls_val, cls_test, n_classes)
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
                                             anomaly_ratio=config.get("anomaly_ratio", 1.0))

    return best_ckpt, best_mse, cls_acc, anom_result


# ── Random baseline ───────────────────────────────────────────────────────────

def run_random(skip_train: bool = False, pretrain_dataset: str = None, forecast_dataset: str = None):
    random_dir = Path(__file__).parent / "random"
    npt_dir    = Path(__file__).parent / "NPT"
    _add_path(random_dir)
    _add_path(npt_dir)

    import importlib.util
    _spec = importlib.util.spec_from_file_location("config_ntp", npt_dir / "config_ntp.py")
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
    "jepa2":           run_jepa2,
    "jepa_simple":     run_jepa_simple,
    "lejepa":          run_lejepa,
    "patchtst":        run_patchtst,
    "patchtst_random": lambda skip_train=False, pretrain_dataset=None, forecast_dataset=None, classification_dataset=None, pretrain_only=False, pred_len=None, checkpoints=None, encoder_layers=None: run_patchtst(skip_train=skip_train, pretrain_dataset=pretrain_dataset, forecast_dataset=forecast_dataset, classification_dataset=classification_dataset, pretrain_only=pretrain_only, pred_len=pred_len, checkpoints=checkpoints, random_encoder=True, encoder_layers=encoder_layers),
    "npt":             run_ntp,
    "random":          run_random,
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
        lr: float = None):
    """
    Unified entry point. Each run handles ONE task.

    task="pretrain"   — pretrain only (same as pretrain_only=True)
    task="forecast"   — skip pretraining, run forecasting only (same as skip_train=True)
    task="classify"   — skip pretraining, run classification only

    Backwards compatible — old flags (skip_train, pretrain_only) still work when task=None.

    Examples:
        run(model="jepa_simple", task="pretrain")
        run(model="jepa_simple", task="forecast", forecast_dataset="etth1", skip_train=True)
        run(model="jepa_simple", task="classify",
            classification_dataset="EthanolConcentration", skip_train=True)

        # Old style still works:
        run(model="jepa_simple", skip_train=False)
        run(model="jepa_simple", pretrain_only=True)
    """
    # ── resolve task → old flags (backwards compat) ───────────────────────────
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
            skip_train    = True
            pretrain_only = False
            forecast_dataset  = None        # force no forecasting
            anomaly_dataset   = None
        elif task == "anomaly":
            skip_train    = True
            pretrain_only = False
            forecast_dataset       = None
            classification_dataset = None
        else:
            raise ValueError(f"Unknown task '{task}'. Choose: pretrain | forecast | classify | anomaly")

    _set_seed()
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
    if 'pred_lens'              in sig.parameters: kwargs['pred_lens']              = pred_lens
    if 'checkpoints'            in sig.parameters: kwargs['checkpoints']            = checkpoints
    if 'pred_len'               in sig.parameters: kwargs['pred_len']               = pred_len
    if 'encoder_layers'         in sig.parameters: kwargs['encoder_layers']         = encoder_layers
    if 'predictor_layers'       in sig.parameters: kwargs['predictor_layers']       = predictor_layers
    if 'lr'                     in sig.parameters: kwargs['lr']                     = lr
    if 'classification_dataset' in sig.parameters: kwargs['classification_dataset'] = classification_dataset
    if 'anomaly_dataset'        in sig.parameters: kwargs['anomaly_dataset']        = anomaly_dataset
    if 'checkpoint'             in sig.parameters: kwargs['checkpoint']             = checkpoint
    return runner(**kwargs)


if __name__ == "__main__":
    from dataset_registry import DATASETS as _DATASETS
    parser = argparse.ArgumentParser(description="Unified training + forecasting runner")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(RUNNERS),
        help="Which model to run: dino | jepa | jepa2 | jepa_simple | lejepa | patchtst | npt | random",
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
                        choices=["pretrain", "forecast", "classify"],
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
        lr=args.lr)
