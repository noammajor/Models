"""
LE-JEPA pretraining entry point.

Run from the LE-JEPA/ directory:
    python main.py [--pretrain_dataset etth1] [--forecast_dataset etth1]
                   [--skip_train] [--pretrain_only]
                   [--pred_lens 96 192 336 720]

Data is loaded via DataPullerDJepa (shared data loaders).
Masks returned by DataPullerDJepa are ignored — LE-JEPA uses augmentation,
not masking, to produce two views.
"""

import sys
import os
import copy
import argparse
import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent.resolve()
_ROOT      = _HERE.parent
_DATALOADER_DIR = _ROOT / "Utils"

def _add(p):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

_add(_HERE)
_add(_ROOT)
_add(_DATALOADER_DIR)   # DataPullerDJepa + MonashDataPullerJEPA + ForcastingDataPullerDescrete


# ── Imports that depend on sys.path ───────────────────────────────────────────
from data_loaders.data_puller import (
    DataPullerDJepa,
    ForcastingDataPullerDescrete,
    MonashDataPullerJEPA,
)
from dataset_registry import get_dataset_info                       # root registry
from LeJepa import LeJEPA


def _resolve(rel_path: str) -> str:
    """Make a path absolute relative to LE-JEPA/."""
    p = Path(rel_path)
    return str((_HERE / p).resolve()) if not p.is_absolute() else rel_path


def _load_config():
    spec = importlib.util.spec_from_file_location(
        "config_lejepa", _HERE / "config_lejepa.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_dataset',  type=str, default=None)
    parser.add_argument('--forecast_dataset',  type=str, default=None)
    parser.add_argument('--skip_train',        action='store_true')
    parser.add_argument('--pretrain_only',     action='store_true')
    parser.add_argument('--pred_lens',         type=int, nargs='+', default=[96, 192, 336, 720])
    parser.add_argument('--checkpoint',        type=str, default=None,
                        help='Path to pretrained encoder checkpoint (.pt) for skip_train mode')
    args = parser.parse_args()

    config = _load_config()

    # ── Dataset routing ───────────────────────────────────────────────────────
    pretrain_on_monash = config.get('pretrain_on_monash', False)
    pretrain_dataset   = args.pretrain_dataset or config.get('pretrain_dataset')
    forecast_dataset   = args.forecast_dataset or config.get('forecast_dataset') or pretrain_dataset

    # Single-dataset mode: align splits so test never leaks into pretraining
    if pretrain_dataset and pretrain_dataset == forecast_dataset:
        config['pretrain_on_monash'] = False
        pretrain_on_monash = False
        config['val_prec']  = config.get('val_prec_forcasting',  0.1)
        config['test_prec'] = config.get('test_prec_forcasting', 0.1)

    if pretrain_on_monash:
        monash_dir = config.get('monash_data_dir', '../Monash')
        if not os.path.isabs(monash_dir):
            config['monash_data_dir'] = str((_HERE / monash_dir).resolve())
        print("\n" + "="*60)
        print(f"  MODEL: LE-JEPA")
        print(f"  pretrain: Monash   forecast: {forecast_dataset or '(pretrain only)'}")
        print("="*60)
    else:
        ds_pre = get_dataset_info(pretrain_dataset)
        n_groups = len(ds_pre["jepa_groups"])
        config["path_data"]       = [_resolve(ds_pre["csv_path"])] * n_groups
        config["timestampcols"]   = [ds_pre["timestamp_col"]] * n_groups
        config["input_variables"] = ds_pre["jepa_groups"]
        print("\n" + "="*60)
        print(f"  MODEL: LE-JEPA")
        print(f"  pretrain: {pretrain_dataset}   forecast: {forecast_dataset}")
        print("="*60)

    if not args.pretrain_only:
        ds_fore = get_dataset_info(forecast_dataset)
        config["path_data_forcasting"]       = [_resolve(ds_fore["csv_path"])]
        config["timestampcols_forcasting"]   = [ds_fore["timestamp_col"]]
        config["input_variables_forcasting"] = [ds_fore["columns"]]

    # ── Pretraining data loaders ──────────────────────────────────────────────
    if args.skip_train and pretrain_on_monash:
        print("[LE-JEPA] skip_train + Monash: skipping Monash data load.")
        # Derive dims from config
        input_dim   = config["patch_size"]
        num_patches = config["ratio_patches"]
        train_loader = val_loader = test_loader = None
    else:
        print("\n[LE-JEPA] Loading pretraining datasets …")
        if pretrain_on_monash:
            train_dataset = MonashDataPullerJEPA(config, which='train')
            val_dataset   = MonashDataPullerJEPA(config, which='val')
            test_dataset  = MonashDataPullerJEPA(config, which='test')
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

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=config["batch_size"], shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_dataset,  batch_size=config["batch_size"], shuffle=False, num_workers=0)

        # Derive dims from the first sample: [P, P_L, C]
        sample = train_dataset[0]
        patches_sample = sample[0] if isinstance(sample, (list, tuple)) else sample
        num_patches, input_dim = patches_sample.shape[0], patches_sample.shape[1]

    # ── Build model ───────────────────────────────────────────────────────────
    model = LeJEPA(
        config      = config,
        input_dim   = input_dim,
        num_patches = num_patches,
        train_loader= train_loader,
        val_loader  = val_loader,
        test_loader = test_loader,
    )
    model.to(model.device)

    # Load checkpoint if skip_train
    if args.skip_train and args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=model.device)
        state = ckpt.get("encoder", ckpt)
        model.encoder.load_state_dict(state)
        print(f"[LE-JEPA] Loaded encoder from {args.checkpoint}")

    # ── Pretraining ───────────────────────────────────────────────────────────
    if not args.skip_train:
        print("\n[LE-JEPA] Starting pretraining …")
        model.train_and_evaluate()
    else:
        print("[LE-JEPA] Skipping pretraining.")

    if args.pretrain_only:
        print("[LE-JEPA] Pretrain-only mode — done.")
        return

    # ── Forecasting downstream ────────────────────────────────────────────────
    # Reuse JEPA's zeroshot forecasting via frozen encoder + linear head.
    # Import from the shared JEPA module.
    _add(_ROOT / "JEPA")
    from JEPA.Forecasting import forcasting_zeroshot

    print("\n[LE-JEPA] Running downstream forecasting …")
    p_s = config["patch_size_forcasting"]

    fore_data = ForcastingDataPullerDescrete(config)
    val_fc    = copy.copy(fore_data); val_fc.which  = "val";  val_fc.rebuild()
    test_fc   = copy.copy(fore_data); test_fc.which = "test"; test_fc.rebuild()

    results = {}
    for pred_len in args.pred_lens:
        h_t = pred_len // p_s
        for ds in [fore_data, val_fc, test_fc]:
            ds.h = h_t
            ds.target_raw_len = h_t * p_s

        fc_train = DataLoader(fore_data, batch_size=config["batch_size"], shuffle=True,  num_workers=0)
        fc_val   = DataLoader(val_fc,    batch_size=config["batch_size"], shuffle=False, num_workers=0)
        fc_test  = DataLoader(test_fc,   batch_size=config["batch_size"], shuffle=False, num_workers=0)

        fore_cfg = dict(config)
        fore_cfg["horizon_t"] = h_t
        fore_cfg["ratio_patches"] = config.get("context_t", 21)

        mse, mae = forcasting_zeroshot(
            self_obj    = model,
            config      = fore_cfg,
            train_loader= fc_train,
            val_loader  = fc_val,
            test_loader = fc_test,
            encoder     = model.encoder,
        )
        results[pred_len] = {"mse": mse, "mae": mae}
        print(f"  pred_len={pred_len:4d}  MSE={mse:.4f}  MAE={mae:.4f}")

    print("\n[LE-JEPA] Forecasting results:")
    for pl, r in results.items():
        print(f"  {pl:4d}: MSE={r['mse']:.4f}  MAE={r['mae']:.4f}")


if __name__ == "__main__":
    main()
