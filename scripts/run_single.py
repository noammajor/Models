#!/usr/bin/env python3
"""
run_single.py — Pretrain a single model on one in-domain dataset, then forecast.

Saves the best checkpoint to:
    {project_root}/Models/{name}/best_chkp_{dataset}.pt

Then runs in-domain forecasting (pred_lens 96/192/336/720) on the same dataset.

Usage:
    python scripts/run_single.py --model dino --dataset etth1 --name my_run
    python scripts/run_single.py --model patchtst --dataset weather \\
        --name ptst_w8 --layers 8 --embed_dim 256 --lr 1e-4 --gpu 0
    python scripts/run_single.py --model ntp --dataset ettm2 --name ntp_test \\
        --skip_pretrain   # skip training, jump straight to forecasting
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

IN_DOMAIN_DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity"]
ALL_MODELS         = ["dino", "jepa", "lejepa", "patchtst", "ntp", "hybrid", "timedart"]

MODEL_DEFAULT_LR = {
    "dino":     5e-4,
    "jepa":     5e-4,
    "lejepa":   5e-4,
    "patchtst": 5e-5,
    "ntp":      5e-5,
    "hybrid":   5e-4,
    "timedart": 1e-4,
}

_python = sys.executable


# ── checkpoint locator ────────────────────────────────────────────────────────

def _find_src_checkpoint(model: str, dataset: str, layers: int, out_dim: int = None, ckpt_tag: str = None) -> Path:
    """Return the path where Train_and_downstream.py saves the best checkpoint."""
    if model == "dino":
        _outdim_tag = f"_outdim{out_dim}" if out_dim is not None else ''
        _ckpt_tag   = f"_{ckpt_tag}" if ckpt_tag else ''
        return ROOT / f"checkpoints_{dataset}_layers{layers}{_outdim_tag}{_ckpt_tag}" / "checkpoint_best.pth"

    elif model == "jepa":
        return ROOT / "output_model" / f"JEPA_{dataset}_layers{layers}" / "best_model.pt"

    elif model == "lejepa":
        return ROOT / "output_model" / f"LE-JEPA_layers{layers}" / "best_model.pt"

    elif model == "patchtst":
        save_dir = (ROOT / "PatchTST_self_supervised" / "saved_models" /
                    dataset / "masked_patchtst" / "based_model" / f"layers{layers}")
        candidates = [p for p in save_dir.glob("*.pth") if "_epoch" not in p.name]
        return candidates[0] if candidates else save_dir / "checkpoint_best.pth"

    elif model == "ntp":
        save_dir = ROOT / "NTP" / "saved_models" / dataset / "ntp" / f"layers{layers}"
        candidates = [p for p in save_dir.glob("*.pt") if "_epoch" not in p.name and "_losses" not in p.name]
        return candidates[0] if candidates else save_dir / "checkpoint_best.pt"

    elif model == "hybrid":
        # in-domain run: path includes dataset tag
        return ROOT / "output_model" / f"Hybrid_{dataset}_layers{layers}" / "best_model.pt"

    elif model == "timedart":
        return ROOT / "outputs" / f"timedart_pretrain_{dataset}_layers{layers}" / f"monash_{dataset}" / "ckpt_best.pth"

    else:
        raise ValueError(f"Unknown model: {model}")


# ── subprocess launcher ───────────────────────────────────────────────────────

def _run(cmd: list, gpu: int, log_path: Path, dry_run: bool, label: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n[{label}]  GPU={gpu}  log={log_path.relative_to(ROOT)}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return 0
    with open(log_path, "w") as fh:
        fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
        result = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"  → {status}")
    return result.returncode


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Single-run in-domain pretrain + forecasting with a unified checkpoint path"
    )
    parser.add_argument("--model",    required=True, choices=ALL_MODELS,
                        help="Model to run")
    parser.add_argument("--dataset",  required=False, default=None, choices=IN_DOMAIN_DATASETS,
                        help="Dataset to pretrain and forecast on (optional when using --pretrain_source)")
    parser.add_argument("--name",     required=True,
                        help="Run name — checkpoint saved as Models/{name}/best_chkp_{dataset}.pt")
    parser.add_argument("--layers",   type=int, default=8,
                        help="Number of encoder layers (default: 8)")
    parser.add_argument("--step_size", type=int, default=None,
                        help="Patch stride override (DINO). num_patches recomputed to keep context window. step_size<patch_len = overlapping patches; omit to use config.py.")
    parser.add_argument("--embed_dim", type=int, default=None,
                        help="Embedding dim / d_model override")
    parser.add_argument("--predictor_embed_dim", type=int, default=None,
                        help="Predictor embedding dim (JEPA only)")
    parser.add_argument("--predictor_layers", type=int, default=None,
                        help="Number of predictor layers (JEPA/LE-JEPA)")
    parser.add_argument("--out_dim", type=int, default=None,
                        help="DINO output bins / prototype count (DINO only)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of pretraining epochs")
    parser.add_argument("--epochs_forecasting", type=int, default=None,
                        help="Number of forecasting fine-tune epochs (DINO)")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Checkpoint epochs to evaluate during forecasting, e.g. --checkpoints 1 3 5 10")
    parser.add_argument("--lr",       type=float, default=None,
                        help="Pretraining LR (default: model-specific)")
    parser.add_argument("--warmup_epochs", type=int, default=None,
                        help="Number of LR warmup epochs (DINO only)")
    parser.add_argument("--ckpt_tag", type=str, default=None,
                        help="Extra tag appended to checkpoint directory name (e.g. 'wrLR')")
    parser.add_argument("--aug_global", type=str, default=None,
                        help="Global (teacher) augmentation type, overrides config")
    parser.add_argument("--aug_local",  type=str, default=None,
                        help="Local (student) augmentation type, overrides config")
    parser.add_argument("--dwt_wavelet_pool", type=str, default=None,
                        help="Comma-separated wavelet pool, e.g. db4,db6 (DINO only)")
    parser.add_argument("--dwt_wavelet",      type=str, default=None,
                        help="Fixed wavelet override, e.g. db6 (DINO only)")
    parser.add_argument("--pretrain_source", type=str, default=None,
                        choices=["monash", "synthetic", "monash+synthetic"],
                        help="Pretrain data source (DINO only)")
    parser.add_argument("--synthetic_data_dir", type=str, default=None,
                        help="Override synthetic data directory path")
    parser.add_argument("--lr_pred",  type=float, default=None,
                        help="Predictor LR (JEPA only)")
    parser.add_argument("--gpu",      type=int, default=0,
                        help="GPU index via CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--seed",     type=int, default=None,
                        help="Random seed")
    parser.add_argument("--phi",        type=float, default=None,
                        help="LEJEPA/NTP mixing weight φ∈[0,1] (hybrid model only)")
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Skip pretraining, go straight to forecasting")
    parser.add_argument("--dry_run",       action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    if args.dataset is None and args.pretrain_source is None:
        parser.error("--dataset is required unless --pretrain_source is set")
    lr          = args.lr or MODEL_DEFAULT_LR[args.model]
    target_dir  = ROOT / "Models" / args.name
    target_ckpt = target_dir / f"best_chkp_{args.dataset}.pt"
    log_base    = ROOT / "logs" / "single" / args.name

    print(f"\n{'='*60}")
    print(f"  model:     {args.model}")
    print(f"  dataset:   {args.dataset}")
    print(f"  layers:    {args.layers}" +
          (f"   embed_dim: {args.embed_dim}" if args.embed_dim else "") +
          (f"   pred_layers: {args.predictor_layers}" if args.predictor_layers else "") +
          (f"   pred_embed_dim: {args.predictor_embed_dim}" if args.predictor_embed_dim else ""))
    print(f"  lr:        {lr}")
    print(f"  gpu:       {args.gpu}")
    print(f"  ckpt  →    {target_ckpt}")
    if args.skip_pretrain:
        print(f"  [skip_pretrain]")
    if args.dry_run:
        print(f"  [dry_run]")
    print(f"{'='*60}")

    # base flags shared by both pretrain and forecast calls
    base_cmd = [
        _python, str(ROOT / "Train_and_downstream.py"),
        "--model",          args.model,
        "--encoder_layers", str(args.layers),
        "--lr",             str(lr),
    ]
    if args.lr_pred:
        base_cmd += ["--lr_pred", str(args.lr_pred)]
    if args.embed_dim:
        base_cmd += ["--embed_dim", str(args.embed_dim)]
    if args.predictor_embed_dim:
        base_cmd += ["--predictor_embed_dim", str(args.predictor_embed_dim)]
    if args.predictor_layers:
        base_cmd += ["--predictor_layers", str(args.predictor_layers)]
    if args.out_dim:
        base_cmd += ["--out_dim", str(args.out_dim)]
    if args.epochs:
        base_cmd += ["--epochs", str(args.epochs)]
    if args.epochs_forecasting:
        base_cmd += ["--epochs_forecasting", str(args.epochs_forecasting)]
    if args.seed is not None:
        base_cmd += ["--seed", str(args.seed)]
    if args.warmup_epochs is not None:
        base_cmd += ["--warmup_epochs", str(args.warmup_epochs)]
    if args.ckpt_tag is not None:
        base_cmd += ["--ckpt_tag", args.ckpt_tag]
    if args.step_size is not None:
        base_cmd += ["--step_size", str(args.step_size)]
    if args.aug_global is not None:
        base_cmd += ["--aug_global", args.aug_global]
    if args.aug_local is not None:
        base_cmd += ["--aug_local", args.aug_local]
    if args.phi is not None:
        base_cmd += ["--phi", str(args.phi)]
    if args.pretrain_source is not None:
        base_cmd += ["--pretrain_source", args.pretrain_source]
    if args.synthetic_data_dir is not None:
        base_cmd += ["--synthetic_data_dir", args.synthetic_data_dir]
    if args.dwt_wavelet_pool is not None:
        base_cmd += ["--dwt_wavelet_pool", args.dwt_wavelet_pool]
    if args.dwt_wavelet is not None:
        base_cmd += ["--dwt_wavelet", args.dwt_wavelet]

    # ── pretrain ──────────────────────────────────────────────────────────────
    if not args.skip_pretrain:
        _pretrain_extra = ["--pretrain_only", "true"]
        if args.dataset is not None:
            # Pin forecast_dataset == pretrain_dataset so in-domain pretraining
            # doesn't fall back to the config's default forecast dataset (which
            # would otherwise get mixed into DINO's CSV pretraining set).
            _pretrain_extra += ["--pretrain_dataset", args.dataset,
                                "--forecast_dataset", args.dataset]
        rc = _run(
            base_cmd + _pretrain_extra,
            args.gpu,
            log_base / "pretrain.log",
            args.dry_run,
            f"pretrain/{args.model}/{args.dataset or 'global'}",
        )
        if rc != 0:
            print("\nPretraining failed — aborting.")
            sys.exit(rc)

    # ── copy best checkpoint to unified path ──────────────────────────────────
    if args.dataset is None:
        print("\nPretrain-only run (no dataset) — skipping checkpoint copy and forecasting.")
        sys.exit(0)
    src_ckpt = _find_src_checkpoint(args.model, args.dataset, args.layers, args.out_dim, args.ckpt_tag)
    if not args.dry_run:
        if src_ckpt.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_ckpt, target_ckpt)
            print(f"\nCheckpoint → {target_ckpt}")
        else:
            print(f"\nWarning: expected checkpoint not found at {src_ckpt}")
    else:
        print(f"\n[dry_run] copy  {src_ckpt}\n         →     {target_ckpt}")

    # ── forecast ──────────────────────────────────────────────────────────────
    forecast_cmd = base_cmd + [
        "--task",             "forecast",
        "--pretrain_dataset", args.dataset,
        "--forecast_dataset", args.dataset,
    ]
    if args.checkpoints:
        forecast_cmd += ["--checkpoints"] + [str(c) for c in args.checkpoints]

    rc = _run(
        forecast_cmd,
        args.gpu,
        log_base / "forecast.log",
        args.dry_run,
        f"forecast/{args.model}/{args.dataset}",
    )
    if rc != 0:
        print("\nForecasting failed.")
        sys.exit(rc)

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"  Checkpoint: {target_ckpt}")
    print(f"  Logs:       {log_base.relative_to(ROOT)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
