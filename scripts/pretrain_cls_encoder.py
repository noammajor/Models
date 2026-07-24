#!/usr/bin/env python3
"""
pretrain_cls_encoder.py — Pre-train 8-layer encoders with 1152-timestep context for classification.

The standard pre-training uses short context windows aligned with forecasting (336–512 ts).
This script pre-trains with num_patches=72 (72 × 16 = 1152 timesteps), covering the full
length of the longest UEA classification datasets (SelfRegulationSCP2 T=1152).

Checkpoint locations:
  dino        → checkpoints_layers8_cw1152/
  jepa → output_model/JEPA_layers8_cw1152/
  lejepa      → output_model/LE-JEPA_layers8_cw1152/
  patchtst    → PatchTST_self_supervised/saved_models/  (context_points=1152)
  ntp         → NTP/saved_models/  (ratio_patches=72)

Usage:
    python pretrain_cls_encoder.py
    python pretrain_cls_encoder.py --models jepa lejepa
    python pretrain_cls_encoder.py --pretrain_source monash+synthetic
    python pretrain_cls_encoder.py --gpu_override 2
    python pretrain_cls_encoder.py --dry_run
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

DEFAULT_ENCODER_LAYERS   = 8
DEFAULT_PREDICTOR_LAYERS = 4       # half of encoder layers
DEFAULT_NUM_PATCHES      = 72      # 72 × 16 = 1152 timesteps
DEFAULT_PATCH_SIZE       = 16

# Per-model LR (same as run_layer_sweep.py for 8-layer configs)
MODEL_LR = {
    "dino":        5e-4,
    "jepa":        5e-4,
    "lejepa":      5e-4,
    "patchtst":    5e-5,
    "ntp":         5e-5,
    "timedart":    5e-5,
    "softclt":     1e-3,
}

MODEL_GPU = {
    "dino":        0,
    "jepa":        1,
    "lejepa":      2,
    "patchtst":    3,
    "ntp":         4,
    "timedart":    5,
    "softclt":     7,
}

ALL_MODELS = list(MODEL_GPU.keys())


def launch_model(model: str, gpu: int, pretrain_source: str,
                 log_dir: Path, dry_run: bool,
                 encoder_layers: int, predictor_layers: int,
                 num_patches: int, patch_size: int,
                 log_tag: str = "", embed_dim: int = None,
                 synthetic_data_dir: str = None):
    lr = MODEL_LR[model]
    cw = num_patches * patch_size
    tag_suffix = f"_{log_tag}" if log_tag else ""
    log_path = log_dir / f"{model}_layers{encoder_layers}_{pretrain_source}_cw{cw}{tag_suffix}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "Train_and_downstream.py"),
        "--model",            model,
        "--pretrain_only",    "true",
        "--encoder_layers",   str(encoder_layers),
        "--predictor_layers", str(predictor_layers),
        "--num_patches",      str(num_patches),
        "--lr",               str(lr),
        "--pretrain_source",  pretrain_source,
    ]
    if embed_dim is not None:
        cmd += ["--embed_dim", str(embed_dim)]
    if synthetic_data_dir is not None:
        cmd += ["--synthetic_data_dir", synthetic_data_dir]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"  [{model:12s}] GPU={gpu}  layers={encoder_layers}  "
          f"num_patches={num_patches}  cw={cw}  lr={lr}"
          f"  log={log_path.relative_to(ROOT)}")

    if dry_run:
        print(f"    CMD: {' '.join(cmd)}")
        return None

    fh = open(log_path, "w")
    fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    proc._log_fh = fh
    proc._label  = model
    return proc


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train encoders with a long context window for classification"
    )
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS, metavar="MODEL",
                        help=f"Models to train (default: all). Choices: {ALL_MODELS}")
    parser.add_argument("--pretrain_source", type=str, default="monash",
                        choices=["monash", "synthetic", "monash+synthetic"],
                        help="Pre-training data source (default: monash)")
    parser.add_argument("--encoder_layers",   type=int, default=DEFAULT_ENCODER_LAYERS,
                        help=f"Encoder depth (default: {DEFAULT_ENCODER_LAYERS})")
    parser.add_argument("--predictor_layers", type=int, default=DEFAULT_PREDICTOR_LAYERS,
                        help=f"Predictor depth (default: {DEFAULT_PREDICTOR_LAYERS})")
    parser.add_argument("--num_patches",      type=int, default=DEFAULT_NUM_PATCHES,
                        help=f"Number of patches in the context window (default: {DEFAULT_NUM_PATCHES})")
    parser.add_argument("--embed_dim",        type=int, default=None,
                        help="Encoder embedding dim (d_model). Omit to use the model's config default.")
    parser.add_argument("--synthetic_data_dir", type=str, default=None,
                        help="Override synthetic .arrow directory (e.g. the Monash-sized subset). Omit for the DATA_PATHS default.")
    parser.add_argument("--patch_size",       type=int, default=DEFAULT_PATCH_SIZE,
                        help=f"Patch size (default: {DEFAULT_PATCH_SIZE}). Used for log naming + cw display; "
                             "actual patch size is set by each model's config file.")
    parser.add_argument("--gpu_override", type=int, default=None,
                        help="Run the task on this GPU (overrides per-model assignment).")
    parser.add_argument("--log_tag", type=str, default="",
                        help="Suffix appended to training log filename (e.g. 'physical_3' → dino_layers8_monash_cw1152_physical_3.log)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    cw = args.num_patches * args.patch_size
    log_dir = ROOT / "logs" / "cls_encoder_pretrain"

    print("=" * 60)
    print(f"  Classification encoder pre-training")
    print(f"  encoder_layers  : {args.encoder_layers}")
    print(f"  predictor_layers: {args.predictor_layers}")
    print(f"  num_patches     : {args.num_patches}  (context = {cw} timesteps)")
    print(f"  patch_size      : {args.patch_size}")
    print(f"  models          : {args.models}")
    print(f"  pretrain_source : {args.pretrain_source}")
    if args.dry_run:
        print("  DRY RUN")
    print("=" * 60 + "\n")

    procs = []
    for model in args.models:
        gpu = args.gpu_override if args.gpu_override is not None else MODEL_GPU[model]
        proc = launch_model(model, gpu, args.pretrain_source, log_dir, args.dry_run,
                            encoder_layers=args.encoder_layers,
                            predictor_layers=args.predictor_layers,
                            num_patches=args.num_patches,
                            patch_size=args.patch_size,
                            log_tag=args.log_tag,
                            embed_dim=args.embed_dim,
                            synthetic_data_dir=args.synthetic_data_dir)
        if proc is not None:
            procs.append(proc)

    if not procs:
        return

    print(f"\nWaiting for {len(procs)} workers …")
    failed = []
    for proc in procs:
        proc.wait()
        proc._log_fh.close()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  {proc._label}: {status}")
        if proc.returncode != 0:
            failed.append(proc._label)

    if failed:
        print(f"\nWARNING: {len(failed)} failed: {failed}")
    else:
        print(f"\nAll {len(procs)} workers completed.")


if __name__ == "__main__":
    main()
