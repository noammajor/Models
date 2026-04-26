#!/usr/bin/env python3
"""
run_monash.py — Pretrain each model on the Monash dataset and save a checkpoint every epoch.

Usage:
    python run_monash.py                    # pretrain all models on GPU 0
    python run_monash.py --gpu 1            # use GPU 1
    python run_monash.py --models dino ntp  # pretrain specific model(s)

Models: dino | jepa | ntp | patchtst
Checkpoints are saved by each model to its own output directory:
  dino        →  TSDiNO/checkpoints/checkpoint{epoch}.pth
  jepa        →  JEPA/output_model/JEPA/_epoch{epoch}/
  ntp         →  NTP/saved_models/monash/ntp/ntp_pretrained_*_epoch{epoch}.pt
  patchtst    →  PatchTST_self_supervised/saved_models/monash/masked_patchtst/based_model/*_epoch{epoch}.pth
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT   = Path(__file__).parent.parent
MODELS = ["jepa", "ntp", "patchtst", "dino"]


def pretrain_one(model: str, gpu: int, log_dir: Path) -> bool:
    log_path = log_dir / f"{model}.log"

    print(f"\n{'='*60}", flush=True)
    print(f"  Pretraining {model.upper()} on Monash  (GPU {gpu})")
    print(f"  log: logs/{model}.log")
    print(f"{'='*60}", flush=True)

    code = (
        f"from Train_and_downstream import run; "
        f"run(model='{model}', pretrain_only=True)"
    )
    cmd = [sys.executable, "-c", code]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29500"

    with open(log_path, "w") as flog:
        flog.write(f"# {model} pretrain on Monash  GPU={gpu}  started {datetime.now()}\n\n")
        flog.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            flog.write(line)
            flog.flush()
        proc.wait()

    ok = proc.returncode == 0
    print(f"\n  [{'OK' if ok else 'FAILED (exit ' + str(proc.returncode) + ')'}]  {model}", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Pretrain models on Monash (no downstream eval)")
    parser.add_argument(
        "--models", nargs="+", default=MODELS,
        choices=MODELS, metavar="MODEL",
        help=f"Models to pretrain (default: all). Choices: {MODELS}",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index to use (sets CUDA_VISIBLE_DEVICES, default: 0)",
    )
    args = parser.parse_args()

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    results = {}
    start   = datetime.now()

    print(f"\nPretraining {len(args.models)} model(s) on Monash  (GPU {args.gpu})")
    print(f"Logs -> logs/\n")

    for model in args.models:
        results[model] = pretrain_one(model, args.gpu, log_dir)

    elapsed = datetime.now() - start
    passed  = [m for m, ok in results.items() if ok]
    failed  = [m for m, ok in results.items() if not ok]

    print(f"\n\n{'='*60}")
    print(f"  SUMMARY  (total time: {elapsed})")
    print(f"{'='*60}")
    print(f"  Passed: {len(passed)}/{len(args.models)}")
    if failed:
        print(f"  Failed:")
        for m in failed:
            print(f"    {m}   (see logs/{m}.log)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
