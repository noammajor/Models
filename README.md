HEY! Thank you for coming to look at our project.
small run down on what we have here:
we have 6 SSL models, who use the same/"sameish" backbones to check the efficency of learning from large datasets.
Our models are as follows:
1. JEPA
2. LE-JEPA
3. Wavelets Dino
4. TimeDart -> Diffusion
5. PatchTST -> MAE
6. NTP

Our datasets for pre-training are as follows:
1. Monash
2. Fully Synthetic
3. Mix of both

Downstream tasks supported:
1. Forecasting
2. Anomoly Detection
3. Classification

under linear probing and fine tuning
---

## Scripts

### scripts/pretrain_cls_encoder.py

Pretrains encoders for **classification** with a longer context window than the
default forecasting setup. Defaults: 8-layer encoder, num_patches=72, patch_size=16
→ 1152-timestep context window, which covers the longest UEA classification
datasets (e.g. SelfRegulationSCP2 has T=1152). All four hyperparameters are
overridable via CLI.

Each model is launched as a subprocess of `Train_and_downstream.py` with
`--pretrain_only true`. By default each model runs on its own GPU in parallel
(per the `MODEL_GPU` table at the top of the script).

Checkpoint locations (with default 8-layer / cw=1152):
- dino     → checkpoints_layers8_cw1152/
- jepa     → output_model/JEPA_layers8_cw1152/
- lejepa   → output_model/LE-JEPA_layers8_cw1152/
- patchtst → PatchTST_self_supervised/saved_models/  (context_points=1152)
- ntp      → NTP/saved_models/  (ratio_patches=72)

CLI options:
- `--models`             one or more of: dino, jepa, lejepa, patchtst, ntp, timedart  (default: all)
- `--pretrain_source`    monash | synthetic | monash+synthetic  (default: monash)
- `--encoder_layers N`   encoder depth (default: 8)
- `--predictor_layers N` predictor depth (default: 4)
- `--num_patches N`      number of patches in context window (default: 72)
- `--patch_size N`       patch size — affects log naming + cw display only; actual
                         patch size is set by each model's config file (default: 16)
- `--gpu_override N`     run the task on GPU N (overrides per-model GPU assignment)
- `--dry_run`            print the commands without launching

Per-model defaults (LR + GPU assignment) live at the top of the script in
`MODEL_LR` and `MODEL_GPU`.

Examples:
    python scripts/pretrain_cls_encoder.py
    python scripts/pretrain_cls_encoder.py --models jepa lejepa
    python scripts/pretrain_cls_encoder.py --pretrain_source monash+synthetic
    python scripts/pretrain_cls_encoder.py --encoder_layers 4 --num_patches 32
    python scripts/pretrain_cls_encoder.py --gpu_override 2
    python scripts/pretrain_cls_encoder.py --dry_run

Logs land at `logs/cls_encoder_pretrain/{model}_layers{N}_{pretrain_source}_cw{cw}.log`
where `cw = num_patches × patch_size`.

### scripts/run_layer_sweep.py

Pretrains all models across multiple encoder layer depths. For each layer
config in `[2, 4, 8, 12, 24]`, all selected models are launched in parallel
(one per GPU); the script waits for each layer config to finish before moving
to the next. Predictor depth (JEPA-family) is always `encoder_layers // 2`.

Per-layer LR is set from three tables at the top of the script:
- `LAYER_LR`                          — jepa, lejepa (SGD + OneCycleLR)
- `DINO_LAYER_LR`                     — dino (AdamW + cosine)
- `NTP_PATCHTST_TIMEDART_LAYER_LR`    — ntp, patchtst, timedart (Adam, ~10× lower)

Checkpoints are saved under layer-suffixed paths, e.g.:
- `output_model/JEPA_layers8/`
- `output_model/LE-JEPA_layers8/`
- `checkpoints_layers8/` (DINO)
- `NTP/saved_models/{src}/ntp/layers8/`
- `PatchTST_self_supervised/saved_models/{src}/masked_patchtst/based_model/layers8/`

CLI options:
- `--models`           one or more of: dino, jepa, lejepa, patchtst, ntp, timedart  (default: all)
- `--layers N [N ...]` encoder layer counts to sweep (default: 2 4 8 12 24) - can run any size
- `--pretrain_source`  monash | synthetic | monash+synthetic  (default: monash)
- `--gpu_override N`   run all models on GPU N (overrides per-model assignment)
- `--dry_run`          print commands without launching

Examples:
    python scripts/run_layer_sweep.py
    python scripts/run_layer_sweep.py --layers 8 12
    python scripts/run_layer_sweep.py --models dino lejepa
    python scripts/run_layer_sweep.py --pretrain_source monash+synthetic
    python scripts/run_layer_sweep.py --gpu_override 5
    python scripts/run_layer_sweep.py --dry_run

Logs land at `logs/layer_sweep/{model}{src_tag}_layers{N}.log` (per-worker stdout).

### scripts/run_anomaly_sweep.py

Runs anomaly detection across (model × encoder_layer_depth × dataset). Each
`(model, encoder_layers)` combination is launched as a subprocess that loops
over all selected datasets. Default: frozen-encoder + trained linear
reconstruction decoder; can be flipped to fine-tune mode.

Per-dataset anomaly ratios match the TSLib reference (SMD=0.5, MSL/SMAP/SWaT/PSM=1.0)
unless overridden with `--anomaly_ratio`.

Results CSV columns:
`model, encoder_layers, pretrain_source, dataset, f1, precision, recall, accuracy, timestamp`

The script reads the CSV at startup and **skips already-completed**
`(model, encoder_layers, pretrain_source, dataset)` rows — re-running after a
crash picks up where it left off.

CLI options:
- `--models`             one or more of: dino, jepa, lejepa, patchtst, ntp, timedart, patchtst_random  (default: all)
- `--layers N [N ...]`   encoder depths to sweep (default: 2 4 8 12 24)
- `--datasets D [D ...]` anomaly datasets (default: SMD MSL SMAP SWaT PSM)
- `--pretrain_source`    monash | synthetic | monash+synthetic  (default: monash)
- `--gpu_override N`     run the task on GPU N (overrides per-model GPU assignment)
- `--linear_probe`       true|false (default: true). false → fine-tune encoder + decoder
- `--anomaly_ratio F`    override TSLib per-dataset ratio with a single value (e.g. 1.0)
- `--out_csv PATH`       output CSV path (default: results/anomaly_sweep.csv)
- `--dry_run`            print commands without launching

Examples:
    python scripts/run_anomaly_sweep.py
    python scripts/run_anomaly_sweep.py --models jepa lejepa --layers 8
    python scripts/run_anomaly_sweep.py --datasets MSL SMAP SMD
    python scripts/run_anomaly_sweep.py --models jepa --layers 8 --linear_probe false
    python scripts/run_anomaly_sweep.py --anomaly_ratio 1.0
    python scripts/run_anomaly_sweep.py --out_csv results/anomaly_sweep_synthetic.csv \
                                        --pretrain_source synthetic
    python scripts/run_anomaly_sweep.py --gpu_override 4

Logs land at `logs/anomaly_sweep/layers{N}/{pretrain_source}/{model}/{dataset}.log`
(per-dataset) plus `logs/anomaly_sweep/{model}_layers{N}.log` (per-worker stdout).

### scripts/run_finetune_8layers.py

End-to-end **fine-tuning** evaluation for all three downstream tasks (forecast,
classify, anomaly) using N-layer pretrained backbones (default: 8). Each model
is launched as a subprocess that runs all selected tasks in sequence with
`linear_probe=False` (i.e. the backbone is unfrozen and trained jointly with
the task head).

> **Name note:** the script file is named `run_finetune_8layers.py` for
> historical reasons. The 8-layer assumption is now just a default —
> `--encoder_layers` accepts any depth.

Before running each model, the script checks that a pretrained checkpoint
exists for the (model, encoder_layers, pretrain_source) triple. If not, the
worker prints `[SKIP]` and exits without writing a CSV row, so re-running
after pretraining picks up where it left off automatically.

Three result CSVs (one per task) are written to:
- `results/finetune_layers{N}_forecast{src_tag}.csv`
- `results/finetune_layers{N}_classification{src_tag}.csv`
- `results/finetune_layers{N}_anomaly{src_tag}.csv`

(`{src_tag}` is empty for monash, `_synthetic`/`_monash_synthetic` otherwise.)

CLI options:
- `--models`                 one or more of: dino, jepa, lejepa, patchtst, ntp, timedart, patchtst_random  (default: all)
- `--tasks`                  one or more of: forecast, classify, anomaly  (default: all)
- `--pretrain_source`        monash | synthetic | monash+synthetic  (default: monash)
- `--encoder_layers N`       backbone depth (default: 8)
- `--pred_lens N [N ...]`    forecast horizons (default: 96 192 336 720)
- `--checkpoint PATH`        explicit pretrained-backbone path; bypasses auto-discovery.
                             Requires a single `--models` entry. If the path doesn't
                             exist, the worker `[SKIP]`s instead of running.
- `--gpu_override N`         run all models on GPU N (overrides per-model assignment)
- `--forecast_datasets`      override the default forecast dataset list
- `--classification_datasets` override the default UEA dataset list
- `--anomaly_datasets`       override the default anomaly dataset list (SMD MSL SMAP SWaT PSM)
- `--dry_run`                print commands without launching

Examples:
    python scripts/run_finetune_8layers.py
    python scripts/run_finetune_8layers.py --models dino lejepa timedart
    python scripts/run_finetune_8layers.py --pretrain_source monash+synthetic
    python scripts/run_finetune_8layers.py --gpu_override 0
    python scripts/run_finetune_8layers.py --tasks forecast anomaly
    python scripts/run_finetune_8layers.py --forecast_datasets etth1 etth2
    python scripts/run_finetune_8layers.py --encoder_layers 4 --pred_lens 96
    python scripts/run_finetune_8layers.py --models jepa \
                                           --checkpoint /path/to/my_jepa.pt
    python scripts/run_finetune_8layers.py --dry_run

Logs land at:
- `logs/finetune_layers{N}/{model}.log` (per-worker stdout)
- `logs/finetune_layers{N}/{forecast|classify|anomaly}/{pretrain_source}/{model}/{dataset}[_pl{N}].log` (per-task)

### scripts/run_layer_classification.py

Sweeps **classification** across (model × encoder_layer_depth × dataset).
Each `(model, layer)` is launched as a subprocess that loops over all selected
classification datasets. By default the encoder is frozen (linear-probe mode);
use `--linear_probe false` to fine-tune.

Results CSV columns:
`model, encoder_layers, dataset, accuracy, timestamp`

The script reads the CSV at startup and **skips already-completed**
`(model, encoder_layers, dataset)` rows.

CLI options:
- `--models`             one or more of: dino, jepa, lejepa, patchtst, ntp, timedart, jepa_random, patchtst_random  (default: all)
- `--layers N [N ...]`   encoder depths to sweep (default: 8)
- `--datasets D [D ...]` UEA classification datasets (default: 10 standard sets)
- `--pretrain_source`    monash | synthetic | monash+synthetic (default: monash)
- `--gpu_override N`     run the task on GPU N (overrides per-model GPU assignment)
- `--linear_probe`       true|false (default: true). false → fine-tune encoder + head
- `--out_csv PATH`       output CSV path (default: results/layer_classification{src_tag}{cw_tag}.csv)
- `--dry_run`            print commands without launching

Examples:
    python scripts/run_layer_classification.py
    python scripts/run_layer_classification.py --layers 2 4 8
    python scripts/run_layer_classification.py --models dino jepa
    python scripts/run_layer_classification.py --datasets EthanolConcentration SelfRegulationSCP2
    python scripts/run_layer_classification.py --gpu_override 5
    python scripts/run_layer_classification.py --linear_probe false       # fine-tune mode
    python scripts/run_layer_classification.py --out_csv results/cls_synth.csv \
                                               --pretrain_source synthetic

Logs land at `logs/layer_classification{src_tag}/layers{N}/{model}/{dataset}.log`
(per-dataset) plus `logs/layer_classification{src_tag}/{model}_layers{N}.log`
(per-worker stdout).

### scripts/run_layer_forecast.py

Sweeps **forecasting** across (model × encoder_layer_depth × dataset) using a
tournament-style checkpoint search at four horizons. For each `(model, layers,
dataset)` combination the default flow is:

    pred96 (all ckpts)  →  top-3
    pred192 (top-3)     →  top-2
    pred336 (top-2)     →  best
    pred720 (best)

If you don't need the tournament, use `--best_only` to evaluate only the best
saved checkpoint at each pred_len (faster but doesn't pick a horizon-specific
optimum).

Result CSV columns:
`model, encoder_layers, dataset, pred_len, mse, mae, timestamp`

CSV resume: rows already containing valid `(model, layers, dataset, pred_len)`
are skipped on re-run.

Per-model output paths (default): `results/layer_forecast_{model}{src_tag}_layers{N}.csv`.

CLI options:
- `--models`           one or more of: dino, jepa, lejepa, patchtst, ntp, timedart, patchtst_random  (default: all)
- `--layers N [N ...]` encoder depths to sweep (default: 2 4 8 12 24)
- `--datasets D ...`   forecast datasets (default: etth1 etth2 ettm1 ettm2 weather electricity traffic)
- `--gpu_override N`   run all models on GPU N (overrides per-model assignment)
- `--best_only`        skip the tournament; only evaluate `checkpoints=["best"]` at each pred_len
- `--linear_probe`     true|false (default: true). false → fine-tune backbone + head
- `--pretrain_source`  monash | synthetic | monash+synthetic  (default: monash)
- `--pred_lens N ...`  custom horizons — **only effective with `--best_only`**.
                       The tournament keeps fixed `[96,192,336,720]` because its
                       top-K filtering depends on those exact stages.
- `--out_csv PATH`     single output CSV for all workers (overrides the per-model auto-path)
- `--dry_run`          print commands without launching

Examples:
    python scripts/run_layer_forecast.py
    python scripts/run_layer_forecast.py --layers 4 12
    python scripts/run_layer_forecast.py --models dino lejepa
    python scripts/run_layer_forecast.py --linear_probe false                  # fine-tune
    python scripts/run_layer_forecast.py --best_only --pred_lens 96 192        # quick eval
    python scripts/run_layer_forecast.py --pretrain_source synthetic
    python scripts/run_layer_forecast.py --dry_run

Logs land at `logs/layer_forecast{src_tag}/layers{N}/{model}/{dataset}/pred{P}_ckpt{K}.log`
(per-checkpoint stage) plus `logs/layer_forecast/{model}{src_tag}_layers{N}.log`
(per-worker stdout).

### scripts/run_seed_analysis.py

Reproducibility sweep: pretrain + all three downstream tasks (forecast,
classify, anomaly) across multiple random seeds at a fixed `encoder_layers=8`.
Default seeds: `[2003, 123, 456, 789, 1337]`, but **any integer(s)** can be
passed via `--seeds`. Per seed, all selected models run their full pipeline in
parallel (one thread per model, pinned to that model's GPU); the script waits
for the seed to finish before moving on.

Each seed's pretrain phase runs **two** pretrains per model:
- standard pretrain (forecast/anomaly context)
- classification pretrain with `num_patches=72` → cw=1152

Checkpoints are saved with a `_seedN` suffix
(e.g. `output_model/JEPA_monash_synthetic_layers8_seed2/`).

CLI options:
- `--seeds N [N ...]`            seeds to sweep (default: 2003 123 456 789 1337)
- `--models`                     one or more of: dino, jepa, lejepa, patchtst, ntp, timedart  (default: all)
- `--pretrain_source`            monash | synthetic | monash+synthetic  (default: monash)
- `--phases P [P ...]`           subset of: pretrain, forecast, classify, anomaly  (default: all)
- `--forecast_datasets D ...`    restrict forecast phase to these datasets
- `--classification_datasets D …`restrict classify phase to these UEA datasets
- `--skip_pretrain`              reuse existing `_seedN` checkpoints, skip the pretrain phase
- `--gpu_override N`             run all models on GPU N (overrides per-model assignment)
- `--dry_run`                    print commands without launching

Examples:
    python scripts/run_seed_analysis.py
    python scripts/run_seed_analysis.py --seeds 0 1 2
    python scripts/run_seed_analysis.py --seeds 42                         # any seed
    python scripts/run_seed_analysis.py --models dino ntp
    python scripts/run_seed_analysis.py --skip_pretrain --phases forecast
    python scripts/run_seed_analysis.py --pretrain_source synthetic
    python scripts/run_seed_analysis.py --gpu_override 3
    python scripts/run_seed_analysis.py --dry_run

Logs land at `logs/forecasting/seed_analysis/seed{N}/{phase}_{model}_{dataset}.log`.
Aggregated results: `results/seed_analysis_{forecast,classification,anomaly}.csv`.
