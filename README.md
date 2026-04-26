HEY! 
Thank you for coming to look at our project.
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

under linear probing and fine tuning.
if you want to add to this project, feel free to branch and try add a PR.

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. point the code at your data (one-time per machine)
$EDITOR data_paths.py     # set monash_data_dir, forecasting_data_dir, etc.

# 3. run a single model end-to-end (pretrain + downstream)
python Train_and_downstream.py --model jepa
python Train_and_downstream.py --model dino     --task forecast
python Train_and_downstream.py --model lejepa   --task classify   --classification_dataset Heartbeat
python Train_and_downstream.py --model patchtst --task anomaly    --anomaly_dataset MSL

# 4. or fan out across encoder depths, datasets, seeds, GPUs (see Scripts below)
python scripts/run_layer_sweep.py
python scripts/run_layer_forecast.py --best_only
python scripts/run_seed_analysis.py --seeds 42
```

[Train_and_downstream.py](Train_and_downstream.py) is the unified entry point.
`--model` is the only required flag; everything else falls back to the
per-model config (see [Config](#config) below). Common overrides:
`--encoder_layers`, `--lr`, `--pretrain_source`, `--num_patches`,
`--checkpoint`, `--seed`, `--task`, `--pretrain_only true`.

## Model architecture & learning rates

All 6 models share patch_size=16 and zero-shot forecasting (frozen encoder +
linear head, no fine-tuning on the target dataset). Defaults below are what
ships in each per-model config; the sweep scripts override depth and
sometimes context window.

| Model    | d_model | n_heads | d_ff | n_layers (default) | context (default) | dropout |
|----------|---------|---------|------|--------------------|-------------------|---------|
| JEPA     | 256     | 8       | 1024 | 5                  | 32 patches (512)  | 0.0     |
| LE-JEPA  | 256     | 8       | 1024 | 5                  | 32 patches (512)  | 0.0     |
| DINO     | 128     | 16      | 512  | 5                  | 21 patches (336)  | 0.1     |
| PatchTST | 128     | 16      | 512  | 3                  | 21 patches (336)  | 0.2     |
| NTP      | 128     | 16      | 512  | 3                  | 21 patches (336)  | 0.2     |
| TimeDART | 256     | 8       | 512  | 3                  | 21 patches (336)  | 0.1     |

Learning rates at the 8-layer sweep depth (see
[scripts/run_layer_sweep.py](scripts/run_layer_sweep.py) for the full
per-depth tables):

| Model    | Optimizer | Pretrain LR (8L) | Forecast LR             |
|----------|-----------|------------------|-------------------------|
| JEPA     | SGD       | 5e-4             | 4e-4                    |
| LE-JEPA  | AdamW     | 5e-4             | 2e-4                    |
| DINO     | AdamW     | 5e-4             | auto `find_lr`          |
| PatchTST | Adam      | 5e-5             | 4e-4                    |
| NTP      | Adam      | 5e-5             | 1e-4                    |
| TimeDART | Adam      | 5e-5             | 1e-4                    |

## Config

Each model is controlled by a single Python `config = {...}` dict. The dict
is loaded by [Train_and_downstream.py](Train_and_downstream.py), merged with
the shared [data_paths.py](data_paths.py), and converted into the runner's
arguments. Editing these dicts is the primary way to change a model's
behavior; the sweep scripts in [scripts/](scripts/) only override a small,
well-defined subset (encoder depth, num_patches, GPU, pretrain source, LR
scaling, etc.) — everything else comes from the config file.

### Where each model's config lives

| Model    | Config file |
|----------|-------------|
| DINO     | [TSDiNO/config.py](TSDiNO/config.py) |
| JEPA     | [JEPA/config_files/config_jepa.py](JEPA/config_files/config_jepa.py) |
| LE-JEPA  | [LE-JEPA/config_lejepa.py](LE-JEPA/config_lejepa.py) |
| PatchTST | [PatchTST_self_supervised/config_patchtst.py](PatchTST_self_supervised/config_patchtst.py) |
| NTP      | [NTP/config_ntp.py](NTP/config_ntp.py) |
| TimeDART | [TimeDART-main/config_timedart.py](TimeDART-main/config_timedart.py) |

### Shared data paths

[data_paths.py](data_paths.py) holds the absolute locations of the Monash
pretraining corpus, synthetic `.arrow` files, forecasting CSVs, classification
sets, and anomaly sets. Edit it once per machine — every model's config gets
merged on top, so per-model overrides still win if needed.

### Keys you'll most often want to change

These keys appear (under similar names) in every config:

- **Pretraining source** — `pretrain_source`: `"monash" | "synthetic" | "monash+synthetic"`.
  Also overridable from every sweep script via `--pretrain_source`.
- **Architecture** — `num_encoder_layers` / `n_layers` / `e_layers`,
  `encoder_embed_dim` / `d_model` / `embed_dim`, `nhead` / `n_heads`, `d_ff`,
  `patch_size` / `patch_len`, `ratio_patches` / `num_patches` /
  `context_points`. The sweep scripts override depth (`--encoder_layers`,
  `--layers`) and the classification context window (`--num_patches`,
  `--patch_size`); everything else is taken from the config.
- **Pretrain optimization** — `num_epochs` / `epochs` / `train_epochs`,
  `batch_size`, `lr` / `learning_rate` / `lr_adamw`, `weight_decay`,
  `warmup_ratio`, `clip_grad`, `optimizer` (where applicable).
- **Forecasting downstream** — `epoch_t` / `epochs_forecasting`,
  `lr_forcasting` / `lr_forecasting`, `batch_size_forecast`, `horizon_t`,
  `forecasting_modes` (`"zeroshot"` is what we ship — all 6 models are
  zero-shot forecasters).
- **Classification downstream** — `epoch_classification`,
  `lr_classification`, `lr_classification_encoder` (encoder LR when
  fine-tuning; `None` falls back to head LR).
- **Anomaly downstream** — `epoch_anomaly`, `lr_anomaly`,
  `lr_anomaly_encoder`.
- **Model-specific knobs** —
  - JEPA / LE-JEPA: `mask_ratio`, `masking_type`, `num_blocks`,
    `predictor_*`, `ema_momentum`, VICReg / SIGReg loss weights.
  - LE-JEPA: `lambda_sigreg`, `sigreg_num_slices`, the augmentation block
    (`aug_*`, `view{1,2}_dwt_mode`, `dwt_*`).
  - DINO: `global_crops` / `local_crops` (augmentation views), `dwt_*`
    parameters, `out_dim`, teacher temperatures, `momentum_teacher`,
    `use_reconstruction` / `recon_*` (MAE-style auxiliary loss).
  - PatchTST: `mask_ratio`, `revin`, `model_type`.
  - NTP: `masking_type` (`"causal"` for next-patch prediction),
    `context_patches`, `horizon_t`.
  - TimeDART: `time_steps`, `scheduler`, `mask_ratio`, `lradj`, `pct_start`.

### How CLI flags interact with configs

The sweep-script flags do **not** rewrite the config files — they override
specific keys at runtime inside
[Train_and_downstream.py](Train_and_downstream.py) and via environment
variables (`TS_FORECAST_BS`, `TS_FORECAST_LR_SCALE`, `TS_CLS_BS`). Anything
not exposed as a flag (e.g. `mask_ratio`, `nhead`, augmentation specs, EMA
momentum) must be edited in the config file itself.

Typical workflow:

1. Set the dataset paths once in [data_paths.py](data_paths.py).
2. Tune model-specific behavior in the per-model config (e.g. `mask_ratio`,
   `lambda_sigreg`, augmentation views).
3. Use the sweep scripts in [scripts/](scripts/) to fan out across encoder depths, datasets, seeds, GPUs, and pretrain sources without touching the config.

---
## Scripts

### scripts/synthetic_data_generation/

Two GP-based generators that produce GluonTS `.arrow` files consumed by
the `synthetic` and `monash+synthetic` pretrain sources. Drop the output
into the directory pointed to by `synthetic_data_dir` (or
`synthetic_mix_data_dir`) in [data_paths.py](data_paths.py); the dataloader
[SyntheticArrowDataPullerJEPA](shared/data_loaders/data_puller.py)
auto-discovers every `.arrow` file in that directory and handles both
univariate (`[T]`) and multivariate (`[C, T]`) targets, so kernel-synth and
LMC outputs can live side-by-side. Sizes used in our experiments:

| Corpus            | Series  | Total timesteps | Notes                                |
|-------------------|---------|-----------------|--------------------------------------|
| Synthetic (full)  | 8,000   | ~1.61 B         | LMC_synth_MTS dominates (400k ts/series) |
| Synthetic (mix)   | 5,800   | ~1.17 B         | curated subset for the mix runs      |

#### kernel-synth.py

Univariate Gaussian-process kernel-synth. Each series is drawn from a GP
whose covariance is a random composition of base kernels (`RBF`,
`ExpSineSquared` at many periodicities, `RationalQuadratic`, `DotProduct`,
`WhiteKernel`, `ConstantKernel`) combined with random `+` or `×` operators.

CLI:
- `-N`  number of series (default 4000)
- `-J`  max kernels per series (default 5)
- `-L`  length per series (default 2500)
- `-P`  parallel jobs (default 4)

Output: `kernel_synth.arrow` next to the script.

    python scripts/synthetic_data_generation/kernel-synth.py -N 4000 -L 2500 -P 8

#### LMC_Synth.py

Multivariate extension via the Linear Coregionalization Model. For each
series:

1. Sample `latent_num ~ Weibull(shape, scale)`, clipped to
   `[max(2, num_channels // 20), num_channels]`.
2. Build `latent_num` independent univariate GP series, each with a random
   composite kernel (same kernel bank as kernel-synth).
3. Sample mixing weights from `Dirichlet(α · 1)` with
   `α ~ Uniform(dirichlet_min, dirichlet_max)`.
4. Combine: `output[C, T] = weights[C, latent_num] @ latent[latent_num, T]`
   → a correlated `num_channels`-variate series.

CLI (in addition to `-N`, `-L`, `-P`, `-J` from kernel-synth):
- `-C`  number of channels per series (default 160)
- `-M`  `dirichlet_min` (lower bound of α) — **required**
- `-X`  `dirichlet_max` (upper bound of α) — **required**
- `-W`  Weibull shape — **required**
- `-Z`  Weibull scale — **required**
- `-O`  output filename (default `LMC_synth_MTS.arrow`)
- `-D`  output directory (default `./`)

Example (8k series × 400k timesteps × 160 channels — what populates the
"Synthetic (full)" row above):

    python scripts/synthetic_data_generation/LMC_Synth.py \
        -N 8000 -L 400000 -C 160 -J 5 -P 16 \
        -M 0.1 -X 1.0 -W 1.5 -Z 2.0 \
        -O LMC_synth_MTS.arrow \
        -D /home/shared/datasets/synthetic_data_TS/

Lower `dirichlet_min` → sparser channel mixtures; higher `weibull_scale` →
more latent functions per series. Tune to taste.

After generation, run any model with `--pretrain_source synthetic` (synth
only) or `--pretrain_source monash+synthetic` (Monash + the curated mix
folder).

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

---
## Visuals

### Visuals/tsne_embeddings.py

Visualizes what each frozen backbone has learned by projecting its embeddings
onto a 2-D plane. For every selected model the script:

1. Loads the **classification pretrained checkpoint** (the one trained with
   `num_patches=72` → cw=1152, see
   [scripts/pretrain_cls_encoder.py](scripts/pretrain_cls_encoder.py)).
2. Encodes all train+test samples of each requested UEA classification
   dataset.
3. L2-normalizes the embeddings, optionally runs PCA (default 50 dims),
   then runs t-SNE (or UMAP via `--method umap`).
4. Saves a scatter plot colored by class label.

Per-model checkpoint paths are auto-resolved from
`(model, encoder_layers, pretrain_source)` — same scheme as the rest of the
repo (e.g. `output_model/classification/JEPA_monash_synthetic_layers8_cw1152/best_model.pt`).
Missing checkpoints print a warning and the model uses random weights, so
you'll see noise instead of structure for that subplot.

CLI options:
- `--datasets D [D ...]`     **required** — UEA classification dataset names
- `--models`                 one or more of: dino, jepa, lejepa, patchtst, ntp, timedart  (default: all)
- `--encoder_layers N`       backbone depth (default: 8)
- `--pretrain_source`        monash | synthetic | monash+synthetic  (default: monash+synthetic)
- `--cls_dir PATH`           root for classification datasets (default: `/home/shared/datasets/Classification_TS`)
- `--output_dir PATH`        figure output directory (default: `plots/`)
- `--max_points N`           stratified subsample per dataset (default: 500)
- `--method`                 tsne | umap (default: tsne; umap needs `pip install umap-learn`)
- `--pca_components N`       PCA dims before t-SNE/UMAP — `0` to skip PCA (default: 50)
- `--perplexity F`           t-SNE perplexity (default: 50)
- `--batch_size N`           encoder batch size (default: 64)
- `--gpu N`                  GPU index (default: 7)
- `--combined`               one figure with all 6 models as subplots (default: one figure per model+dataset)
- `--suffix STR`             suffix appended to output filenames

Examples:

    python Visuals/tsne_embeddings.py --datasets EthanolConcentration Heartbeat
    python Visuals/tsne_embeddings.py --datasets JapaneseVowels --models lejepa ntp
    python Visuals/tsne_embeddings.py --datasets SpokenArabicDigits --combined --suffix final
    python Visuals/tsne_embeddings.py --datasets Handwriting --method umap --pca_components 0
    python Visuals/tsne_embeddings.py --datasets FaceDetection PEMS-SF --output_dir plots/

Output filenames encode the reduction settings:
`{method}{pca_tag}{perp_tag}_{model}_{dataset}.png` (per-model) or
`{method}{pca_tag}{perp_tag}_combined_{dataset}{suffix}.png` (combined).
