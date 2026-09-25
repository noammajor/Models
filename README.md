# Quantifying the Pre-training Dividend: Generative versus Latent Self-Supervised Learning for Time Series Foundation Models

Code and experimental setup for a study comparing generative and latent
self-supervised pre-training objectives for time-series foundation models.
Seven objectives are benchmarked under a shared encoder and a unified
evaluation protocol, so that differences downstream can be attributed to the
pre-training objective rather than to architectural asymmetries.

Every number in the paper comes from one of the scripts in
[scripts/](scripts/); [Running the experiments](#running-the-experiments) maps
each protocol to its script.

## Objectives evaluated

Latent / joint-embedding:

| Name | Objective |
| --- | --- |
| **JEPA** | joint-embedding predictive architecture |
| **LE-JEPA** | JEPA with VICReg/SIGReg-style latent regularization |
| **DINO** | self-distillation with a momentum teacher and DWT-based augmentations |
| **SoftCLT** | soft contrastive learning with instance- and temporal-level soft assignments |

Generative / reconstructive:

| Name | Objective |
| --- | --- |
| **MAE** | masked-patch autoencoding (PatchTST backbone) |
| **Diffusion** | denoising-diffusion patch reconstruction (TimeDART) |
| **NTP** | next-patch prediction (causal masking) |

A randomly initialised, never-pre-trained encoder (`mae_random`) is the control
every objective is measured against.

## Installation

```bash
pip install -r requirements.txt
```

Developed against Python 3.12 with a CUDA-capable GPU. All training scripts are single-GPU; there
is no distributed training anywhere in the repository.

## Datasets

Five corpora. Fill in the link column with the source you download from, then
point [data_paths.py](data_paths.py) at where you put each one — that file is
the only place paths are configured, and every script reads it.

| Corpus | Used for | Download | `data_paths.py` key |
| --- | --- | --- | --- |
| **Monash** | pre-training | [HuggingFace](https://huggingface.co/datasets/Monash-University/monash_tsf) | `monash_data_dir` |
| **Synthetic** | pre-training | generated locally, see [below](#generating-the-synthetic-corpora) | `synthetic_data_dir`, `synthetic_mix_data_dir` |
| **Forecasting** | downstream | [HuggingFace](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets) | `forecasting_data_dir` |
| **Classification** (UEA) | downstream | [timeseriesclassification.com](https://www.timeseriesclassification.com/dataset.php) | `classification_data_dir` |
| **Anomaly detection** | downstream | *(link)* | `anomaly_data_dir` |

### Expected layout

Each loader discovers its own files, so only the shape of each directory
matters:

| Corpus | Layout |
| --- | --- |
| Monash | a flat directory of `*.tsf` files |
| Synthetic | a flat directory of `*.arrow` files (GluonTS); univariate `[T]` and multivariate `[C, T]` targets may be mixed |
| Forecasting | `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`, `weather.csv`, `electricity.csv`, `traffic.csv` — a flat directory, no subfolders |
| Classification | one directory per dataset: `<Name>/<Name>_TRAIN.ts` and `<Name>/<Name>_TEST.ts` |
| Anomaly detection | one directory per dataset: `<NAME>/train.npy` `[T, C]`, `<NAME>/test.npy` `[T, C]`, `<NAME>/test_labels.npy` `[T]` |

The classification data comes from the UEA multivariate archive. That page
offers bulk archives only, in several formats — take the **multivariate, aeon
`ts` format** one (~1.5 GB), which is the `.ts` layout the loader reads. It
extracts to one directory per dataset; these nine are the ones this study uses:

```
EthanolConcentration   JapaneseVowels        SelfRegulationSCP2
FaceDetection          SelfRegulationSCP1    SpokenArabicDigits
Handwriting            Heartbeat             UWaveGestureLibrary
```

Each must end up as `<classification_data_dir>/<Name>/<Name>_TRAIN.ts` and
`_TEST.ts`; the rest of the archive can be deleted. The site also documents
loading individual datasets programmatically through the aeon toolkit, which
avoids the full download.

The Monash repository stores the archive as one `.zip` per dataset under
`data/`, not as loose `.tsf` files, so it has to be extracted — `monash_data_dir`
must end up a flat directory of `.tsf` files. Its dataset viewer is disabled
because the repository ships a loading script; the raw archives are what this
code wants, so download the files rather than calling `datasets.load_dataset`:

```bash
huggingface-cli download Monash-University/monash_tsf --repo-type dataset \
    --include "data/*.zip" --local-dir monash_raw
cd monash_raw/data && for z in *.zip; do unzip -o -q "$z"; done
```

The forecasting repository linked above carries all seven CSVs (alongside
`exchange_rate.csv` and `national_illness.csv`, which this study does not use).
Its dataset viewer reports a schema error because the files do not share
columns; the CSVs themselves download normally.

The anomaly datasets are distributed in several different native formats.
[shared/prep_anomaly_data.py](shared/prep_anomaly_data.py) converts them into
the layout above:

```bash
python shared/prep_anomaly_data.py --in_dir <raw downloads> --out_dir <anomaly_data_dir>
```

Pre-training uses only series of at least 512 steps (`monash_min_len`), which
is what the corpus sizes below are counted under. Check what you have with:

```bash
python scripts/count_dataset_sizes.py --min_len 512
python scripts/count_dataset_sizes.py --only <one .arrow directory>
```

### Datasets used

- **Forecasting** (7): ETTh1, ETTh2, ETTm1, ETTm2, weather, electricity, traffic —
  horizons 96, 192, 336 and 720 from a 336-step context.
- **Classification** (9): EthanolConcentration, FaceDetection, Handwriting,
  Heartbeat, JapaneseVowels, SelfRegulationSCP1, SelfRegulationSCP2,
  SpokenArabicDigits, UWaveGestureLibrary.
- **Anomaly detection** (5): MSL, PSM, SMAP, SMD, SWaT.

### Generating the synthetic corpora

The synthetic corpora are generated, not downloaded. One script builds all
three:

```bash
JOBS=16 ./scripts/run_synthetic_generation.sh all
```

| Stage | Written to | Read by |
| --- | --- | --- |
| `full` | `synthetic_data_dir` | `--pretrain_source synthetic` |
| `mix` | `synthetic_mix_data_dir` | `--pretrain_source monash+synthetic` |
| `subset` | `<synthetic_data_dir>_monashsize` | `run_corpus.sh <model> synthetic_small` |

Each corpus is an equal number of rows from two Gaussian-process generators
(see [scripts/synthetic_data_generation/](scripts/synthetic_data_generation/)):
`LMC_Synth.py` draws multivariate rows via the Linear Coregionalization Model
(160 channels x 2,500 steps = 400k timesteps per row, so it dominates the
totals), and `kernel-synth.py` draws univariate rows of 2,500 steps. 4,000 +
4,000 rows give the full corpus, 2,900 + 2,900 the mix.

Note that the hybrid source reads the **mix** directory, not the full corpus:
the hybrid runs pair Monash with a smaller synthetic half, so `all` is needed
before the hybrid protocol will run.

`subset` is a random draw from the full corpus whose total filtered timestep
count matches Monash. "Synthetic vs Monash" otherwise confounds the
data-generating process with corpus size; the subset holds size fixed so that
only the process differs.

| Corpus | Series | Filtered timesteps |
| --- | ---: | ---: |
| Monash | 45,042 | 445,011,429 |
| Synthetic (full) | 8,000 | 1,610,000,000 |
| Synthetic (mix) | 5,800 | 1,167,250,000 |

Generation is CPU-bound and takes hours at these sizes — every series is a draw
from a GP prior. Set `JOBS` to the core count and run it detached.

---

## Running the experiments

Every protocol in the paper is one bash script in [scripts/](scripts/), all
with the same shape:

```
./scripts/<protocol>.sh <model> [<protocol options>] <task> <seed> [<seed> ...]
```

- `model` — `dino jepa lejepa ntp mae diffusion softclt`
- `task` — `forecast anomaly classify`
- seeds — omit to run all five (123, 456, 789, 1337, 2003)

Each script pre-trains and then evaluates every dataset of the task, on one
GPU, sequentially. Logs land in `logs/<protocol>/<model>_<task>/`.

| Script | Protocol | Varies from the main table |
| --- | --- | --- |
| `run_per_model.sh` | Monash, per-objective tuned LR and batch, 20 epochs | — (this *is* the main table) |
| `run_equal_budget.sh` | LR 1e-4, batch 64, 20 epochs for every objective | the pre-training budget |
| `run_corpus.sh` | `synthetic`, `synthetic_small`, `hybrid` | the pre-training corpus |
| `run_indomain.sh` | pre-train on the target forecasting dataset itself | the corpus (and its size) |
| `run_long_pretrain.sh` | 40 epochs instead of 20 | the pre-training length |
| `run_probe_head.sh` | `linear`, `mlp`, `finetune` | the evaluation protocol |
| `run_random_baseline.sh` | untrained encoder, frozen, probed | no pre-training at all |

Everything shared between them — the dataset lists, the per-objective learning
rates and batch sizes, the pretrain-then-evaluate loop — lives in
`scripts/_common.sh`, so the difference between two protocols is the difference
between their scripts. See [scripts/README.md](scripts/README.md) for more.

### Examples

```bash
# one model, one task, one seed, on GPU 0
CUDA_VISIBLE_DEVICES=0 ./scripts/run_per_model.sh ntp forecast 123

# the equal-budget classification table for MAE, all five seeds
CUDA_VISIBLE_DEVICES=1 ./scripts/run_equal_budget.sh mae classify

# the hybrid corpus (needs the synthetic mix generated first)
CUDA_VISIBLE_DEVICES=2 ./scripts/run_corpus.sh softclt hybrid forecast 123

# an MLP probe on the per-model backbone (run run_per_model.sh first)
CUDA_VISIBLE_DEVICES=3 ./scripts/run_probe_head.sh lejepa mlp classify 123
```

`run_probe_head.sh` pre-trains nothing — it reads the backbone
`run_per_model.sh` produced, so run that for the same model and task first.

### Across several GPUs

`launch.sh` expands `%M` over the objectives and `%S` over the seeds, and keeps
one job per GPU until the queue drains:

```bash
# the main Monash forecasting table: 7 models x 5 seeds over 8 GPUs
./scripts/launch.sh "0 1 2 3 4 5 6 7" ./scripts/run_per_model.sh %M forecast %S

# one model, one seed per GPU
MODELS=mae ./scripts/launch.sh "0 1 2 3 4" ./scripts/run_equal_budget.sh %M classify %S

# the random control, which takes no model argument
MODELS=- ./scripts/launch.sh "0 1 2 3 4" ./scripts/run_random_baseline.sh forecast %S
```

`DRY_RUN=1` prints the commands instead of running them, on both the protocol
scripts and `launch.sh`. Use it to check a protocol before committing GPUs to
it.

### Running a single configuration by hand

The scripts are thin wrappers over one entry point, which can be called
directly:

```bash
python Train_and_downstream.py --model ntp --task pretrain \
    --pretrain_source monash --encoder_layers 8 --embed_dim 128 \
    --num_patches 21 --lr 1e-4 --epochs 20 --ckpt_tag monash --seed 123

python Train_and_downstream.py --model ntp --task forecast \
    --forecast_dataset etth1 --encoder_layers 8 --embed_dim 128 \
    --num_patches 21 --ckpt_tag monash --seed 123
```

`--task` is one of `pretrain forecast classify anomaly`, and `python
Train_and_downstream.py --help` lists every flag. Checkpoint directories are
tagged by pre-training source, depth, context window, epoch count, `--ckpt_tag`
and seed, so the downstream call must repeat the flags the pre-training call
was given or the backbone will not be found.

---

## Figures and analyses

These run forward passes through frozen, already-pre-trained encoders; none of
them train anything. All default to the 8-layer backbones and take `--models`,
`--seed` and `--output_dir`.

```bash
# effective rank of the embedding space vs the forecasting gain over random
python Visuals/e4_isotropy.py --models dino jepa lejepa ntp mae diffusion softclt \
    --datasets etth1 weather --encoder_layers 8 --seed 123

# latent drift under time-shift and time-masking perturbations
python Visuals/latent_drift.py --datasets weather etth1 --context forecast

# t-SNE of classification embeddings, one figure per model
python Visuals/tsne_embeddings.py --datasets JapaneseVowels Heartbeat

# is the LE-JEPA embedding distribution the isotropic Gaussian SIGReg pushes for?
python Visuals/lejepa_gaussianity.py --dataset SpokenArabicDigits
```

`e4_isotropy.py` also takes `--ckpt_override MODEL=PATH` to pin an individual
model's backbone, and `latent_drift.py` takes `--context classify` for the
1152-step encoders.

## Utilities

| Script | |
| --- | --- |
| `scripts/run_synthetic_generation.sh` | build all three synthetic corpora |
| `scripts/synthetic_data_generation/` | the two generators it calls |
| `scripts/subsample_synthetic.py` | draw a Monash-sized subset of an existing corpus |
| `scripts/count_dataset_sizes.py` | series and timestep counts per corpus |
| `shared/prep_anomaly_data.py` | convert raw anomaly datasets into the expected layout |

---

## Configuration

Each objective is controlled by a single Python `config = {...}` dict, loaded
by [Train_and_downstream.py](Train_and_downstream.py), merged with the shared
[data_paths.py](data_paths.py), and turned into the runner's arguments. The
scripts override only a small, well-defined subset at runtime (depth, width,
context, LR, epochs, batch size, corpus, seed, checkpoint tag) — everything
else comes from the config file.

| Objective | Config file |
| --- | --- |
| DINO | [DINO/config.py](DINO/config.py) |
| JEPA | [JEPA/config_files/config_jepa.py](JEPA/config_files/config_jepa.py) |
| LE-JEPA | [LeJEPA/config_lejepa.py](LeJEPA/config_lejepa.py) |
| MAE | [MAE/config_patchtst.py](MAE/config_patchtst.py) |
| NTP | [NTP/config_ntp.py](NTP/config_ntp.py) |
| Diffusion | [Diffusion/config_timedart.py](Diffusion/config_timedart.py) |
| SoftCLT | [SoftCLT/config_softclt.py](SoftCLT/config_softclt.py) |

### Keys worth knowing

The same quantities appear under different names in each config, because each
objective's implementation keeps its original vocabulary:

- **Pre-training corpus** — `pretrain_source`: `"monash" | "synthetic" | "monash+synthetic"`.
- **Architecture** — `num_encoder_layers` / `n_layers` / `e_layers`;
  `encoder_embed_dim` / `d_model` / `embed_dim`; `nhead` / `n_heads`; `d_ff`;
  `patch_size` / `patch_len`; `ratio_patches` / `num_patches` / `context_points`.
- **Pre-training optimization** — `num_epochs` / `epochs` / `train_epochs`,
  `batch_size`, `lr` / `learning_rate`, `weight_decay`, `warmup_ratio`,
  `clip_grad`.
- **Forecasting** — `epoch_t` / `epochs_forecasting`, `lr_forcasting` /
  `lr_forecasting`, `batch_size_forecast`, `horizon_t`, `forecasting_modes`
  (`"zeroshot"` throughout: every objective is evaluated as a frozen encoder
  with a linear head).
- **Classification** — `epoch_classification`, `lr_classification`,
  `lr_classification_encoder`.
- **Anomaly detection** — `epoch_anomaly`, `lr_anomaly`, `lr_anomaly_encoder`.
- **Objective-specific** —
  - JEPA / LE-JEPA: `mask_ratio`, `masking_type`, `num_blocks`, `predictor_*`,
    `ema_momentum`, VICReg / SIGReg loss weights.
  - LE-JEPA: `lambda_sigreg`, `sigreg_num_slices`, the augmentation block.
  - DINO: `global_crops` / `local_crops`, `dwt_*`, `out_dim`, teacher
    temperatures, `momentum_teacher`.
  - MAE: `mask_ratio`, `revin`, `model_type`.
  - NTP: `masking_type` (`"causal"`), `context_patches`, `horizon_t`.
  - Diffusion: `time_steps`, `scheduler`, `mask_ratio`, `lradj`, `pct_start`.
  - SoftCLT: the soft-assignment temperatures `tau_inst` and `tau_temp`.

CLI flags never rewrite the config files; they override keys at runtime, some
through environment variables (`TS_PRETRAIN_BS`, `TS_PATCHTST_BS`,
`TS_TIMEDART_BS`, `TS_SOFTCLT_BS`, `TS_DINO_BS`, `TS_FORECAST_BS`, `TS_CLS_BS`,
`TS_CKPT_TAG`). Anything not exposed as a flag — `mask_ratio`, `nhead`,
augmentation specifications, EMA momentum — has to be edited in the config.

## Repository layout

```
Train_and_downstream.py    single entry point: pre-training and all three downstream tasks
data_paths.py              where every dataset lives (edit once per machine)
dataset_registry.py        the forecasting CSVs and their columns
DINO/ JEPA/ LeJEPA/        one directory per objective: model, config, training loop
MAE/ NTP/ Diffusion/ SoftCLT/
shared/                    dataloaders shared across objectives, anomaly preprocessing
Monash_data_utils/         .tsf reader
scripts/                   one script per experimental protocol (see scripts/README.md)
Visuals/                   figures and embedding-space diagnostics
```

## Acknowledgements

This codebase builds on prior open-source releases:

1. PatchTST — [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
2. TimeDART — [arXiv:2410.05711](https://arxiv.org/abs/2410.05711)
3. DINO — [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
4. I-JEPA — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
5. SoftCLT — [arXiv:2312.16424](https://arxiv.org/abs/2312.16424)
6. TimesPFN — [arXiv:2502.16294](https://arxiv.org/abs/2502.16294)
7. Chronos, for the kernel-synth procedure — [arXiv:2403.07815](https://arxiv.org/abs/2403.07815)
