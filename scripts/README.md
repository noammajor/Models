# Experiment scripts

One script per protocol in the paper. They all take the same shape:

```
./scripts/<protocol>.sh <model> [<protocol options>] <task> <seed> [<seed> ...]
```

with `model` one of `dino jepa lejepa ntp mae diffusion softclt` and `task` one
of `forecast anomaly classify`. Omitting the seeds runs all five (123, 456, 789,
1337, 2003). Each script pre-trains and then evaluates every dataset of the task,
on one GPU, sequentially; logs land under `logs/<protocol>/<model>_<task>/`.

Everything shared — the dataset lists, the per-objective learning rates and batch
sizes, the pretrain→evaluate loop — lives in `_common.sh`, so the difference
between two protocols is the difference between their scripts.

| Script | Protocol | Varies from the main table |
| --- | --- | --- |
| `run_per_model.sh` | Monash, per-objective tuned LR and batch, 20 epochs | — (this *is* the main table) |
| `run_equal_budget.sh` | identical LR 1e-4, batch 64, 20 epochs for every objective | the pre-training budget |
| `run_corpus.sh` | `synthetic`, `synthetic_small`, `hybrid` | the pre-training corpus |
| `run_indomain.sh` | pre-train on the target forecasting dataset itself | the corpus (and its size) |
| `run_long_pretrain.sh` | 40 epochs instead of 20 | the pre-training length |
| `run_probe_head.sh` | `linear`, `mlp`, `finetune` on the per-model backbone | the evaluation protocol |
| `run_random_baseline.sh` | untrained encoder, frozen, probed | no pre-training at all |

The synthetic corpora are built from scratch by `run_synthetic_generation.sh`:

```bash
JOBS=16 ./scripts/run_synthetic_generation.sh all      # all three
./scripts/run_synthetic_generation.sh mix              # just one of them
```

| Stage | Written to | Read by |
| --- | --- | --- |
| `full` | `synthetic_data_dir` | `--pretrain_source synthetic` |
| `mix` | `synthetic_mix_data_dir` | `--pretrain_source monash+synthetic` |
| `subset` | `<synthetic_data_dir>_monashsize` | `run_corpus.sh <model> synthetic_small` |

Note the hybrid source reads the *mix* directory, not the full corpus: the
hybrid runs pair Monash with a smaller synthetic half (5,800 rows against
8,000), so `all` is needed before the hybrid protocol will run.

`run_probe_head.sh` pre-trains nothing — run `run_per_model.sh` for that model and
task first, or there is no backbone to probe.

## Running them

Prefix with `CUDA_VISIBLE_DEVICES` for a single job:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_per_model.sh ntp forecast 123
```

or hand a whole table to `launch.sh`, which expands `%M` over the seven
objectives and `%S` over the five seeds and keeps one job per GPU:

```bash
# 35 jobs over 8 GPUs
./scripts/launch.sh "0 1 2 3 4 5 6 7" ./scripts/run_per_model.sh %M forecast %S

# one model, one seed per GPU
MODELS=mae ./scripts/launch.sh "0 1 2 3 4" ./scripts/run_equal_budget.sh %M classify %S
```

`DRY_RUN=1` prints the commands instead of running them, and works on both the
protocol scripts and `launch.sh`.

## Utilities

| Script | |
| --- | --- |
| `run_synthetic_generation.sh` | build the synthetic corpus and its Monash-sized subset |
| `synthetic_data_generation/` | the two generators it calls (LMC and kernel-synth) |
| `subsample_synthetic.py` | draw the Monash-sized subset of an existing corpus |
| `count_dataset_sizes.py` | series and timestep counts per corpus (the dataset table) |
