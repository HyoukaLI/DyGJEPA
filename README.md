# DYGJEPA Experiment Runner

This repository contains the code, configurations, and processed datasets
needed to run the dynamic link-prediction comparisons. The much larger Tmall
and Patent node datasets are prepared separately as described below.

## 1. Clone the repository

Use Git rather than GitHub's **Download ZIP** option so the checked-out commit
and later updates are reproducible. The processed link datasets are included;
Tmall and Patent are downloaded separately because of their size.

```bash
git clone https://github.com/HyoukaLI/DyGJEPA.git
cd DyGJEPA
```

For a private repository, use a GitHub personal access token when Git asks for
a password, or clone with an authorized SSH key:

```bash
git clone git@github.com:HyoukaLI/DyGJEPA.git
```

## 2. Create the Python environment

Python 3.10 or newer is required.

### Conda

```bash
conda create -n dygjepa python=3.10 -y
conda activate dygjepa
python -m pip install -e .
```

### Python venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On a GPU cluster, install the PyTorch build required by the cluster before
installing this project if it is not already available.

All experiment configurations use `device: auto`, which selects CUDA first,
then Apple MPS, and finally CPU. The runners print `runtime_device` before
training. Set `REQUIRE_CUDA=1` to fail instead of silently falling back to CPU:

```bash
REQUIRE_CUDA=1 bash scripts/run_link_datasets.sh wikipedia
```

## 3. Check the datasets

The following command should list the processed `.npz` files:

```bash
ls -lh data/processed/*.npz
```

The unified link-prediction configuration includes:

```text
wikipedia  mooc  lastfm  canparl  contacts  flights
untrade    unvote  uslegis  enron  uci
```

## 4. Run the experiments

Run all configured datasets sequentially:

```bash
bash scripts/run_link_datasets.sh
```

The unified link configuration follows the DyGLib transductive protocol:
chronological `70/15/15`, one random destination negative per positive,
batch size 200 positive events, every recorded validation/test event (including
self-interactions when present in the raw stream), fixed
validation/test negative seeds `0/2`, and model seeds `0, 1, 2, 3, 4`.
AP and AUC are averaged over evaluation batches as in DyGLib. MRR and
Recall@10 are intentionally not computed by this DyGLib-aligned comparison.
RCPS-JEPA may use multiple random destinations per positive during training as
a model-specific hyperparameter; validation and test always retain the shared
one-positive/one-negative protocol.

Each model is reinitialized independently for every seed. To override the seed list:

```bash
SEEDS="0 1" bash scripts/run_link_datasets.sh wikipedia
```

Run only selected datasets with the same configuration and protocol:

```bash
bash scripts/run_link_datasets.sh wikipedia mooc lastfm
```

The script uses `./.venv/bin/python` when that environment exists. Otherwise,
it uses `python3` from the currently activated environment. A specific Python
executable can be selected explicitly:

```bash
PYTHON_BIN=/path/to/python bash scripts/run_link_datasets.sh enron uci
```

To run selected models directly:

```bash
python -m jepa_compare.compare_link_prediction \
  --config configs/link_comparison_all.yaml \
  --datasets wikipedia mooc \
  --models tgn tgat dygformer rcps_jepa
```

Run all three node-prediction datasets under the same model/probe protocol:

```bash
bash scripts/run_node_datasets.sh
```

On the configured Slurm cluster, submit GPU-enforced link or node jobs with:

```bash
sbatch run_link_datasets.sbatch
sbatch run_node_datasets.sbatch
```

Both Slurm launchers stop immediately if the selected PyTorch environment
cannot see CUDA. Override datasets without changing the shared protocol, for
example `DATASET_NAMES="tmall patent" sbatch run_node_datasets.sbatch`.

Model forward/backward passes, negative sampling, masking, and link metrics run
on CUDA when available. Dataset parsing and the official temporal-neighbor
indices remain on CPU by design: they use NumPy search structures and moving
them per query would add synchronization overhead. Checkpoints are also copied
to CPU so validation snapshots do not occupy extra GPU memory.

The node runner uses seeds `42, 44, 46, 48, 50` by default. Override them for
a smoke test or a partial rerun with:

```bash
SEEDS="42" EPOCHS=1 BASELINES="" bash scripts/run_node_datasets.sh tmall
```

Run only Tmall and Patent (the `tsmall` alias is also accepted):

```bash
bash scripts/run_node_datasets.sh tmall patent
```

For a short pipeline check, disable node baselines and override all JEPA
training to one epoch:

```bash
BASELINES="" EPOCHS=1 bash scripts/run_node_datasets.sh tmall
```

The shared node configuration is `configs/node_comparison_all.yaml`; it
inherits the established DBLP settings and changes only dataset paths and the
node batch size (1024 for Tmall, 2048 for Patent).

Multi-seed node results are written to:

```text
results/node_comparison_<dataset>_seed<seed>.json
results/node_comparison_<dataset>.json
results/node_comparison_all.json
```

The seed-specific files contain raw metrics. Dataset-level files contain all
five runs plus population mean and standard deviation for every numeric metric.

## 5. Run on Slurm

Run every dataset:

```bash
sbatch run_link_datasets.sbatch
```

Run selected datasets:

```bash
DATASET_NAMES="enron uci canparl" sbatch run_link_datasets.sbatch
```

Override the project directory or Python executable when necessary:

```bash
PROJECT_DIR=/path/to/DyGJEPA \
PYTHON_BIN=/path/to/python \
DATASET_NAMES="wikipedia mooc" \
sbatch run_link_datasets.sbatch
```

## 6. Results

Per-dataset and combined results are written to:

```text
results/link_comparison_<dataset>_seed<seed>.json
results/link_comparison_<dataset>.json
results/link_comparison_all.json
```

The seed-specific files contain raw metrics. The dataset-level and combined
files additionally contain population mean and standard deviation across the
five runs.

Local logs and result files are ignored by Git. Send the generated JSON files
back to the experiment owner separately unless explicitly asked to commit them.

## 7. Weights & Biases (optional)

Experiment tracking is off by default and the runners are fully usable without
it. To install the extra:

```bash
python -m pip install -e ".[tracking]"     # or: pip install -r requirements-tracking.txt
wandb login                                 # once per machine
```

Enable it per run:

```bash
python -m jepa_compare.compare_link_prediction --config configs/link_comparison_all.yaml --wandb
python -m jepa_compare.compare_node_prediction --config configs/node_comparison_all.yaml --wandb
```

or per environment, which is what the Slurm scripts forward:

```bash
DYGJEPA_WANDB=1 sbatch run_link_datasets.sbatch
```

### What gets logged

One run per `(dataset, model, seed)` triple, so seeds and models stay
separable:

```text
name      wikipedia-rcps_jepa-s0
group     link-wikipedia          # every model and seed of one dataset
job_type  rcps_jepa               # group by this to average over seeds
tags      link, wikipedia, rcps_jepa, seed-0
```

Metrics use the epoch as the step: `train/*` every epoch, `val/*` on every
evaluation epoch, and the selected checkpoint's `final/val/*` and
`final/test/*` in the run summary. SSL baselines log their pretraining stage
under `pretrain/*` and continue the probe stage after the last pretraining
epoch, so the two never share a step. The JSON files under `results/` are
written exactly as before — wandb is an addition, not a replacement.

### Offline clusters

`mode: auto` (the default) tries an online run once. If the compute node
cannot reach `api.wandb.ai`, it prints the reason, switches to offline for the
rest of the process, and writes to `$WANDB_DIR/wandb`. Upload afterwards from a
node that has network:

```bash
wandb sync wandb/offline-run-*
```

Set `mode: offline` to skip the online attempt entirely, or `mode: disabled`
to turn tracking off from the config. Every failure path is non-fatal: a
missing package, a failed login, or a dropped connection prints a warning and
training continues.

### Configuration

Each comparison config carries a `wandb` block:

```yaml
wandb:
  enabled: false
  project: dygjepa
  entity: null        # team or user; null uses your default
  mode: auto          # auto | online | offline | disabled
  group: null         # null derives "<task>-<dataset>"
  tags: []            # appended to the automatic tags
  dir: null           # parent of the offline "wandb" directory
  init_timeout: 30    # seconds to wait before falling back to offline
```

Precedence is environment > command line > config file. The environment
variables are `DYGJEPA_WANDB`, `WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY`
and `WANDB_DIR`; the flags are `--wandb` / `--no-wandb`, `--wandb-project`,
`--wandb-entity`, `--wandb-mode`, `--wandb-group`, `--wandb-tags` and
`--wandb-dir`.

## Troubleshooting

### Missing processed datasets

```bash
git pull
git checkout -- data/processed
```

### Python executable not found

Activate the intended Conda/venv environment or specify it explicitly:

```bash
PYTHON_BIN=/absolute/path/to/python bash scripts/run_link_datasets.sh
```
