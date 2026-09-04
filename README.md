# DYGJEPA Experiment Runner

This repository contains the code, configurations, and processed datasets
needed to run the dynamic link-prediction comparisons.

## 1. Clone the repository

Use Git rather than GitHub's **Download ZIP** option so the checked-out commit
and later updates are reproducible. The processed datasets are included in the
repository.

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

The unified configuration runs seeds `42, 44, 46, 48, 50`. Each model is
reinitialized independently for every seed. To override the seed list:

```bash
SEEDS="42 44" bash scripts/run_link_datasets.sh wikipedia
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
