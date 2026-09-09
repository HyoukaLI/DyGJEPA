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

### Historical negative sampling (separate run)

DyGLib's historical negatives (Poursafaei et al., 2022;
`evaluate_link_prediction.py --negative_sample_strategy historical`) are a
separate run with their own config, launcher and Slurm script, so they never
touch the random-negative results:

```bash
bash scripts/run_link_datasets_historical.sh              # all datasets
bash scripts/run_link_datasets_historical.sh canparl uci  # a subset
sbatch run_link_datasets_historical.sbatch                # cluster; DATASET_NAMES/MODELS/SEEDS/EPOCHS as usual
```

`configs/link_comparison_all_historical.yaml` is an overlay
(`base_config: link_comparison_all.yaml`): every dataset entry, model setting
and optimizer recipe is inherited from the random config, and only
`link.negative_strategy: historical` plus the result location differ, so
tuning done in `link_comparison_all.yaml` carries over. Training and
checkpoint selection still use random negatives; the final validation/test
pass of every model then scores one shared pre-sampled historical negative per
event (an edge observed before the 200-event batch but absent from it, both
endpoints replaced, random pairs filling short pools, validation seed 0 / test
seed 2). Results go to `results/historical/*_historical.json` and
`results/historical/link_comparison_all_historical.json`. EdgeBank keeps the
configured `edgebank.memory_mode`; DyGLib's strategy-specific EdgeBank memory
modes are not reproduced. Setting `link.negative_strategy: historical`
directly in any config has the same effect and suffixes its result files with
`_historical`.

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

### Node datasets need SpikeNet DeepWalk features first

The SG-JEPA paper (and SpikeNet, whose DBLP/Tmall/Patent release it uses)
feeds every model 80-dimensional per-snapshot DeepWalk node features
(`<dataset>.npy`, shape `[T, N, 80]`). Without them the converters write a 4-D
structural placeholder and **every** model (SG-JEPA, EvolveGCN, ROLAND, ...)
collapses to the majority class (DBLP: ~0.05 Macro-F1 / ~0.29 Micro-F1 instead
of the paper's ~0.74 / ~0.75). `load_npz` now emits a `RuntimeWarning` when an
archive carries the fallback. Two ways to obtain the features:

1. **Official files** (fastest, exactly what the paper used): SpikeNet's
   Dropbox folder holds `dblp.npy`; `tmall.npy` and `patent.npy` are on the
   Aliyun Drive link in the SpikeNet README (rename the downloaded `.txt` to
   `.npy`). Place them at `data/raw/<dataset>/<dataset>.npy`; expected shapes
   are `[27, 28085, 80]`, `[19, 577314, 80]` and `[13, 2738012, 80]`
   (Tmall/Patent snapshots are merged by 10/2 as in the official loader).
2. **Regenerate** with the official recipe
   (`DeepWalk(80, 10, 128, window_size=10, negative=1)`, `--normalize` for
   Tmall/Patent only), resumable and streamed so Patent fits in memory:

   ```bash
   DATASET=dblp   sbatch generate_node_features.sbatch
   DATASET=tmall  sbatch generate_node_features.sbatch   # hours
   DATASET=patent sbatch generate_node_features.sbatch   # ~1 day; SCRATCH_DIR needs ~25 GB
   ```

   or directly: `python scripts/generate_spikenet_deepwalk.py --dataset tmall`.

Then rebuild the archives (the `.npy` next to the raw data is auto-detected):

```bash
python scripts/prepare_dblp.py
python scripts/prepare_spikenet_node.py --dataset tmall
python scripts/prepare_spikenet_node.py --dataset patent
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
