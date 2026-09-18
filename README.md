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
DyGJEPA may use multiple random destinations per positive during training as
a model-specific hyperparameter; validation and test always retain the shared
one-positive/one-negative protocol.

One dataset-specific exception: on UN Trade the only event feature is the raw
trade volume (up to ~5.25e7, while every other dataset stays below 3e2). The
TGAT adapter has no per-layer normalization, so the unscaled values overflow
fp32 and its loss is non-finite from the first epoch (the run then has no
`tgat` entry). The `untrade` block therefore sets
`tgat: {edge_feature_transform: log1p}`, which applies `sign(x) * log1p(|x|)`
to TGAT's own copy of the event features. Nothing else changes: the default is
`none`, every other model reads the stored features unchanged, and TGAT on
every other dataset is bit-identical to the previous runs.

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

### Inductive negative sampling (separate run)

DyGLib's inductive negatives (`--negative_sample_strategy inductive`, the
"ind" columns of DyGFormer / DyG-Mamba style tables) are a third separate run:

```bash
bash scripts/run_link_datasets_inductive.sh              # all datasets
bash scripts/run_link_datasets_inductive.sh canparl uci  # a subset
sbatch run_link_datasets_inductive.sbatch                # cluster; DATASET_NAMES/MODELS/SEEDS/EPOCHS as usual
```

`configs/link_comparison_all_inductive.yaml` overlays the random config like
the historical one. The mechanics are the historical protocol with a smaller
pool: a negative must be an edge that first appeared during the evaluation
period itself -- after the last training timestamp for validation, after the
last validation timestamp for test (DyGLib's `observed_edges` boundary) -- was
observed before the 200-event batch and is absent from it. Short pools are
again filled with random pairs, so early evaluation batches are close to the
random protocol and later ones increasingly test whether a model picks up the
recurrence of edges it never saw in training. Results go to
`results/inductive/*_inductive.json`.

### Inductive (new-node) setting with random negatives (separate run)

The "inductive" tables of DyGFormer / DyG-Mamba style papers are a different
thing from the inductive *negatives* above: they follow DyGLib's
`get_link_prediction_data`, which holds nodes out of training and evaluates on
the events that touch a node the model never saw. This is a fourth separate
run:

```bash
bash scripts/run_link_datasets_inductive_setting.sh              # all datasets
bash scripts/run_link_datasets_inductive_setting.sh canparl uci  # a subset
sbatch run_link_datasets_inductive_setting.sbatch                # cluster; DATASET_NAMES/MODELS/SEEDS/EPOCHS as usual
```

`configs/link_comparison_all_inductive_setting.yaml` overlays the random
config with `link.setting: inductive` (`new_node_ratio: 0.1`,
`new_node_seed: 2020`). The driver (`jepa_compare/inductive_setting.py`) draws
10% of all nodes among those that interact after the training period, removes
every training event that touches one of them (their message edges, features
and activity in the training snapshots as well), trains every model on the
reduced snapshots and still selects its checkpoint on the ordinary
(transductive) validation set with random negatives. The final
validation/test pass then scores only the events with at least one endpoint
absent from the reduced training data, against one random destination per
event drawn from those events' own destinations (validation seed 1, test
seed 3), 200 positives per batch. At evaluation time the context snapshots,
DyGLib neighbour samplers and EdgeBank's memory see the full graph (DyGLib's
`full_neighbor_sampler`), memory models replay the reduced training stream,
and DyGJEPA's causal event history is built from the reduced training
snapshots followed by the full evaluation ones. The run prints an
`inductive_setting` summary (nodes held out, events removed, events scored)
and writes `results/inductive_setting/*_inductive_setting.json`; the
multi-seed files carry `"setting": "inductive"`. It cannot be combined with
historical/inductive negatives.

### DyGJEPA module ablations (separate runs)

Each ablation removes one module of the full model and is otherwise the main
run: `configs/ablation/<variant>.yaml` is an overlay of
`link_comparison_all.yaml` (same datasets, recipes, seeds 42/44/46/48/50 and
random-negative protocol) whose `ablation.rcps_jepa` / `ablation.rcps_training`
keys are applied *after* the per-dataset overrides, so the module is removed on
every dataset. Results go to `results/ablation/<variant>/`.

| variant | removes |
|---|---|
| `prior_only` | all training (`epochs: 0`): the epoch-0 evaluation of the parameter-free recurrence prior |
| `no_history` | the causal event-prefix history (history latent and recurrence prior) |
| `no_signature` | node and pair path signatures |
| `no_subgraph` | the pair-conditioned subgraph context |
| `no_trajectories` | the individual and relational GRU trajectories of the node context |
| `no_jepa` | the JEPA objectives (node, relation, variance, covariance losses): a purely supervised model |
| `no_id` | the transductive ID embeddings of the link head |

A removed module is replaced by zeros of the same width, so every layer keeps
its shape and initialisation; with all switches at their defaults the code
path of the main run is unchanged (`bash scripts/check_rcps_ablation.sh`
verifies this bit-for-bit against `git HEAD` on CPU).

```bash
bash scripts/run_link_ablation.sh --list                        # variants
bash scripts/run_link_ablation.sh no_history wikipedia uci canparl  # datasets are named explicitly
bash scripts/run_link_ablation.sh no_signature wikipedia canparl
VARIANT=no_history sbatch run_link_ablation.sbatch              # cluster; DATASET_NAMES/SEEDS/EPOCHS as usual
python scripts/summarize_ablation.py                            # table vs results/link_comparison_<ds>.json
python scripts/summarize_ablation.py --metric auc --format latex
```

### Efficiency (separate run)

Every model of every comparison run now records an `efficiency` block next to
its `validation` / `test` metrics (`jepa_compare/efficiency.py`): trainable and
total parameters, setup time, seconds per training epoch, number of epochs run,
time to the selected checkpoint (setup + pretraining + training epochs and
validation passes up to `best_epoch`), total training time, final test
inference time (and examples per second), total wall-clock, and on CUDA the
peak allocated memory during the run versus the memory already resident before
it. Phase timings synchronise the device, so the metrics of the comparison are
unchanged. For the paper the numbers come from one dedicated single-seed pass
in which every model of a dataset runs sequentially on the same GPU:

```bash
bash scripts/run_link_efficiency.sh enron wikipedia            # one seed, all models, results/efficiency/
DATASET_NAMES="enron" sbatch run_link_efficiency.sbatch         # cluster; one job per dataset, same --gres
python scripts/summarize_efficiency.py --results results/efficiency            # markdown table per dataset
python scripts/summarize_efficiency.py --results results/efficiency --format latex
python scripts/summarize_efficiency.py --results results/efficiency --plot     # bubble + relative-cost figures
```

`configs/link_comparison_all_efficiency.yaml` is an overlay of
`link_comparison_all.yaml` with `seed: [42]` and `output_dir: results/efficiency`,
so every recipe is the main run's. Do not split the models of one dataset over
different GPUs or nodes when timing them.

### Noise robustness (separate run)

DyG-Mamba's robustness test (§5.5): every model is trained and checkpoint-
selected on the clean data with the main protocol, then the selected checkpoint
is re-scored on the clean test positives while 10%–60% random noisy events are
inserted into the history it sees. Noise is generated per snapshot (bin) so the
binning and the split never move; noisy events enter the context snapshots'
structure and activity, DyGJEPA's causal event history, and the neighbor
samplers / event streams of the continuous-time baselines, while the test
positives and their negatives stay identical across rates
(`jepa_compare/compare_link_robustness.py`).

```bash
bash scripts/run_link_robustness.sh wikipedia                       # tgn tgat dvgmae rcps_jepa, rates 0-0.6
MODELS="rcps_jepa tgn" RATES="0 0.3 0.6" bash scripts/run_link_robustness.sh uci
EPOCHS=1 MODELS="rcps_jepa dvgmae" bash scripts/run_link_robustness.sh uci  # smoke test
```

`configs/link_comparison_all_robustness.yaml` (overlay of
`link_comparison_all.yaml`) fixes the models, one seed, the noise rates, the
noise seed and how noisy events get edge features (`resample` from real events
of the same bin, or `zeros`). Results: `results/robustness/link_robustness_<ds>.json`
(per model: clean test metrics and metrics at every rate), the same as
`.csv` with the relative AP drop, and `results/robustness/figures/<ds>_noise_ap.{pdf,png}`
(AP versus noise rate, drop at the last rate annotated). A rate-0 evaluation
that differs from the training test pass is reported as a warning.

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

### Reproducing the node experiments on another machine

`data/raw/` is not in git. Everything needed lives on the GitHub release
`node-data-v1` (raw SpikeNet files; DeepWalk features when uploaded):

```bash
git clone git@github.com:HyoukaLI/DyGJEPA.git && cd DyGJEPA
pip install -e '.[features]'                 # + torch/pyyaml for the runs
scripts/download_node_data.sh                # data/raw/{dblp,tmall,patent}/...
# if the release has no <dataset>.npy yet, generate it (CPU, resumable):
sbatch generate_node_features.sbatch         # array: 0=dblp 1=tmall 2=patent
# otherwise just build the archives:
python scripts/prepare_dblp.py
python scripts/prepare_spikenet_node.py --dataset tmall
python scripts/prepare_spikenet_node.py --dataset patent
sbatch run_node_datasets.sbatch              # 3 datasets x 5 seeds
```

`scripts/upload_node_data.sh` (needs `gh`) publishes the raw files, and with
`WITH_NPY=1` the generated features split into <2 GB parts, to that release.

### Node datasets on a fresh clone (one command)

```bash
bash scripts/run_node_datasets.sh tmall      # prepares the data if needed, then trains
```

When `data/processed/<dataset>.npz` is missing or still carries the 4-D
structural fallback, the launcher first runs
`bash scripts/prepare_node_dataset.sh <dataset>`, which (1) takes the raw files
from the repository (Tmall: `data/raw/tmall/tmall.txt.gz` + `node2label.txt`,
read transparently) or from the GitHub release (DBLP, Patent), (2) downloads the
pre-computed DeepWalk features `<dataset>.npy` from the release (split parts are
reassembled), and (3) builds the archive with them. The downloads need network,
so on a cluster run the preparation once on a login node before submitting jobs.
`GENERATE=1` regenerates the features with the official recipe when the release
has none (hours for Tmall); `AUTO_PREPARE=0` restores the old fail-fast behaviour.
The rebuilt archives embed the features, so `tmall.npz` grows to about 3.5 GB
and `patent.npz` to about 11 GB (both are git-ignored).

### Node datasets need SpikeNet DeepWalk features first

The SG-JEPA paper (and SpikeNet, whose DBLP/Tmall/Patent release it uses)
feeds every model 80-dimensional per-snapshot DeepWalk node features
(`<dataset>.npy`, shape `[T, N, 80]`). Without them the converters write a 4-D
structural placeholder and **every** model (SG-JEPA, EvolveGCN, ROLAND, ...)
collapses to the majority class (DBLP: ~0.05 Macro-F1 / ~0.29 Micro-F1 instead
of the paper's ~0.74 / ~0.75). `load_npz` therefore refuses to load such an
archive, and `scripts/run_node_datasets.sh` (hence the Slurm job) checks every
requested archive's `feature_source` before starting; set
`DYGJEPA_ALLOW_STRUCTURAL_FALLBACK=1` to run the fallback protocol on purpose.
Two ways to obtain the features:

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

Per-job runs (one model, one seed, one labelled-training ratio) use the same
knobs as the link launcher; each job writes its own file, so any number of
them can run in parallel:

```bash
MODELS=cawn SEED=42 TRAIN_RATIO=0.6 OUTPUT_DIR=results/tmall OUTPUT_NAME=cawn \
  bash scripts/run_node_datasets.sh tmall
# -> results/tmall/cawn_ratio0.6_seed42.json  (keys: cawn only)
MODELS="sg_jepa rcps_jepa" SEED=42 TRAIN_RATIO=0.4 OUTPUT_DIR=results/tmall OUTPUT_NAME=jepa \
  bash scripts/run_node_datasets.sh tmall
# -> results/tmall/jepa_ratio0.4_seed42.json  (keys: sg_jepa, rcps_jepa)
```

`MODELS` picks any subset of `sg_jepa rcps_jepa evolvegcn_h roland tgn tgat
cawn tcl graphmixer dygformer cldg maskdgnn dvgmae` (default: both JEPA models
plus `node_baselines.enabled`); `TRAIN_RATIO` overrides `probe.train_ratio`
(default 0.4) and appends `_ratio<r>` to the file stem; `SEED`/`SEEDS`,
`EPOCHS` and `BASELINES` work as before. The same options exist on the driver
as `--models`, `--train-ratio`, `--output`, `--output-name`.

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
