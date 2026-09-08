# Dataset layout

The comparison scripts load the portable archives in `data/processed/`.
Processed archives are tracked with Git LFS; `data/raw/` is intentionally
ignored because it contains large, duplicate preprocessing inputs.

## Ready to run

| Dataset | Archive | Config | Graph type |
| --- | --- | --- | --- |
| Wikipedia | `processed/wikipedia.npz` | `../configs/link_comparison_all.yaml` | bipartite link prediction |
| MOOC | `processed/mooc.npz` | `../configs/link_comparison_all.yaml` | bipartite link prediction |
| LastFM | `processed/lastfm.npz` | `../configs/link_comparison_all.yaml` | bipartite link prediction |
| CanParl | `processed/canparl.npz` | `../configs/link_comparison_all.yaml` | homogeneous link prediction |
| Contacts | `processed/contacts.npz` | `../configs/link_comparison_all.yaml` | homogeneous link prediction |
| Flights | `processed/flights.npz` | `../configs/link_comparison_all.yaml` | homogeneous link prediction |
| UNtrade | `processed/untrade.npz` | `../configs/link_comparison_all.yaml` | homogeneous link prediction |
| UNvote | `processed/unvote.npz` | `../configs/link_comparison_all.yaml` | homogeneous link prediction |
| USLegis | `processed/uslegis.npz` | `../configs/link_comparison_all.yaml` | homogeneous link prediction |
| Enron | `processed/enron.npz` | `../configs/link_comparison_all.yaml` | homogeneous link prediction |
| UCI | `processed/uci.npz` | `../configs/link_comparison_all.yaml` | homogeneous link prediction |
| DBLP | `processed/dblp.npz` | `../configs/node_comparison_dblp.yaml` | homogeneous node prediction |

MOOC uses four-dimensional interaction features and state-change labels.
LastFM uses two-dimensional interaction features and has no positive
state-change labels, so its JODIE auxiliary state objective is disabled.

## Node archives (DBLP, Tmall, Patent) require DeepWalk features

The SG-JEPA/SpikeNet protocol uses 80-dimensional per-snapshot DeepWalk node
features, `data/raw/<dataset>/<dataset>.npy` with shape `[T, N, 80]`
(DBLP `[27, 28085, 80]`, Tmall `[19, 577314, 80]`, Patent
`[13, 2738012, 80]`). `prepare_dblp.py` and `prepare_spikenet_node.py` pick the
file up automatically; when it is missing they print a warning and write a 4-D
structural placeholder, and `load_npz` warns again at run time. That
placeholder carries no class information: on DBLP a linear probe on it reaches
4.5 Macro-F1 / 29 Micro-F1 (majority class = 29%), while the same probe on the
DeepWalk features reaches 66.7 / 66.7. Obtain the features from the SpikeNet
release (Dropbox for DBLP, Aliyun Drive for Tmall/Patent, rename `.txt` to
`.npy`) or regenerate them with the official recipe:

```bash
python scripts/generate_spikenet_deepwalk.py --dataset dblp
python scripts/generate_spikenet_deepwalk.py --dataset tmall    # --normalize by default
python scripts/generate_spikenet_deepwalk.py --dataset patent   # --normalize by default
# or on Slurm: DATASET=patent sbatch generate_node_features.sbatch
python scripts/prepare_dblp.py
python scripts/prepare_spikenet_node.py --dataset tmall
python scripts/prepare_spikenet_node.py --dataset patent
```

The generator is resumable (`<dataset>.npy.progress.json`) and streams the
random walks through gensim's `corpus_file` reader; Patent writes ~25 GB of
walks per snapshot to `--scratch-dir` (default `$TMPDIR`).

## Homogeneous event conversion

CanParl, Contacts, Flights, UNtrade, UNvote, USLegis, Enron and UCI are
homogeneous temporal graphs. Their standard DyGLib triples are stored under
`data/raw/<dataset>/` locally:

- `ml_<dataset>.csv`: timestamped interactions;
- `ml_<dataset>.npy`: interaction features;
- `ml_<dataset>_node.npy`: node features (not node-class labels).

They must not be passed through the bipartite Wikipedia converter. Rebuild all
eight archives with:

```bash
scripts/prepare_all_homogeneous.sh
```

The converter preserves directed duplicate events, exact timestamps and stable
event order. It creates 50 equal-event snapshots so every dataset receives the
same sliding-window split, while continuous-time baselines still read the
original timestamps. Feature normalization is fit only on the chronological
75% training prefix.

## Rebuilding the bipartite archives

```bash
python scripts/prepare_wikipedia.py \
  --input data/raw/mooc/mooc.csv \
  --output data/processed/mooc.npz \
  --event-bins 50

python scripts/prepare_wikipedia.py \
  --input data/raw/lastfm/lastfm.csv \
  --output data/processed/lastfm.npz \
  --event-bins 50
```

## Running all ready link datasets

One config contains the shared protocol plus dataset-specific overrides. The
driver loops over datasets and reinitializes/releases every model per run:

```bash
scripts/run_link_datasets.sh
```

Pass dataset names to run a subset:

```bash
scripts/run_link_datasets.sh mooc lastfm
```

The direct equivalent is:

```bash
python -m jepa_compare.compare_link_prediction \
  --config configs/link_comparison_all.yaml
```

On Slurm, submit `run_link_datasets.sbatch`. Override `PROJECT_DIR`,
`PYTHON_BIN`, or `DATASET_NAMES` in the submission environment when needed.
