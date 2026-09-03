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

The current local DBLP archive contains the four-dimensional structural
fallback. Regenerate it with `--features data/raw/dblp/dblp.npy` before a
paper-aligned SG-JEPA comparison that requires 80-dimensional DeepWalk input.

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
