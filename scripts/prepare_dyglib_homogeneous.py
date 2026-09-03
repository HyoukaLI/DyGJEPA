from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _resolve_files(dataset_dir: Path) -> tuple[Path, Path, Path]:
    csv_files = list(dataset_dir.glob("ml_*.csv"))
    node_files = list(dataset_dir.glob("ml_*_node.npy"))
    edge_files = [
        path
        for path in dataset_dir.glob("ml_*.npy")
        if not path.name.endswith("_node.npy")
    ]
    if len(csv_files) != 1 or len(edge_files) != 1 or len(node_files) != 1:
        raise ValueError(
            f"{dataset_dir} must contain one ml_*.csv, ml_*.npy and ml_*_node.npy"
        )
    return csv_files[0], edge_files[0], node_files[0]


def _read_events(
    csv_path: Path, edge_feature_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge_table = np.load(edge_feature_path, mmap_mode="r")
    if edge_table.ndim != 2 or edge_table.shape[0] < 2:
        raise ValueError("DyGLib edge features must include row-zero padding")
    event_count = edge_table.shape[0] - 1
    sources = np.empty(event_count, dtype=np.int64)
    destinations = np.empty(event_count, dtype=np.int64)
    timestamps = np.empty(event_count, dtype=np.float64)
    labels = np.empty(event_count, dtype=np.int64)
    edge_ids = np.empty(event_count, dtype=np.int64)

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"u", "i", "ts", "label", "idx"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"{csv_path} does not follow the DyGLib CSV schema")
        rows = 0
        for rows, row in enumerate(reader, start=1):
            if rows > event_count:
                raise ValueError("CSV has more events than the feature array")
            position = rows - 1
            sources[position] = int(row["u"])
            destinations[position] = int(row["i"])
            timestamps[position] = float(row["ts"])
            labels[position] = int(float(row["label"]))
            edge_ids[position] = int(row["idx"])
    if rows != event_count:
        raise ValueError(
            f"CSV/feature event mismatch: {rows} rows versus {event_count} features"
        )
    if edge_ids.min() < 1 or edge_ids.max() >= edge_table.shape[0]:
        raise ValueError("edge idx lies outside the feature array")

    raw_features = np.asarray(edge_table[edge_ids], dtype=np.float32)
    order = np.argsort(timestamps, kind="stable")
    return (
        sources[order], destinations[order], timestamps[order],
        raw_features[order], labels[order],
    )


def _event_bins(timestamps: np.ndarray, requested_bins: int) -> list[np.ndarray]:
    if requested_bins > len(timestamps):
        raise ValueError("event_bins cannot exceed the number of interactions")
    # Match the bipartite converter: equal-event bins make the downstream
    # window split identical across datasets. Exact timestamps remain attached
    # to every event for continuous-time models, including stable ordering of
    # events that share a timestamp.
    return list(np.array_split(np.arange(len(timestamps)), requested_bins))


def convert(
    dataset_dir: Path,
    output_path: Path,
    *,
    event_bins: int = 50,
    identity_dim: int = 16,
    event_feature_dim: int = 16,
    train_ratio: float = 0.75,
    seed: int = 42,
) -> None:
    csv_path, edge_feature_path, node_feature_path = _resolve_files(dataset_dir)
    sources, destinations, timestamps, raw_features, labels = _read_events(
        csv_path, edge_feature_path
    )
    if event_bins < 6 or identity_dim < 1 or event_feature_dim < 1:
        raise ValueError("event_bins must be >= 6 and feature dimensions positive")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not np.isfinite(timestamps).all() or not np.isfinite(raw_features).all():
        raise ValueError("timestamps and interaction features must be finite")

    raw_node_features = np.load(node_feature_path, mmap_mode="r")
    num_nodes = int(max(sources.max(), destinations.max()))
    if sources.min() < 1 or destinations.min() < 1:
        raise ValueError("expected DyGLib's one-based node ids with row-zero padding")
    if raw_node_features.ndim != 2 or raw_node_features.shape[0] <= num_nodes:
        raise ValueError("node feature array does not cover every node id")
    sources = sources - 1
    destinations = destinations - 1

    fit_end = max(1, int(len(timestamps) * train_ratio))
    mean = raw_features[:fit_end].mean(axis=0, keepdims=True)
    std = raw_features[:fit_end].std(axis=0, keepdims=True)
    normalized = (raw_features - mean) / np.maximum(std, 1e-6)
    rng = np.random.default_rng(seed)
    if normalized.shape[1] > event_feature_dim:
        projection = rng.normal(
            0.0,
            1.0 / np.sqrt(normalized.shape[1]),
            size=(normalized.shape[1], event_feature_dim),
        ).astype(np.float32)
        event_features = normalized @ projection
    elif normalized.shape[1] < event_feature_dim:
        event_features = np.pad(
            normalized,
            ((0, 0), (0, event_feature_dim - normalized.shape[1])),
        )
    else:
        event_features = normalized
    event_features = np.asarray(event_features, dtype=np.float32)

    identity = rng.normal(size=(num_nodes, identity_dim)).astype(np.float32)
    identity /= np.maximum(np.linalg.norm(identity, axis=1, keepdims=True), 1e-6)
    feature_dim = identity_dim + event_feature_dim + 4
    bins = _event_bins(timestamps, event_bins)
    features = np.zeros((len(bins), num_nodes, feature_dim), dtype=np.float32)
    active = np.ones((len(bins), num_nodes), dtype=bool)
    cumulative_out = np.zeros(num_nodes, dtype=np.float32)
    cumulative_in = np.zeros(num_nodes, dtype=np.float32)
    last_out = np.full(num_nodes, timestamps[0], dtype=np.float64)
    last_in = np.full(num_nodes, timestamps[0], dtype=np.float64)
    archive: dict[str, np.ndarray] = {}
    bin_timestamps: list[float] = []

    for bin_index, event_indices in enumerate(bins):
        source = sources[event_indices]
        destination = destinations[event_indices]
        queries = np.stack([source, destination])
        messages = np.concatenate([queries, queries[[1, 0]]], axis=1)
        messages = np.unique(messages, axis=1).astype(np.int64, copy=False)
        archive[f"edges_{bin_index}"] = messages
        archive[f"queries_{bin_index}"] = queries.astype(np.int64, copy=False)
        archive[f"query_timestamps_{bin_index}"] = timestamps[event_indices].astype(
            np.float32, copy=False
        )
        archive[f"query_features_{bin_index}"] = raw_features[event_indices]
        archive[f"query_labels_{bin_index}"] = labels[event_indices]

        aggregated = np.zeros((num_nodes, event_feature_dim), dtype=np.float32)
        counts = np.zeros(num_nodes, dtype=np.float32)
        np.add.at(aggregated, source, event_features[event_indices])
        np.add.at(aggregated, destination, event_features[event_indices])
        np.add.at(counts, source, 1.0)
        np.add.at(counts, destination, 1.0)
        aggregated /= np.maximum(counts[:, None], 1.0)
        np.add.at(cumulative_out, source, 1.0)
        np.add.at(cumulative_in, destination, 1.0)
        current_time = float(timestamps[event_indices[-1]])
        last_out[np.unique(source)] = current_time
        last_in[np.unique(destination)] = current_time
        span = max(1.0, current_time - float(timestamps[0]))

        out_count = np.log1p(cumulative_out)[:, None]
        in_count = np.log1p(cumulative_in)[:, None]
        count_scale = max(1.0, float(max(out_count.max(), in_count.max())))
        out_count /= count_scale
        in_count /= count_scale
        out_recency = np.clip((current_time - last_out) / span, 0.0, 1.0)[:, None]
        in_recency = np.clip((current_time - last_in) / span, 0.0, 1.0)[:, None]
        features[bin_index] = np.concatenate(
            [identity, aggregated, out_count, in_count, out_recency, in_recency],
            axis=1,
        )
        bin_timestamps.append(current_time)

    archive.update(
        {
            "features": features,
            "active": active,
            "timestamps": np.asarray(bin_timestamps, dtype=np.float64),
            "node_ids": np.arange(num_nodes, dtype=np.int64),
            "feature_source": np.asarray(
                "causal_events+fixed_identity+directional_activity", dtype="U"
            ),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **archive)
    print(
        f"wrote {output_path}: events={len(timestamps)}, nodes={num_nodes}, "
        f"bins={len(bins)}, raw_event_dim={raw_features.shape[1]}, "
        f"node_feature_dim={feature_dim}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one homogeneous DyGLib event dataset to snapshot NPZ"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-bins", type=int, default=50)
    parser.add_argument("--identity-dim", type=int, default=16)
    parser.add_argument("--event-feature-dim", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    convert(
        args.dataset_dir,
        args.output,
        event_bins=args.event_bins,
        identity_dim=args.identity_dim,
        event_feature_dim=args.event_feature_dim,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
