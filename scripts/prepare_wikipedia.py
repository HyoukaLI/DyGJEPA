from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_events(
    path: Path,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    users: list[str] = []
    items: list[str] = []
    timestamps: list[float] = []
    labels: list[int] = []
    features: list[list[float]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) < 5:
            raise ValueError("Wikipedia CSV needs the JODIE header and feature columns")
        for line_number, row in enumerate(reader, start=2):
            if len(row) < 5:
                raise ValueError(f"row {line_number} has fewer than five columns")
            users.append(row[0])
            items.append(row[1])
            timestamps.append(float(row[2]))
            label = int(row[3])
            if label not in (0, 1):
                raise ValueError(f"row {line_number} has a non-binary state label")
            labels.append(label)
            features.append([float(value) for value in row[4:]])
    if not timestamps:
        raise ValueError("Wikipedia CSV contains no interactions")
    width = len(features[0])
    if width < 1 or any(len(row) != width for row in features):
        raise ValueError("interaction feature vectors must have a fixed positive width")
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    feature_array = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    if not np.isfinite(timestamp_array).all() or not np.isfinite(feature_array).all():
        raise ValueError("timestamps and interaction features must be finite")
    order = np.argsort(timestamp_array, kind="stable")
    return (
        [users[index] for index in order],
        [items[index] for index in order],
        timestamp_array[order],
        feature_array[order],
        label_array[order],
    )


def _ordered_ids(values: list[str]) -> tuple[dict[str, int], np.ndarray]:
    mapping: dict[str, int] = {}
    ordered = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
            ordered.append(value)
    return mapping, np.asarray(ordered)


def convert(
    input_path: Path,
    output_path: Path,
    event_bins: int = 50,
    identity_dim: int = 16,
    event_feature_dim: int = 16,
    seed: int = 42,
    train_ratio: float = 0.70,
) -> None:
    users, items, timestamps, raw_features, state_labels = _read_events(input_path)
    if event_bins < 6 or event_bins > len(users):
        raise ValueError("event_bins must be between 6 and the number of interactions")
    if identity_dim < 1 or event_feature_dim < 1:
        raise ValueError("identity_dim and event_feature_dim must be positive")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")

    user_map, user_names = _ordered_ids(users)
    item_map, item_names = _ordered_ids(items)
    num_users = len(user_map)
    num_items = len(item_map)
    num_nodes = num_users + num_items
    source = np.asarray([user_map[value] for value in users], dtype=np.int64)
    destination = np.asarray(
        [num_users + item_map[value] for value in items], dtype=np.int64
    )

    # Fit feature normalization on the chronological training prefix only.
    fit_end = max(1, int(len(users) * train_ratio))
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
        padding = np.zeros(
            (normalized.shape[0], event_feature_dim - normalized.shape[1]),
            dtype=np.float32,
        )
        event_features = np.concatenate([normalized, padding], axis=1)
    else:
        event_features = normalized

    identity = rng.normal(size=(num_nodes, identity_dim)).astype(np.float32)
    identity /= np.maximum(np.linalg.norm(identity, axis=1, keepdims=True), 1e-6)
    node_type = np.zeros((num_nodes, 2), dtype=np.float32)
    node_type[:num_users, 0] = 1.0
    node_type[num_users:, 1] = 1.0
    feature_dim = identity_dim + 2 + event_feature_dim + 2
    features = np.zeros((event_bins, num_nodes, feature_dim), dtype=np.float32)
    active = np.ones((event_bins, num_nodes), dtype=bool)
    cumulative_count = np.zeros(num_nodes, dtype=np.float32)
    last_seen = np.full(num_nodes, timestamps[0], dtype=np.float64)
    bins = np.array_split(np.arange(len(users)), event_bins)
    archive: dict[str, np.ndarray] = {}
    bin_timestamps = []

    for bin_index, event_indices in enumerate(bins):
        u = source[event_indices]
        v = destination[event_indices]
        directed_queries = np.stack([u, v])
        unique_pairs = np.unique(directed_queries, axis=1)
        message_edges = np.concatenate(
            [unique_pairs, unique_pairs[[1, 0]]], axis=1
        ).astype(np.int64, copy=False)
        archive[f"edges_{bin_index}"] = message_edges
        # Preserve directed duplicates for the actual future-interaction task.
        archive[f"queries_{bin_index}"] = directed_queries.astype(
            np.int64, copy=False
        )
        # Preserve the original event stream for continuous-time baselines.
        # These arrays stay aligned with the duplicate-preserving query edges.
        archive[f"query_timestamps_{bin_index}"] = timestamps[event_indices].astype(
            np.float32, copy=False
        )
        # JODIE consumes the original interaction feature vector.  The
        # projected features above are reserved for fixed-width snapshot node
        # features used by RCPS-JEPA.
        archive[f"query_features_{bin_index}"] = raw_features[event_indices].astype(
            np.float32, copy=False
        )
        archive[f"query_labels_{bin_index}"] = state_labels[event_indices].astype(
            np.int64, copy=False
        )

        messages = np.zeros((num_nodes, event_feature_dim), dtype=np.float32)
        counts = np.zeros(num_nodes, dtype=np.float32)
        np.add.at(messages, u, event_features[event_indices])
        np.add.at(messages, v, event_features[event_indices])
        np.add.at(counts, u, 1.0)
        np.add.at(counts, v, 1.0)
        messages /= np.maximum(counts[:, None], 1.0)
        cumulative_count += counts
        current_time = float(timestamps[event_indices[-1]])
        touched = counts > 0
        last_seen[touched] = current_time
        count_feature = np.log1p(cumulative_count)[:, None]
        count_feature /= max(1.0, float(np.log1p(cumulative_count.max())))
        total_span = max(1.0, current_time - float(timestamps[0]))
        recency = np.clip((current_time - last_seen) / total_span, 0.0, 1.0)[:, None]
        features[bin_index] = np.concatenate(
            [identity, node_type, messages, count_feature, recency.astype(np.float32)],
            axis=1,
        )
        bin_timestamps.append(current_time)

    archive.update(
        {
            "features": features,
            "active": active,
            "timestamps": np.asarray(bin_timestamps, dtype=np.float64),
            "num_source_nodes": np.asarray(num_users, dtype=np.int64),
            "user_ids": user_names,
            "item_ids": item_names,
            "feature_source": np.asarray(
                "causal_jodie_events+type+fixed_identity", dtype="U"
            ),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **archive)
    print(
        f"wrote {output_path}: events={len(users)}, users={num_users}, "
        f"items={num_items}, bins={event_bins}, feature_dim={feature_dim}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the JODIE Wikipedia event CSV to causal snapshots"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw/wikipedia/wikipedia.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/wikipedia.npz")
    )
    parser.add_argument("--event-bins", type=int, default=50)
    parser.add_argument("--identity-dim", type=int, default=16)
    parser.add_argument("--event-feature-dim", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    convert(
        args.input,
        args.output,
        event_bins=args.event_bins,
        identity_dim=args.identity_dim,
        event_feature_dim=args.event_feature_dim,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
