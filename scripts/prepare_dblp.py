from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def structural_features(edges: list[np.ndarray], num_nodes: int) -> np.ndarray:
    """Featureless fallback: cumulative degree, log-degree, activity, first-seen time."""
    outputs = []
    first_seen = np.full(num_nodes, len(edges), dtype=np.float32)
    for t, edge_index in enumerate(edges):
        degree = np.bincount(edge_index[0], minlength=num_nodes).astype(np.float32)
        active = degree > 0
        first_seen[(first_seen == len(edges)) & active] = t
        degree_z = (degree - degree.mean()) / (degree.std() + 1e-6)
        log_degree = np.log1p(degree)
        log_degree = (log_degree - log_degree.mean()) / (log_degree.std() + 1e-6)
        seen = np.where(first_seen < len(edges), first_seen / max(len(edges) - 1, 1), 1.0)
        outputs.append(np.stack([degree_z, log_degree, active.astype(np.float32), seen], axis=-1))
    return np.stack(outputs).astype(np.float32)


def load_deepwalk(path: Path, steps: int, nodes: int) -> np.ndarray:
    features = np.load(path).astype(np.float32)
    expected = (steps, nodes, 80)
    if features.shape != expected:
        raise ValueError(
            f"SpikeNet-compatible DBLP DeepWalk features must have shape {expected}; "
            f"got {features.shape}. 80 is the embedding dimension and 128 is "
            "the number of walks per node."
        )
    flat = features.reshape(steps, -1)
    flat = (flat - flat.mean(axis=1, keepdims=True)) / (flat.std(axis=1, keepdims=True) + 1e-6)
    return flat.reshape(features.shape).astype(np.float32)


def convert(edge_path: Path, label_path: Path, output: Path, feature_path: Path | None) -> None:
    grouped: dict[float, list[tuple[int, int]]] = defaultdict(list)
    max_node = -1
    with edge_path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{edge_path}:{line_no}: expected src dst timestamp")
            src, dst, timestamp = int(fields[0]), int(fields[1]), float(fields[2])
            grouped[timestamp].append((src, dst))
            max_node = max(max_node, src, dst)
    num_nodes = max_node + 1
    labels = np.full(num_nodes, -1, dtype=np.int64)
    with label_path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            node, label = map(int, line.split())
            if not 0 <= node < num_nodes:
                raise ValueError(f"{label_path}:{line_no}: invalid node {node}")
            labels[node] = label
    if np.any(labels < 0):
        raise ValueError("labels are missing for one or more nodes")

    cumulative: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    for timestamp in sorted(grouped):
        now = np.asarray(grouped[timestamp], dtype=np.int64).T
        chunks.append(np.concatenate([now, now[::-1]], axis=1))
        cumulative.append(np.concatenate(chunks, axis=1))

    if feature_path is None:
        features = structural_features(cumulative, num_nodes)
        feature_source = "structural-fallback"
    else:
        features = load_deepwalk(feature_path, len(cumulative), num_nodes)
        feature_source = str(feature_path)
    active = np.stack([np.bincount(e[0], minlength=num_nodes) > 0 for e in cumulative])
    payload: dict[str, np.ndarray] = {
        "features": features,
        "active": active,
        "labels": labels,
        "timestamps": np.asarray(sorted(grouped), dtype=np.float32),
        "feature_source": np.asarray(feature_source),
    }
    payload.update({f"edges_{t}": edge for t, edge in enumerate(cumulative)})
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(
        f"saved {output}: T={len(cumulative)}, N={num_nodes}, "
        f"events={sum(len(v) for v in grouped.values())}, F={features.shape[-1]}, "
        f"feature_source={feature_source}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SpikeNet-format DBLP to SG-JEPA NPZ")
    parser.add_argument("--edges", type=Path, default=Path("data/raw/dblp/dblp.txt"))
    parser.add_argument("--labels", type=Path, default=Path("data/raw/dblp/node2label.txt"))
    parser.add_argument(
        "--features", type=Path, default=None,
        help="optional SpikeNet-compatible dblp.npy [27, 28085, 80]",
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/dblp.npz"))
    args = parser.parse_args()
    convert(args.edges, args.labels, args.output, args.features)


if __name__ == "__main__":
    main()
