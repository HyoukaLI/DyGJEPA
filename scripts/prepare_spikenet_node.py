from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.prepare_dblp import structural_features
except ModuleNotFoundError:  # direct invocation from the repository root
    from prepare_dblp import structural_features


OFFICIAL_MERGE_STEPS = {"tmall": 10, "patent": 2}


def canonical_dataset(name: str) -> str:
    """Accept the common Tsmall typo while keeping the official Tmall name."""
    normalized = name.lower()
    if normalized == "tsmall":
        return "tmall"
    if normalized not in OFFICIAL_MERGE_STEPS:
        raise ValueError(f"unsupported dataset: {name}")
    return normalized


def _encode_labels(values: list[str]) -> np.ndarray:
    classes = {value: index for index, value in enumerate(sorted(set(values)))}
    return np.asarray([classes[value] for value in values], dtype=np.int64)


def _read_tmall(
    edge_path: Path, label_path: Path
) -> tuple[list[np.ndarray], list[str], int, np.ndarray, np.ndarray]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    max_node = -1
    with edge_path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{edge_path}:{line_no}: expected src dst timestamp")
            src, dst, timestamp = int(fields[0]), int(fields[1]), fields[2]
            if src < 0 or dst < 0:
                raise ValueError(f"{edge_path}:{line_no}: node ids must be non-negative")
            grouped[timestamp].append((src, dst))
            max_node = max(max_node, src, dst)
    if not grouped:
        raise ValueError(f"{edge_path} contains no edges")

    labeled_nodes: list[int] = []
    raw_labels: list[str] = []
    with label_path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{label_path}:{line_no}: expected node label")
            node = int(fields[0])
            if node < 0 or node > max_node:
                raise ValueError(f"{label_path}:{line_no}: invalid node {node}")
            labeled_nodes.append(node)
            raw_labels.append(fields[1])
    if not labeled_nodes:
        raise ValueError(f"{label_path} contains no labels")
    if len(set(labeled_nodes)) != len(labeled_nodes):
        raise ValueError(f"{label_path} contains duplicate node labels")

    num_nodes = max_node + 1
    labeled = np.asarray(labeled_nodes, dtype=np.int64)
    # Keep the exact official implementation's integer-set iteration order;
    # the released tmall.npy was generated after this same reindexing.
    unlabeled = np.asarray(
        list(set(range(num_nodes)) - set(labeled_nodes)), dtype=np.int64
    )
    # The official SpikeNet loader places labeled nodes first before generating
    # Tmall DeepWalk features. Reproduce that order so tmall.npy stays aligned.
    original_node_ids = np.concatenate([labeled, unlabeled])
    old_to_new = np.empty(num_nodes, dtype=np.int64)
    old_to_new[original_node_ids] = np.arange(num_nodes, dtype=np.int64)

    labels = np.full(num_nodes, -1, dtype=np.int64)
    labels[: labeled.size] = _encode_labels(raw_labels)
    timestamps = sorted(grouped)
    increments = [
        old_to_new[np.asarray(grouped[t], dtype=np.int64).T] for t in timestamps
    ]
    return increments, timestamps, num_nodes, labels, original_node_ids


def _literal_tuple(path: Path, line_no: int, line: str, length: int) -> tuple:
    try:
        value = ast.literal_eval(line)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path}:{line_no}: invalid Python/JSON tuple") from exc
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{path}:{line_no}: expected a {length}-field tuple")
    return tuple(value)


def _read_patent(
    edge_path: Path, node_path: Path
) -> tuple[list[np.ndarray], list[int], int, np.ndarray, np.ndarray]:
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    max_edge_node = -1
    with edge_path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            src, dst, date, _, _ = _literal_tuple(edge_path, line_no, line, 5)
            src, dst, year = int(src), int(dst), int(date) // 10_000
            if src < 0 or dst < 0:
                raise ValueError(f"{edge_path}:{line_no}: node ids must be non-negative")
            grouped[year].append((src, dst))
            max_edge_node = max(max_edge_node, src, dst)
    if not grouped:
        raise ValueError(f"{edge_path} contains no edges")

    labels_by_node: dict[int, int] = {}
    max_label_node = -1
    with node_path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            node, _, _, label = _literal_tuple(node_path, line_no, line, 4)
            node, label = int(node), int(label) - 1
            if node < 0 or label < 0:
                raise ValueError(f"{node_path}:{line_no}: invalid node or one-based label")
            if node in labels_by_node:
                raise ValueError(f"{node_path}:{line_no}: duplicate node {node}")
            labels_by_node[node] = label
            max_label_node = max(max_label_node, node)

    num_nodes = max(max_edge_node, max_label_node) + 1
    labels = np.full(num_nodes, -1, dtype=np.int64)
    for node, label in labels_by_node.items():
        labels[node] = label
    if np.any(labels < 0):
        raise ValueError("Patent requires exactly one label for every node id")

    timestamps = sorted(grouped)
    increments = [np.asarray(grouped[t], dtype=np.int64).T for t in timestamps]
    return increments, timestamps, num_nodes, labels, np.arange(num_nodes, dtype=np.int64)


def _merge_snapshots(
    increments: list[np.ndarray], timestamps: list[str | int], step: int
) -> tuple[list[np.ndarray], list[str | int]]:
    if step < 1:
        raise ValueError("merge step must be positive")
    merged_edges, merged_times = [], []
    for start in range(0, len(increments), step):
        block = increments[start : start + step]
        merged_edges.append(np.concatenate(block, axis=1))
        merged_times.append(timestamps[min(start + step, len(timestamps)) - 1])
    return merged_edges, merged_times


def _cumulative_undirected(increments: list[np.ndarray]) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    cumulative: list[np.ndarray] = []
    for edge_index in increments:
        chunks.append(np.concatenate([edge_index, edge_index[::-1]], axis=1))
        cumulative.append(np.concatenate(chunks, axis=1))
    return cumulative


def _load_features(path: Path, steps: int, nodes: int) -> np.ndarray:
    source = np.load(path, mmap_mode="r")
    expected_prefix = (steps, nodes)
    if source.ndim != 3 or source.shape[:2] != expected_prefix:
        raise ValueError(
            f"SpikeNet features must have shape [{steps}, {nodes}, F]; got {source.shape}"
        )
    if source.shape[2] != 80:
        raise ValueError(
            f"SpikeNet DeepWalk uses 80 feature dimensions; got {source.shape[2]}"
        )
    # SpikeNet standardizes every snapshot as one flattened vector. Normalizing
    # one slice at a time avoids an additional full-size temporary array.
    features = np.empty(source.shape, dtype=np.float32)
    for t in range(steps):
        current = np.asarray(source[t], dtype=np.float32)
        features[t] = (current - current.mean()) / (current.std() + 1e-6)
    return features


def convert(
    dataset: str,
    input_dir: Path,
    output: Path,
    feature_path: Path | None,
    merge_step: int | None = None,
) -> None:
    dataset = canonical_dataset(dataset)
    if dataset == "tmall":
        increments, timestamps, num_nodes, labels, original_node_ids = _read_tmall(
            input_dir / "tmall.txt", input_dir / "node2label.txt"
        )
    else:
        increments, timestamps, num_nodes, labels, original_node_ids = _read_patent(
            input_dir / "patent_edges.json", input_dir / "patent_nodes.json"
        )

    step = OFFICIAL_MERGE_STEPS[dataset] if merge_step is None else merge_step
    increments, timestamps = _merge_snapshots(increments, timestamps, step)
    cumulative = _cumulative_undirected(increments)
    if feature_path is None:
        features = structural_features(cumulative, num_nodes)
        feature_source = "structural-fallback"
    else:
        features = _load_features(feature_path, len(cumulative), num_nodes)
        feature_source = str(feature_path)
    active = np.stack(
        [np.bincount(edge[0], minlength=num_nodes) > 0 for edge in cumulative]
    )

    payload: dict[str, np.ndarray] = {
        "features": features,
        "active": active,
        "labels": labels,
        "original_node_ids": original_node_ids,
        "timestamps": np.asarray(timestamps),
        "dataset": np.asarray(dataset),
        "merge_step": np.asarray(step, dtype=np.int64),
        "feature_source": np.asarray(feature_source),
    }
    payload.update({f"edges_{t}": edge for t, edge in enumerate(cumulative)})
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    labeled = int((labels >= 0).sum())
    print(
        f"saved {output}: dataset={dataset}, T={len(cumulative)}, N={num_nodes}, "
        f"labeled={labeled}, E_final={cumulative[-1].shape[1] // 2}, "
        f"F={features.shape[-1]}, merge_step={step}, feature_source={feature_source}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official SpikeNet Tmall/Patent data to DYGJEPA NPZ"
    )
    parser.add_argument("--dataset", required=True, choices=["tmall", "tsmall", "patent"])
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--merge-step",
        type=int,
        default=None,
        help="override the official temporal merge (Tmall=10, Patent=2)",
    )
    parser.add_argument(
        "--structural-features",
        action="store_true",
        help="force the 4-D featureless fallback even if <dataset>.npy exists",
    )
    args = parser.parse_args()
    dataset = canonical_dataset(args.dataset)
    input_dir = args.input_dir or Path("data/raw") / dataset
    output = args.output or Path("data/processed") / f"{dataset}.npz"
    if args.structural_features:
        feature_path = None
    else:
        default_features = input_dir / f"{dataset}.npy"
        feature_path = args.features or (
            default_features if default_features.is_file() else None
        )
        if args.features is not None and not args.features.is_file():
            parser.error(f"feature file does not exist: {args.features}")
        if feature_path is None:
            print(
                f"{default_features} not found; using the same 4-D "
                "structural-fallback protocol as the packaged DBLP archive"
            )
    convert(dataset, input_dir, output, feature_path, args.merge_step)


if __name__ == "__main__":
    main()
