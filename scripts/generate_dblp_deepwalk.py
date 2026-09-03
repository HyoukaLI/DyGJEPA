from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np


# Exact interpretation of SpikeNet's positional call:
# DeepWalk(80, 10, 128, window_size=10, negative=1, workers=16).
DEFAULT_DIMENSIONS = 80
DEFAULT_WALK_LENGTH = 10
DEFAULT_WALKS_PER_NODE = 128
DEFAULT_WINDOW_SIZE = 10
DEFAULT_NEGATIVE = 1
DEFAULT_WORKERS = 16


def require_feature_dependencies():
    try:
        import scipy.sparse as sp
        from gensim.models import Word2Vec
        from numba import njit
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            "DeepWalk dependencies are missing. Run: pip install -e '.[features]'"
        ) from exc
    return sp, Word2Vec, njit, tqdm


def read_temporal_edges(path: Path) -> tuple[list[np.ndarray], int]:
    grouped: dict[float, list[tuple[int, int]]] = defaultdict(list)
    max_node = -1
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_no}: expected src dst timestamp")
            src, dst, timestamp = int(fields[0]), int(fields[1]), float(fields[2])
            grouped[timestamp].append((src, dst))
            max_node = max(max_node, src, dst)
    if not grouped:
        raise ValueError(f"{path} contains no temporal edges")
    return [np.asarray(grouped[t], dtype=np.int64).T for t in sorted(grouped)], max_node + 1


def cumulative_adjacencies(path: Path):
    sp, _, _, _ = require_feature_dependencies()
    increments, num_nodes = read_temporal_edges(path)
    cumulative = None
    snapshots = []
    for edge_index in increments:
        now = sp.csr_matrix(
            (np.ones(edge_index.shape[1], dtype=np.float32), edge_index),
            shape=(num_nodes, num_nodes),
        )
        now = now.maximum(now.T)
        now.data[:] = 1.0
        cumulative = now if cumulative is None else (cumulative + now)
        cumulative.data[:] = 1.0
        cumulative.eliminate_zeros()
        snapshots.append(cumulative.copy())
    return snapshots


def make_walk_kernel(njit):
    @njit(cache=True)
    def random_walks(indices, indptr, walk_length, walks_per_node, seed):
        np.random.seed(seed)
        nodes = indptr.size - 1
        total = nodes * walks_per_node
        walks = np.empty((total, walk_length), dtype=np.int32)
        lengths = np.ones(total, dtype=np.int16)
        row = 0
        for _ in range(walks_per_node):
            for start in range(nodes):
                current = start
                walks[row, 0] = start
                for step in range(1, walk_length):
                    begin, end = indptr[current], indptr[current + 1]
                    if begin == end:
                        break
                    current = indices[np.random.randint(begin, end)]
                    walks[row, step] = current
                    lengths[row] = step + 1
                row += 1
        return walks, lengths

    return random_walks


class WalkCorpus:
    def __init__(self, walks: np.ndarray, lengths: np.ndarray) -> None:
        self.walks = walks
        self.lengths = lengths

    def __iter__(self) -> Iterator[list[str]]:
        for walk, length in zip(self.walks, self.lengths):
            yield [str(int(node)) for node in walk[: int(length)]]


def generate(
    edge_path: Path,
    output: Path,
    dimensions: int = DEFAULT_DIMENSIONS,
    walk_length: int = DEFAULT_WALK_LENGTH,
    walks_per_node: int = DEFAULT_WALKS_PER_NODE,
    window_size: int = DEFAULT_WINDOW_SIZE,
    negative: int = DEFAULT_NEGATIVE,
    workers: int = DEFAULT_WORKERS,
    seed: int = 42,
    max_snapshots: int | None = None,
) -> None:
    _, Word2Vec, njit, tqdm = require_feature_dependencies()
    snapshots = cumulative_adjacencies(edge_path)
    if max_snapshots is not None:
        snapshots = snapshots[:max_snapshots]
    walk_kernel = make_walk_kernel(njit)
    feature_snapshots = []
    for t, graph in enumerate(tqdm(snapshots, desc="DeepWalk snapshots")):
        walks, lengths = walk_kernel(
            graph.indices.astype(np.int32), graph.indptr.astype(np.int64),
            walk_length, walks_per_node, seed + t,
        )
        model = Word2Vec(
            sentences=WalkCorpus(walks, lengths), vector_size=dimensions,
            window=window_size, min_count=0, sg=1, hs=0, alpha=0.025,
            epochs=1, negative=negative, workers=workers, seed=seed + t,
        )
        feature_snapshots.append(
            np.stack([model.wv[str(node)] for node in range(graph.shape[0])]).astype(np.float32)
        )
        del walks, lengths, model
    features = np.stack(feature_snapshots)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, features)
    print(f"saved {output}: shape={features.shape}, dtype={features.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SpikeNet-compatible DBLP DeepWalk features")
    parser.add_argument("--edges", type=Path, default=Path("data/raw/dblp/dblp.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/dblp/dblp.npy"))
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--walk-length", type=int, default=DEFAULT_WALK_LENGTH)
    parser.add_argument("--walks-per-node", type=int, default=DEFAULT_WALKS_PER_NODE)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--negative", type=int, default=DEFAULT_NEGATIVE)
    parser.add_argument("--workers", type=int, default=min(DEFAULT_WORKERS, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-snapshots", type=int, default=None, help="debug only")
    args = parser.parse_args()
    generate(args.edges, args.output, args.dimensions, args.walk_length,
             args.walks_per_node, args.window_size, args.negative,
             args.workers, args.seed, args.max_snapshots)


if __name__ == "__main__":
    main()
