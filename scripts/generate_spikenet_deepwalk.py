"""Generate SpikeNet-style per-snapshot DeepWalk node features for DBLP / Tmall / Patent.

This reproduces the official SpikeNet recipe (EdisonLeeeee/SpikeNet,
``generate_feature.py``)::

    model = DeepWalk(80, 10, 128, window_size=10, negative=1, workers=16)
    for g in data.adj:                      # cumulative, merged snapshots
        model.fit(g)
        xs.append(model.get_embedding(normalize=args.normalize))
    np.save(f"{name}.npy", np.stack(xs))    # [T, N, 80]

with the official invocations::

    python generate_feature.py --dataset dblp
    python generate_feature.py --dataset tmall --normalize
    python generate_feature.py --dataset patent --normalize

Snapshots are built by exactly the same readers/merging as
``scripts/prepare_spikenet_node.py`` (Tmall: labeled-first reindexing and
merge step 10 -> 19 snapshots; Patent: merge step 2 -> 13 snapshots) and
``scripts/prepare_dblp.py`` (DBLP: 27 raw snapshots), so the generated
``data/raw/<dataset>/<dataset>.npy`` is picked up automatically by those
converters and stays aligned with the node ids inside the NPZ archives.

Differences from the reference implementation that do not change the model:

* Random walks are streamed to a plain-text corpus and consumed through
  gensim's multi-threaded ``corpus_file`` reader instead of materialising
  hundreds of millions of Python lists (Patent alone has 350M walks per
  snapshot).  Word2Vec hyperparameters are identical.
* Every finished snapshot is written straight into the output ``.npy``
  (``np.lib.format.open_memmap``) and recorded in ``<output>.progress.json``,
  so an interrupted job resumes where it stopped and Slurm array jobs can
  split snapshots with ``--start/--end``.

Dependencies: ``pip install -e '.[features]'`` (gensim, numba, scipy, tqdm).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_spikenet_node import (  # noqa: E402
    OFFICIAL_MERGE_STEPS,
    _merge_snapshots,
    _read_patent,
    _read_tmall,
    canonical_dataset,
)

# DeepWalk(80, 10, 128, window_size=10, negative=1, workers=16) in SpikeNet:
# positional arguments are (dimensions, walk_length, walk_number).
DIMENSIONS = 80
WALK_LENGTH = 10
WALKS_PER_NODE = 128
WINDOW_SIZE = 10
NEGATIVE = 1
LEARNING_RATE = 0.025
EPOCHS = 1
# generate_feature.py passes --normalize for Tmall and Patent but not DBLP.
OFFICIAL_NORMALIZE = {"dblp": False, "tmall": True, "patent": True}


# --------------------------------------------------------------------------- #
# Snapshot construction (shared with the NPZ converters)
# --------------------------------------------------------------------------- #
def read_dblp_increments(edge_path: Path) -> tuple[list[np.ndarray], int]:
    """Same grouping as scripts/prepare_dblp.py (one snapshot per timestamp)."""
    from collections import defaultdict

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
    if not grouped:
        raise ValueError(f"{edge_path} contains no edges")
    increments = [np.asarray(grouped[t], dtype=np.int64).T for t in sorted(grouped)]
    return increments, max_node + 1


def load_increments(dataset: str, input_dir: Path) -> tuple[list[np.ndarray], int]:
    """Return the per-snapshot *new* edges after the official temporal merge."""
    if dataset == "dblp":
        return read_dblp_increments(input_dir / "dblp.txt")
    if dataset == "tmall":
        increments, timestamps, num_nodes, _, _ = _read_tmall(
            input_dir / "tmall.txt", input_dir / "node2label.txt"
        )
    else:
        increments, timestamps, num_nodes, _, _ = _read_patent(
            input_dir / "patent_edges.json", input_dir / "patent_nodes.json"
        )
    increments, _ = _merge_snapshots(increments, timestamps, OFFICIAL_MERGE_STEPS[dataset])
    return increments, num_nodes


def iter_cumulative_csr(increments: list[np.ndarray], num_nodes: int):
    """Yield the cumulative, symmetric, binary CSR adjacency of every snapshot."""
    import scipy.sparse as sp

    cumulative = None
    for edge_index in increments:
        now = sp.csr_matrix(
            (np.ones(edge_index.shape[1], dtype=np.float32), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes),
        )
        now = now.maximum(now.T)
        cumulative = now if cumulative is None else cumulative.maximum(now)
        cumulative.data[:] = 1.0
        cumulative.eliminate_zeros()
        cumulative.sort_indices()
        yield cumulative


# --------------------------------------------------------------------------- #
# Random walks -> text corpus
# --------------------------------------------------------------------------- #
def _walk_kernels():
    """Return (random_walks, format_walks); numba-compiled when available."""
    try:
        from numba import njit
    except ImportError:  # pragma: no cover - exercised only without numba
        def njit(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda fn: fn
        print("numba not installed; falling back to pure Python walks (very slow)",
              file=sys.stderr)

    @njit
    def random_walks(indices, indptr, first, last, walk_length, walks_per_node, seed):
        np.random.seed(seed)
        count = last - first
        total = count * walks_per_node
        walks = np.empty((total, walk_length), dtype=np.int32)
        lengths = np.ones(total, dtype=np.int16)
        row = 0
        for _ in range(walks_per_node):
            for start in range(first, last):
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

    @njit
    def format_walks(walks, lengths, buf):
        pos = 0
        for r in range(walks.shape[0]):
            length = lengths[r]
            for k in range(length):
                value = walks[r, k]
                if value == 0:
                    buf[pos] = 48
                    pos += 1
                else:
                    begin = pos
                    while value > 0:
                        buf[pos] = 48 + value % 10
                        value //= 10
                        pos += 1
                    i, j = begin, pos - 1
                    while i < j:
                        tmp = buf[i]
                        buf[i] = buf[j]
                        buf[j] = tmp
                        i += 1
                        j -= 1
                if k + 1 < length:
                    buf[pos] = 32
                    pos += 1
            buf[pos] = 10
            pos += 1
        return pos

    return random_walks, format_walks


def write_walk_corpus(
    graph, corpus_path: Path, walks_per_node: int, walk_length: int,
    seed: int, node_chunk: int, kernels,
) -> int:
    """Stream ``walks_per_node`` walks from every node into ``corpus_path``.

    Returns the number of sentences written.  Node chunks keep peak memory
    around ``node_chunk * walks_per_node * walk_length * 4`` bytes.
    """
    random_walks, format_walks = kernels
    num_nodes = graph.shape[0]
    indices = graph.indices.astype(np.int32)
    indptr = graph.indptr.astype(np.int64)
    digits = len(str(max(num_nodes - 1, 1)))
    sentences = 0
    with corpus_path.open("wb") as handle:
        for chunk_id, first in enumerate(range(0, num_nodes, node_chunk)):
            last = min(first + node_chunk, num_nodes)
            walks, lengths = random_walks(
                indices, indptr, first, last, walk_length, walks_per_node,
                seed + 1000 * chunk_id,
            )
            buf = np.empty(walks.shape[0] * (walk_length * (digits + 1) + 1), dtype=np.uint8)
            used = format_walks(walks, lengths, buf)
            handle.write(buf[:used].tobytes())
            sentences += walks.shape[0]
            del walks, lengths, buf
    return sentences


# --------------------------------------------------------------------------- #
# Word2Vec
# --------------------------------------------------------------------------- #
def train_word2vec(corpus_path: Path, num_nodes: int, workers: int, seed: int) -> np.ndarray:
    from gensim.models import Word2Vec

    model = Word2Vec(
        corpus_file=str(corpus_path),
        vector_size=DIMENSIONS,
        window=WINDOW_SIZE,
        min_count=0,
        sg=1,
        hs=0,
        negative=NEGATIVE,
        alpha=LEARNING_RATE,
        epochs=EPOCHS,
        workers=workers,
        seed=seed,
        compute_loss=True,
    )
    key_to_index = model.wv.key_to_index
    if len(key_to_index) != num_nodes:
        raise RuntimeError(
            f"vocabulary has {len(key_to_index)} nodes, expected {num_nodes}; "
            "every node must start at least one walk"
        )
    order = np.fromiter((key_to_index[str(node)] for node in range(num_nodes)),
                        dtype=np.int64, count=num_nodes)
    return np.ascontiguousarray(model.wv.vectors[order], dtype=np.float32)


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """sklearn.preprocessing.normalize(X) semantics: zero rows stay zero."""
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (embedding / norms).astype(np.float32)


# --------------------------------------------------------------------------- #
# Driver with resume support
# --------------------------------------------------------------------------- #
def _progress_path(output: Path) -> Path:
    return output.with_name(output.name + ".progress.json")


def _load_progress(output: Path) -> dict:
    path = _progress_path(output)
    if path.is_file():
        with path.open() as handle:
            return json.load(handle)
    return {"done": []}


def _save_progress(output: Path, progress: dict) -> None:
    tmp = _progress_path(output).with_suffix(".tmp")
    with tmp.open("w") as handle:
        json.dump(progress, handle, indent=1, sort_keys=True)
    os.replace(tmp, _progress_path(output))


def open_output(output: Path, steps: int, num_nodes: int, resume: bool):
    shape = (steps, num_nodes, DIMENSIONS)
    if output.is_file() and resume:
        array = np.lib.format.open_memmap(output, mode="r+")
        if array.shape != shape or array.dtype != np.float32:
            raise ValueError(
                f"{output} has shape {array.shape}/{array.dtype}, expected {shape}/float32; "
                "delete it (and its .progress.json) or pass --no-resume"
            )
        return array
    output.parent.mkdir(parents=True, exist_ok=True)
    if _progress_path(output).exists():
        _progress_path(output).unlink()
    return np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=shape)


def generate(
    dataset: str,
    input_dir: Path,
    output: Path,
    normalize: bool,
    workers: int,
    seed: int,
    node_chunk: int,
    scratch_dir: Path,
    start: int,
    end: int | None,
    resume: bool,
    keep_corpus: bool,
    max_snapshots: int | None,
) -> None:
    increments, num_nodes = load_increments(dataset, input_dir)
    if max_snapshots is not None:
        increments = increments[:max_snapshots]
    steps = len(increments)
    end = steps if end is None else min(end, steps)
    print(
        f"{dataset}: T={steps} snapshots, N={num_nodes} nodes, "
        f"E_final={increments[-1].shape[1] if increments else 0} new edges in last increment, "
        f"normalize={normalize}, walks={WALKS_PER_NODE}x{WALK_LENGTH}, dim={DIMENSIONS}",
        flush=True,
    )

    features = open_output(output, steps, num_nodes, resume)
    progress = _load_progress(output) if resume else {"done": []}
    done = set(progress["done"])
    scratch_dir.mkdir(parents=True, exist_ok=True)
    kernels = _walk_kernels()

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda x, **_: x  # noqa: E731

    for t, graph in enumerate(tqdm(iter_cumulative_csr(increments, num_nodes), total=steps,
                                   desc=f"DeepWalk {dataset}")):
        if t < start or t >= end or t in done:
            continue
        tic = time.time()
        corpus = scratch_dir / f"{dataset}_snapshot{t:03d}.walks.txt"
        sentences = write_walk_corpus(
            graph, corpus, WALKS_PER_NODE, WALK_LENGTH, seed + t, node_chunk, kernels
        )
        walk_seconds = time.time() - tic
        embedding = train_word2vec(corpus, num_nodes, workers, seed + t)
        if normalize:
            embedding = l2_normalize(embedding)
        features[t] = embedding
        features.flush()
        if not keep_corpus:
            corpus.unlink(missing_ok=True)
        done.add(t)
        progress["done"] = sorted(done)
        progress["shape"] = list(features.shape)
        progress["normalize"] = normalize
        _save_progress(output, progress)
        print(
            f"snapshot {t}: nnz={graph.nnz}, sentences={sentences}, "
            f"walks={walk_seconds:.0f}s, total={time.time() - tic:.0f}s", flush=True,
        )

    if len(done) == steps:
        print(f"saved {output}: shape={features.shape}, dtype={features.dtype}; all snapshots done")
    else:
        missing = sorted(set(range(steps)) - done)
        print(f"{output}: {len(done)}/{steps} snapshots done, missing {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SpikeNet-compatible DeepWalk features [T, N, 80]"
    )
    parser.add_argument("--dataset", required=True, choices=["dblp", "tmall", "tsmall", "patent"])
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="default data/raw/<dataset>")
    parser.add_argument("--output", type=Path, default=None,
                        help="default data/raw/<dataset>/<dataset>.npy (auto-detected by the converters)")
    normalize = parser.add_mutually_exclusive_group()
    normalize.add_argument("--normalize", dest="normalize", action="store_true", default=None,
                           help="L2-normalize embeddings (official for Tmall/Patent)")
    normalize.add_argument("--no-normalize", dest="normalize", action="store_false",
                           help="raw embeddings (official for DBLP)")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--node-chunk", type=int, default=50_000,
                        help="nodes per walk batch (memory ~ chunk*128*10*4 bytes)")
    parser.add_argument("--scratch-dir", type=Path, default=None,
                        help="where walk corpora are written (default $TMPDIR or data/raw/<dataset>/walks); "
                             "Patent needs ~25 GB free per snapshot")
    parser.add_argument("--start", type=int, default=0, help="first snapshot index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="last snapshot index (exclusive)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="overwrite an existing output instead of resuming")
    parser.add_argument("--keep-corpus", action="store_true", help="keep the walk text files")
    parser.add_argument("--max-snapshots", type=int, default=None, help="debug only")
    args = parser.parse_args()

    dataset = canonical_dataset(args.dataset) if args.dataset != "dblp" else "dblp"
    input_dir = args.input_dir or Path("data/raw") / dataset
    output = args.output or input_dir / f"{dataset}.npy"
    normalize = OFFICIAL_NORMALIZE[dataset] if args.normalize is None else args.normalize
    scratch = args.scratch_dir or Path(os.environ.get("TMPDIR") or input_dir / "walks")
    generate(
        dataset, input_dir, output, normalize, args.workers, args.seed, args.node_chunk,
        scratch, args.start, args.end, args.resume, args.keep_corpus, args.max_snapshots,
    )


if __name__ == "__main__":
    main()
