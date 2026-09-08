import json
from pathlib import Path

import numpy as np
import pytest

from scripts import generate_spikenet_deepwalk as gen
from scripts.prepare_spikenet_node import convert


def _fake_train(corpus_path: Path, num_nodes: int, workers: int, seed: int) -> np.ndarray:
    """Deterministic stand-in for gensim: row i encodes (snapshot seed, node i)."""
    lines = corpus_path.read_text().splitlines()
    starts = sorted({int(line.split()[0]) for line in lines})
    assert starts == list(range(num_nodes)), "every node must start walks"
    assert len(lines) == num_nodes * gen.WALKS_PER_NODE
    out = np.zeros((num_nodes, gen.DIMENSIONS), dtype=np.float32)
    out[:, 0] = seed
    out[:, 1] = np.arange(num_nodes)
    return out


@pytest.fixture
def toy_tmall(tmp_path: Path) -> Path:
    raw = tmp_path / "tmall"
    raw.mkdir()
    (raw / "tmall.txt").write_text("0 1 00\n1 2 01\n2 3 02\n3 0 03\n0 2 04\n1 3 05\n")
    (raw / "node2label.txt").write_text("2 shop\n0 user\n1 shop\n")
    return raw


def test_official_hyperparameters() -> None:
    assert (gen.DIMENSIONS, gen.WALK_LENGTH, gen.WALKS_PER_NODE) == (80, 10, 128)
    assert (gen.WINDOW_SIZE, gen.NEGATIVE, gen.EPOCHS) == (10, 1, 1)
    assert gen.OFFICIAL_NORMALIZE == {"dblp": False, "tmall": True, "patent": True}


def test_walk_corpus_matches_graph(tmp_path: Path) -> None:
    import scipy.sparse as sp

    increments = [np.array([[0, 1], [1, 2]]), np.array([[3], [0]])]
    graphs = list(gen.iter_cumulative_csr(increments, 4))
    assert graphs[0].nnz == 4 and graphs[1].nnz == 6
    assert (graphs[1] != graphs[1].T).nnz == 0  # symmetric
    assert set(graphs[1].data.tolist()) == {1.0}
    corpus = tmp_path / "walks.txt"
    sentences = gen.write_walk_corpus(
        graphs[1], corpus, walks_per_node=3, walk_length=5, seed=1, node_chunk=3,
        kernels=gen._walk_kernels(),
    )
    lines = corpus.read_text().splitlines()
    assert sentences == len(lines) == 4 * 3
    adjacency = sp.csr_matrix(graphs[1])
    for line in lines:
        walk = [int(token) for token in line.split()]
        assert 1 <= len(walk) <= 5
        for a, b in zip(walk, walk[1:]):
            assert adjacency[a, b] == 1.0


def test_generate_resume_and_converter_alignment(
    tmp_path: Path, toy_tmall: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gen, "train_word2vec", _fake_train)
    monkeypatch.setattr(gen, "OFFICIAL_MERGE_STEPS", {"tmall": 2, "patent": 2})
    output = toy_tmall / "tmall.npy"
    common = dict(
        dataset="tmall", input_dir=toy_tmall, output=output, normalize=False, workers=1,
        seed=10, node_chunk=2, scratch_dir=tmp_path / "walks", resume=True,
        keep_corpus=False, max_snapshots=None,
    )
    gen.generate(start=0, end=1, **common)
    progress = json.loads((output.parent / "tmall.npy.progress.json").read_text())
    assert progress["done"] == [0]
    gen.generate(start=0, end=None, **common)
    progress = json.loads((output.parent / "tmall.npy.progress.json").read_text())
    assert progress["done"] == [0, 1, 2]
    features = np.load(output)
    assert features.shape == (3, 4, 80)
    assert features[:, 0, 0].tolist() == [10, 11, 12]  # per-snapshot seeds
    assert features[2, :, 1].tolist() == [0, 1, 2, 3]  # reindexed node order
    assert not list((tmp_path / "walks").glob("*.txt"))

    archive_path = tmp_path / "tmall.npz"
    convert("tmall", toy_tmall, archive_path, feature_path=output, merge_step=2)
    archive = np.load(archive_path)
    assert archive["features"].shape == (3, 4, 80)
    assert str(archive["feature_source"]) == str(output)


def test_l2_normalize_keeps_zero_rows() -> None:
    embedding = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = gen.l2_normalize(embedding)
    assert np.allclose(normalized, [[0.6, 0.8], [0.0, 0.0]])
