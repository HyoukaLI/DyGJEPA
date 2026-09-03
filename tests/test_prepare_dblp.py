from pathlib import Path

import numpy as np

from scripts.generate_dblp_deepwalk import (
    DEFAULT_DIMENSIONS, DEFAULT_WALK_LENGTH, DEFAULT_WALKS_PER_NODE,
)
from scripts.prepare_dblp import convert, load_deepwalk


def test_convert_dblp_cumulative_snapshots(tmp_path: Path) -> None:
    edges = tmp_path / "dblp.txt"
    labels = tmp_path / "node2label.txt"
    edges.write_text("0 1 0.0\n1 2 0.5\n2 0 1.0\n")
    labels.write_text("0 0\n1 1\n2 0\n")
    output = tmp_path / "dblp.npz"
    convert(edges, labels, output, None)
    raw = np.load(output)
    assert raw["features"].shape == (3, 3, 4)
    assert raw["edges_0"].shape == (2, 2)
    assert raw["edges_2"].shape == (2, 6)
    assert raw["labels"].tolist() == [0, 1, 0]


def test_spikenet_deepwalk_parameter_interpretation() -> None:
    assert DEFAULT_DIMENSIONS == 80
    assert DEFAULT_WALK_LENGTH == 10
    assert DEFAULT_WALKS_PER_NODE == 128


def test_rejects_128_dimensional_feature_confusion(tmp_path: Path) -> None:
    path = tmp_path / "wrong.npy"
    np.save(path, np.zeros((3, 4, 128), dtype=np.float32))
    try:
        load_deepwalk(path, steps=3, nodes=4)
    except ValueError as exc:
        assert "80 is the embedding dimension" in str(exc)
    else:
        raise AssertionError("128-dimensional features should be rejected")
