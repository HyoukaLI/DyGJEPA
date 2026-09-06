from pathlib import Path

import numpy as np
import torch

from jepa_compare.node_evaluation import stratified_split
from scripts.prepare_spikenet_node import canonical_dataset, convert


def test_tsmall_alias_and_tmall_partial_labels(tmp_path: Path) -> None:
    raw = tmp_path / "tmall"
    raw.mkdir()
    (raw / "tmall.txt").write_text(
        "0 1 00\n1 2 01\n2 3 02\n3 0 03\n0 2 04\n1 3 05\n"
    )
    (raw / "node2label.txt").write_text("2 shop\n0 user\n1 shop\n")
    feature_path = raw / "tmall.npy"
    np.save(
        feature_path,
        np.arange(3 * 4 * 80, dtype=np.float32).reshape(3, 4, 80),
    )
    output = tmp_path / "tmall.npz"

    convert("tsmall", raw, output, feature_path=feature_path, merge_step=2)

    archive = np.load(output)
    assert canonical_dataset("Tsmall") == "tmall"
    assert archive["features"].shape == (3, 4, 80)
    assert np.allclose(archive["features"].mean(axis=(1, 2)), 0.0, atol=1e-5)
    assert archive["edges_0"].shape == (2, 4)
    assert archive["edges_2"].shape == (2, 12)
    assert archive["labels"].tolist() == [0, 1, 0, -1]
    assert archive["original_node_ids"].tolist() == [2, 0, 1, 3]
    assert archive["timestamps"].tolist() == ["01", "03", "05"]


def test_tmall_unlabeled_nodes_are_excluded_from_probe_split(tmp_path: Path) -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 1, -1, -1])
    split = stratified_split(labels, train_ratio=0.5, seed=42)
    selected = torch.cat([split.train, split.validation, split.test])
    assert (labels[selected] >= 0).all()
    assert set(selected.tolist()).isdisjoint({6, 7})


def test_patent_official_tuple_format_and_two_step_merge(tmp_path: Path) -> None:
    raw = tmp_path / "patent"
    raw.mkdir()
    (raw / "patent_edges.json").write_text(
        "(0, 1, 20000101, 'a', 'b')\n"
        "(1, 2, 20010101, 'b', 'c')\n"
        "(2, 0, 20020101, 'c', 'a')\n"
    )
    (raw / "patent_nodes.json").write_text(
        "(2, 'c', 20020101, 3)\n"
        "(0, 'a', 20000101, 1)\n"
        "(1, 'b', 20010101, 2)\n"
    )
    output = tmp_path / "patent.npz"

    convert("patent", raw, output, feature_path=None)

    archive = np.load(output)
    assert archive["features"].shape == (2, 3, 4)
    assert archive["edges_0"].shape == (2, 4)
    assert archive["edges_1"].shape == (2, 6)
    assert archive["labels"].tolist() == [0, 1, 2]
    assert archive["timestamps"].tolist() == [2001, 2002]
    assert int(archive["merge_step"]) == 2
