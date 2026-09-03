from pathlib import Path

import numpy as np

from jepa_compare.data import load_npz
from scripts.prepare_dyglib_homogeneous import convert as convert_homogeneous
from scripts.prepare_wikipedia import convert


def test_convert_wikipedia_preserves_bipartite_duplicate_events(tmp_path: Path) -> None:
    source = tmp_path / "wikipedia.csv"
    source.write_text(
        "user,item,timestamp,state_label,f0,f1\n"
        "u0,p0,0.0,0,1.0,0.0\n"
        "u0,p0,1.0,0,0.5,0.5\n"
        "u1,p1,2.0,1,0.0,1.0\n"
        "u0,p1,3.0,0,0.2,0.8\n"
        "u1,p0,4.0,0,0.8,0.2\n"
        "u1,p1,5.0,0,0.4,0.6\n"
        "u0,p1,6.0,0,0.3,0.7\n"
    )
    output = tmp_path / "wikipedia.npz"
    convert(
        source,
        output,
        event_bins=6,
        identity_dim=2,
        event_feature_dim=2,
        seed=7,
    )
    raw = np.load(output)
    assert int(raw["num_source_nodes"]) == 2
    assert raw["features"].shape == (6, 4, 8)
    assert raw["queries_0"].shape == (2, 2)
    assert np.array_equal(raw["queries_0"][:, 0], raw["queries_0"][:, 1])
    assert raw["query_timestamps_0"].shape == (2,)
    assert raw["query_features_0"].shape == (2, 2)
    assert raw["query_labels_1"].tolist() == [1]
    assert np.array_equal(raw["query_timestamps_0"], np.asarray([0.0, 1.0]))
    graph = load_npz(output)
    assert graph.num_source_nodes == 2
    assert graph.snapshots[0].query_edge_index is not None
    assert graph.snapshots[0].query_timestamps is not None
    assert graph.snapshots[0].query_features is not None
    assert graph.snapshots[1].query_labels is not None
    assert graph.snapshots[1].query_labels.tolist() == [1]
    assert graph.snapshots[0].active.all()


def test_convert_homogeneous_dyglib_events_without_bipartite_split(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "UCI"
    dataset_dir.mkdir()
    (dataset_dir / "ml_uci.csv").write_text(
        ",u,i,ts,label,idx\n"
        "0,1,2,0,0,1\n"
        "1,2,3,1,0,2\n"
        "2,3,1,2,0,3\n"
        "3,1,3,3,0,4\n"
        "4,2,1,4,0,5\n"
        "5,3,2,5,0,6\n"
    )
    np.save(dataset_dir / "ml_uci.npy", np.arange(14).reshape(7, 2))
    np.save(dataset_dir / "ml_uci_node.npy", np.zeros((4, 2)))
    output = tmp_path / "uci.npz"

    convert_homogeneous(
        dataset_dir,
        output,
        event_bins=6,
        identity_dim=2,
        event_feature_dim=2,
        train_ratio=0.75,
        seed=7,
    )

    raw = np.load(output)
    assert "num_source_nodes" not in raw
    assert raw["features"].shape == (6, 3, 8)
    assert raw["query_features_0"].shape == (1, 2)
    graph = load_npz(output)
    assert graph.num_source_nodes is None
    assert graph.snapshots[0].query_edge_index is not None
