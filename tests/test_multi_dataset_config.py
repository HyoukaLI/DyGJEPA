from pathlib import Path

import yaml

from jepa_compare.compare_link_prediction import _dataset_configs


def test_link_multi_dataset_config_expands_aligned_overrides() -> None:
    path = Path("configs/link_comparison_all.yaml")
    config = yaml.safe_load(path.read_text())
    expanded = dict(_dataset_configs(config))

    assert set(expanded) == {
        "wikipedia", "mooc", "lastfm", "canparl", "contacts", "flights",
        "untrade", "unvote", "uslegis", "enron", "uci",
    }
    assert expanded["wikipedia"]["jodie"]["interaction_feature_dim"] == 172
    assert expanded["mooc"]["jodie"]["interaction_feature_dim"] == 4
    assert expanded["lastfm"]["jodie"]["interaction_feature_dim"] == 2
    assert expanded["lastfm"]["jodie"]["state_change"] is False
    assert expanded["mooc"]["tgn"]["interaction_feature_dim"] == 172
    assert expanded["lastfm"]["dygformer"]["interaction_feature_dim"] == 172
    assert expanded["mooc"]["tgat"]["interaction_feature_dim"] == 172

    assert expanded["mooc"]["tgn"]["dropout"] == 0.2
    assert expanded["lastfm"]["cawn"]["num_neighbors"] == 128
    assert expanded["mooc"]["graphmixer"]["num_neighbors"] == 20
    assert expanded["lastfm"]["dygformer"]["patch_size"] == 16
    assert "jodie" not in expanded["canparl"]["models"]
    assert "rcps_jepa" in expanded["canparl"]["models"]
    assert expanded["canparl"]["tgat"]["uniform_neighbors"] is True
    assert expanded["unvote"]["tgn"]["sample_neighbor_strategy"] == "uniform"
    assert expanded["enron"]["dygformer"]["dropout"] == 0.0

    common_link = expanded["wikipedia"]["link"]
    assert expanded["mooc"]["link"] == common_link
    assert expanded["lastfm"]["link"] == common_link
    assert expanded["contacts"]["link"] == common_link
