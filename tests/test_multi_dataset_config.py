from pathlib import Path

import pytest
import yaml

from jepa_compare.compare_link_prediction import (
    _aggregate_seed_runs,
    _dataset_configs,
    _seed_values,
)


def test_link_multi_dataset_config_expands_aligned_overrides() -> None:
    path = Path("configs/link_comparison_all.yaml")
    config = yaml.safe_load(path.read_text())
    assert _seed_values(config) == [42, 44, 46, 48, 50]
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


def test_seed_aggregation_reports_mean_and_population_std() -> None:
    runs = {
        "42": {"model": {"test": {"ap": 0.6, "best_epoch": 2.0}}},
        "44": {"model": {"test": {"ap": 0.8, "best_epoch": 4.0}}},
    }
    aggregate = _aggregate_seed_runs(runs)

    assert aggregate["model"]["test"]["ap"]["mean"] == pytest.approx(0.7)
    assert aggregate["model"]["test"]["ap"]["std"] == pytest.approx(0.1)
    assert aggregate["model"]["test"]["best_epoch"]["mean"] == 3.0
