from pathlib import Path

import pytest
import yaml

from jepa_compare.compare_link_prediction import (
    _aggregate_seed_runs,
    _dataset_configs,
    _seed_output_path,
    _seed_values,
)


def test_link_multi_dataset_config_expands_aligned_overrides() -> None:
    path = Path("configs/link_comparison_all.yaml")
    config = yaml.safe_load(path.read_text())
    assert _seed_values(config) == [0, 1, 2, 3, 4]
    expanded = dict(_dataset_configs(config))

    assert set(expanded) == {
        "wikipedia", "mooc", "lastfm", "canparl", "contacts", "flights",
        "untrade", "unvote", "uslegis", "enron", "uci",
    }
    assert expanded["wikipedia"]["jodie"]["interaction_feature_dim"] == 172
    assert expanded["wikipedia"]["rcps_jepa"]["hidden_dim"] == 128
    assert expanded["wikipedia"]["rcps_jepa"]["id_embedding_dim"] == 128
    assert expanded["wikipedia"]["rcps_jepa"]["id_embedding_dropout"] == 0.2
    assert expanded["wikipedia"]["rcps_jepa"]["initial_id_score_scale"] == 0.5
    assert expanded["wikipedia"]["rcps_jepa"]["train_negative_ratio"] == 4.0
    assert expanded["wikipedia"]["rcps_jepa"]["rank_loss_weight"] == 1.0
    assert expanded["wikipedia"]["rcps_training"]["pair_batch_size"] == 500
    assert expanded["wikipedia"]["rcps_training"]["learning_rate"] == 0.0001
    assert expanded["mooc"]["jodie"]["interaction_feature_dim"] == 4
    assert expanded["lastfm"]["jodie"]["interaction_feature_dim"] == 2
    assert expanded["lastfm"]["jodie"]["state_change"] is False
    assert expanded["mooc"]["tgn"]["interaction_feature_dim"] == 172
    assert expanded["lastfm"]["dygformer"]["interaction_feature_dim"] == 172
    assert expanded["mooc"]["tgat"]["interaction_feature_dim"] == 172
    assert expanded["mooc"]["dyrep"]["interaction_feature_dim"] == 172
    assert expanded["lastfm"]["dyrep"]["interaction_feature_dim"] == 172

    assert expanded["mooc"]["tgn"]["dropout"] == 0.2
    assert expanded["lastfm"]["cawn"]["num_neighbors"] == 128
    assert expanded["mooc"]["graphmixer"]["num_neighbors"] == 20
    assert expanded["lastfm"]["dygformer"]["patch_size"] == 16
    assert "jodie" not in expanded["canparl"]["models"]
    assert "rcps_jepa" in expanded["canparl"]["models"]
    assert expanded["canparl"]["tgat"]["uniform_neighbors"] is True
    assert expanded["canparl"]["dyrep"]["sample_neighbor_strategy"] == "uniform"
    assert expanded["contacts"]["dyrep"]["dropout"] == 0.0
    assert expanded["flights"]["dyrep"]["dropout"] == 0.1
    assert expanded["unvote"]["tgn"]["sample_neighbor_strategy"] == "uniform"
    assert expanded["enron"]["dygformer"]["dropout"] == 0.0

    common_link = expanded["wikipedia"]["link"]
    assert expanded["mooc"]["link"] == common_link
    assert expanded["lastfm"]["link"] == common_link
    assert expanded["contacts"]["link"] == common_link
    assert common_link["negative_ratio"] == 1.0
    assert common_link["max_positive_pairs"] is None
    assert common_link["allow_negative_collisions"] is True
    assert common_link["eval_positive_batch_size"] == 200
    assert expanded["enron"]["split"] == {
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
    }


def test_seed_aggregation_reports_mean_and_population_std() -> None:
    runs = {
        "42": {"model": {"test": {"ap": 0.6, "best_epoch": 2.0}}},
        "44": {"model": {"test": {"ap": 0.8, "best_epoch": 4.0}}},
    }
    aggregate = _aggregate_seed_runs(runs)

    assert aggregate["model"]["test"]["ap"]["mean"] == pytest.approx(0.7)
    assert aggregate["model"]["test"]["ap"]["std"] == pytest.approx(0.1)
    assert aggregate["model"]["test"]["best_epoch"]["mean"] == 3.0


def test_seed_aggregation_preserves_provenance_strings() -> None:
    runs = {
        "0": {"model": {"test": {"ap": 0.6, "protocol": "dyglib"}}},
        "1": {"model": {"test": {"ap": 0.8, "protocol": "dyglib"}}},
    }
    aggregate = _aggregate_seed_runs(runs)
    assert aggregate["model"]["test"]["protocol"] == "dyglib"


def test_output_name_claims_a_per_model_filename() -> None:
    """Every model of a dataset otherwise writes the same result filenames."""
    config = {
        "datasets": [
            {
                "name": "wikipedia",
                "path": "data/processed/wikipedia.npz",
                "interaction_feature_dim": 172,
            }
        ],
        "output_dir": "results/wikipedia",
        "output_name": "rcps_jepa",
    }
    (name, expanded), = _dataset_configs(config)
    assert name == "wikipedia"
    assert expanded["output_path"] == "results/wikipedia/rcps_jepa.json"
    assert (
        str(_seed_output_path(Path(expanded["output_path"]), 3))
        == "results/wikipedia/rcps_jepa_seed3.json"
    )


def test_output_name_without_it_keeps_the_dataset_filename() -> None:
    config = {
        "datasets": [
            {
                "name": "wikipedia",
                "path": "data/processed/wikipedia.npz",
                "interaction_feature_dim": 172,
            }
        ],
        "output_dir": "results",
    }
    (_, expanded), = _dataset_configs(config)
    assert expanded["output_path"] == "results/link_comparison_wikipedia.json"


def test_output_name_is_rejected_for_more_than_one_dataset() -> None:
    """Two datasets sharing one filename stem would overwrite each other."""
    config = {
        "datasets": [
            {"name": "wikipedia", "path": "a.npz", "interaction_feature_dim": 172},
            {"name": "mooc", "path": "b.npz", "interaction_feature_dim": 4},
        ],
        "output_name": "rcps_jepa",
    }
    with pytest.raises(ValueError, match="output_name needs exactly one dataset"):
        _dataset_configs(config)
