from pathlib import Path

import pytest

from jepa_compare.compare_node_prediction import (
    _aggregate_seed_runs,
    _dataset_configs,
    _load_config,
    _seed_output_path,
    _seed_values,
)


ROOT = Path(__file__).resolve().parents[1]


def test_node_all_config_expands_dblp_tmall_and_patent() -> None:
    config = _load_config(ROOT / "configs" / "node_comparison_all.yaml")
    expanded = dict(_dataset_configs(config))

    assert _seed_values(config) == [42, 44, 46, 48, 50]
    assert set(expanded) == {"dblp", "tmall", "patent"}
    assert expanded["tmall"]["data"]["path"] == "data/processed/tmall.npz"
    assert expanded["patent"]["data"]["path"] == "data/processed/patent.npz"
    assert expanded["tmall"]["training"]["node_batch_size"] == 1024
    assert expanded["patent"]["training"]["node_batch_size"] == 2048
    assert expanded["tmall"]["probe"] == expanded["dblp"]["probe"]
    assert expanded["patent"]["node_baselines"]["enabled"] == expanded["dblp"]["node_baselines"]["enabled"]


def test_node_seed_output_and_flat_metric_aggregation() -> None:
    assert _seed_output_path(Path("results/node_comparison_tmall.json"), 44) == Path(
        "results/node_comparison_tmall_seed44.json"
    )
    runs = {
        "42": {
            "sg_jepa": {
                "macro_f1": 0.7,
                "micro_f1": 0.8,
                "protocol": "ssl_probe",
            }
        },
        "44": {
            "sg_jepa": {
                "macro_f1": 0.9,
                "micro_f1": 0.6,
                "protocol": "ssl_probe",
            }
        },
    }
    aggregate = _aggregate_seed_runs(runs)["sg_jepa"]
    assert aggregate["macro_f1"]["mean"] == pytest.approx(0.8)
    assert aggregate["macro_f1"]["std"] == pytest.approx(0.1)
    assert aggregate["micro_f1"]["mean"] == pytest.approx(0.7)
    assert aggregate["micro_f1"]["std"] == pytest.approx(0.1)
    assert aggregate["metadata"] == {"protocol": "ssl_probe"}
