from pathlib import Path

from jepa_compare.compare_node_prediction import _dataset_configs, _load_config


ROOT = Path(__file__).resolve().parents[1]


def test_node_all_config_expands_dblp_tmall_and_patent() -> None:
    config = _load_config(ROOT / "configs" / "node_comparison_all.yaml")
    expanded = dict(_dataset_configs(config))

    assert set(expanded) == {"dblp", "tmall", "patent"}
    assert expanded["tmall"]["data"]["path"] == "data/processed/tmall.npz"
    assert expanded["patent"]["data"]["path"] == "data/processed/patent.npz"
    assert expanded["tmall"]["training"]["node_batch_size"] == 1024
    assert expanded["patent"]["training"]["node_batch_size"] == 2048
    assert expanded["tmall"]["probe"] == expanded["dblp"]["probe"]
    assert expanded["patent"]["node_baselines"]["enabled"] == expanded["dblp"]["node_baselines"]["enabled"]
