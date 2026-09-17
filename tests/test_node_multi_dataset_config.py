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


def test_node_run_overrides_select_models_ratio_and_output_naming() -> None:
    from types import SimpleNamespace

    from jepa_compare.compare_node_prediction import (
        _NODE_MODELS,
        _apply_run_overrides,
        _explicit_output,
    )

    config = _load_config(ROOT / "configs" / "node_comparison_all.yaml")
    tmall = dict(_dataset_configs(config))["tmall"]
    original_enabled = list(tmall["node_baselines"]["enabled"])
    original_ratio = tmall["probe"]["train_ratio"]

    # No override: configuration untouched (default path unchanged).
    untouched = _load_config(ROOT / "configs" / "node_comparison_all.yaml")
    untouched = dict(_dataset_configs(untouched))["tmall"]
    _apply_run_overrides(untouched, models=None, train_ratio=None, output_dir=None, output_name=None)
    assert "models" not in untouched
    assert untouched["node_baselines"]["enabled"] == original_enabled
    assert untouched["probe"]["train_ratio"] == original_ratio
    assert untouched["output_path"] == "results/node_comparison_tmall.json"

    # Launcher-style job: one baseline, one ratio, its own directory and stem.
    _apply_run_overrides(
        tmall, models=["cawn"], train_ratio=0.6, output_dir=Path("results/tmall"), output_name="cawn"
    )
    assert tmall["models"] == ["cawn"]
    assert tmall["node_baselines"]["enabled"] == ["cawn"]
    assert tmall["probe"]["train_ratio"] == 0.6
    assert tmall["output_path"] == "results/tmall/cawn_ratio0.6.json"
    assert _seed_output_path(Path(tmall["output_path"]), 42) == Path(
        "results/tmall/cawn_ratio0.6_seed42.json"
    )

    # JEPA-only job keeps every baseline off; 0.4 formats without trailing zeros.
    jepa = dict(_dataset_configs(config))["tmall"]
    _apply_run_overrides(jepa, models=["rcps_jepa"], train_ratio=0.4, output_dir=None, output_name=None)
    assert jepa["models"] == ["rcps_jepa"]
    assert jepa["node_baselines"]["enabled"] == []
    assert jepa["output_path"] == "results/node_comparison_tmall_ratio0.4.json"

    with pytest.raises(ValueError, match="unknown node models"):
        _apply_run_overrides(jepa, models=["jodie"], train_ratio=None, output_dir=None, output_name=None)
    assert set(_NODE_MODELS) >= {"sg_jepa", "rcps_jepa", "cawn", "dvgmae"}

    # Explicit naming forces the per-seed file even for one seed.
    assert _explicit_output(SimpleNamespace(output=None, output_name=None, train_ratio=None)) is False
    assert _explicit_output(SimpleNamespace(output=None, output_name="cawn", train_ratio=None)) is True
    assert _explicit_output(SimpleNamespace(output=None, output_name=None, train_ratio=0.8)) is True
