"""Efficiency accounting: the meter itself and its wiring into the driver."""
import time

import torch
from torch import nn

from jepa_compare.compare_link_prediction import (
    _efficiency_record,
    _phase,
    _train_one,
    _train_snapshot_ssl_one,
)
from jepa_compare.efficiency import EfficiencyMeter, count_parameters


def test_meter_records_phases_and_time_to_best() -> None:
    meter = EfficiencyMeter("cpu")
    with meter.measure("setup"):
        time.sleep(0.01)
    with meter.measure("validation", epoch=0):
        time.sleep(0.01)
    for epoch in (1, 2, 3):
        with meter.measure("train_epoch"):
            time.sleep(0.01)
        with meter.measure("validation", epoch=epoch):
            time.sleep(0.005)
    with meter.measure("final_validation"):
        pass
    with meter.measure("test"):
        time.sleep(0.02)
    model = nn.Linear(4, 2)
    frozen = nn.Linear(2, 1)
    frozen.requires_grad_(False)
    both = nn.Sequential(model, frozen)
    record = meter.summary(both, best_epoch=2, test_examples=200.0)

    assert record["train_epochs"] == 3.0
    assert record["validation_passes"] == 4.0
    assert record["parameters"] == 10.0  # 4*2 + 2 trainable
    assert record["parameters_total"] == 13.0  # + 2*1 + 1 frozen
    assert record["test_seconds"] >= 0.02
    assert record["test_examples_per_second"] == 200.0 / record["test_seconds"]
    # Time to best: setup + epochs 1-2 + validation passes at epochs 0, 1, 2.
    phases = meter.phases
    expected = (
        sum(phases["setup"])
        + sum(phases["train_epoch"][:2])
        + sum(phases["validation"][:3])
    )
    assert abs(record["time_to_best_seconds"] - expected) < 1e-9
    assert record["time_to_best_seconds"] < record["wall_seconds_total"]
    assert "peak_memory_mb" not in record  # CPU: no CUDA allocator statistics
    assert all(isinstance(value, float) for value in record.values())


def test_meter_epoch_zero_checkpoint_counts_only_setup_and_first_validation() -> None:
    meter = EfficiencyMeter("cpu")
    with meter.measure("validation", epoch=0):
        pass
    with meter.measure("train_epoch"):
        time.sleep(0.005)
    with meter.measure("validation", epoch=1):
        time.sleep(0.005)
    record = meter.summary(None, best_epoch=0)
    assert record["time_to_best_seconds"] == meter.phases["validation"][0]
    assert record["parameters"] == 0.0


def test_meter_validation_requires_epoch_and_rejects_unknown_phase() -> None:
    meter = EfficiencyMeter("cpu")
    try:
        with meter.measure("validation"):
            pass
    except ValueError as error:
        assert "epoch" in str(error)
    else:
        raise AssertionError("validation without epoch was accepted")
    try:
        with meter.measure("inference"):
            pass
    except ValueError:
        pass
    else:
        raise AssertionError("unknown phase was accepted")


def test_phase_and_record_helpers_tolerate_a_missing_meter() -> None:
    with _phase(None, "train_epoch"):
        pass
    assert _efficiency_record(None, nn.Linear(1, 1), {"best_epoch": 3.0}) is None
    meter = EfficiencyMeter("cpu")
    record = _efficiency_record(meter, nn.Linear(1, 1), {"best_epoch": 3.0, "examples": 10.0})
    assert record["parameters"] == 2.0
    assert "test_examples_per_second" not in record  # no test pass was timed


def test_count_parameters_separates_trainable_from_total() -> None:
    model = nn.Linear(3, 3)
    assert count_parameters(model) == (12, 12)
    model.requires_grad_(False)
    assert count_parameters(model) == (0, 12)


def test_train_one_accepts_a_meter_keyword() -> None:
    # The driver passes the meter as a keyword; older callers omit it.
    import inspect

    assert "meter" in inspect.signature(_train_one).parameters
    assert "meter" in inspect.signature(_train_snapshot_ssl_one).parameters
    assert inspect.signature(_train_one).parameters["meter"].default is None


def test_train_one_fills_the_meter_end_to_end() -> None:
    from jepa_compare.data import make_synthetic
    from jepa_compare.link_prediction import TemporalWindowSplit, sliding_windows
    from jepa_compare.rcps_jepa import RCPSJEPA

    graph = make_synthetic(16, 6, 6, 2, 0.2, seed=43)
    windows = sliding_windows(graph.snapshots, 3)
    split = TemporalWindowSplit(
        train=windows[:1], validation=windows[1:2], test=windows[2:]
    )
    meter = EfficiencyMeter("cpu")
    with meter.measure("setup"):
        model = RCPSJEPA(
            feature_dim=6,
            hidden_dim=8,
            rwpe_dim=2,
            rwpe_walks=4,
            time_dim=4,
            gnn_layers=1,
            window_size=3,
            predictor_hidden_dim=16,
            event_dim=3,
            signature_depth=2,
            subgraph_budget=6,
            max_positive_pairs=6,
        )
    validation, test = _train_one(
        "rcps_jepa",
        model,
        split,
        {
            "epochs": 2,
            "pair_batch_size": 8,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
            "eval_every": 1,
            "evaluate_before_training": True,
        },
        seed=11,
        meter=meter,
    )
    record = _efficiency_record(meter, model, test)
    assert record["train_epochs"] == 2.0
    assert record["validation_passes"] == 3.0  # epoch 0, 1, 2
    assert meter.validation_epochs == [0, 1, 2]
    assert len(meter.phases["final_validation"]) == 1
    assert len(meter.phases["test"]) == 1
    assert record["setup_seconds"] > 0.0
    assert record["train_epoch_seconds_mean"] > 0.0
    assert record["test_examples_per_second"] > 0.0
    assert record["parameters"] == count_parameters(model)[0]
    assert 0.0 < record["time_to_best_seconds"] <= record["wall_seconds_total"]
    assert 0.0 <= validation["ap"] <= 1.0
