"""Historical (DyGLib) negative sampling wired through every evaluator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pathlib import Path

import yaml

from jepa_compare.compare_link_prediction import (
    _dataset_configs,
    _negative_strategy,
    _train_one,
    _with_strategy_suffix,
    load_config,
)
from jepa_compare.data import Snapshot
from jepa_compare.dyglib_baselines import DyGLibLinkBaseline, EdgeBankLinkBaseline
from jepa_compare.jodie_baseline import JODIELinkBaseline
from jepa_compare.link_prediction import (
    TemporalWindowSplit,
    negative_edge_table_from_snapshots,
    sample_link_queries,
    sliding_windows,
)
from jepa_compare.negative_edges import NegativeEdgeTable
from jepa_compare.rcps_jepa import RCPSJEPA
from jepa_compare.temporal_event_utils import unique_snapshots
from jepa_compare.tgat_baseline import TGATLinkBaseline


def event_windows(
    *, bipartite: bool, num_snapshots: int = 7, events: int = 10, seed: int = 0
) -> tuple[list[list[Snapshot]], int | None, int]:
    """Repeating-edge event snapshots with chronological timestamps."""
    generator = torch.Generator().manual_seed(seed)
    if bipartite:
        num_users, num_nodes = 3, 7
        source_choices = torch.arange(3)
        destination_choices = torch.arange(3, 7)
    else:
        num_users, num_nodes = None, 5
        source_choices = torch.arange(5)
        destination_choices = torch.arange(5)
    source_weights = 1.0 / torch.arange(1, source_choices.numel() + 1, dtype=torch.float)
    destination_weights = 1.0 / torch.arange(
        1, destination_choices.numel() + 1, dtype=torch.float
    )
    snapshots = []
    for time in range(num_snapshots):
        sources = source_choices[
            torch.multinomial(source_weights, events, replacement=True, generator=generator)
        ]
        destinations = destination_choices[
            torch.multinomial(
                destination_weights, events, replacement=True, generator=generator
            )
        ]
        timestamps = torch.sort(
            torch.rand(events, generator=generator) * 9.0 + 10.0 * time
        ).values
        queries = torch.stack([sources, destinations])
        messages = torch.cat([queries, queries.flip(0)], dim=1)
        snapshots.append(
            Snapshot(
                x=torch.randn(num_nodes, 6, generator=generator),
                edge_index=messages,
                active=torch.ones(num_nodes, dtype=torch.bool),
                time=time,
                query_edge_index=queries,
                query_timestamps=timestamps.to(torch.float32),
                query_features=torch.randn(events, 2, generator=generator),
                query_labels=torch.zeros(events, dtype=torch.long),
            )
        )
    return sliding_windows(snapshots, 3), num_users, num_nodes


def make_split(windows: list[list[Snapshot]]) -> TemporalWindowSplit:
    return TemporalWindowSplit(
        train=windows[:2], validation=windows[2:3], test=windows[3:]
    )


def make_table(
    windows: list[list[Snapshot]], num_users: int | None, batch_size: int = 4
) -> tuple[NegativeEdgeTable, TemporalWindowSplit]:
    split = make_split(windows)
    table = negative_edge_table_from_snapshots(
        unique_snapshots(windows),
        {
            "validation": unique_snapshots(split.validation, targets_only=True),
            "test": unique_snapshots(split.test, targets_only=True),
        },
        {"validation": 0, "test": 2},
        strategy="historical",
        batch_size=batch_size,
        bipartite_source_count=num_users,
    )
    assert table is not None
    return table, split


def destination_pool(windows: list[list[Snapshot]]) -> torch.Tensor:
    return torch.unique(
        torch.cat([s.query_edge_index[1] for s in unique_snapshots(windows)])
    )


def query_kwargs(windows, num_users, table=None) -> dict:
    return dict(
        negative_ratio=1.0,
        max_positive=None,
        new_edges_only=False,
        undirected=False,
        bipartite_source_count=num_users,
        negative_destination_candidates=destination_pool(windows),
        allow_negative_collisions=True,
        negative_edges=table,
    )


def test_random_strategy_builds_no_table() -> None:
    windows, num_users, _ = event_windows(bipartite=True)
    split = make_split(windows)
    assert (
        negative_edge_table_from_snapshots(
            unique_snapshots(windows),
            {"validation": unique_snapshots(split.validation, targets_only=True)},
            {"validation": 0},
            strategy="random",
            batch_size=4,
            bipartite_source_count=num_users,
        )
        is None
    )


@pytest.mark.parametrize("bipartite", [True, False])
def test_queries_pair_each_event_with_its_table_negative(bipartite: bool) -> None:
    windows, num_users, _ = event_windows(bipartite=bipartite)
    table, split = make_table(windows, num_users)
    target_window = split.test[0]
    target = target_window[-1]
    entry = table.for_snapshot(target.time)
    assert entry is not None
    queries = sample_link_queries(
        target, target_window[-2], seed=5, **query_kwargs(windows, num_users, table)
    )
    events = target.query_edge_index.shape[1]
    assert queries.labels.numel() == 2 * events
    assert int(queries.labels.sum().item()) == events
    for group in range(events):
        rows = torch.nonzero(queries.group_ids == group).flatten()
        assert rows.numel() == 2
        positive_row = rows[queries.labels[rows] == 1][0]
        negative_row = rows[queries.labels[rows] == 0][0]
        assert queries.pairs[positive_row].tolist() == [
            int(entry.positive_sources[group]),
            int(entry.positive_destinations[group]),
        ]
        assert queries.pairs[negative_row].tolist() == [
            int(entry.sources[group]),
            int(entry.destinations[group]),
        ]
        # DyGLib scores the negative at the paired positive's timestamp.
        assert queries.timestamps[positive_row] == queries.timestamps[negative_row]
    # A negative never repeats a positive of its own evaluation batch.
    entry_positives = set(
        zip(entry.positive_sources.tolist(), entry.positive_destinations.tolist())
    )
    for start in range(0, events, 4):
        batch_positives = set(
            zip(
                entry.positive_sources[start : start + 4].tolist(),
                entry.positive_destinations[start : start + 4].tolist(),
            )
        )
        batch_negatives = set(
            zip(entry.sources[start : start + 4].tolist(), entry.destinations[start : start + 4].tolist())
        )
        assert batch_negatives.isdisjoint(batch_positives)
    assert entry_positives  # sanity: the stream had events


def test_training_targets_keep_the_random_protocol_byte_for_byte() -> None:
    windows, num_users, _ = event_windows(bipartite=True)
    table, split = make_table(windows, num_users)
    train_window = split.train[-1]
    assert table.for_snapshot(train_window[-1].time) is None
    with_table = sample_link_queries(
        train_window[-1], train_window[-2], seed=9, **query_kwargs(windows, num_users, table)
    )
    without_table = sample_link_queries(
        train_window[-1], train_window[-2], seed=9, **query_kwargs(windows, num_users)
    )
    assert torch.equal(with_table.pairs, without_table.pairs)
    assert torch.equal(with_table.labels, without_table.labels)
    assert torch.equal(with_table.group_ids, without_table.group_ids)
    assert torch.equal(with_table.timestamps, without_table.timestamps)


def test_table_queries_reject_non_dyglib_settings() -> None:
    windows, num_users, _ = event_windows(bipartite=True)
    table, split = make_table(windows, num_users)
    window = split.test[0]
    base = query_kwargs(windows, num_users, table)
    for override in (
        {"negative_ratio": 2.0},
        {"max_positive": 3},
        {"new_edges_only": True},
        {"undirected": True, "bipartite_source_count": None},
    ):
        with pytest.raises(ValueError):
            sample_link_queries(window[-1], window[-2], seed=1, **{**base, **override})
    stray = Snapshot(
        x=torch.zeros(7, 6),
        edge_index=torch.tensor([[0], [3]]),
        active=torch.ones(7, dtype=torch.bool),
        time=99,
        query_edge_index=torch.tensor([[0], [3]]),
        query_timestamps=torch.tensor([0.0]),
    )
    with pytest.raises(KeyError):
        sample_link_queries(stray, None, seed=1, **base)


def test_table_build_rejects_misaligned_streams() -> None:
    windows, num_users, _ = event_windows(bipartite=True)
    snapshots = unique_snapshots(windows)
    split = make_split(windows)
    targets = {
        "validation": unique_snapshots(split.validation, targets_only=True),
        "test": unique_snapshots(split.test, targets_only=True),
    }
    seeds = {"validation": 0, "test": 2}
    # Homogeneous partition claims for a bipartite stream are fine, but a
    # bipartite claim that excludes events is not.
    with pytest.raises(ValueError):
        negative_edge_table_from_snapshots(
            snapshots, targets, seeds, strategy="historical", batch_size=4,
            bipartite_source_count=5,
        )
    shuffled = list(snapshots)
    last = shuffled[-1]
    shuffled[-1] = Snapshot(
        x=last.x,
        edge_index=last.edge_index,
        active=last.active,
        time=last.time,
        query_edge_index=last.query_edge_index,
        query_timestamps=last.query_timestamps.flip(0),
        query_features=last.query_features,
    )
    with pytest.raises(ValueError):
        negative_edge_table_from_snapshots(
            shuffled, targets, seeds, strategy="historical", batch_size=4,
            bipartite_source_count=num_users,
        )


def tcl_model(windows, num_users, num_nodes) -> DyGLibLinkBaseline:
    model = DyGLibLinkBaseline(
        model_name="tcl",
        feature_dim=6,
        num_nodes=num_nodes,
        bipartite_source_count=num_users,
        interaction_feature_dim=2,
        time_feat_dim=2,
        num_layers=1,
        num_heads=1,
        num_neighbors=2,
        train_batch_size=4,
        eval_pair_batch_size=4,
        negative_ratio=1.0,
        max_positive_pairs=None,
        negative_destination_candidates=destination_pool(windows),
        allow_negative_collisions=True,
        eval_positive_batch_size=4,
    )
    split = make_split(windows)
    model.prepare_streams(unique_snapshots(windows), unique_snapshots(split.train))
    return model


def test_dyglib_batched_evaluator_scores_both_table_endpoints() -> None:
    windows, num_users, num_nodes = event_windows(bipartite=False)
    table, split = make_table(windows, num_users)
    model = tcl_model(windows, num_users, num_nodes)
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    original = model._embeddings

    def spy(sources, destinations, timestamps, **kwargs):
        calls.append((np.asarray(sources).copy(), np.asarray(destinations).copy()))
        return original(sources, destinations, timestamps, **kwargs)

    model._embeddings = spy  # type: ignore[method-assign]
    random_metrics = model.evaluate_protocol(split.test, [*split.train, *split.validation], query_seed=2)
    random_negative_calls = calls[1::2]
    positive_calls = calls[0::2]
    # Random protocol: the negative keeps the positive's source.
    for (positive_sources, _), (negative_sources, _) in zip(positive_calls, random_negative_calls):
        assert np.array_equal(positive_sources, negative_sources)
    calls.clear()

    model.negative_edge_table = table
    historical_metrics = model.evaluate_protocol(
        split.test, [*split.train, *split.validation], query_seed=2
    )
    expected = table.for_snapshots([w[-1].time for w in split.test])
    negative_sources = np.concatenate([call[0] for call in calls[1::2]])
    negative_destinations = np.concatenate([call[1] for call in calls[1::2]])
    assert np.array_equal(negative_sources - 1, expected.sources)
    assert np.array_equal(negative_destinations - 1, expected.destinations)
    events = sum(w[-1].query_edge_index.shape[1] for w in split.test)
    assert historical_metrics["examples"] == random_metrics["examples"] == 2.0 * events


def test_edgebank_unlimited_memory_recognises_every_pool_negative() -> None:
    windows, num_users, num_nodes = event_windows(bipartite=True, events=12)
    table, split = make_table(windows, num_users)
    edge_bank = EdgeBankLinkBaseline(
        num_nodes=num_nodes,
        bipartite_source_count=num_users,
        negative_ratio=1.0,
        max_positive_pairs=None,
        negative_destination_candidates=destination_pool(windows),
        allow_negative_collisions=True,
        eval_positive_batch_size=4,
        memory_mode="unlimited_memory",
    )
    captured: dict[str, torch.Tensor] = {}

    def capture(labels, scores, groups):
        captured.update(labels=labels, scores=scores, groups=groups)
        return {"ap": 0.0, "auc": 0.0, "examples": float(labels.numel())}

    edge_bank.metrics = capture  # type: ignore[method-assign]
    edge_bank.negative_edge_table = table
    edge_bank.evaluate_protocol(split.test, [*split.train, *split.validation], query_seed=2)
    expected = table.for_snapshots([w[-1].time for w in split.test])
    negative_scores = captured["scores"][captured["labels"] == 0]
    assert negative_scores.numel() == len(expected)
    # Every pool negative was observed before its batch, so unlimited-memory
    # EdgeBank scores it exactly like a repeated positive.
    assert torch.all(negative_scores[torch.as_tensor(expected.from_pool)] == 1.0)
    assert expected.from_pool.any()


def test_tgat_evaluator_scores_table_negatives() -> None:
    windows, num_users, num_nodes = event_windows(bipartite=True)
    table, split = make_table(windows, num_users)
    model = TGATLinkBaseline(
        feature_dim=6,
        num_nodes=num_nodes,
        bipartite_source_count=num_users,
        interaction_feature_dim=2,
        num_layers=1,
        num_heads=1,
        num_neighbors=2,
        train_batch_size=4,
        eval_group_batch_size=4,
        negative_ratio=1.0,
        max_positive_pairs=None,
        negative_destination_candidates=destination_pool(windows),
        allow_negative_collisions=True,
        eval_positive_batch_size=4,
    )
    model.prepare_streams(unique_snapshots(windows), unique_snapshots(split.train))
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = model._score_pairs

    def spy(sources, destinations, *args, **kwargs):
        calls.append((sources.detach().clone(), destinations.detach().clone()))
        return original(sources, destinations, *args, **kwargs)

    model._score_pairs = spy  # type: ignore[method-assign]
    model.negative_edge_table = table
    metrics = model.evaluate_protocol(split.test, [*split.train, *split.validation], query_seed=2)
    expected = table.for_snapshots([w[-1].time for w in split.test])
    negative_sources = torch.cat([call[0] for call in calls[1::2]]).cpu().numpy()
    negative_destinations = torch.cat([call[1] for call in calls[1::2]]).cpu().numpy()
    assert np.array_equal(negative_sources, expected.sources)
    assert np.array_equal(negative_destinations, expected.destinations)
    assert metrics["examples"] == 2.0 * len(expected)


def test_jodie_scores_historical_negatives_with_their_own_user() -> None:
    windows, num_users, num_nodes = event_windows(bipartite=True)
    table, split = make_table(windows, num_users)
    model = JODIELinkBaseline(
        feature_dim=6,
        num_nodes=num_nodes,
        bipartite_source_count=num_users,
        hidden_dim=8,
        interaction_feature_dim=2,
        negative_ratio=1.0,
        max_positive_pairs=None,
        new_edges_only=False,
        negative_destination_candidates=destination_pool(windows),
        allow_negative_collisions=True,
        eval_positive_batch_size=4,
        tbatch_count=4,
    )
    model.fit_stream_statistics(unique_snapshots(windows))
    scored_users: list[int] = []
    original = model._score_candidates

    def spy(state, user, candidate_items, timestamp):
        scored_users.append(int(user))
        return original(state, user, candidate_items, timestamp)

    model._score_candidates = spy  # type: ignore[method-assign]
    model.negative_edge_table = table
    metrics = model.evaluate_protocol(split.test, [*split.train, *split.validation], query_seed=2)
    expected = table.for_snapshots([w[-1].time for w in split.test])
    assert metrics["examples"] == 2.0 * len(expected)
    # Negatives whose user differs from the positive's are scored for their own user.
    differing = {
        int(source)
        for source, positive in zip(expected.sources, expected.positive_sources)
        if source != positive
    }
    assert differing and differing <= set(scored_users)


def test_rcps_evaluates_table_negatives_and_trains_randomly() -> None:
    windows, num_users, num_nodes = event_windows(bipartite=True)
    table, split = make_table(windows, num_users)
    model = RCPSJEPA(
        feature_dim=6,
        num_nodes=num_nodes,
        hidden_dim=8,
        rwpe_dim=2,
        rwpe_walks=4,
        time_dim=4,
        gnn_layers=1,
        window_size=3,
        predictor_hidden_dim=16,
        event_dim=3,
        subgraph_budget=4,
        negative_ratio=1.0,
        train_negative_ratio=2.0,
        max_positive_pairs=None,
        new_edges_only=False,
        undirected=False,
        bipartite_source_count=num_users,
        negative_destination_candidates=destination_pool(windows),
        allow_negative_collisions=True,
        eval_positive_batch_size=4,
    )
    model.negative_edge_table = table
    loss, _ = model.loss_windows(split.train, pair_batch_size=8, query_seed=3)
    assert torch.isfinite(loss)  # training targets fall back to random negatives
    metrics = model.evaluate_windows(split.test, pair_batch_size=8, query_seed=2)
    events = sum(w[-1].query_edge_index.shape[1] for w in split.test)
    assert metrics["examples"] == 2.0 * events


def test_driver_attaches_table_only_after_checkpoint_selection() -> None:
    windows, num_users, num_nodes = event_windows(bipartite=False)
    table, split = make_table(windows, num_users)
    model = tcl_model(windows, num_users, num_nodes)
    seen_tables: list[NegativeEdgeTable | None] = []
    original = model.evaluate_protocol

    def spy(*args, **kwargs):
        seen_tables.append(model.negative_edge_table)
        return original(*args, **kwargs)

    model.evaluate_protocol = spy  # type: ignore[method-assign]
    validation, test = _train_one(
        "tcl",
        model,
        split,
        {
            "epochs": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "eval_every": 1,
        },
        seed=3,
        negative_edge_table=table,
    )
    # Two per-epoch validations pick the checkpoint with random negatives;
    # only the final validation/test pass sees the historical table.
    assert [t is None for t in seen_tables] == [True, True, False, False]
    assert all(t is table for t in seen_tables[2:])
    assert 0.0 <= validation["ap"] <= 1.0 and 0.0 <= test["ap"] <= 1.0


def test_result_paths_get_a_strategy_suffix() -> None:
    assert _with_strategy_suffix("results/x.json", "random") == "results/x.json"
    assert _with_strategy_suffix("results/x.json", "historical") == "results/x_historical.json"
    config = {
        "seed": 0,
        "output_dir": "results",
        "link": {"negative_strategy": "historical"},
        "datasets": [
            {"name": "wikipedia", "path": "a.npz", "interaction_feature_dim": 172},
        ],
    }
    (_, expanded), = _dataset_configs(config)
    assert expanded["output_path"] == "results/link_comparison_wikipedia_historical.json"
    config["link"] = {}
    (_, expanded), = _dataset_configs(config)
    assert expanded["output_path"] == "results/link_comparison_wikipedia.json"


def test_historical_overlay_config_inherits_the_random_run() -> None:
    base_path = Path("configs/link_comparison_all.yaml")
    overlay_path = Path("configs/link_comparison_all_historical.yaml")
    base = yaml.safe_load(base_path.read_text())
    config = load_config(overlay_path)
    assert "base_config" not in config
    assert _negative_strategy(config) == "historical"
    assert _negative_strategy(load_config(base_path)) == "random"
    # Every dataset entry and every model/optimizer recipe is the base's.
    assert config["datasets"] == base["datasets"]
    for section in ("rcps_jepa", "rcps_training", "dygformer", "tgn", "edgebank", "training"):
        assert config[section] == base[section]
    assert {k: v for k, v in config["link"].items() if k != "negative_strategy"} == {
        k: v for k, v in base["link"].items() if k != "negative_strategy"
    }
    assert config["output_dir"] == "results/historical"
    assert "historical" in config["wandb"]["tags"]
    expanded = dict(_dataset_configs(config))
    assert set(expanded) == {entry["name"] for entry in base["datasets"]}
    assert expanded["canparl"]["output_path"] == (
        "results/historical/link_comparison_canparl_historical.json"
    )
    assert expanded["canparl"]["link"]["negative_strategy"] == "historical"
    # Result files never collide with the random run's.
    random_expanded = dict(_dataset_configs(load_config(base_path)))
    assert random_expanded["canparl"]["output_path"] == "results/link_comparison_canparl.json"
