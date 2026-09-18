"""DyGLib's inductive (new-node) setting: held-out nodes, reduced training data,
new-node-only evaluation with the new-node destination pool."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_compare.compare_link_prediction import (
    _dataset_configs,
    _link_setting,
    _train_one,
    _with_setting_suffix,
    load_config,
)
from jepa_compare.data import Snapshot
from jepa_compare.dyglib_baselines import DyGLibLinkBaseline, EdgeBankLinkBaseline
from jepa_compare.inductive_setting import (
    InductiveSetting,
    build_inductive_setting,
    event_touches,
    normalize_setting,
    remove_nodes,
    restrict_queries,
    sample_new_nodes,
)
from jepa_compare.link_prediction import TemporalWindowSplit, sliding_windows
from jepa_compare.rcps_jepa import RCPSJEPA
from jepa_compare.temporal_event_utils import unique_snapshots
from jepa_compare.tgat_baseline import TGATLinkBaseline

NUM_NODES = 40
FEATURES = 6


def make_snapshots(num_snapshots: int = 8, events: int = 30, seed: int = 0) -> list[Snapshot]:
    generator = torch.Generator().manual_seed(seed)
    snapshots = []
    for time in range(num_snapshots):
        sources = torch.randint(0, NUM_NODES, (events,), generator=generator)
        destinations = torch.randint(0, NUM_NODES, (events,), generator=generator)
        timestamps = torch.sort(torch.rand(events, generator=generator) * 9.0 + 10.0 * time).values
        queries = torch.stack([sources, destinations])
        messages = torch.cat([queries, queries.flip(0)], dim=1)
        active = torch.zeros(NUM_NODES, dtype=torch.bool)
        active[queries.flatten()] = True
        snapshots.append(
            Snapshot(
                x=torch.randn(NUM_NODES, FEATURES, generator=generator),
                edge_index=messages,
                active=active,
                time=time,
                query_edge_index=queries,
                query_timestamps=timestamps.to(torch.float32),
                query_features=torch.randn(events, 2, generator=generator),
                query_labels=torch.zeros(events, dtype=torch.long),
            )
        )
    return snapshots


def make_split(snapshots: list[Snapshot]) -> TemporalWindowSplit:
    windows = sliding_windows(snapshots, 3)          # 6 windows, targets 2..7
    return TemporalWindowSplit(train=windows[:3], validation=windows[3:4], test=windows[4:])


def make_setting(ratio: float = 0.2) -> tuple[list[Snapshot], TemporalWindowSplit, InductiveSetting]:
    snapshots = make_snapshots()
    split = make_split(snapshots)
    return snapshots, split, build_inductive_setting(snapshots, split, ratio=ratio, seed=2020)


def link_kwargs(pool: torch.Tensor, num_users=None) -> dict:
    return dict(
        negative_ratio=1.0,
        max_positive_pairs=None,
        new_edges_only=False,
        undirected=False,
        bipartite_source_count=num_users,
        negative_destination_candidates=pool,
        allow_negative_collisions=True,
        eval_positive_batch_size=4,
    )


def test_new_node_draw_is_deterministic_and_sized_like_dyglib() -> None:
    candidates = torch.arange(10, 30)
    first = sample_new_nodes(candidates, num_total_nodes=37, ratio=0.1, seed=2020)
    second = sample_new_nodes(candidates, num_total_nodes=37, ratio=0.1, seed=2020)
    assert torch.equal(first, second) and first.numel() == int(0.1 * 37) == 3
    assert set(first.tolist()) <= set(range(10, 30))
    assert not torch.equal(first, sample_new_nodes(candidates, 37, 0.1, seed=7))
    with pytest.raises(ValueError, match="cannot hold out"):
        sample_new_nodes(torch.arange(2), num_total_nodes=100, ratio=0.1, seed=2020)
    with pytest.raises(ValueError, match="new_node_ratio"):
        sample_new_nodes(candidates, 37, ratio=1.5, seed=0)


def test_reduced_training_snapshots_hide_the_held_out_nodes_entirely() -> None:
    snapshots, split, setting = make_setting()
    held_out = torch.zeros(NUM_NODES, dtype=torch.bool)
    held_out[setting.sampled_nodes] = True
    assert setting.sampled_nodes.numel() == int(0.2 * setting.summary["num_interacting_nodes"])
    train_times = {s.time for s in unique_snapshots(split.train)}
    assert [s.time for s in setting.train_snapshots] == sorted(train_times)
    for original, reduced in zip(unique_snapshots(split.train), setting.train_snapshots):
        assert reduced.time == original.time
        assert not event_touches(reduced, held_out).any()
        assert not (held_out[reduced.edge_index[0]] | held_out[reduced.edge_index[1]]).any()
        assert torch.equal(reduced.x[held_out], torch.zeros(int(held_out.sum()), FEATURES))
        assert not reduced.active[held_out].any()
        # Nothing else changed: the kept events are the original ones in order.
        keep = ~event_touches(original, held_out)
        assert torch.equal(reduced.query_edge_index, original.query_edge_index[:, keep])
        assert torch.equal(reduced.query_timestamps, original.query_timestamps[keep])
        assert torch.equal(reduced.query_features, original.query_features[keep])
        assert torch.equal(reduced.x[~held_out], original.x[~held_out])
    removed = sum(
        int(o.query_edge_index.shape[1]) - int(r.query_edge_index.shape[1])
        for o, r in zip(unique_snapshots(split.train), setting.train_snapshots)
    )
    assert removed == setting.summary["training_events_removed"] > 0
    # The training windows are built from the reduced snapshots only.
    for window in setting.train_windows:
        for snapshot in window:
            assert not event_touches(snapshot, held_out).any()


def test_new_node_set_is_every_node_absent_from_reduced_training() -> None:
    snapshots, split, setting = make_setting()
    seen = torch.zeros(NUM_NODES, dtype=torch.bool)
    for snapshot in setting.train_snapshots:
        seen[snapshot.query_edge_index.flatten()] = True
    assert torch.equal(setting.new_node_mask, ~seen)
    assert bool(setting.new_node_mask[setting.sampled_nodes].all())
    assert np.array_equal(setting.new_node_ids, torch.nonzero(~seen).flatten().numpy())


def test_evaluation_windows_keep_full_context_and_new_node_positives_only() -> None:
    snapshots, split, setting = make_setting()
    for original_windows, windows in (
        (split.validation, setting.validation_windows),
        (split.test, setting.test_windows),
    ):
        assert len(windows) == len(original_windows)
        for original, window in zip(original_windows, windows):
            assert all(a is b for a, b in zip(original[:-1], window[:-1]))   # context untouched
            target, full = window[-1], original[-1]
            assert target.time == full.time
            assert target.x is full.x and target.edge_index is full.edge_index
            assert event_touches(target, setting.new_node_mask).all()
            keep = event_touches(full, setting.new_node_mask)
            assert torch.equal(target.query_edge_index, full.query_edge_index[:, keep])
            assert 0 < int(keep.sum()) < int(keep.numel())
    validation_targets = unique_snapshots(setting.validation_windows, targets_only=True)
    expected_pool = torch.unique(torch.cat([t.query_edge_index[1] for t in validation_targets]))
    assert torch.equal(setting.validation_pool, expected_pool)
    assert setting.summary["validation_events"] == sum(
        int(t.query_edge_index.shape[1]) for t in validation_targets
    )
    assert setting.summary["validation_events"] < setting.summary["validation_events_total"]
    # History for DyGJEPA: reduced training snapshots, then the full later ones.
    assert [s.time for s in setting.history_snapshots] == [s.time for s in snapshots]
    train_times = {s.time for s in setting.train_snapshots}
    for snapshot, original in zip(setting.history_snapshots, snapshots):
        if snapshot.time in train_times:
            assert snapshot is not original
        else:
            assert snapshot is original
    assert [w[-1].time for w in setting.final_split(split).validation] == [
        w[-1].time for w in split.validation
    ]
    assert setting.validation_query_seed == 1 and setting.test_query_seed == 3


def test_restrict_and_remove_are_pure() -> None:
    snapshot = make_snapshots(1)[0]
    mask = torch.zeros(NUM_NODES, dtype=torch.bool)
    mask[:5] = True
    before = snapshot.query_edge_index.clone()
    restricted = restrict_queries(snapshot, mask)
    reduced = remove_nodes(snapshot, mask)
    assert torch.equal(snapshot.query_edge_index, before)
    assert restricted.query_edge_index.shape[1] + reduced.query_edge_index.shape[1] == before.shape[1]


def tcl_model(snapshots, setting: InductiveSetting | None) -> DyGLibLinkBaseline:
    pool = torch.unique(torch.cat([s.query_edge_index[1] for s in snapshots]))
    model = DyGLibLinkBaseline(
        model_name="tcl",
        feature_dim=FEATURES,
        num_nodes=NUM_NODES,
        interaction_feature_dim=2,
        time_feat_dim=2,
        num_layers=1,
        num_heads=1,
        num_neighbors=2,
        train_batch_size=4,
        eval_pair_batch_size=4,
        **link_kwargs(pool),
    )
    split = make_split(snapshots)
    train = unique_snapshots(split.train) if setting is None else setting.train_snapshots
    model.prepare_streams(
        snapshots, train, exclude_nodes=None if setting is None else setting.sampled_nodes.numpy()
    )
    return model


def test_dyglib_adapter_trains_on_the_reduced_stream_with_original_edge_ids() -> None:
    snapshots, split, setting = make_setting()
    full = tcl_model(snapshots, None)
    reduced = tcl_model(snapshots, setting)
    held = np.zeros(NUM_NODES + 1, dtype=bool)
    held[setting.sampled_nodes.numpy() + 1] = True
    stream = reduced._train_stream
    assert not (held[stream.sources] | held[stream.destinations]).any()
    assert len(stream) == setting.summary["training_events_kept"]
    # Edge ids are the full stream's ids for the very same events.
    full_stream = full._train_stream
    keep = ~(held[full_stream.sources] | held[full_stream.destinations])
    assert np.array_equal(stream.edge_ids, full_stream.edge_ids[keep])
    assert np.array_equal(stream.timestamps, full_stream.timestamps[keep])
    assert not held[reduced._train_destinations].any()
    # The full stream / full sampler used at evaluation time are unchanged.
    assert np.array_equal(reduced._full_destinations, full._full_destinations)
    for time in {s.time for s in setting.train_snapshots}:
        assert len(reduced._snapshot_streams[time]) == len(full._snapshot_streams[time])
        assert len(reduced._train_snapshot_streams[time]) <= len(full._snapshot_streams[time])


def test_dyglib_adapter_scores_new_node_events_against_the_new_node_pool() -> None:
    snapshots, split, setting = make_setting()
    model = tcl_model(snapshots, setting)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train_epoch(split.train, optimizer, grad_clip=1.0, seed=3)
    model.eval()
    calls: list[np.ndarray] = []
    original = model._embeddings

    def spy(sources, destinations, timestamps, *, edge_ids=None, positive=False):
        calls.append(np.asarray(destinations) - 1)
        return original(sources, destinations, timestamps, edge_ids=edge_ids, positive=positive)

    model._embeddings = spy  # type: ignore[method-assign]
    transductive = model.evaluate_protocol(split.validation, split.train, query_seed=0)
    assert transductive["examples"] == 2.0 * setting.summary["validation_events_total"]
    calls.clear()
    pool = setting.validation_pool.numpy()
    model.inductive_evaluation = (setting.new_node_ids, pool)
    inductive = model.evaluate_protocol(setting.validation_windows, setting.full_train_windows, query_seed=1)
    assert inductive["examples"] == 2.0 * setting.summary["validation_events"]
    # Stateless backbones embed the positives first, then the negatives, per batch.
    positives = np.concatenate(calls[0::2])
    drawn = np.concatenate(calls[1::2])
    assert positives.size == setting.summary["validation_events"]
    assert drawn.size == setting.summary["validation_events"]
    assert np.isin(drawn, pool).all()
    # Passing the wrong targets is caught.
    with pytest.raises(RuntimeError, match="new-node events"):
        model.evaluate_protocol(split.validation, split.train, query_seed=1)


def test_tgat_and_edgebank_score_only_the_restricted_events() -> None:
    snapshots, split, setting = make_setting()
    tgat = TGATLinkBaseline(
        feature_dim=FEATURES,
        num_nodes=NUM_NODES,
        interaction_feature_dim=2,
        num_layers=1,
        num_heads=1,
        num_neighbors=2,
        train_batch_size=4,
        eval_group_batch_size=4,
        **link_kwargs(setting.test_pool),
    )
    tgat.prepare_streams(snapshots, setting.train_snapshots)
    held = torch.zeros(NUM_NODES, dtype=torch.bool)
    held[setting.sampled_nodes] = True
    assert not (held[tgat._train_stream.sources] | held[tgat._train_stream.destinations]).any()
    seen: list[torch.Tensor] = []
    original = tgat._score_pairs

    def spy(sources, destinations, *args, **kwargs):
        seen.append(destinations.detach().clone())
        return original(sources, destinations, *args, **kwargs)

    tgat._score_pairs = spy  # type: ignore[method-assign]
    result = tgat.evaluate_protocol(setting.test_windows, setting.full_train_windows, query_seed=3)
    assert result["examples"] == 2.0 * setting.summary["test_events"]
    negatives = torch.cat(seen[1::2])
    assert torch.isin(negatives, setting.test_pool).all()

    edge_bank = EdgeBankLinkBaseline(
        num_nodes=NUM_NODES, memory_mode="unlimited_memory", **link_kwargs(setting.test_pool)
    )
    result = edge_bank.evaluate_protocol(
        setting.test_windows, [*setting.full_train_windows, *split.validation], query_seed=3
    )
    assert result["examples"] == 2.0 * setting.summary["test_events"]


def test_driver_selects_on_transductive_validation_and_reports_new_node_metrics() -> None:
    snapshots, split, setting = make_setting()
    reduced_split = TemporalWindowSplit(
        train=setting.train_windows, validation=split.validation, test=split.test
    )
    model = tcl_model(snapshots, setting)
    calls: list[tuple[int, int, tuple | None, int]] = []
    original = model.evaluate_protocol

    def spy(windows, history_windows, query_seed):
        pool = model.negative_destination_candidates
        calls.append(
            (
                int(sum(w[-1].query_edge_index.shape[1] for w in windows)),
                int(pool.numel()),
                model.inductive_evaluation,
                query_seed,
            )
        )
        return original(windows, history_windows, query_seed)

    model.evaluate_protocol = spy  # type: ignore[method-assign]
    validation, test = _train_one(
        "tcl",
        model,
        reduced_split,
        {"epochs": 2, "learning_rate": 1e-4, "weight_decay": 0.0, "grad_clip": 1.0, "eval_every": 1},
        seed=3,
        inductive=setting,
    )
    total_val = setting.summary["validation_events_total"]
    full_pool = int(model._full_destinations.size)
    # Two per-epoch validations: full positives, full pool, seed 0, no inductive hook.
    assert [c[0] for c in calls[:2]] == [total_val, total_val]
    assert all(c[1] == full_pool and c[2] is None and c[3] == 0 for c in calls[:2])
    # Final validation / test: new-node positives, new-node pools, seeds 1 / 3.
    assert calls[2][0] == setting.summary["validation_events"] and calls[2][3] == 1
    assert calls[2][1] == int(setting.validation_pool.numel()) and calls[2][2] is not None
    assert calls[3][0] == setting.summary["test_events"] and calls[3][3] == 3
    assert calls[3][1] == int(setting.test_pool.numel())
    assert validation["examples"] == 2.0 * setting.summary["validation_events"]
    assert test["examples"] == 2.0 * setting.summary["test_events"]
    assert test["best_epoch"] in (1.0, 2.0)


def test_dygjepa_evaluates_restricted_windows_with_the_reduced_history() -> None:
    snapshots, split, setting = make_setting()
    torch.manual_seed(0)
    model = RCPSJEPA(
        feature_dim=FEATURES,
        num_nodes=NUM_NODES,
        hidden_dim=8,
        rwpe_dim=2,
        rwpe_walks=4,
        time_dim=4,
        gnn_layers=1,
        window_size=3,
        predictor_hidden_dim=16,
        event_dim=3,
        subgraph_budget=4,
        train_negative_ratio=2.0,
        **link_kwargs(setting.validation_pool),
    )
    model.prepare_causal_history(setting.history_snapshots)
    model.eval()
    result = model.evaluate_windows(setting.validation_windows, query_seed=1)
    assert result["examples"] == 2.0 * setting.summary["validation_events"]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = model.train_epoch(setting.train_windows, optimizer, 1.0, seed=3)
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_inductive_setting_overlay_and_result_paths() -> None:
    base = load_config(Path("configs/link_comparison_all.yaml"))
    config = load_config(Path("configs/link_comparison_all_inductive_setting.yaml"))
    assert _link_setting(base) == "transductive"
    assert _link_setting(config) == "inductive"
    assert config["link"]["new_node_ratio"] == 0.1 and config["link"]["new_node_seed"] == 2020
    assert config["link"].get("negative_strategy", "random") == "random"
    assert config["datasets"] == base["datasets"]
    for section in ("rcps_jepa", "rcps_training", "dygformer", "edgebank", "training"):
        assert config[section] == base[section]
    expanded = dict(_dataset_configs(config))
    assert expanded["canparl"]["output_path"] == (
        "results/inductive_setting/link_comparison_canparl_inductive_setting.json"
    )
    assert expanded["canparl"]["link"]["setting"] == "inductive"
    transductive = dict(_dataset_configs(base))
    assert transductive["canparl"]["output_path"] == "results/link_comparison_canparl.json"
    assert _with_setting_suffix("results/x.json", "transductive") == "results/x.json"
    assert _with_setting_suffix("results/x.json", "inductive") == "results/x_inductive_setting.json"
    assert normalize_setting(None) == "transductive"
    with pytest.raises(ValueError, match="link.setting"):
        normalize_setting("new_node")
