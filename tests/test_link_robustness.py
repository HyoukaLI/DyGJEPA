"""Noise injection and evaluation plumbing of the robustness run."""
import json
from pathlib import Path

import pytest
import torch

from jepa_compare.compare_link_robustness import (
    DEFAULT_RATES,
    SUPPORTED_MODELS,
    evaluate_under_noise,
    inject_noise,
    noisy_windows,
    plot,
    write_csv,
)
from jepa_compare.data import Snapshot
from jepa_compare.link_prediction import TemporalWindowSplit, sliding_windows
from jepa_compare.rcps_jepa import RCPSJEPA

NUM_NODES = 12
NUM_SOURCES = 4


def _snapshots(bipartite: bool = True) -> list[Snapshot]:
    generator = torch.Generator().manual_seed(3)
    snapshots = []
    for time in range(6):
        events = 20
        if bipartite:
            source = torch.randint(0, NUM_SOURCES, (events,), generator=generator)
            destination = torch.randint(NUM_SOURCES, NUM_NODES, (events,), generator=generator)
        else:
            source = torch.randint(0, NUM_NODES, (events,), generator=generator)
            destination = (source + 1 + torch.randint(0, NUM_NODES - 1, (events,), generator=generator)) % NUM_NODES
        queries = torch.stack([source, destination])
        edge_index = torch.unique(torch.cat([queries, queries.flip(0)], dim=1), dim=1)
        active = torch.zeros(NUM_NODES, dtype=torch.bool)
        active[queries.flatten()] = True
        snapshots.append(
            Snapshot(
                x=torch.randn(NUM_NODES, 5, generator=generator),
                edge_index=edge_index,
                active=active,
                time=time,
                query_edge_index=queries,
                query_timestamps=torch.sort(torch.rand(events, generator=generator) * 5 + time * 10).values,
                query_features=torch.randn(events, 3, generator=generator),
            )
        )
    return snapshots


def test_rate_zero_returns_the_original_snapshots() -> None:
    snapshots = _snapshots()
    noisy, masks = inject_noise(
        snapshots, 0.0, num_nodes=NUM_NODES, num_source_nodes=NUM_SOURCES, seed=0
    )
    assert all(a is b for a, b in zip(noisy, snapshots))
    assert all(not bool(mask.any()) and mask.numel() == 20 for mask in masks.values())


@pytest.mark.parametrize("bipartite", [True, False])
def test_noise_is_added_per_bin_and_flagged(bipartite: bool) -> None:
    snapshots = _snapshots(bipartite)
    rate = 0.3
    noisy, masks = inject_noise(
        snapshots,
        rate,
        num_nodes=NUM_NODES,
        num_source_nodes=NUM_SOURCES if bipartite else None,
        seed=7,
    )
    for clean, dirty in zip(snapshots, noisy):
        mask = masks[clean.time]
        assert dirty.time == clean.time
        assert dirty.query_edge_index.shape[1] == 26  # 20 + round(0.3 * 20)
        assert int(mask.sum()) == 6 and mask.shape[0] == 26
        # Real events survive untouched, in their original (sorted) order.
        assert torch.equal(dirty.query_edge_index[:, ~mask], clean.query_edge_index)
        assert torch.equal(dirty.query_timestamps[~mask], clean.query_timestamps)
        assert torch.equal(dirty.query_features[~mask], clean.query_features)
        # Noisy events are sorted in with the rest and stay inside the bin.
        assert torch.all(dirty.query_timestamps[1:] >= dirty.query_timestamps[:-1])
        noise_times = dirty.query_timestamps[mask]
        assert noise_times.min() >= clean.query_timestamps.min()
        assert noise_times.max() <= clean.query_timestamps.max()
        src, dst = dirty.query_edge_index[:, mask]
        if bipartite:
            assert bool((src < NUM_SOURCES).all()) and bool((dst >= NUM_SOURCES).all())
        else:
            assert bool((src != dst).all())
        # Structure and activity follow the noisy events; node features do not.
        assert dirty.edge_index.shape[1] >= clean.edge_index.shape[1]
        assert bool(dirty.active[src.long()].all()) and bool(dirty.active[dst.long()].all())
        assert dirty.x is clean.x
        # Resampled features are copies of real feature rows of the same bin.
        for row in dirty.query_features[mask]:
            assert any(torch.equal(row, real) for real in clean.query_features)


def test_noise_is_deterministic_and_rate_specific() -> None:
    snapshots = _snapshots()
    first, _ = inject_noise(snapshots, 0.5, num_nodes=NUM_NODES, num_source_nodes=NUM_SOURCES, seed=1)
    second, _ = inject_noise(snapshots, 0.5, num_nodes=NUM_NODES, num_source_nodes=NUM_SOURCES, seed=1)
    other_seed, _ = inject_noise(snapshots, 0.5, num_nodes=NUM_NODES, num_source_nodes=NUM_SOURCES, seed=2)
    assert all(torch.equal(a.query_edge_index, b.query_edge_index) for a, b in zip(first, second))
    assert any(not torch.equal(a.query_edge_index, b.query_edge_index) for a, b in zip(first, other_seed))
    zeros, _ = inject_noise(
        snapshots, 0.5, num_nodes=NUM_NODES, num_source_nodes=NUM_SOURCES, seed=1, feature_mode="zeros"
    )
    _, masks = inject_noise(snapshots, 0.5, num_nodes=NUM_NODES, num_source_nodes=NUM_SOURCES, seed=1)
    for dirty, mask in zip(zeros, masks.values()):
        assert torch.count_nonzero(dirty.query_features[mask]) == 0


def test_noisy_windows_keep_the_clean_target() -> None:
    snapshots = _snapshots()
    noisy, _ = inject_noise(snapshots, 0.4, num_nodes=NUM_NODES, num_source_nodes=NUM_SOURCES, seed=0)
    by_time = {s.time: s for s in noisy}
    windows = noisy_windows(sliding_windows(snapshots, 3), by_time)
    for clean, dirty in zip(sliding_windows(snapshots, 3), windows):
        assert dirty[-1] is clean[-1]
        assert all(d is by_time[c.time] and d is not c for c, d in zip(clean[:-1], dirty[:-1]))


def _tiny_rcps() -> RCPSJEPA:
    torch.manual_seed(0)
    return RCPSJEPA(
        feature_dim=5,
        num_nodes=NUM_NODES,
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
        max_positive_pairs=None,
        new_edges_only=False,
        undirected=False,
        bipartite_source_count=NUM_SOURCES,
        negative_destination_candidates=torch.arange(NUM_SOURCES, NUM_NODES),
        allow_negative_collisions=True,
        eval_positive_batch_size=10,
        use_causal_history=True,
    )


class _Graph:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.num_nodes = NUM_NODES
        self.num_source_nodes = NUM_SOURCES


def test_rcps_noisy_evaluation_matches_clean_at_rate_zero_and_changes_with_noise() -> None:
    snapshots = _snapshots()
    windows = sliding_windows(snapshots, 3)
    split = TemporalWindowSplit(train=windows[:2], validation=windows[2:3], test=windows[3:])
    model = _tiny_rcps()
    model.prepare_causal_history(snapshots)
    model.eval()
    clean = model.evaluate_windows(split.test, pair_batch_size=None, query_seed=2)
    training = {"pair_batch_size": None, "test_query_seed": 2}
    graph = _Graph(snapshots)
    at_zero = evaluate_under_noise(
        "rcps_jepa", model, graph, split, 0.0, noise_seed=0, feature_mode="resample", training=training
    )
    assert at_zero["ap"] == clean["ap"] and at_zero["examples"] == clean["examples"]
    noisy = evaluate_under_noise(
        "rcps_jepa", model, graph, split, 0.6, noise_seed=0, feature_mode="resample", training=training
    )
    # Same clean positives and negatives (same example count), different history.
    assert noisy["examples"] == clean["examples"]
    assert noisy["ap"] != clean["ap"] or noisy["auc"] != clean["auc"]
    # The model can be re-pointed at the clean history afterwards.
    again = evaluate_under_noise(
        "rcps_jepa", model, graph, split, 0.0, noise_seed=0, feature_mode="resample", training=training
    )
    assert again["ap"] == clean["ap"]


def test_csv_and_plot_from_a_result_record(tmp_path: Path) -> None:
    result = {
        "dataset": "uci",
        "seed": 42,
        "noise_rates": list(DEFAULT_RATES),
        "noise_seed": 0,
        "noise_features": "resample",
        "models": {
            name: {
                "validation": {"ap": 0.9},
                "clean_test": {"ap": 0.9 - 0.05 * i, "auc": 0.9},
                "by_rate": {
                    f"{r:.2f}": {"ap": 0.9 - 0.05 * i - 0.1 * r * (i + 1), "auc": 0.9 - 0.1 * r}
                    for r in DEFAULT_RATES
                },
            }
            for i, name in enumerate(["rcps_jepa", "tgn", "tgat", "dvgmae"])
        },
    }
    write_csv(result, tmp_path / "r.csv")
    lines = (tmp_path / "r.csv").read_text().splitlines()
    assert lines[0] == "dataset,model,noise_rate,ap,auc,ap_drop_pct"
    assert len(lines) == 1 + 4 * len(DEFAULT_RATES)
    pytest.importorskip("matplotlib")
    written = plot(result, tmp_path / "fig" / "uci_noise_ap")
    assert {p.suffix for p in written} == {".pdf", ".png"}
    assert json.loads(json.dumps(result))  # serialisable
    assert set(["tgn", "tgat", "dvgmae", "rcps_jepa"]) <= set(SUPPORTED_MODELS)
