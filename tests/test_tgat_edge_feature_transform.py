"""TGAT's per-dataset edge-feature guard (UN Trade raw trade volumes)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from jepa_compare.tgat_baseline import TGATLinkBaseline
from jepa_compare.temporal_event_utils import unique_snapshots

from test_historical_negatives import destination_pool, event_windows, make_split


def _tgat(windows, num_users, num_nodes, **kwargs) -> TGATLinkBaseline:
    torch.manual_seed(0)
    model = TGATLinkBaseline(
        feature_dim=6,
        num_nodes=num_nodes,
        bipartite_source_count=num_users,
        interaction_feature_dim=2,
        num_layers=2,
        num_heads=1,
        num_neighbors=2,
        train_batch_size=4,
        eval_group_batch_size=4,
        negative_ratio=1.0,
        max_positive_pairs=None,
        negative_destination_candidates=destination_pool(windows),
        allow_negative_collisions=True,
        eval_positive_batch_size=4,
        **kwargs,
    )
    split = make_split(windows)
    model.prepare_streams(unique_snapshots(windows), unique_snapshots(split.train))
    return model


def _scale_features(windows, scale: float):
    """Event features rescaled to UN-Trade-like magnitudes (raw trade volumes)."""
    scaled = []
    for window in windows:
        scaled.append(
            [
                dataclasses.replace(
                    snapshot, query_features=snapshot.query_features.abs() * scale
                )
                for snapshot in window
            ]
        )
    return scaled


def test_default_transform_leaves_edge_features_untouched() -> None:
    windows, num_users, num_nodes = event_windows(bipartite=False)
    model = _tgat(windows, num_users, num_nodes)
    assert model.edge_feature_transform == "none"
    stream = model._full_stream
    assert stream is not None
    assert torch.equal(model.edge_features[:-1], stream.features)
    assert torch.equal(model.edge_features[-1], torch.zeros(2))


def test_log1p_transform_is_sign_preserving_and_compresses_magnitudes() -> None:
    windows, num_users, num_nodes = event_windows(bipartite=False)
    model = _tgat(windows, num_users, num_nodes, edge_feature_transform="log1p")
    stream = model._full_stream
    assert stream is not None
    expected = torch.sign(stream.features) * torch.log1p(stream.features.abs())
    assert torch.allclose(model.edge_features[:-1], expected)
    huge = torch.tensor([[5.25e7, -5.25e7], [0.0, 276.0]])
    transformed = model._transform_edge_features(huge)
    assert transformed[0, 0] == pytest.approx(17.776, abs=1e-3)
    assert transformed[0, 1] == pytest.approx(-17.776, abs=1e-3)
    assert transformed[1, 0] == 0.0
    assert transformed[1, 1] == pytest.approx(5.624, abs=1e-3)


def test_unknown_transform_is_rejected() -> None:
    windows, num_users, num_nodes = event_windows(bipartite=False)
    with pytest.raises(ValueError, match="edge_feature_transform"):
        _tgat(windows, num_users, num_nodes, edge_feature_transform="standardize")


def test_log1p_keeps_training_finite_on_trade_volume_features() -> None:
    windows, num_users, num_nodes = event_windows(bipartite=False)
    scaled = _scale_features(windows, 5.0e7)
    split = make_split(scaled)
    model = _tgat(scaled, num_users, num_nodes, edge_feature_transform="log1p")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(3):
        metrics = model.train_epoch(split.train, optimizer, grad_clip=1.0, seed=3)
        assert torch.isfinite(torch.tensor(metrics["loss"])), metrics
    assert all(torch.isfinite(p).all() for p in model.parameters())
    test = model.evaluate_protocol(split.test, [*split.train, *split.validation], query_seed=2)
    assert 0.0 <= test["ap"] <= 1.0 and 0.0 <= test["auc"] <= 1.0
    assert torch.isfinite(torch.tensor(test["mean_probability"]))
