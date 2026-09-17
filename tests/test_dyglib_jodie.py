"""DyGLib's JODIE variant runs through the shared protocol on any graph."""

from __future__ import annotations

import torch

from jepa_compare.compare_link_prediction import _ALL_COMPARISON_MODELS, _requested_models
from jepa_compare.dyglib_baselines import DyGLibLinkBaseline
from jepa_compare.dyglib_official import MemoryModel
from jepa_compare.temporal_event_utils import unique_snapshots

from test_historical_negatives import destination_pool, event_windows, make_split


def jodie_model(windows, num_users, num_nodes) -> DyGLibLinkBaseline:
    model = DyGLibLinkBaseline(
        model_name="jodie",
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


def test_dyglib_jodie_is_a_memory_model_with_time_projection() -> None:
    windows, num_users, num_nodes = event_windows(bipartite=False)
    model = jodie_model(windows, num_users, num_nodes)
    assert model.is_memory_model
    assert isinstance(model.backbone, MemoryModel)
    assert model.backbone.model_name == "JODIE"
    assert type(model.backbone.embedding_module).__name__ == "TimeProjectionEmbedding"
    assert type(model.backbone.memory_updater).__name__ == "RNNMemoryUpdater"


def test_dyglib_jodie_trains_and_evaluates_on_homogeneous_and_bipartite_graphs() -> None:
    for bipartite in (False, True):
        windows, num_users, num_nodes = event_windows(bipartite=bipartite)
        split = make_split(windows)
        model = jodie_model(windows, num_users, num_nodes)
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=1e-4
        )
        metrics = model.train_epoch(split.train, optimizer, grad_clip=1.0, seed=3)
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        validation = model.evaluate_protocol(split.validation, split.train, query_seed=0)
        test = model.evaluate_protocol(split.test, [*split.train, *split.validation], query_seed=2)
        events = sum(w[-1].query_edge_index.shape[1] for w in split.test)
        assert test["examples"] == 2.0 * events
        assert 0.0 <= validation["ap"] <= 1.0 and 0.0 <= test["auc"] <= 1.0


def test_jodie_author_is_opt_in_and_jodie_is_default() -> None:
    assert "jodie" in _ALL_COMPARISON_MODELS and "jodie_author" in _ALL_COMPARISON_MODELS
    assert _requested_models({}) is None
    assert _requested_models({"models": ["jodie_author"]}) == {"jodie_author"}
