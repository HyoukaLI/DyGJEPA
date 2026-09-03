import torch

from jepa_compare.compare_link_prediction import _train_one
from jepa_compare.data import Snapshot, make_synthetic
from jepa_compare.dyrep_baseline import DyRepLinkBaseline
from jepa_compare.dyglib_baselines import DyGLibLinkBaseline, EdgeBankLinkBaseline
from jepa_compare.jodie_baseline import JODIELinkBaseline
from jepa_compare.link_prediction import (
    NODE_EVENT_DIM,
    TemporalWindowSplit,
    canonical_pairs,
    grouped_ranking_metrics,
    node_transition_statistics,
    sample_link_queries,
    sliding_windows,
    temporal_node_increments,
    temporal_window_split,
)
from jepa_compare.rcps_jepa import RCPSJEPA
from jepa_compare.signature import truncated_signature
from jepa_compare.temporal_event_utils import unique_snapshots
from jepa_compare.tgat_baseline import TGATLinkBaseline


def test_signature_preserves_second_order_event_order() -> None:
    ab = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    ba = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    sig_ab = truncated_signature(ab, depth=2)
    sig_ba = truncated_signature(ba, depth=2)
    assert torch.equal(sig_ab[:, :2], sig_ba[:, :2])
    assert not torch.equal(sig_ab[:, 2:], sig_ba[:, 2:])


def test_link_query_sampling_is_deterministic_and_disjoint() -> None:
    graph = make_synthetic(16, 4, 6, 2, 0.2, seed=7)
    first = sample_link_queries(
        graph.snapshots[-1], graph.snapshots[-2], max_positive=8, seed=19
    )
    second = sample_link_queries(
        graph.snapshots[-1], graph.snapshots[-2], max_positive=8, seed=19
    )
    assert torch.equal(first.pairs, second.pairs)
    assert torch.equal(first.labels, second.labels)
    assert torch.equal(first.group_ids, second.group_ids)
    positive = {tuple(pair) for pair in first.pairs[first.labels.bool()].tolist()}
    negative = {tuple(pair) for pair in first.pairs[~first.labels.bool()].tolist()}
    target = {tuple(pair) for pair in canonical_pairs(graph.snapshots[-1].edge_index, 16).tolist()}
    assert positive <= target
    assert negative.isdisjoint(target)


def test_node_relation_path_encodes_structural_transitions() -> None:
    graph = make_synthetic(16, 4, 6, 2, 0.2, seed=17)
    first, second = graph.snapshots[:2]
    transition = node_transition_statistics(second, first)
    increments = temporal_node_increments([first, second])
    assert transition.shape == (16, NODE_EVENT_DIM)
    assert increments.shape == (16, 2, NODE_EVENT_DIM + 1)
    assert torch.allclose(increments[:, 1, 1:], transition)
    assert torch.any(transition[:, 2] > 0) or torch.any(transition[:, 3] > 0)


def tiny_rcps() -> RCPSJEPA:
    return RCPSJEPA(
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


def bipartite_windows() -> list[list[Snapshot]]:
    generator = torch.Generator().manual_seed(23)
    snapshots = []
    for time in range(5):
        source = torch.tensor([0, 1, 2])
        destination = torch.tensor(
            [3 + time % 4, 3 + (time + 1) % 4, 3 + (time + 2) % 4]
        )
        queries = torch.stack([source, destination])
        messages = torch.cat([queries, queries.flip(0)], dim=1)
        snapshots.append(
            Snapshot(
                x=torch.randn(7, 6, generator=generator),
                edge_index=messages,
                active=torch.ones(7, dtype=torch.bool),
                time=time,
                query_edge_index=queries,
                query_timestamps=torch.arange(3, dtype=torch.float32) + time * 10,
                query_features=torch.randn(3, 2, generator=generator),
                query_labels=torch.tensor([time % 2, 0, 1]),
            )
        )
    return sliding_windows(snapshots, 3)


def homogeneous_windows() -> list[list[Snapshot]]:
    generator = torch.Generator().manual_seed(29)
    snapshots = []
    for time in range(5):
        source = torch.tensor([0, 1, 2])
        destination = torch.tensor([1 + time % 3, 2, 3])
        queries = torch.stack([source, destination])
        messages = torch.cat([queries, queries.flip(0)], dim=1)
        snapshots.append(
            Snapshot(
                x=torch.randn(4, 6, generator=generator),
                edge_index=messages,
                active=torch.ones(4, dtype=torch.bool),
                time=time,
                query_edge_index=queries,
                query_timestamps=torch.arange(3, dtype=torch.float32) + time * 10,
                query_features=torch.randn(3, 2, generator=generator),
                query_labels=torch.zeros(3, dtype=torch.long),
            )
        )
    return sliding_windows(snapshots, 3)


def test_rcps_loss_gradients_ema_and_survival_monotonicity() -> None:
    graph = make_synthetic(16, 5, 6, 2, 0.2, seed=8)
    window = sliding_windows(graph.snapshots, 3)[0]
    model = tiny_rcps()
    loss, metrics = model.loss_windows([window], pair_batch_size=16, query_seed=11)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics["link_loss"] > 0
    loss.backward()
    assert model.event_projector[0].weight.grad is not None
    assert model.node_event_projector[0].weight.grad is not None
    assert model.node_relation_gru.weight_ih_l0.grad is not None
    assert model.node_context_gate.weight.grad is not None
    assert next(model.online_encoder.parameters()).grad is not None
    assert next(model.target_encoder.parameters()).grad is None

    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.online_encoder.parameters()).add_(0.1)
    model.update_target_encoder()
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(before, after)

    queries = model.sample_queries(window, seed=13)
    output = model.forward_pairs(window, queries.pairs[:4])
    target = window[-1]
    longer_target = Snapshot(
        x=target.x,
        edge_index=target.edge_index,
        active=target.active,
        time=target.time + 2,
    )
    longer_output = model.forward_pairs([*window[:-1], longer_target], queries.pairs[:4])
    assert torch.allclose(longer_output.intensity, output.intensity)
    assert torch.all(longer_output.probability >= output.probability)


def test_untrained_node_inference_stays_on_context_homophily() -> None:
    graph = make_synthetic(16, 5, 6, 2, 0.2, seed=18)
    window = graph.snapshots[-3:]
    model = tiny_rcps()
    embeddings, node_ids = model.infer_nodes(window)
    assert embeddings.shape == (16, model.hidden_dim)
    assert torch.equal(node_ids, torch.arange(16))
    hop2 = model._homophily_state(window[-1])
    hop5 = model._content_views(window[-1])["hop5"]
    blend = model.blend_content_hops(model._content_views(window[-1]))
    # Zero-init residual keeps fused near the learned hop blend prior.
    assert torch.allclose(embeddings, blend, atol=0.15)
    assert not torch.allclose(hop2, model.feature_skip(window[-1].x))
    views, view_ids = model.infer_node_views(window)
    assert torch.equal(view_ids, node_ids)
    assert set(views) == {
        "content",
        "hop1",
        "hop2",
        "hop4",
        "hop5",
        "hop6",
        "hop8",
        "blend",
        "raw_hop5",
        "encoder",
        "enc_hop4",
        "enc_hop5",
        "temporal",
        "fused",
        "prediction",
    }
    assert torch.allclose(views["hop2"], hop2)
    assert torch.allclose(views["blend"], blend, atol=1e-4)
    assert torch.allclose(views["fused"], blend, atol=1e-4)
    assert torch.allclose(views["content"], model.feature_skip(window[-1].x))


def test_node_objective_uses_relation_signature_and_latent_target() -> None:
    graph = make_synthetic(16, 5, 6, 2, 0.2, seed=18)
    window = sliding_windows(graph.snapshots, 3)[0]
    model = tiny_rcps()
    loss, metrics = model.node_loss_windows([window], node_batch_size=8)
    assert torch.isfinite(loss)
    assert metrics["node_loss"] > 0
    assert metrics["contrastive_loss"] > 0
    assert 0 <= metrics["relation_gate"] <= 1
    assert 0 <= metrics["graph_scale"] <= 1
    assert 0 <= metrics["history_scale"] <= 1
    assert 0 <= metrics["homophily_scale"] <= 1
    assert 0 <= metrics["dynamics_scale"] <= 1
    loss.backward()
    assert model.node_event_projector[0].weight.grad is not None
    assert model.node_relation_gru.weight_ih_l0.grad is not None
    assert model.node_context_encoder[0].weight.grad is not None
    assert model.snapshot_graph_logit.grad is not None
    assert model.node_history_logit.grad is not None
    assert model.node_homophily_logit.grad is not None
    assert model.node_predictor[-1].weight.grad is not None


def test_jodie_link_baseline_uses_coupled_updates_and_time_projection() -> None:
    window = bipartite_windows()[0]
    model = JODIELinkBaseline(
        feature_dim=6,
        num_nodes=7,
        bipartite_source_count=3,
        hidden_dim=8,
        interaction_feature_dim=2,
        max_positive_pairs=3,
        tbatch_count=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    metrics = model.train_epoch([window], optimizer, grad_clip=1.0)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["prediction_loss"] >= 0
    assert metrics["state_loss"] > 0
    assert model.user_rnn.weight_ih.grad is not None
    assert model.item_rnn.weight_ih.grad is not None
    assert model.time_projection.weight.grad is not None
    assert model.prediction_layer.weight.grad is not None
    assert model.prediction_layer.in_features == 8 * 2 + 3 + 5
    assert model.prediction_layer.out_features == 8 + 5


def test_dyrep_keeps_point_process_and_sampled_survival_objective() -> None:
    windows = bipartite_windows()
    snapshots = unique_snapshots(windows)
    model = DyRepLinkBaseline(
        feature_dim=6,
        num_nodes=7,
        bipartite_source_count=3,
        hidden_dim=8,
        neighbor_count=2,
        survival_samples=2,
        train_batch_size=3,
        negative_ratio=1.0,
        max_positive_pairs=2,
    )
    model.prepare_streams(snapshots, unique_snapshots(windows[:1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = model.train_epoch(windows[:1], optimizer, grad_clip=1.0, seed=5)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["survival_loss"] > 0
    assert model.intensity_projection.weight.grad is not None
    validation = model.evaluate_protocol(windows[1:2], windows[:1], query_seed=7)
    assert validation["examples"] == 4
    assert 0 <= validation["ap"] <= 1


def test_tgat_keeps_recursive_temporal_attention_and_raw_edge_features() -> None:
    windows = bipartite_windows()
    snapshots = unique_snapshots(windows)
    model = TGATLinkBaseline(
        feature_dim=6,
        num_nodes=7,
        bipartite_source_count=3,
        interaction_feature_dim=2,
        num_layers=1,
        num_heads=2,
        num_neighbors=2,
        train_batch_size=3,
        eval_group_batch_size=2,
        negative_ratio=1.0,
        max_positive_pairs=2,
    )
    model.prepare_streams(snapshots, unique_snapshots(windows[:1]))
    assert model.edge_features.shape == (15 + 1, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = model.train_epoch(windows[:1], optimizer, grad_clip=1.0, seed=5)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert model.time_encoder.frequency.grad is not None
    validation = model.evaluate_protocol(windows[1:2], windows[:1], query_seed=7)
    assert validation["examples"] == 4
    assert 0 <= validation["auc"] <= 1


def test_jodie_keeps_stream_state_under_shared_query_protocol() -> None:
    windows = bipartite_windows()
    model = JODIELinkBaseline(
        feature_dim=6,
        num_nodes=7,
        bipartite_source_count=3,
        hidden_dim=8,
        interaction_feature_dim=2,
        negative_ratio=2.0,
        max_positive_pairs=3,
        tbatch_count=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train_epoch(windows[:2], optimizer, grad_clip=1.0)
    metrics = model.evaluate_protocol(
        windows[2:], windows[:2], query_seed=41
    )
    assert metrics["examples"] == 9.0
    assert 0.0 <= metrics["ap"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0
    assert 0.0 <= metrics["recall_at_10"] <= 1.0


def test_rcps_train_epoch_steps_each_pair_batch() -> None:
    graph = make_synthetic(16, 6, 6, 2, 0.2, seed=41)
    windows = sliding_windows(graph.snapshots, 3)[:2]
    model = tiny_rcps()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = next(model.online_encoder.parameters()).detach().clone()
    metrics = model.train_epoch(
        windows, optimizer, grad_clip=1.0, seed=7, pair_batch_size=8
    )
    after = next(model.online_encoder.parameters()).detach()
    assert metrics["steps"] >= 2
    assert metrics["link_loss"] > 0
    assert not torch.equal(before, after)


def test_rcps_driver_uses_minibatched_train_epoch() -> None:
    graph = make_synthetic(16, 6, 6, 2, 0.2, seed=43)
    windows = sliding_windows(graph.snapshots, 3)
    split = TemporalWindowSplit(
        train=windows[:1], validation=windows[1:2], test=windows[2:]
    )
    model = tiny_rcps()
    before = next(model.online_encoder.parameters()).detach().clone()
    validation, test = _train_one(
        "rcps_jepa",
        model,
        split,
        {
            "epochs": 1,
            "pair_batch_size": 8,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
            "eval_every": 1,
        },
        seed=11,
    )
    after = next(model.online_encoder.parameters()).detach()
    assert not torch.equal(before, after)
    assert 0.0 <= validation["ap"] <= 1.0
    assert 0.0 <= test["ap"] <= 1.0
    assert test["best_epoch"] == 1.0
    graph = make_synthetic(16, 6, 6, 2, 0.2, seed=29)
    windows = sliding_windows(graph.snapshots, 3)[:2]

    rcps = tiny_rcps()
    rcps.zero_grad(set_to_none=True)
    rcps_loss, rcps_metrics = rcps.loss_windows(
        windows,
        pair_batch_size=4,
        query_seed=31,
        backward=True,
    )
    assert torch.isfinite(rcps_loss)
    assert rcps_metrics["link_loss"] > 0
    assert rcps.event_projector[0].weight.grad is not None
    assert rcps.node_gru.weight_ih_l0.grad is not None
    assert next(rcps.online_encoder.parameters()).grad is not None

    jodie = JODIELinkBaseline(
        feature_dim=6,
        num_nodes=7,
        bipartite_source_count=3,
        hidden_dim=8,
        interaction_feature_dim=2,
        max_positive_pairs=3,
        tbatch_count=4,
    )
    optimizer = torch.optim.Adam(jodie.parameters(), lr=1e-3, weight_decay=1e-5)
    jodie_metrics = jodie.train_epoch(
        bipartite_windows()[:2], optimizer, grad_clip=1.0
    )
    assert torch.isfinite(torch.tensor(jodie_metrics["loss"]))
    assert jodie_metrics["prediction_loss"] > 0
    assert jodie.user_rnn.weight_ih.grad is not None
    assert jodie.prediction_layer.weight.grad is not None


def test_shared_driver_runs_native_jodie_train_validation_and_test() -> None:
    windows = bipartite_windows()
    split = TemporalWindowSplit(
        train=windows[:1], validation=windows[1:2], test=windows[2:]
    )
    model = JODIELinkBaseline(
        feature_dim=6,
        num_nodes=7,
        bipartite_source_count=3,
        hidden_dim=8,
        interaction_feature_dim=2,
        negative_ratio=2.0,
        max_positive_pairs=3,
        tbatch_count=4,
    )
    all_snapshots = model._unique_snapshots(windows)
    model.fit_stream_statistics(all_snapshots)
    validation, test = _train_one(
        "jodie",
        model,
        split,
        {
            "epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
            "eval_every": 1,
        },
        seed=17,
    )
    assert 0.0 <= validation["ap"] <= 1.0
    assert 0.0 <= test["mrr"] <= 1.0
    assert test["best_epoch"] == 1.0


def test_bipartite_queries_corrupt_only_destination_and_rank() -> None:
    x = torch.randn(7, 4)
    # Three users [0, 3), four items [3, 7); reverse edges are message-only.
    query_edges = torch.tensor([[0, 1, 0], [3, 4, 3]])
    message_edges = torch.cat([query_edges, query_edges.flip(0)], dim=1)
    snapshot = Snapshot(
        x,
        message_edges,
        torch.ones(7, dtype=torch.bool),
        1,
        query_edges,
    )
    queries = sample_link_queries(
        snapshot,
        None,
        negative_ratio=2.0,
        seed=3,
        new_edges_only=False,
        undirected=False,
        bipartite_source_count=3,
    )
    assert torch.all(queries.pairs[:, 0] < 3)
    assert torch.all(queries.pairs[:, 1] >= 3)
    assert int(queries.labels.sum().item()) == 3  # duplicate edits are preserved
    scores = torch.where(queries.labels.bool(), 1.0, 0.0)
    mrr, recall = grouped_ranking_metrics(
        queries.labels, scores, queries.group_ids, recall_k=1
    )
    assert mrr == 1.0
    assert recall == 1.0


def test_temporal_window_split_is_chronological() -> None:
    graph = make_synthetic(12, 9, 6, 2, 0.15, seed=10)
    split = temporal_window_split(graph.snapshots, 3, 0.5, 0.25)
    assert split.train[-1][-1].time < split.validation[0][-1].time
    assert split.validation[-1][-1].time < split.test[0][-1].time


def test_official_dyglib_backbones_follow_shared_protocol() -> None:
    windows = bipartite_windows()
    split = TemporalWindowSplit(
        train=windows[:1], validation=windows[1:2], test=windows[2:]
    )
    all_snapshots = unique_snapshots(windows)
    train_snapshots = unique_snapshots(split.train)
    settings = {
        "tgn": dict(num_layers=1, num_heads=1, num_neighbors=2),
        "cawn": dict(
            walk_length=1,
            num_walk_heads=1,
            num_neighbors=2,
            position_feat_dim=2,
            sample_neighbor_strategy="time_interval_aware",
            time_scaling_factor=1e-6,
        ),
        "tcl": dict(num_layers=1, num_heads=1, num_neighbors=2),
        "graphmixer": dict(num_layers=1, num_neighbors=2, time_gap=3),
        "dygformer": dict(
            num_layers=1,
            num_heads=1,
            channel_embedding_dim=2,
            max_input_sequence_length=4,
        ),
    }
    for model_name, model_settings in settings.items():
        model = DyGLibLinkBaseline(
            model_name=model_name,
            feature_dim=6,
            num_nodes=7,
            bipartite_source_count=3,
            interaction_feature_dim=2,
            time_feat_dim=2,
            train_batch_size=3,
            eval_pair_batch_size=4,
            negative_ratio=1.0,
            max_positive_pairs=2,
            **model_settings,
        )
        model.prepare_streams(all_snapshots, train_snapshots)
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-4,
        )
        train_metrics = model.train_epoch(
            split.train, optimizer, grad_clip=1.0, seed=31
        )
        validation = model.evaluate_protocol(
            split.validation, split.train, query_seed=37
        )
        assert torch.isfinite(torch.tensor(train_metrics["loss"]))
        assert 0.0 <= validation["ap"] <= 1.0
        assert validation["examples"] == 4.0

    edge_bank = EdgeBankLinkBaseline(
        num_nodes=7,
        bipartite_source_count=3,
        negative_ratio=1.0,
        max_positive_pairs=2,
    )
    edge_metrics = edge_bank.evaluate_protocol(
        split.validation, split.train, query_seed=37
    )
    assert 0.0 <= edge_metrics["auc"] <= 1.0


def test_dyglib_adapter_zero_pads_low_dimensional_event_features() -> None:
    windows = bipartite_windows()
    snapshots = unique_snapshots(windows)
    model = DyGLibLinkBaseline(
        model_name="tcl",
        feature_dim=6,
        num_nodes=7,
        bipartite_source_count=3,
        interaction_feature_dim=4,
        time_feat_dim=2,
        num_layers=1,
        num_heads=1,
        num_neighbors=2,
        train_batch_size=3,
        eval_pair_batch_size=4,
        negative_ratio=1.0,
        max_positive_pairs=2,
    )
    model.prepare_streams(snapshots, snapshots[:1])

    assert model._train_stream is not None
    assert model._train_stream.features.shape[1] == 4
    assert (model._train_stream.features[:, 2:] == 0.0).all()


def test_event_baselines_accept_homogeneous_destination_corruption() -> None:
    windows = homogeneous_windows()
    snapshots = unique_snapshots(windows)
    train_snapshots = snapshots[:3]
    models = [
        TGATLinkBaseline(
            feature_dim=6,
            num_nodes=4,
            bipartite_source_count=None,
            interaction_feature_dim=4,
            num_layers=1,
            num_heads=1,
            num_neighbors=2,
            train_batch_size=3,
            eval_group_batch_size=2,
            negative_ratio=1.0,
            max_positive_pairs=2,
        ),
        DyRepLinkBaseline(
            feature_dim=6,
            num_nodes=4,
            bipartite_source_count=None,
            hidden_dim=4,
            neighbor_count=2,
            survival_samples=2,
            train_batch_size=3,
            negative_ratio=1.0,
            max_positive_pairs=2,
        ),
        DyGLibLinkBaseline(
            model_name="tcl",
            feature_dim=6,
            num_nodes=4,
            bipartite_source_count=None,
            interaction_feature_dim=4,
            time_feat_dim=2,
            num_layers=1,
            num_heads=1,
            num_neighbors=2,
            train_batch_size=3,
            eval_pair_batch_size=4,
            negative_ratio=1.0,
            max_positive_pairs=2,
        ),
    ]
    for model in models:
        model.prepare_streams(snapshots, train_snapshots)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        metrics = model.train_epoch(windows[:1], optimizer, grad_clip=1.0, seed=41)
        assert torch.isfinite(torch.tensor(metrics["loss"]))
