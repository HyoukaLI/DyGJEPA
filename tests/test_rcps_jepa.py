import numpy as np
import torch

from jepa_compare.compare_link_prediction import _train_one
from jepa_compare.data import Snapshot, make_synthetic
from jepa_compare.dyglib_baselines import DyGLibLinkBaseline, EdgeBankLinkBaseline
from jepa_compare.jodie_baseline import JODIELinkBaseline
from jepa_compare.link_prediction import (
    NODE_EVENT_DIM,
    TemporalWindowSplit,
    binary_average_precision,
    binary_roc_auc,
    canonical_pairs,
    link_prediction_metrics,
    node_transition_statistics,
    sample_link_queries,
    sliding_windows,
    temporal_node_increments,
    temporal_window_split,
)
from jepa_compare.rcps_jepa import RCPSJEPA
from jepa_compare.signature import truncated_signature
from jepa_compare.temporal_event_utils import (
    EventStream,
    TemporalNeighborIndex,
    unique_snapshots,
)
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


def test_dyglib_random_queries_use_full_destination_pool_and_allow_collisions() -> None:
    graph = make_synthetic(8, 4, 6, 2, 0.2, seed=9)
    target = graph.snapshots[-1]
    positive_destination = canonical_pairs(
        target.edge_index, 8
    )[0, 1].reshape(1)
    queries = sample_link_queries(
        target,
        graph.snapshots[-2],
        negative_ratio=1.0,
        max_positive=1,
        seed=3,
        negative_destination_candidates=positive_destination,
        allow_negative_collisions=True,
    )
    positive = queries.pairs[queries.labels.bool()][0]
    negative = queries.pairs[~queries.labels.bool()][0]
    assert negative[0] == positive[0]
    assert negative[1] == positive_destination[0]


def test_raw_event_queries_preserve_self_interactions() -> None:
    x = torch.zeros(3, 2)
    target = Snapshot(
        x=x,
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        active=torch.ones(3, dtype=torch.bool),
        time=1,
        query_edge_index=torch.tensor([[0, 1, 2], [0, 2, 2]]),
        query_timestamps=torch.tensor([1.0, 1.1, 1.2]),
    )
    queries = sample_link_queries(
        target,
        None,
        negative_ratio=1.0,
        seed=5,
        new_edges_only=False,
        undirected=False,
        negative_destination_candidates=torch.arange(3),
        allow_negative_collisions=True,
    )
    positives = queries.pairs[queries.labels.bool()]
    assert sorted(map(tuple, positives.tolist())) == [(0, 0), (1, 2), (2, 2)]
    assert queries.timestamps is not None
    positive_times = queries.timestamps[queries.labels.bool()]
    assert torch.allclose(
        positive_times.sort().values, target.query_timestamps.sort().values
    )
    assert queries.labels.numel() == 2 * target.query_edge_index.shape[1]


def test_rcps_separates_training_and_evaluation_negative_ratios() -> None:
    window = bipartite_windows()[0]
    model = RCPSJEPA(
        feature_dim=6,
        num_nodes=7,
        hidden_dim=8,
        rwpe_dim=2,
        rwpe_walks=2,
        time_dim=4,
        gnn_layers=1,
        window_size=3,
        predictor_hidden_dim=16,
        event_dim=2,
        signature_depth=1,
        negative_ratio=1.0,
        train_negative_ratio=4.0,
        new_edges_only=False,
        undirected=False,
        bipartite_source_count=3,
        negative_destination_candidates=torch.arange(3, 7),
        allow_negative_collisions=True,
    )
    evaluation = model.sample_queries(window, seed=3)
    training = model.sample_queries(
        window, seed=3, negative_ratio=model.train_negative_ratio
    )
    positives = int(evaluation.labels.sum().item())
    assert evaluation.labels.numel() == positives * 2
    assert training.labels.numel() == positives * 5


def test_rcps_history_is_strictly_causal_within_a_snapshot() -> None:
    x = torch.zeros(3, 2)
    first = Snapshot(
        x=x,
        edge_index=torch.tensor([[0, 2], [2, 0]]),
        active=torch.ones(3, dtype=torch.bool),
        time=0,
        query_edge_index=torch.tensor([[0], [2]]),
        query_timestamps=torch.tensor([1.0]),
    )
    second = Snapshot(
        x=x,
        edge_index=torch.tensor([[0, 2], [2, 0]]),
        active=torch.ones(3, dtype=torch.bool),
        time=1,
        query_edge_index=torch.tensor([[0, 0, 0], [2, 2, 2]]),
        query_timestamps=torch.tensor([10.0, 10.0, 11.0]),
    )
    model = RCPSJEPA(
        feature_dim=2,
        num_nodes=3,
        hidden_dim=4,
        rwpe_dim=1,
        rwpe_walks=1,
        time_dim=2,
        gnn_layers=1,
        window_size=2,
        predictor_hidden_dim=8,
        event_dim=2,
        signature_depth=1,
        use_causal_history=True,
        history_semantic_dim=0,
    )
    model.prepare_causal_history([first, second])
    pairs = torch.tensor([[0, 2], [0, 2], [0, 2]])
    features = model._causal_history_features(
        second, pairs, torch.tensor([10.0, 10.0, 11.0])
    )
    # Events tied at timestamp 10 cannot observe one another. The event at 11
    # observes both timestamp-10 interactions plus the earlier snapshot event.
    expected_counts = torch.log1p(torch.tensor([1.0, 1.0, 3.0]))
    assert torch.allclose(features[:, 0], expected_counts)


def test_dyglib_metrics_average_ap_auc_by_positive_batch() -> None:
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    probabilities = torch.tensor([0.9, 0.8, 0.1, 0.2])
    groups = torch.tensor([0, 0, 1, 1])
    metrics = link_prediction_metrics(
        labels, probabilities, groups, positive_batch_size=1
    )
    assert metrics["ap"] == 0.75
    assert metrics["auc"] == 0.5


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


def test_tgat_keeps_recursive_temporal_attention_and_raw_edge_features() -> None:
    windows = bipartite_windows()
    snapshots = unique_snapshots(windows)
    model = TGATLinkBaseline(
        feature_dim=6,
        num_nodes=7,
        bipartite_source_count=3,
        interaction_feature_dim=2,
        num_layers=2,
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


def test_temporal_neighbor_index_accepts_an_empty_batch() -> None:
    stream = EventStream(
        sources=torch.tensor([0]),
        destinations=torch.tensor([1]),
        timestamps=torch.tensor([1.0]),
        features=torch.zeros(1, 2),
    )
    index = TemporalNeighborIndex(stream, num_nodes=2)
    nodes, events, times, mask = index.sample(
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
        count=3,
        uniform=False,
        rng=np.random.default_rng(1),
        device=torch.device("cpu"),
    )
    assert nodes.shape == events.shape == times.shape == mask.shape == (0, 3)


def test_tgat_empty_pair_batch_returns_empty_logits() -> None:
    model = TGATLinkBaseline(
        feature_dim=2,
        num_nodes=2,
        bipartite_source_count=None,
        interaction_feature_dim=2,
        num_layers=2,
        num_heads=2,
        num_neighbors=2,
    )
    stream = EventStream(
        sources=torch.tensor([0]),
        destinations=torch.tensor([1]),
        timestamps=torch.tensor([1.0]),
        features=torch.zeros(1, 2),
    )
    index = TemporalNeighborIndex(stream, num_nodes=2)
    logits = model._score_pairs(
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
        index,
        np.random.default_rng(1),
    )
    assert logits.shape == (0,)


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
    assert 0.0 <= test["auc"] <= 1.0
    assert test["best_epoch"] == 1.0


def test_bipartite_queries_corrupt_only_destination() -> None:
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


def test_temporal_window_split_is_chronological() -> None:
    graph = make_synthetic(12, 9, 6, 2, 0.15, seed=10)
    split = temporal_window_split(graph.snapshots, 3, 0.5, 0.25)
    assert split.train[-1][-1].time < split.validation[0][-1].time
    assert split.validation[-1][-1].time < split.test[0][-1].time


def test_vectorized_binary_metrics_preserve_ties() -> None:
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    scores = torch.tensor([0.5, 0.5, 1.0, 0.0])
    assert binary_average_precision(labels, scores) == 1.0
    assert binary_roc_auc(labels, scores) == 0.875


def test_official_dyglib_backbones_follow_shared_protocol() -> None:
    windows = bipartite_windows()
    split = TemporalWindowSplit(
        train=windows[:1], validation=windows[1:2], test=windows[2:]
    )
    all_snapshots = unique_snapshots(windows)
    train_snapshots = unique_snapshots(split.train)
    settings = {
        "dyrep": dict(num_layers=1, num_heads=1, num_neighbors=2),
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


def test_dyglib_random_event_evaluator_uses_all_target_events() -> None:
    windows = bipartite_windows()
    snapshots = unique_snapshots(windows)
    destination_pool = torch.unique(
        torch.cat([snapshot.query_edge_index[1] for snapshot in snapshots])
    )
    model = DyGLibLinkBaseline(
        model_name="tcl",
        feature_dim=6,
        num_nodes=7,
        bipartite_source_count=3,
        interaction_feature_dim=2,
        time_feat_dim=2,
        num_layers=1,
        num_heads=1,
        num_neighbors=2,
        train_batch_size=2,
        eval_pair_batch_size=2,
        negative_ratio=1.0,
        max_positive_pairs=None,
        negative_destination_candidates=destination_pool,
        allow_negative_collisions=True,
        eval_positive_batch_size=2,
    )
    model.prepare_streams(snapshots, snapshots[:3])
    metrics = model.evaluate_protocol(
        windows[2:], windows[:2], query_seed=2
    )
    target_events = sum(window[-1].query_edge_index.shape[1] for window in windows[2:])
    assert metrics["examples"] == 2.0 * target_events


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


def test_dyglib_backbone_created_after_to_follows_adapter_device() -> None:
    snapshots = unique_snapshots(bipartite_windows())
    model = DyGLibLinkBaseline(
        model_name="dyrep",
        feature_dim=6,
        num_nodes=7,
        bipartite_source_count=3,
        interaction_feature_dim=4,
        time_feat_dim=2,
        num_layers=1,
        num_heads=1,
        num_neighbors=2,
    ).to("meta")

    model.prepare_streams(snapshots, snapshots[:1])

    assert model.backbone is not None
    assert all(parameter.device.type == "meta" for parameter in model.parameters())


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
        DyGLibLinkBaseline(
            model_name="dyrep",
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
