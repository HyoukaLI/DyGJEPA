import torch

from jepa_compare.data import DynamicGraph, Snapshot
from jepa_compare.node_tgnn_baselines import (
    DyGLibStatelessNodeSSL,
    EvolveGCNHNodeClassifier,
    ROLANDNodeClassifier,
    TGATNodeSSL,
    TGNNodeSSL,
    snapshot_addition_events,
)


def _graph() -> DynamicGraph:
    generator = torch.Generator().manual_seed(7)
    edges = [
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
    ]
    snapshots = [
        Snapshot(
            x=torch.randn(6, 4, generator=generator),
            edge_index=edge,
            active=torch.ones(6, dtype=torch.bool),
            time=time,
        )
        for time, edge in enumerate(edges)
    ]
    return DynamicGraph(snapshots, labels=torch.tensor([0, 0, 1, 1, 0, 1]))


def test_snapshot_addition_events_deduplicates_reverse_edges() -> None:
    events = snapshot_addition_events(_graph().snapshots)
    assert len(events) == 3
    assert list(zip(events.sources, events.destinations)) == [(0, 1), (1, 2), (2, 3)]
    assert events.timestamps.tolist() == [0.0, 1.0, 2.0]


def test_snapshot_tgnn_classifiers_forward_and_backward() -> None:
    graph = _graph()
    for model in (
        EvolveGCNHNodeClassifier(4, 4, 2, layers=2),
        ROLANDNodeClassifier(4, 4, 2, layers=2, bptt_steps=2),
    ):
        logits = model(graph.snapshots)
        assert logits.shape == (6, 2)
        torch.nn.functional.cross_entropy(logits, graph.labels).backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_tgn_ssl_one_step_and_node_readout() -> None:
    graph = _graph()
    model = TGNNodeSSL(
        feature_dim=4,
        num_nodes=6,
        time_dim=4,
        num_layers=1,
        num_heads=2,
        num_neighbors=2,
        batch_size=3,
        inference_batch_size=3,
    )
    model.prepare(graph)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    metrics = model.train_epoch(optimizer, 1.0, 42)
    embedding = model.node_embeddings()
    assert metrics["events"] == 3.0
    assert embedding.shape == (6, 4)
    assert torch.isfinite(embedding).all()


def test_tgat_ssl_one_step_and_node_readout() -> None:
    graph = _graph()
    model = TGATNodeSSL(
        feature_dim=4,
        num_nodes=6,
        hidden_dim=4,
        num_layers=1,
        num_heads=2,
        num_neighbors=2,
        batch_size=3,
        inference_batch_size=3,
    )
    model.prepare(graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = model.train_epoch(optimizer, 1.0, 42)
    embedding = model.node_embeddings()
    assert metrics["events"] == 3.0
    assert embedding.shape == (6, 4)
    assert torch.isfinite(embedding).all()


def test_stateless_dyglib_ssl_one_step_and_node_readout() -> None:
    graph = _graph()
    for name in ("cawn", "tcl", "graphmixer", "dygformer"):
        model = DyGLibStatelessNodeSSL(
            model_name=name,
            feature_dim=4,
            num_nodes=6,
            time_dim=4,
            num_layers=1,
            num_heads=2,
            num_neighbors=2,
            channel_embedding_dim=2,
            position_feat_dim=4,
            walk_length=1,
            num_walk_heads=2,
            max_input_sequence_length=4,
            time_gap=2,
            batch_size=3,
            inference_batch_size=3,
        )
        model.prepare(graph)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = model.train_epoch(optimizer, 1.0, 42)
        embedding = model.node_embeddings()
        assert metrics["events"] == 3.0
        assert embedding.shape == (6, 4)
        assert torch.isfinite(embedding).all()
