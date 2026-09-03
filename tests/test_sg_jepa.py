import torch

from jepa_compare.data import make_synthetic
from jepa_compare.encoding import random_walk_positional_encoding
from jepa_compare.node_evaluation import MLPProbe, fit_probe_ensemble, stratified_split
from jepa_compare.sg_jepa import SGJEPA


def tiny_model() -> SGJEPA:
    return SGJEPA(
        feature_dim=6,
        hidden_dim=8,
        rwpe_dim=2,
        time_dim=4,
        gnn_layers=1,
        window_size=3,
        predictor_hidden_dim=16,
    )


def test_forward_loss_and_gradients() -> None:
    graph = make_synthetic(12, 6, 6, 3, 0.15, seed=1)
    model = tiny_model()
    loss, metrics = model.loss(graph)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert 0 <= metrics["spike_rate"] <= 1
    loss.backward()
    assert model.plif.beta.grad is not None
    assert model.prefix_projection.grad is not None


def test_precision_control() -> None:
    graph = make_synthetic(10, 3, 6, 2, 0.2, seed=2)
    model = tiny_model()
    low, nodes = model.infer(graph, precision=1, representation="prediction")
    high, nodes2 = model.infer(graph, precision=2, representation="prediction")
    assert low.shape == high.shape == (10, 8)
    assert torch.equal(nodes, nodes2)
    assert not torch.equal(low, high)
    encoder, encoder_nodes = model.infer(graph, representation="encoder")
    assert encoder.shape == (10, 8)
    assert torch.equal(encoder_nodes, nodes)
    views, view_nodes = model.infer_node_views(graph)
    assert set(views) == {"encoder", "prediction"}
    assert torch.equal(view_nodes, nodes)


def test_incomplete_tail_is_dropped() -> None:
    graph = make_synthetic(8, 7, 6, 2, 0.2, seed=3)
    assert len(list(graph.windows(3))) == 2


def test_rwpe_is_deterministic_and_cached() -> None:
    graph = make_synthetic(10, 3, 6, 2, 0.2, seed=4)
    snapshot = graph.snapshots[0]
    a = random_walk_positional_encoding(snapshot.edge_index, 10, 3, walks=8, seed=7)
    b = random_walk_positional_encoding(snapshot.edge_index, 10, 3, walks=8, seed=7)
    assert torch.equal(a, b)
    model = tiny_model()
    model.encode_snapshot(snapshot)
    assert len(model._rwpe_cache) == 1
    model.encode_snapshot(snapshot)
    assert len(model._rwpe_cache) == 1


def test_stratified_split_and_mlp_probe() -> None:
    labels = torch.arange(3).repeat_interleave(10)
    split = stratified_split(labels, train_ratio=0.4, validation_ratio_within_train=0.25)
    assert split.train.numel() == 9
    assert split.validation.numel() == 3
    assert split.test.numel() == 18
    assert set(split.train.tolist()).isdisjoint(split.test.tolist())
    probe = MLPProbe(8, 3, hidden_dim=4)
    assert probe(torch.randn(5, 8)).shape == (5, 3)


def test_multiscale_probe_ensemble() -> None:
    labels = torch.arange(3).repeat_interleave(10)
    split = stratified_split(labels, train_ratio=0.4, validation_ratio_within_train=0.25)
    embeddings = {
        "local": torch.randn(30, 8),
        "global": torch.randn(30, 5),
    }
    result = fit_probe_ensemble(
        embeddings, labels, split.train, split.validation, epochs=2, seed=7, hidden_dim=4
    )
    assert set(result) == {"macro_f1", "micro_f1"}
    assert 0 <= result["macro_f1"] <= 1
