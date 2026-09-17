"""Module-level ablation switches of DyGJEPA and their config overlays."""
from pathlib import Path

import pytest
import torch
import yaml

from jepa_compare.compare_link_prediction import (
    _dataset_configs,
    _rcps_ablation,
    load_config,
)
from jepa_compare.data import Snapshot
from jepa_compare.link_prediction import sliding_windows
from jepa_compare.rcps_jepa import RCPSJEPA

ABLATION_DIR = Path("configs/ablation")
BASE_CONFIG = Path("configs/link_comparison_all.yaml")


def _model(**flags: object) -> RCPSJEPA:
    torch.manual_seed(0)
    defaults = dict(use_causal_history=True, history_semantic_dim=2, id_embedding_dim=4)
    defaults.update(flags)
    return RCPSJEPA(
        feature_dim=6,
        num_nodes=16,
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
        allow_negative_collisions=True,
        negative_destination_candidates=torch.arange(NUM_NODES),
        **defaults,
    )


NUM_NODES = 16


def _event_snapshots() -> list[Snapshot]:
    """Six snapshots of a homogeneous event stream with repeated pairs."""
    generator = torch.Generator().manual_seed(5)
    snapshots = []
    for time in range(6):
        events = 12
        source = torch.randint(NUM_NODES, (events,), generator=generator)
        # Small destination alphabet so pairs repeat across and within bins.
        destination = torch.randint(6, (events,), generator=generator)
        queries = torch.stack([source, destination])
        undirected = torch.cat([queries, queries.flip(0)], dim=1)
        edge_index = torch.unique(undirected, dim=1)
        snapshots.append(
            Snapshot(
                x=torch.randn(NUM_NODES, 6, generator=generator),
                edge_index=edge_index,
                active=torch.ones(NUM_NODES, dtype=torch.bool),
                time=time,
                query_edge_index=queries,
                query_timestamps=torch.sort(
                    torch.rand(events, generator=generator) + time * 10
                ).values,
                query_features=torch.randn(events, 3, generator=generator),
            )
        )
    return snapshots


def _window():
    snapshots = _event_snapshots()
    return snapshots, sliding_windows(snapshots, 3)[2]


def _forward(model: RCPSJEPA, snapshots, window):
    model.prepare_causal_history(snapshots)
    model.eval()
    queries = model.sample_queries(window, seed=13)
    return model.forward_pairs(window, queries.pairs, timestamps=queries.timestamps)


def test_defaults_keep_every_module_and_are_recorded() -> None:
    model = _model()
    flags = model.ablation_flags()
    assert flags["use_node_trajectories"] is True
    assert flags["use_path_signature"] is True
    assert flags["use_relation_subgraph"] is True
    assert flags["use_causal_history"] is True
    assert flags["id_embedding_dim"] == 4
    assert flags["rank_loss_weight"] == model.rank_loss_weight


def test_full_model_is_deterministic_across_constructions() -> None:
    snapshots, window = _window()
    first = _forward(_model(), snapshots, window)
    second = _forward(_model(), snapshots, window)
    assert torch.equal(first.logit, second.logit)
    assert torch.equal(first.relation_prediction, second.relation_prediction)


@pytest.mark.parametrize(
    "flag, parameter",
    [
        ("use_node_trajectories", "node_gru.weight_ih_l0"),
        ("use_node_trajectories", "node_relation_gru.weight_ih_l0"),
        ("use_path_signature", "event_projector.0.weight"),
        ("use_path_signature", "node_event_projector.0.weight"),
        ("use_relation_subgraph", "graph_gru.weight_ih_l0"),
        ("use_causal_history", "history_prior_weights"),
    ],
)
def test_disabling_a_module_changes_the_output_and_detaches_its_parameters(
    flag: str, parameter: str
) -> None:
    snapshots, window = _window()
    full = _forward(_model(), snapshots, window)
    ablated_model = _model(**{flag: False})
    ablated = _forward(ablated_model, snapshots, window)
    assert full.logit.shape == ablated.logit.shape
    # The link head's last layer is zero-initialised (the untrained model is
    # the parameter-free recurrence prior), so compare the pair context the
    # head consumes rather than the logit itself.
    assert not torch.allclose(full.context, ablated.context)
    assert ablated_model.ablation_flags()[flag] is False

    # The removed module receives no gradient; everything else still trains.
    ablated_model.train()
    loss, _ = ablated_model.loss_windows([window], pair_batch_size=16, query_seed=11)
    loss.backward()
    parameters = dict(ablated_model.named_parameters())
    grad = parameters[parameter].grad
    # Unused modules get no gradient; the history prior still multiplies an
    # all-zero feature vector, so its gradient is exactly zero.
    assert grad is None or not bool(grad.any())
    assert parameters["context_encoder.0.weight"].grad is not None
    assert bool(parameters["intensity_head.2.weight"].grad.any())


def test_ablated_modules_contribute_zeros_of_the_same_width() -> None:
    snapshots, window = _window()
    model = _model(use_node_trajectories=False, use_path_signature=False)
    model.prepare_causal_history(snapshots)
    prepared = model.prepare_window(window)
    individual, relation = model._node_trajectories(
        prepared.context_snapshots, prepared.context_embeddings
    )
    assert individual.shape == (NUM_NODES, model.hidden_dim)
    assert torch.count_nonzero(individual) == 0 and torch.count_nonzero(relation) == 0
    signature = model._node_signature_context(
        prepared.context_snapshots, prepared.context_embeddings
    )
    assert signature.shape == (NUM_NODES, model.hidden_dim)
    assert torch.count_nonzero(signature) == 0
    # The temporal readout shares the fusion path and stays usable.
    temporal = model.encode_temporal_state(window)
    assert temporal.shape == (NUM_NODES, model.hidden_dim)
    assert torch.isfinite(temporal).all()


def test_without_jepa_objectives_only_the_link_head_trains() -> None:
    snapshots, window = _window()
    model = _model(
        node_loss_weight=0.0,
        relation_loss_weight=0.0,
        variance_loss_weight=0.0,
        covariance_loss_weight=0.0,
    )
    model.prepare_causal_history(snapshots)
    loss, values = model.loss_windows([window], pair_batch_size=16, query_seed=11)
    loss.backward()
    parameters = dict(model.named_parameters())
    # Zero-weighted objectives still sit in the graph, so their branches get
    # an exactly-zero gradient rather than none at all.
    for name in ("relation_predictor.0.weight", "node_predictor.0.weight"):
        grad = parameters[name].grad
        assert grad is None or not bool(grad.any()), name
    # The link head's last layer is zero-initialised, so only that layer (and
    # the recurrence prior) receives a non-zero gradient at initialisation.
    assert bool(parameters["intensity_head.2.weight"].grad.any())
    assert bool(parameters["history_prior_weights"].grad.any())
    assert values["link_loss"] > 0


def test_ablation_section_is_validated() -> None:
    assert _rcps_ablation({}) == {}
    assert _rcps_ablation({"ablation": {"name": "x", "rcps_jepa": {"use_path_signature": False}}}) == {
        "name": "x",
        "rcps_jepa": {"use_path_signature": False},
    }
    with pytest.raises(ValueError, match="unknown ablation fields"):
        _rcps_ablation({"ablation": {"rcps": {}}})
    with pytest.raises(ValueError, match="must be a mapping"):
        _rcps_ablation({"ablation": {"rcps_jepa": [1]}})
    with pytest.raises(ValueError, match="must be a mapping"):
        _rcps_ablation({"ablation": "no_history"})


def _effective_rcps_settings(dataset_config: dict) -> tuple[dict, dict]:
    """Model kwargs and training recipe exactly as run_comparison merges them."""
    ablation = _rcps_ablation(dataset_config)
    model = {
        **dict(dataset_config.get("rcps_jepa", {})),
        **dict(ablation.get("rcps_jepa", {})),
    }
    training = {
        **dict(dataset_config.get("training", {})),
        **dict(dataset_config.get("rcps_training", {})),
        **dict(ablation.get("rcps_training", {})),
    }
    return model, training


def test_ablation_overlays_inherit_the_main_run_and_beat_dataset_overrides() -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text())
    overlays = sorted(ABLATION_DIR.glob("*.yaml"))
    assert overlays, "no ablation overlays found"
    for overlay_path in overlays:
        config = load_config(overlay_path)
        name = overlay_path.stem
        assert config["ablation"]["name"] == name
        assert config["models"] == ["rcps_jepa"]
        assert config["seed"] == [42, 44, 46, 48, 50]
        assert config["output_dir"] == f"results/ablation/{name}"
        assert config["datasets"] == base["datasets"]
        for section in ("rcps_jepa", "rcps_training", "link", "common_model", "training"):
            assert config[section] == base[section]
        expanded = dict(_dataset_configs(config))
        assert set(expanded) == {entry["name"] for entry in base["datasets"]}
        assert expanded["canparl"]["output_path"] == (
            f"results/ablation/{name}/link_comparison_canparl.json"
        )
        for dataset_config in expanded.values():
            model, training = _effective_rcps_settings(dataset_config)
            for key, value in config["ablation"].get("rcps_jepa", {}).items():
                assert model[key] == value, (name, key)
            for key, value in config["ablation"].get("rcps_training", {}).items():
                assert training[key] == value, (name, key)
            # Every model flag is a real constructor argument.
            RCPSJEPA(
                feature_dim=4,
                num_nodes=8,
                **{
                    key: value
                    for key, value in model.items()
                    if key in RCPSJEPA.__init__.__code__.co_varnames
                },
            )

    # Wikipedia tunes id_embedding_dim=128 and epochs=60 in its overrides; the
    # ablation still removes the module / skips training there.
    no_id = dict(_dataset_configs(load_config(ABLATION_DIR / "no_id.yaml")))
    assert no_id["wikipedia"]["rcps_jepa"]["id_embedding_dim"] == 128
    assert _effective_rcps_settings(no_id["wikipedia"])[0]["id_embedding_dim"] == 0
    prior_only = dict(_dataset_configs(load_config(ABLATION_DIR / "prior_only.yaml")))
    assert prior_only["wikipedia"]["rcps_training"]["epochs"] == 60
    assert _effective_rcps_settings(prior_only["wikipedia"])[1]["epochs"] == 0
    assert _effective_rcps_settings(prior_only["wikipedia"])[1]["evaluate_before_training"] is True


def test_main_run_config_has_no_ablation() -> None:
    config = load_config(BASE_CONFIG)
    assert "ablation" not in config
    for _, dataset_config in _dataset_configs(config):
        assert _rcps_ablation(dataset_config) == {}
