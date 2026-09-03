from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

from .compare_link_prediction import _build_graph
from .node_evaluation import (
    final_probe,
    final_probe_ensemble,
    macro_micro_f1,
    stratified_split,
    validation_probe,
)
from .node_tgnn_baselines import (
    DyGLibStatelessNodeSSL,
    EvolveGCNHNodeClassifier,
    ROLANDNodeClassifier,
    TGATNodeSSL,
    TGNNodeSSL,
)
from .rcps_jepa import RCPSJEPA
from .sg_jepa import SGJEPA
from .snapshot_ssl_baselines import (
    CLDGLinkBaseline,
    DVGMAELinkBaseline,
    MaskDGNNLinkBaseline,
    SnapshotSSLLinkBaseline,
)
from .train_sg_jepa import choose_device, cpu_state_dict


def _require_global_node_order(node_ids: torch.Tensor, num_nodes: int) -> None:
    expected = torch.arange(num_nodes, device=node_ids.device)
    if not torch.equal(node_ids, expected):
        raise ValueError(
            "node comparison requires every node to be active in the final target snapshot"
        )


def _infer_views(
    model: nn.Module, graph
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if isinstance(model, SGJEPA):
        return model.infer_node_views(graph)
    return model.infer_node_views(graph.snapshots[-model.window_size :])


def _select_node_embeddings(
    model: nn.Module,
    graph,
    probe_split,
    probe_epochs: int,
    seed: int,
    probe_hidden_dim,
    candidates: tuple[str, ...],
) -> tuple[torch.Tensor, dict[str, float], str]:
    """Select a checkpoint view using validation labels only."""
    with torch.no_grad():
        views, node_ids = _infer_views(model, graph)
    _require_global_node_order(node_ids, graph.num_nodes)
    missing = [name for name in candidates if name not in views]
    if missing:
        raise KeyError(f"model is missing checkpoint views: {missing}")

    best_name = candidates[0]
    best_embeddings = views[best_name]
    best_validation = validation_probe(
        best_embeddings,
        graph.labels,
        probe_split,
        probe_epochs,
        seed,
        probe_hidden_dim,
    )
    for name in candidates[1:]:
        embeddings = views[name]
        validation = validation_probe(
            embeddings,
            graph.labels,
            probe_split,
            probe_epochs,
            seed,
            probe_hidden_dim,
        )
        if validation["macro_f1"] > best_validation["macro_f1"]:
            best_name = name
            best_embeddings = embeddings
            best_validation = validation

    selected = dict(best_validation)
    selected["selected_view"] = best_name
    return best_embeddings.detach(), selected, best_name


def _final_rcps_multiscale_probe(
    model: RCPSJEPA,
    graph,
    probe_split,
    training: dict,
    seed: int,
) -> dict[str, float]:
    names = tuple(training.get("rcps_multiscale_views", ()))
    if len(names) < 2:
        raise ValueError("training.rcps_multiscale_views needs at least two views")
    with torch.no_grad():
        views, node_ids = _infer_views(model, graph)
    _require_global_node_order(node_ids, graph.num_nodes)
    missing = [name for name in names if name not in views]
    if missing:
        raise KeyError(f"RCPS multi-scale probe is missing views: {missing}")

    result = final_probe_ensemble(
        {name: views[name] for name in names},
        graph.labels,
        probe_split,
        int(training.get("probe_epochs", 100)),
        seed,
        training.get("probe_hidden_dim"),
    )
    result["selected_view"] = f"logit_ensemble[{','.join(names)}]"
    result["protocol"] = "ssl_multiscale_probe"
    return result


def _train_node_model(
    name: str,
    model: nn.Module,
    graph,
    windows,
    probe_split,
    training: dict,
    seed: int,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.95),
        eps=1e-10,
    )
    is_rcps = isinstance(model, RCPSJEPA)
    checkpoint_views = (
        tuple(training.get("rcps_checkpoint_views", ("fused",)))
        if is_rcps
        else ("encoder", "prediction")
    )
    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    best_view = checkpoint_views[0]
    min_checkpoint_epoch = int(training.get("min_checkpoint_epoch", 1))
    eval_every = int(training.get("eval_every", 1))
    dense_eval_epochs = int(training.get("dense_eval_epochs", 0))
    probe_epochs = int(training.get("selection_probe_epochs", 50))
    batch_size = training.get("node_batch_size")
    probe_hidden_dim = training.get("probe_hidden_dim")

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, SGJEPA):
            loss, metrics = model.loss(graph, batch_size=batch_size)
        else:
            loss, metrics = model.node_loss_windows(
                windows, node_batch_size=batch_size
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced a non-finite loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
        optimizer.step()
        if hasattr(model, "update_target_encoder"):
            model.update_target_encoder()

        should_evaluate = (
            epoch == 1
            or epoch <= dense_eval_epochs
            or epoch % eval_every == 0
            or epoch == int(training["epochs"])
        )
        if not should_evaluate:
            continue

        model.eval()
        _, validation, view_name = _select_node_embeddings(
            model,
            graph,
            probe_split,
            probe_epochs,
            seed,
            probe_hidden_dim,
            checkpoint_views,
        )
        print(
            json.dumps(
                {
                    "model": name,
                    "epoch": epoch,
                    "train": metrics,
                    "validation_probe": validation,
                }
            )
        )
        if validation["macro_f1"] > best_score and epoch >= min_checkpoint_epoch:
            best_score = validation["macro_f1"]
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            best_view = view_name

    if best_state is None:
        raise RuntimeError(f"{name} did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    if is_rcps:
        result = _final_rcps_multiscale_probe(
            model, graph, probe_split, training, seed
        )
    else:
        with torch.no_grad():
            views, node_ids = _infer_views(model, graph)
        _require_global_node_order(node_ids, graph.num_nodes)
        result = final_probe(
            views[best_view],
            graph.labels,
            probe_split,
            int(training.get("probe_epochs", 100)),
            seed,
            probe_hidden_dim,
        )
        result["selected_view"] = best_view
        result["protocol"] = "ssl_probe"
    result["supervision"] = "self_supervised_latent"
    result["best_epoch"] = float(best_epoch)
    return result


def _classification_metrics(logits, labels, indices) -> dict[str, float]:
    prediction = logits[indices].argmax(dim=-1)
    macro, micro = macro_micro_f1(
        labels[indices], prediction, int(labels.max().item()) + 1
    )
    return {"macro_f1": macro, "micro_f1": micro}


def _save_partial_results(config: dict, result: dict[str, dict[str, float]]) -> None:
    output_path = config.get("output_path")
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))


def _train_supervised_tgnn(
    name: str,
    model: nn.Module,
    graph,
    probe_split,
    settings: dict,
) -> dict[str, float]:
    """Validation-select, then refit on the complete label budget.

    The validation run sees ``probe_split.train`` labels only.  After choosing
    the epoch, the model is reset to its original initialization and trained
    for exactly that many epochs on train+validation, matching the final probe
    budget used by the self-supervised models.
    """
    initial_state = cpu_state_dict(model)
    initial_cpu_rng = torch.random.get_rng_state()
    initial_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def optimizer_for(module: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            module.parameters(),
            lr=float(settings["learning_rate"]),
            weight_decay=float(settings.get("weight_decay", 0.0)),
        )

    optimizer = optimizer_for(model)
    epochs = int(settings["epochs"])
    eval_every = int(settings.get("eval_every", 1))
    grad_clip = float(settings.get("grad_clip", 1.0))
    best_score, best_epoch, best_validation = float("-inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph.snapshots)
        loss = torch.nn.functional.cross_entropy(
            logits[probe_split.train], graph.labels[probe_split.train]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced a non-finite loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                validation_logits = model(graph.snapshots)
                validation = _classification_metrics(
                    validation_logits, graph.labels, probe_split.validation
                )
            print(
                json.dumps(
                    {
                        "model": name,
                        "epoch": epoch,
                        "train": {"loss": float(loss.detach())},
                        "validation": validation,
                    }
                )
            )
            if validation["macro_f1"] > best_score:
                best_score = validation["macro_f1"]
                best_epoch = epoch
                best_validation = validation
    if best_epoch == 0 or best_validation is None:
        raise RuntimeError(f"{name} did not produce a validation checkpoint")

    model.load_state_dict(initial_state)
    torch.random.set_rng_state(initial_cpu_rng)
    if initial_cuda_rng is not None:
        torch.cuda.set_rng_state_all(initial_cuda_rng)
    optimizer = optimizer_for(model)
    full_train = torch.cat([probe_split.train, probe_split.validation])
    for _ in range(best_epoch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph.snapshots)
        loss = torch.nn.functional.cross_entropy(
            logits[full_train], graph.labels[full_train]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        test_logits = model(graph.snapshots)
        result = _classification_metrics(test_logits, graph.labels, probe_split.test)
    result.update(
        {
            "best_epoch": float(best_epoch),
            "protocol": "supervised_node_classification",
            "supervision": "node_labels",
            "architecture": (
                "evolvegcn_h_topk_matrix_gru"
                if name == "evolvegcn_h"
                else "roland_graphsage_hierarchical_gru"
            ),
            "validation_macro_f1": best_validation["macro_f1"],
            "validation_micro_f1": best_validation["micro_f1"],
        }
    )
    return result


def _train_temporal_ssl_node_baseline(
    name: str,
    model: TGNNodeSSL | TGATNodeSSL | DyGLibStatelessNodeSSL,
    graph,
    probe_split,
    settings: dict,
    probe_settings: dict,
    seed: int,
) -> dict[str, float]:
    model.prepare(graph)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 0.0)),
    )
    epochs = int(settings["epochs"])
    eval_every = int(settings.get("eval_every", 1))
    grad_clip = float(settings.get("grad_clip", 1.0))
    selection_probe_epochs = int(
        settings.get(
            "selection_probe_epochs",
            probe_settings.get("selection_probe_epochs", 50),
        )
    )
    probe_hidden_dim = probe_settings.get("probe_hidden_dim")
    best_score, best_epoch, best_state = float("-inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        metrics = model.train_epoch(optimizer, grad_clip, seed + epoch)
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            embeddings = (
                model.node_embeddings(seed)
                if isinstance(model, (TGATNodeSSL, DyGLibStatelessNodeSSL))
                else model.node_embeddings()
            )
            validation = validation_probe(
                embeddings,
                graph.labels,
                probe_split,
                selection_probe_epochs,
                seed,
                probe_hidden_dim,
            )
            print(
                json.dumps(
                    {
                        "model": name,
                        "epoch": epoch,
                        "train": metrics,
                        "validation_probe": validation,
                    }
                )
            )
            if validation["macro_f1"] > best_score:
                best_score = validation["macro_f1"]
                best_epoch = epoch
                best_state = cpu_state_dict(model)
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    embeddings = (
        model.node_embeddings(seed)
        if isinstance(model, (TGATNodeSSL, DyGLibStatelessNodeSSL))
        else model.node_embeddings()
    )
    result = final_probe(
        embeddings,
        graph.labels,
        probe_split,
        int(probe_settings.get("probe_epochs", 100)),
        seed,
        probe_hidden_dim,
    )
    result.update(
        {
            "best_epoch": float(best_epoch),
            "selected_view": "temporal_embedding",
            "protocol": "ssl_link_pretrain_then_node_probe",
            "supervision": "self_supervised_edges",
            "snapshot_adapter": "undirected_new_edges_with_tied_snapshot_time",
        }
    )
    if isinstance(model, DyGLibStatelessNodeSSL):
        result["implementation"] = "vendored_dyglib_official_backbone"
        result["node_readout"] = "self_conditioned_final_time"
    return result


def _train_snapshot_ssl_node_baseline(
    name: str,
    model: SnapshotSSLLinkBaseline,
    graph,
    probe_split,
    settings: dict,
    probe_settings: dict,
    seed: int,
) -> dict[str, float]:
    """Self-supervise a snapshot encoder, then use the common frozen probe."""
    parameters = model.pretrain_parameters()
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 0.0)),
    )
    epochs = int(settings["epochs"])
    eval_every = int(settings.get("eval_every", 1))
    grad_clip = float(settings.get("grad_clip", 1.0))
    selection_probe_epochs = int(
        settings.get(
            "selection_probe_epochs",
            probe_settings.get("selection_probe_epochs", 50),
        )
    )
    probe_hidden_dim = probe_settings.get("probe_hidden_dim")
    best_score, best_epoch, best_state = float("-inf"), 0, None
    for epoch in range(1, epochs + 1):
        metrics = model.pretrain_epoch(
            graph.snapshots, optimizer, grad_clip, seed + epoch
        )
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                embeddings = model.encode_context(graph.snapshots)
            validation = validation_probe(
                embeddings,
                graph.labels,
                probe_split,
                selection_probe_epochs,
                seed,
                probe_hidden_dim,
            )
            print(
                json.dumps(
                    {
                        "model": name,
                        "epoch": epoch,
                        "train": metrics,
                        "validation_probe": validation,
                    }
                )
            )
            if validation["macro_f1"] > best_score:
                best_score = validation["macro_f1"]
                best_epoch = epoch
                best_state = cpu_state_dict(model)
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embeddings = model.encode_context(graph.snapshots)
    result = final_probe(
        embeddings,
        graph.labels,
        probe_split,
        int(probe_settings.get("probe_epochs", 100)),
        seed,
        probe_hidden_dim,
    )
    result.update(
        {
            "best_epoch": float(best_epoch),
            "selected_view": "final_snapshot_embedding",
            "protocol": "snapshot_ssl_then_node_probe",
            "supervision": "self_supervised_snapshots",
            "implementation": model.implementation,
        }
    )
    return result


def run(config: dict) -> dict[str, dict[str, float]]:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(config.get("device", "auto"))
    graph = _build_graph(config, seed).to(device)
    if graph.labels is None:
        raise ValueError("node prediction comparison requires node labels")

    common = dict(config["common_model"])
    windows = list(graph.windows(int(common["window_size"])))
    probe_cfg = config.get("probe", {})
    probe_split = stratified_split(
        graph.labels,
        float(probe_cfg.get("train_ratio", 0.6)),
        float(probe_cfg.get("validation_ratio_within_train", 0.1)),
        seed,
    )

    torch.manual_seed(seed)
    sg_model = SGJEPA(
        feature_dim=graph.feature_dim,
        **{**common, **dict(config.get("sg_jepa", {}))},
    ).to(device)
    sg_result = _train_node_model(
        "sg_jepa", sg_model, graph, windows, probe_split, config["training"], seed
    )

    torch.manual_seed(seed)
    rcps_model = RCPSJEPA(
        feature_dim=graph.feature_dim,
        **{**common, **dict(config.get("rcps_jepa", {}))},
    ).to(device)
    rcps_result = _train_node_model(
        "rcps_jepa", rcps_model, graph, windows, probe_split, config["training"], seed
    )

    result = {"sg_jepa": sg_result, "rcps_jepa": rcps_result}
    _save_partial_results(config, result)
    # The baselines are intentionally run one at a time.  Releasing the JEPA
    # modules here avoids retaining several full temporal computation graphs on
    # the same GPU (important for the 28k-node DBLP run).
    del sg_model, rcps_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    baseline_cfg = dict(config.get("node_baselines", {}))
    enabled = tuple(name.lower() for name in baseline_cfg.get("enabled", ()))
    classes = int(graph.labels.max().item()) + 1
    for name in enabled:
        torch.manual_seed(seed)
        settings = dict(baseline_cfg.get(name, {}))
        if name == "evolvegcn_h":
            model = EvolveGCNHNodeClassifier(
                feature_dim=graph.feature_dim,
                hidden_dim=int(settings.get("hidden_dim", common["hidden_dim"])),
                classes=classes,
                layers=int(settings.get("layers", 2)),
            ).to(device)
            result[name] = _train_supervised_tgnn(
                name, model, graph, probe_split, settings
            )
        elif name == "roland":
            model = ROLANDNodeClassifier(
                feature_dim=graph.feature_dim,
                hidden_dim=int(settings.get("hidden_dim", common["hidden_dim"])),
                classes=classes,
                layers=int(settings.get("layers", 2)),
                dropout=float(settings.get("dropout", 0.0)),
                bptt_steps=int(settings.get("bptt_steps", 4)),
            ).to(device)
            result[name] = _train_supervised_tgnn(
                name, model, graph, probe_split, settings
            )
        elif name == "tgn":
            model = TGNNodeSSL(
                feature_dim=graph.feature_dim,
                num_nodes=graph.num_nodes,
                time_dim=int(settings.get("time_dim", 100)),
                num_layers=int(settings.get("layers", 1)),
                num_heads=int(settings.get("heads", 2)),
                num_neighbors=int(settings.get("num_neighbors", 10)),
                dropout=float(settings.get("dropout", 0.1)),
                batch_size=int(settings.get("batch_size", 100)),
                inference_batch_size=int(settings.get("inference_batch_size", 512)),
                seed=seed,
            ).to(device)
            result[name] = _train_temporal_ssl_node_baseline(
                name, model, graph, probe_split, settings, config["training"], seed
            )
        elif name == "tgat":
            model = TGATNodeSSL(
                feature_dim=graph.feature_dim,
                num_nodes=graph.num_nodes,
                hidden_dim=int(settings.get("hidden_dim", 100)),
                num_layers=int(settings.get("layers", 2)),
                num_heads=int(settings.get("heads", 2)),
                num_neighbors=int(settings.get("num_neighbors", 20)),
                dropout=float(settings.get("dropout", 0.1)),
                uniform_neighbors=bool(settings.get("uniform_neighbors", False)),
                batch_size=int(settings.get("batch_size", 200)),
                inference_batch_size=int(settings.get("inference_batch_size", 256)),
            ).to(device)
            result[name] = _train_temporal_ssl_node_baseline(
                name, model, graph, probe_split, settings, config["training"], seed
            )
        elif name in {"cawn", "tcl", "graphmixer", "dygformer"}:
            model = DyGLibStatelessNodeSSL(
                model_name=name,
                feature_dim=graph.feature_dim,
                num_nodes=graph.num_nodes,
                time_dim=int(settings.get("time_dim", 100)),
                num_layers=int(settings.get("layers", 2)),
                num_heads=int(settings.get("heads", 2)),
                num_neighbors=int(settings.get("num_neighbors", 20)),
                dropout=float(settings.get("dropout", 0.1)),
                channel_embedding_dim=int(
                    settings.get("channel_embedding_dim", 50)
                ),
                position_feat_dim=int(
                    settings.get("position_feat_dim", graph.feature_dim)
                ),
                walk_length=int(settings.get("walk_length", 1)),
                num_walk_heads=int(settings.get("num_walk_heads", 8)),
                patch_size=int(settings.get("patch_size", 1)),
                max_input_sequence_length=int(
                    settings.get("max_input_sequence_length", 32)
                ),
                time_gap=int(settings.get("time_gap", 2000)),
                batch_size=int(settings.get("batch_size", 200)),
                inference_batch_size=int(settings.get("inference_batch_size", 256)),
                sample_neighbor_strategy=str(
                    settings.get("sample_neighbor_strategy", "recent")
                ),
                time_scaling_factor=float(
                    settings.get("time_scaling_factor", 0.0)
                ),
                seed=seed,
            ).to(device)
            result[name] = _train_temporal_ssl_node_baseline(
                name, model, graph, probe_split, settings, config["training"], seed
            )
        elif name in {"cldg", "maskdgnn", "dvgmae"}:
            link_kwargs = dict(
                negative_ratio=1.0,
                max_positive_pairs=None,
                new_edges_only=False,
                undirected=True,
                bipartite_source_count=None,
            )
            if name == "cldg":
                model = CLDGLinkBaseline(
                    feature_dim=graph.feature_dim,
                    hidden_dim=int(settings.get("hidden_dim", 128)),
                    embedding_dim=int(settings.get("embedding_dim", 128)),
                    num_layers=int(settings.get("layers", 2)),
                    dropout=float(settings.get("dropout", 0.0)),
                    num_spans=int(settings.get("num_spans", 4)),
                    num_views=int(settings.get("num_views", 4)),
                    view_strategy=str(settings.get("view_strategy", "sequential")),
                    temperature=float(settings.get("temperature", 0.07)),
                    contrastive_batch_size=int(
                        settings.get("contrastive_batch_size", 1024)
                    ),
                    probe_hidden_dim=int(settings.get("probe_hidden_dim", 128)),
                    **link_kwargs,
                ).to(device)
            elif name == "maskdgnn":
                model = MaskDGNNLinkBaseline(
                    feature_dim=graph.feature_dim,
                    hidden_dim=int(settings.get("hidden_dim", 64)),
                    num_layers=int(settings.get("layers", 2)),
                    dropout=float(settings.get("dropout", 0.1)),
                    window_size=int(settings.get("window_size", 4)),
                    mask_ratio=float(settings.get("mask_ratio", 0.3)),
                    dynamics_ratio=float(settings.get("dynamics_ratio", 0.7)),
                    dynamics_weight=float(settings.get("dynamics_weight", 1.0)),
                    existing_offset=float(settings.get("existing_offset", 2.0)),
                    new_node_offset=float(settings.get("new_node_offset", -0.5)),
                    pagerank_damping=float(settings.get("pagerank_damping", 0.85)),
                    pagerank_steps=int(settings.get("pagerank_steps", 10)),
                    pretrain_pair_limit=int(settings.get("pretrain_pair_limit", 4096)),
                    probe_hidden_dim=int(settings.get("probe_hidden_dim", 128)),
                    **link_kwargs,
                ).to(device)
            else:
                model = DVGMAELinkBaseline(
                    feature_dim=graph.feature_dim,
                    hidden_dim=int(settings.get("hidden_dim", 64)),
                    num_layers=int(settings.get("layers", 2)),
                    dropout=float(settings.get("dropout", 0.1)),
                    window_size=int(settings.get("window_size", 4)),
                    mask_ratio=float(settings.get("mask_ratio", 0.3)),
                    history_balance=float(settings.get("history_balance", 0.5)),
                    kl_weight=float(settings.get("kl_weight", 0.001)),
                    feature_weight=float(settings.get("feature_weight", 0.1)),
                    pretrain_pair_limit=int(settings.get("pretrain_pair_limit", 4096)),
                    probe_hidden_dim=int(settings.get("probe_hidden_dim", 128)),
                    **link_kwargs,
                ).to(device)
            result[name] = _train_snapshot_ssl_node_baseline(
                name, model, graph, probe_split, settings, config["training"], seed
            )
        else:
            raise ValueError(f"unknown node baseline: {name}")
        _save_partial_results(config, result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({"node_comparison": result}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SG-JEPA/RCPS-JEPA with dynamic-GNN node baselines"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/node_comparison_synthetic.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--baselines",
        nargs="*",
        choices=[
            "evolvegcn_h",
            "roland",
            "tgn",
            "tgat",
            "cawn",
            "tcl",
            "graphmixer",
            "dygformer",
            "cldg",
            "maskdgnn",
            "dvgmae",
        ],
        default=None,
        help="override node_baselines.enabled (pass no values to disable all baselines)",
    )
    args = parser.parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        config["training"]["min_checkpoint_epoch"] = min(
            int(config["training"].get("min_checkpoint_epoch", 1)), args.epochs
        )
        for settings in config.get("node_baselines", {}).values():
            if isinstance(settings, dict) and "epochs" in settings:
                settings["epochs"] = args.epochs
    if args.baselines is not None:
        config.setdefault("node_baselines", {})["enabled"] = args.baselines
    run(config)


if __name__ == "__main__":
    main()
