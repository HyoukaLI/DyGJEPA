from __future__ import annotations

import argparse
from copy import deepcopy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

from .data import load_npz, make_synthetic
from .dyglib_baselines import DyGLibLinkBaseline, EdgeBankLinkBaseline
from .jodie_baseline import JODIELinkBaseline
from .link_prediction import TemporalWindowSplit, temporal_window_split
from .rcps_jepa import RCPSJEPA
from .snapshot_ssl_baselines import (
    CLDGLinkBaseline,
    DVGMAELinkBaseline,
    MaskDGNNLinkBaseline,
    SnapshotSSLLinkBaseline,
)
from .temporal_event_utils import unique_snapshots
from .tgat_baseline import TGATLinkBaseline
from .train_sg_jepa import choose_device, cpu_state_dict, release_device_memory


def _build_graph(config: dict, seed: int):
    data_cfg = config["data"]
    if data_cfg.get("path"):
        graph = load_npz(data_cfg["path"])
    else:
        graph = make_synthetic(
            **{k: v for k, v in data_cfg.items() if k not in {"path", "fanout"}},
            seed=seed,
        )
    if data_cfg.get("fanout"):
        graph = graph.sample_neighbors(int(data_cfg["fanout"]), seed=seed)
    return graph


def _train_one(
    name: str,
    model: nn.Module,
    split: TemporalWindowSplit,
    training: dict,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    native_jodie = isinstance(model, JODIELinkBaseline)
    native_event_model = isinstance(
        model,
        (JODIELinkBaseline, TGATLinkBaseline, DyGLibLinkBaseline),
    )
    betas = tuple(float(value) for value in training.get("betas", (0.9, 0.999)))
    optimizer = (
        torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            betas=betas,
        )
        if native_event_model
        else torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            betas=(0.9, 0.95),
            eps=1e-10,
        )
    )
    epochs = int(training["epochs"])
    pair_batch_size = training.get("pair_batch_size")
    eval_every = int(training.get("eval_every", 1))
    best_ap = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = int(training.get("patience", 0))
    evaluations_without_improvement = 0

    def evaluate(
        windows: list[list], history_windows: list[list], query_seed: int
    ) -> dict[str, float]:
        if native_event_model:
            return model.evaluate_protocol(  # type: ignore[attr-defined]
                windows, history_windows, query_seed=query_seed
            )
        return model.evaluate_windows(
            windows,
            pair_batch_size=pair_batch_size,
            query_seed=query_seed,
        )

    if bool(training.get("evaluate_before_training", False)):
        model.eval()
        validation = evaluate(split.validation, split.train, seed + 1_000_000)
        print(json.dumps({"model": name, "epoch": 0, "validation": validation}))
        best_ap = validation["ap"]
        best_epoch = 0
        best_state = cpu_state_dict(model)

    for epoch in range(1, epochs + 1):
        model.train()
        if native_event_model:
            if native_jodie:
                metrics = model.train_epoch(  # type: ignore[attr-defined]
                    split.train,
                    optimizer,
                    float(training["grad_clip"]),
                )
            elif isinstance(model, DyGLibLinkBaseline):
                metrics = model.train_epoch(  # type: ignore[attr-defined]
                    split.train,
                    optimizer,
                    float(training["grad_clip"]),
                    seed=seed,
                )
            else:
                metrics = model.train_epoch(  # type: ignore[attr-defined]
                    split.train,
                    optimizer,
                    float(training["grad_clip"]),
                    seed=seed + epoch * 10_000,
                )
            loss_value = float(metrics["loss"])
        elif isinstance(model, RCPSJEPA):
            metrics = model.train_epoch(
                split.train,
                optimizer,
                float(training["grad_clip"]),
                seed=seed + epoch * 10_000,
                pair_batch_size=pair_batch_size,
            )
            loss_value = float(metrics["loss"])
        else:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.loss_windows(
                split.train,
                pair_batch_size=pair_batch_size,
                query_seed=seed + epoch * 10_000,
                backward=True,
            )
            loss_value = float(loss.detach().item())
            if loss.requires_grad:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["grad_clip"])
            )
            optimizer.step()
        if not np.isfinite(loss_value):
            raise RuntimeError(f"{name} produced a non-finite loss at epoch {epoch}")
        if hasattr(model, "update_target_encoder") and not (
            isinstance(model, RCPSJEPA)
            and model.ema_steps_per_epoch is not None
        ):
            model.update_target_encoder()

        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            validation = evaluate(
                split.validation, split.train, seed + 1_000_000
            )
            print(json.dumps({"model": name, "epoch": epoch, "train": metrics, "validation": validation}))
            if validation["ap"] > best_ap:
                best_ap = validation["ap"]
                best_epoch = epoch
                best_state = cpu_state_dict(model)
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
                if patience > 0 and evaluations_without_improvement >= patience:
                    break

    if best_state is None:
        raise RuntimeError(f"{name} did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation = evaluate(
        split.validation, split.train, seed + 1_000_000
    )
    test = evaluate(
        split.test,
        [*split.train, *split.validation],
        seed + 2_000_000,
    )
    test["best_epoch"] = float(best_epoch)
    return validation, test


def _train_snapshot_ssl_one(
    name: str,
    model: SnapshotSSLLinkBaseline,
    split: TemporalWindowSplit,
    training: dict,
    seed: int,
) -> tuple[dict[str, float], dict[str, float | str]]:
    """Run native SSL pretraining followed by a shared frozen link probe."""
    pretrain_optimizer = torch.optim.Adam(
        model.pretrain_parameters(),
        lr=float(training["pretrain_learning_rate"]),
        weight_decay=float(training.get("pretrain_weight_decay", 0.0)),
    )
    train_snapshots = unique_snapshots(split.train)
    pretrain_epochs = int(training["pretrain_epochs"])
    grad_clip = float(training.get("grad_clip", 1.0))
    for epoch in range(1, pretrain_epochs + 1):
        metrics = model.pretrain_epoch(
            train_snapshots,
            pretrain_optimizer,
            grad_clip,
            seed + epoch * 10_000,
        )
        if not np.isfinite(float(metrics["loss"])):
            raise RuntimeError(f"{name} produced a non-finite SSL loss at epoch {epoch}")
        if epoch == 1 or epoch % int(training.get("pretrain_log_every", 10)) == 0 or epoch == pretrain_epochs:
            print(json.dumps({"model": name, "stage": "ssl_pretrain", "epoch": epoch, "train": metrics}))

    model.freeze_encoder()
    model.eval()
    model.probe.train()
    probe_optimizer = torch.optim.Adam(
        model.probe.parameters(),
        lr=float(training["probe_learning_rate"]),
        weight_decay=float(training.get("probe_weight_decay", 0.0)),
    )
    probe_epochs = int(training["probe_epochs"])
    eval_every = int(training.get("eval_every", 1))
    pair_batch_size = int(training.get("pair_batch_size", 512))
    patience = int(training.get("patience", 0))
    stale_evaluations = 0
    best_ap = float("-inf")
    best_epoch = 0
    best_probe_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, probe_epochs + 1):
        metrics = model.train_probe_epoch(
            split.train,
            probe_optimizer,
            grad_clip,
            pair_batch_size,
            seed + epoch * 10_000,
        )
        if epoch == 1 or epoch % eval_every == 0 or epoch == probe_epochs:
            validation = model.evaluate_windows(
                split.validation,
                pair_batch_size=pair_batch_size,
                query_seed=seed + 1_000_000,
            )
            print(json.dumps({"model": name, "stage": "frozen_link_probe", "epoch": epoch, "train": metrics, "validation": validation}))
            if validation["ap"] > best_ap:
                best_ap = validation["ap"]
                best_epoch = epoch
                best_probe_state = cpu_state_dict(model.probe)
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if patience > 0 and stale_evaluations >= patience:
                break
    if best_probe_state is None:
        raise RuntimeError(f"{name} did not produce a frozen-probe checkpoint")
    model.probe.load_state_dict(best_probe_state)
    model.eval()
    validation = model.evaluate_windows(
        split.validation,
        pair_batch_size=pair_batch_size,
        query_seed=seed + 1_000_000,
    )
    test: dict[str, float | str] = model.evaluate_windows(
        split.test,
        pair_batch_size=pair_batch_size,
        query_seed=seed + 2_000_000,
    )
    test.update(
        {
            "best_epoch": float(best_epoch),
            "pretrain_epochs": float(pretrain_epochs),
            "protocol": "ssl_pretrain_then_frozen_link_probe",
            "implementation": model.implementation,
        }
    )
    return validation, test


_ALL_COMPARISON_MODELS = (
    "jodie",
    "dyrep",
    "tgat",
    "edgebank",
    "tgn",
    "cawn",
    "tcl",
    "graphmixer",
    "dygformer",
    "cldg",
    "maskdgnn",
    "dvgmae",
    "rcps_jepa",
)


def _requested_models(config: dict) -> set[str] | None:
    requested = config.get("models")
    if requested is None:
        return None
    names = {str(name).lower() for name in requested}
    unknown = names - set(_ALL_COMPARISON_MODELS)
    if unknown:
        raise ValueError(f"unknown comparison models: {sorted(unknown)}")
    return names


def run(config: dict) -> dict[str, dict[str, dict[str, float]]]:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(config.get("device", "auto"))
    graph = _build_graph(config, seed).to(device)
    requested = _requested_models(config)

    def should_run(name: str) -> bool:
        return requested is None or name in requested

    common = dict(config["common_model"])
    split_cfg = config.get("split", {})
    split = temporal_window_split(
        graph.snapshots,
        int(common["window_size"]),
        float(split_cfg.get("train_ratio", 0.6)),
        float(split_cfg.get("validation_ratio", 0.2)),
    )
    link_cfg = dict(config.get("link", {}))
    if graph.num_source_nodes is not None:
        configured = link_cfg.get("bipartite_source_count")
        if configured is not None and int(configured) != graph.num_source_nodes:
            raise ValueError("configured bipartite split disagrees with the dataset")
        link_cfg["bipartite_source_count"] = graph.num_source_nodes
    rcps_args = {
        "num_nodes": graph.num_nodes,
        **common,
        **link_cfg,
        **dict(config.get("rcps_jepa", {})),
    }
    if should_run("jodie") and graph.num_source_nodes is None:
        raise ValueError("JODIE comparison requires a bipartite user-item graph")
    jodie_args = {
        "num_nodes": graph.num_nodes,
        "bipartite_source_count": graph.num_source_nodes,
        "hidden_dim": int(common["hidden_dim"]),
        **dict(config.get("jodie", {})),
        **link_cfg,
    }
    tgat_args = {
        "num_nodes": graph.num_nodes,
        "bipartite_source_count": graph.num_source_nodes,
        **dict(config.get("tgat", {})),
        **link_cfg,
    }
    shared_training = dict(config.get("training", {}))
    jodie_training = {**shared_training, **dict(config.get("jodie_training", {}))}
    tgat_training = {**shared_training, **dict(config.get("tgat_training", {}))}
    rcps_training = {**shared_training, **dict(config.get("rcps_training", {}))}
    result: dict[str, dict[str, dict[str, float]]] = {}
    train_snapshots = unique_snapshots(split.train)

    if should_run("jodie"):
        torch.manual_seed(seed)
        jodie_model = JODIELinkBaseline(
            feature_dim=graph.feature_dim, **jodie_args
        ).to(device)
        # The author implementation standardizes event gaps and chooses the
        # t-batch span from the complete stream.  This is a JODIE preprocessing
        # detail, not a learned use of validation/test labels.
        jodie_model.fit_stream_statistics(
            graph.snapshots, JODIELinkBaseline._unique_snapshots(split.train)
        )
        jodie_validation, jodie_test = _train_one(
            "jodie", jodie_model, split, jodie_training, seed
        )
        result["jodie"] = {"validation": jodie_validation, "test": jodie_test}
        del jodie_model
        release_device_memory(device)

    if should_run("tgat"):
        torch.manual_seed(seed)
        tgat_model = TGATLinkBaseline(
            feature_dim=graph.feature_dim, **tgat_args
        ).to(device)
        tgat_model.prepare_streams(graph.snapshots, train_snapshots)
        tgat_validation, tgat_test = _train_one(
            "tgat", tgat_model, split, tgat_training, seed
        )
        result["tgat"] = {"validation": tgat_validation, "test": tgat_test}
        del tgat_model
        release_device_memory(device)

    # These backbones are the author-maintained DyGLib implementations.  The
    # adapter changes only ids/data splits/query candidates/metrics, while each
    # model keeps its native sampler, architecture and one-negative BCE loss.
    enabled_additional = list(
        config.get("additional_baselines", {}).get(
            "enabled", ["edgebank", "dyrep", "tgn", "cawn", "tcl", "graphmixer", "dygformer"]
        )
    )
    supported_additional = {
        "edgebank", "dyrep", "tgn", "cawn", "tcl", "graphmixer", "dygformer"
    }
    unknown = set(enabled_additional) - supported_additional
    if unknown:
        raise ValueError(f"unknown additional baselines: {sorted(unknown)}")
    enabled_additional = [name for name in enabled_additional if should_run(name)]

    if "edgebank" in enabled_additional:
        edge_bank = EdgeBankLinkBaseline(
            num_nodes=graph.num_nodes,
            **link_cfg,
        ).to(device)
        edge_bank.eval()
        edge_validation = edge_bank.evaluate_protocol(
            split.validation, split.train, query_seed=seed + 1_000_000
        )
        edge_test = edge_bank.evaluate_protocol(
            split.test,
            [*split.train, *split.validation],
            query_seed=seed + 2_000_000,
        )
        edge_test["best_epoch"] = 0.0
        result["edgebank"] = {
            "validation": edge_validation,
            "test": edge_test,
        }
        del edge_bank
        release_device_memory(device)

    for baseline_name in ["dyrep", "tgn", "cawn", "tcl", "graphmixer", "dygformer"]:
        if baseline_name not in enabled_additional:
            continue
        torch.manual_seed(seed)
        baseline_model = DyGLibLinkBaseline(
            model_name=baseline_name,
            feature_dim=graph.feature_dim,
            num_nodes=graph.num_nodes,
            **dict(config.get(baseline_name, {})),
            **link_cfg,
        ).to(device)
        baseline_model.prepare_streams(graph.snapshots, train_snapshots)
        baseline_training = {
            **shared_training,
            **dict(config.get(f"{baseline_name}_training", {})),
        }
        baseline_validation, baseline_test = _train_one(
            baseline_name, baseline_model, split, baseline_training, seed
        )
        result[baseline_name] = {
            "validation": baseline_validation,
            "test": baseline_test,
        }
        del baseline_model
        release_device_memory(device)

    # Discrete-time self-supervised baselines use only the chronological
    # training prefix for representation pretraining.  Their encoders are then
    # frozen and evaluated through the same pair MLP and candidate queries.
    snapshot_ssl_cfg = dict(config.get("snapshot_ssl_baselines", {}))
    enabled_snapshot_ssl = [
        str(name).lower() for name in snapshot_ssl_cfg.get("enabled", ())
    ]
    supported_snapshot_ssl = {"cldg", "maskdgnn", "dvgmae"}
    unknown_snapshot_ssl = set(enabled_snapshot_ssl) - supported_snapshot_ssl
    if unknown_snapshot_ssl:
        raise ValueError(
            f"unknown snapshot SSL baselines: {sorted(unknown_snapshot_ssl)}"
        )
    snapshot_ssl_classes = {
        "cldg": CLDGLinkBaseline,
        "maskdgnn": MaskDGNNLinkBaseline,
        "dvgmae": DVGMAELinkBaseline,
    }
    for baseline_name in ("cldg", "maskdgnn", "dvgmae"):
        if baseline_name not in enabled_snapshot_ssl or not should_run(baseline_name):
            continue
        torch.manual_seed(seed)
        baseline_model = snapshot_ssl_classes[baseline_name](
            feature_dim=graph.feature_dim,
            **dict(config.get(baseline_name, {})),
            **link_cfg,
        ).to(device)
        baseline_training = {
            **dict(config.get("snapshot_ssl_training", {})),
            **dict(config.get(f"{baseline_name}_training", {})),
        }
        baseline_validation, baseline_test = _train_snapshot_ssl_one(
            baseline_name, baseline_model, split, baseline_training, seed
        )
        result[baseline_name] = {
            "validation": baseline_validation,
            "test": baseline_test,
        }
        del baseline_model
        release_device_memory(device)

    if should_run("rcps_jepa"):
        torch.manual_seed(seed)
        rcps_model = RCPSJEPA(feature_dim=graph.feature_dim, **rcps_args).to(device)
        rcps_model.prepare_causal_history(graph.snapshots)
        rcps_validation, rcps_test = _train_one(
            "rcps_jepa", rcps_model, split, rcps_training, seed
        )
        result["rcps_jepa"] = {"validation": rcps_validation, "test": rcps_test}
        del rcps_model
        release_device_memory(device)

    output_path = config.get("output_path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"comparison": result}, indent=2))
    return result


def _deep_update(target: dict, updates: dict) -> dict:
    """Recursively apply dataset-specific overrides to a copied config."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def _seed_values(config: dict) -> list[int]:
    """Normalize a scalar or list-valued seed configuration."""
    raw = config.get("seed", 42)
    values = raw if isinstance(raw, list) else [raw]
    if not values:
        raise ValueError("seed list must not be empty")
    seeds: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("seeds must be integers")
        seed = int(value)
        if seed in seeds:
            raise ValueError(f"duplicate seed: {seed}")
        seeds.append(seed)
    return seeds


def _aggregate_seed_runs(runs: dict[str, dict]) -> dict:
    """Compute population mean/std for every numeric result metric."""
    if not runs:
        raise ValueError("cannot aggregate an empty set of seed runs")
    run_results = list(runs.values())
    model_names = set(run_results[0])
    if any(set(result) != model_names for result in run_results[1:]):
        raise ValueError("all seed runs must contain the same models")
    aggregate: dict[str, dict] = {}
    for model_name in sorted(model_names):
        aggregate[model_name] = {}
        split_names = set(run_results[0][model_name])
        for split_name in sorted(split_names):
            metric_names = set(run_results[0][model_name][split_name])
            if any(
                set(result[model_name][split_name]) != metric_names
                for result in run_results[1:]
            ):
                raise ValueError("all seed runs must contain the same metrics")
            aggregate[model_name][split_name] = {}
            for metric_name in sorted(metric_names):
                values = np.asarray(
                    [
                        float(result[model_name][split_name][metric_name])
                        for result in run_results
                    ],
                    dtype=np.float64,
                )
                aggregate[model_name][split_name][metric_name] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
    return aggregate


def _seed_output_path(path: Path, seed: int) -> Path:
    return path.with_name(f"{path.stem}_seed{seed}{path.suffix}")


def _dataset_configs(config: dict) -> list[tuple[str, dict]]:
    """Expand one multi-dataset YAML into ordinary single-dataset configs."""
    entries = config.get("datasets")
    if not entries:
        return [(str(config.get("dataset_name", "single")), config)]
    if not isinstance(entries, list) or not entries:
        raise ValueError("datasets must be a non-empty list")

    base = deepcopy(config)
    base.pop("datasets", None)
    summary_output_path = base.pop("summary_output_path", None)
    output_dir = Path(base.pop("output_dir", "results"))
    expanded: list[tuple[str, dict]] = []
    seen: set[str] = set()
    # JODIE consumes the original event width. TGAT and all DyGLib backbones,
    # including DyRep, retain the repository's shared 172-D width and zero-pad
    # lower-dimensional event features in their adapters.
    feature_consumers = ("jodie",)
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("each datasets entry must be a mapping")
        entry = deepcopy(raw_entry)
        name = str(entry.pop("name")).lower()
        if name in seen:
            raise ValueError(f"duplicate dataset entry: {name}")
        seen.add(name)
        path = entry.pop("path")
        interaction_dim = int(entry.pop("interaction_feature_dim"))
        state_change = bool(entry.pop("state_change", False))
        dataset_models = entry.pop("models", None)
        overrides = entry.pop("overrides", {})
        if entry:
            raise ValueError(
                f"unknown fields for dataset {name}: {sorted(entry)}"
            )

        current = deepcopy(base)
        current["dataset_name"] = name
        current["data"] = {**dict(current.get("data", {})), "path": str(path)}
        current["output_path"] = str(output_dir / f"link_comparison_{name}.json")
        if dataset_models is not None:
            allowed = [str(model).lower() for model in dataset_models]
            globally_requested = current.get("models")
            if globally_requested is None:
                current["models"] = allowed
            else:
                requested = {str(model).lower() for model in globally_requested}
                current["models"] = [model for model in allowed if model in requested]
                if not current["models"]:
                    raise ValueError(
                        f"no requested models support dataset {name}"
                    )
        for section in feature_consumers:
            current.setdefault(section, {})["interaction_feature_dim"] = interaction_dim
        current.setdefault("jodie", {})["state_change"] = state_change
        _deep_update(current, dict(overrides))
        expanded.append((name, current))

    # Retain this only on the parent config; it is consumed by main after all
    # independent model/data runs finish.
    if summary_output_path is not None:
        config["summary_output_path"] = summary_output_path
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare temporal link predictors and RCPS-JEPA under one protocol"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/link_comparison_wikipedia.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="override comparison epochs")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="override the configured seed list",
    )
    parser.add_argument(
        "--max-positive-pairs",
        type=int,
        default=None,
        help="override validation/test positives per target window",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="run only these methods, e.g. rcps_jepa",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="run only these entries from a multi-dataset config",
    )
    parser.add_argument("--output", type=Path, default=None, help="override output JSON path")
    args = parser.parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    if args.seeds is not None:
        config["seed"] = args.seeds[0] if len(args.seeds) == 1 else args.seeds
    if args.datasets:
        if not config.get("datasets"):
            raise ValueError("--datasets requires a multi-dataset config")
        requested_datasets = {name.lower() for name in args.datasets}
        available_datasets = {
            str(entry["name"]).lower() for entry in config["datasets"]
        }
        unknown_datasets = requested_datasets - available_datasets
        if unknown_datasets:
            raise ValueError(
                f"unknown datasets in config: {sorted(unknown_datasets)}"
            )
        config["datasets"] = [
            entry
            for entry in config["datasets"]
            if str(entry["name"]).lower() in requested_datasets
        ]
    if args.models:
        config["models"] = args.models
    if args.max_positive_pairs is not None:
        if args.max_positive_pairs < 1:
            raise ValueError("--max-positive-pairs must be positive")
        config.setdefault("link", {})["max_positive_pairs"] = (
            args.max_positive_pairs
        )
    if args.output is not None:
        if config.get("datasets"):
            config["output_dir"] = str(args.output)
            config["summary_output_path"] = str(
                args.output / "link_comparison_all.json"
            )
        else:
            config["output_path"] = str(args.output)
    if args.epochs is not None:
        for model_name in [
            "jodie",
            "dyrep",
            "tgat",
            "tgn",
            "cawn",
            "tcl",
            "graphmixer",
            "dygformer",
            "rcps",
        ]:
            config.setdefault(f"{model_name}_training", {})["epochs"] = args.epochs
        for model_name in ("cldg", "maskdgnn", "dvgmae"):
            settings = config.setdefault(f"{model_name}_training", {})
            settings["pretrain_epochs"] = args.epochs
            settings["probe_epochs"] = args.epochs
    seeds = _seed_values(config)
    dataset_results = {}
    for dataset_name, dataset_config in _dataset_configs(config):
        base_output = Path(dataset_config["output_path"])
        if len(seeds) == 1:
            dataset_config["seed"] = seeds[0]
            print(
                json.dumps(
                    {
                        "dataset": dataset_name,
                        "seed": seeds[0],
                        "data": dataset_config["data"]["path"],
                        "output": str(base_output),
                    }
                )
            )
            dataset_results[dataset_name] = run(dataset_config)
            continue

        runs: dict[str, dict] = {}
        for seed in seeds:
            seeded_config = deepcopy(dataset_config)
            seeded_config["seed"] = seed
            seeded_output = _seed_output_path(base_output, seed)
            seeded_config["output_path"] = str(seeded_output)
            print(
                json.dumps(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "data": seeded_config["data"]["path"],
                        "output": str(seeded_output),
                    }
                )
            )
            runs[str(seed)] = run(seeded_config)
        dataset_summary = {
            "seeds": seeds,
            "runs": runs,
            "aggregate": _aggregate_seed_runs(runs),
        }
        base_output.parent.mkdir(parents=True, exist_ok=True)
        base_output.write_text(json.dumps(dataset_summary, indent=2))
        dataset_results[dataset_name] = dataset_summary
    if config.get("datasets") and config.get("summary_output_path"):
        summary_path = Path(config["summary_output_path"])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(dataset_results, indent=2))


if __name__ == "__main__":
    main()
