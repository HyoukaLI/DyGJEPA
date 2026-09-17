"""Noise-robustness evaluation of temporal link predictors (DyG-Mamba §5.5).

Protocol
--------
Every model is trained and its checkpoint selected on the *clean* data with
the shared random-negative protocol of ``compare_link_prediction``.  The
selected checkpoint is then re-evaluated on the test split while a fraction
``rate`` of random noisy events is inserted into the history it can see:

* noise is generated per snapshot (bin) so the equal-event binning and the
  train/validation/test split never move: a bin with ``E`` real events
  receives ``round(rate * E)`` extra events between uniformly drawn existing
  nodes (bipartite datasets keep the source -> destination direction),
  with timestamps uniform inside the bin's time range and edge features
  either resampled from the bin's real events or zero;
* noisy events enter everything a model treats as history - the structural
  ``edge_index`` and node activity of the context snapshots, the duplicate-
  preserving event lists behind DyGJEPA's causal history, and the neighbor
  samplers / event streams of the continuous-time baselines;
* the *positives* of the test split stay the original events, so every model
  is scored on the same clean queries with the same negatives.

Results are written to ``results/robustness/link_robustness_<dataset>.json``
(plus a CSV and, when matplotlib is present, an AP-vs-noise figure in the
style of DyG-Mamba Fig. 5).  Launch with

    python -m jepa_compare.compare_link_robustness \\
        --config configs/link_comparison_all_robustness.yaml --datasets wikipedia

or ``bash scripts/run_link_robustness.sh wikipedia``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .compare_link_prediction import (
    _build_graph,
    _negative_destination_pool,
    _rcps_ablation,
    _requested_models,
    _train_one,
    _train_snapshot_ssl_one,
    _dataset_configs,
    load_config,
)
from .data import Snapshot
from .dyglib_baselines import (
    DyGLibLinkBaseline,
    _concatenate,
    _neighbor_sampler,
    _snapshot_stream,
)
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
from .train_sg_jepa import choose_device, device_description, release_device_memory
from . import wandb_logging

DYGLIB_MODELS = ("jodie", "dyrep", "tgn", "cawn", "tcl", "graphmixer", "dygformer")
SNAPSHOT_SSL_MODELS = {
    "cldg": CLDGLinkBaseline,
    "maskdgnn": MaskDGNNLinkBaseline,
    "dvgmae": DVGMAELinkBaseline,
}
SUPPORTED_MODELS = (*DYGLIB_MODELS, "tgat", *SNAPSHOT_SSL_MODELS, "rcps_jepa")
DEFAULT_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
NOISE_FEATURE_MODES = ("resample", "zeros")


# --------------------------------------------------------------------------
# Noise injection
# --------------------------------------------------------------------------
def inject_noise(
    snapshots: Sequence[Snapshot],
    rate: float,
    *,
    num_nodes: int,
    num_source_nodes: int | None,
    seed: int,
    feature_mode: str = "resample",
) -> tuple[list[Snapshot], dict[int, torch.Tensor]]:
    """Return noisy copies of ``snapshots`` and, per snapshot time, a boolean
    mask over the noisy snapshot's (timestamp-sorted) events flagging noise.

    ``rate = 0`` returns the original snapshot objects and all-False masks.
    """
    if rate < 0:
        raise ValueError("noise rate must be non-negative")
    if feature_mode not in NOISE_FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {NOISE_FEATURE_MODES}")
    noisy: list[Snapshot] = []
    masks: dict[int, torch.Tensor] = {}
    for snapshot in snapshots:
        events = snapshot.query_edge_index
        if events is None:
            raise ValueError("noise injection requires query_edge_index events")
        device = events.device
        count = int(events.shape[1])
        extra = int(round(rate * count)) if rate > 0 else 0
        if extra == 0:
            noisy.append(snapshot)
            masks[int(snapshot.time)] = torch.zeros(count, dtype=torch.bool, device=device)
            continue
        generator = torch.Generator().manual_seed(
            int(seed) * 1_000_003 + int(round(rate * 1000)) * 1_009 + int(snapshot.time)
        )
        if num_source_nodes is not None:
            sources = torch.randint(0, num_source_nodes, (extra,), generator=generator)
            destinations = torch.randint(
                num_source_nodes, num_nodes, (extra,), generator=generator
            )
        else:
            sources = torch.randint(0, num_nodes, (extra,), generator=generator)
            destinations = torch.randint(0, num_nodes, (extra,), generator=generator)
            same = destinations == sources
            while bool(same.any()):
                destinations[same] = torch.randint(
                    0, num_nodes, (int(same.sum()),), generator=generator
                )
                same = destinations == sources
        sources = sources.to(device=device, dtype=events.dtype)
        destinations = destinations.to(device=device, dtype=events.dtype)
        noise_events = torch.stack([sources, destinations])

        timestamps = snapshot.query_timestamps
        noise_timestamps = None
        if timestamps is not None:
            low, high = float(timestamps.min()), float(timestamps.max())
            uniform = torch.rand(extra, generator=generator).to(
                device=device, dtype=timestamps.dtype
            )
            noise_timestamps = low + (high - low) * uniform

        features = snapshot.query_features
        noise_features = None
        if features is not None:
            if feature_mode == "resample":
                rows = torch.randint(0, count, (extra,), generator=generator).to(device)
                noise_features = features[rows]
            else:
                noise_features = torch.zeros(
                    extra, features.shape[1], dtype=features.dtype, device=device
                )
        labels = snapshot.query_labels
        noise_labels = (
            None if labels is None else torch.zeros(extra, dtype=labels.dtype, device=device)
        )

        all_events = torch.cat([events, noise_events], dim=1)
        flags = torch.cat(
            [
                torch.zeros(count, dtype=torch.bool, device=device),
                torch.ones(extra, dtype=torch.bool, device=device),
            ]
        )
        if timestamps is not None:
            all_timestamps = torch.cat([timestamps, noise_timestamps])
            order = torch.argsort(all_timestamps, stable=True)
        else:
            all_timestamps = None
            order = torch.arange(count + extra, device=device)
        all_events = all_events[:, order]
        flags = flags[order]
        all_timestamps = None if all_timestamps is None else all_timestamps[order]
        all_features = (
            None if features is None else torch.cat([features, noise_features])[order]
        )
        all_labels = None if labels is None else torch.cat([labels, noise_labels])[order]

        undirected = torch.cat([noise_events, noise_events.flip(0)], dim=1)
        edge_index = torch.unique(
            torch.cat([snapshot.edge_index, undirected.to(snapshot.edge_index.dtype)], dim=1),
            dim=1,
        )
        active = snapshot.active.clone()
        active[sources.long()] = True
        active[destinations.long()] = True
        noisy.append(
            Snapshot(
                x=snapshot.x,
                edge_index=edge_index,
                active=active,
                time=snapshot.time,
                query_edge_index=all_events,
                query_timestamps=all_timestamps,
                query_features=all_features,
                query_labels=all_labels,
            )
        )
        masks[int(snapshot.time)] = flags
    return noisy, masks


def noisy_windows(
    windows: Sequence[Sequence[Snapshot]], noisy_by_time: dict[int, Snapshot]
) -> list[list[Snapshot]]:
    """Context snapshots from the noisy stream, the clean target for positives."""
    return [
        [noisy_by_time[int(snapshot.time)] for snapshot in window[:-1]] + [window[-1]]
        for window in windows
    ]


# --------------------------------------------------------------------------
# Model construction and training (mirrors compare_link_prediction.run)
# --------------------------------------------------------------------------
def _link_cfg(config: dict, graph) -> dict:
    link_cfg = dict(config.get("link", {}))
    strategy = link_cfg.pop("negative_strategy", "random")
    if strategy != "random":
        raise ValueError("the robustness run uses the random-negative protocol only")
    link_cfg["negative_destination_candidates"] = _negative_destination_pool(graph)
    link_cfg["bipartite_source_count"] = graph.num_source_nodes
    return link_cfg


def build_and_train(
    name: str,
    config: dict,
    graph,
    split: TemporalWindowSplit,
    link_cfg: dict,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, float], dict[str, float]]:
    """Construct, train and checkpoint-select one model exactly as the main run."""
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model {name!r}; choose from {SUPPORTED_MODELS}")
    common = dict(config["common_model"])
    train_snapshots = unique_snapshots(split.train)
    torch.manual_seed(seed)
    # Argument dictionaries are merged exactly as compare_link_prediction.run
    # does (later sections override earlier ones).
    if name in DYGLIB_MODELS:
        args = {
            "model_name": name,
            "feature_dim": graph.feature_dim,
            "num_nodes": graph.num_nodes,
            **dict(config.get(name, {})),
            **link_cfg,
        }
        model = DyGLibLinkBaseline(**args).to(device)
        model.prepare_streams(graph.snapshots, train_snapshots)
        validation, test = _train_one(
            name, model, split, _training_section(name, config), seed
        )
    elif name == "tgat":
        args = {
            "feature_dim": graph.feature_dim,
            "num_nodes": graph.num_nodes,
            "bipartite_source_count": graph.num_source_nodes,
            **dict(config.get("tgat", {})),
            **link_cfg,
        }
        model = TGATLinkBaseline(**args).to(device)
        model.prepare_streams(graph.snapshots, train_snapshots)
        validation, test = _train_one(
            name, model, split, _training_section(name, config), seed
        )
    elif name in SNAPSHOT_SSL_MODELS:
        args = {
            "feature_dim": graph.feature_dim,
            **dict(config.get(name, {})),
            **link_cfg,
        }
        model = SNAPSHOT_SSL_MODELS[name](**args).to(device)
        validation, test = _train_snapshot_ssl_one(
            name, model, split, _training_section(name, config), seed
        )
    else:
        ablation = _rcps_ablation(config)
        args = {
            "feature_dim": graph.feature_dim,
            "num_nodes": graph.num_nodes,
            **common,
            **link_cfg,
            **dict(config.get("rcps_jepa", {})),
            **dict(ablation.get("rcps_jepa", {})),
        }
        model = RCPSJEPA(**args).to(device)
        model.prepare_causal_history(graph.snapshots)
        validation, test = _train_one(
            name, model, split, _training_section(name, config), seed
        )
    model.eval()
    return model, validation, test


# --------------------------------------------------------------------------
# Evaluation under noise
# --------------------------------------------------------------------------
def _attach_noisy_streams(
    model: DyGLibLinkBaseline,
    noisy_snapshots: Sequence[Snapshot],
    noise_masks: dict[int, torch.Tensor],
    clean_target_times: set[int],
) -> None:
    """Point a trained DyGLib adapter at the noisy event history.

    ``prepare_streams`` rebuilds the backbone, so the trained model's sampler,
    per-snapshot streams and edge-feature table are replaced in place instead.
    Streams of the test targets are restricted to the real events (the
    positives), while the neighbor sampler and the memory replay of earlier
    snapshots see the noise.  The destination pool of the random negatives is
    left untouched so every rate scores against the same negatives.
    """
    backbone, _ = model._require_prepared()
    streams = []
    next_edge_id = 1
    for snapshot in sorted(noisy_snapshots, key=lambda item: item.time):
        stream = _snapshot_stream(
            snapshot,
            num_users=model.num_users,
            num_nodes=model.num_nodes,
            feature_dim=model.dimension,
            first_edge_id=next_edge_id,
        )
        time = int(snapshot.time)
        if time in clean_target_times:
            keep = ~noise_masks[time].detach().cpu().numpy()
            if keep.shape[0] != len(stream):
                raise RuntimeError(
                    f"noise mask of snapshot {time} does not align with its event stream"
                )
            model._snapshot_streams[time] = stream.take(keep)
        else:
            model._snapshot_streams[time] = stream
        streams.append(stream)
        next_edge_id += len(stream)
    full_stream = _concatenate(streams, model.dimension)
    model._full_sampler = _neighbor_sampler(
        full_stream,
        model.num_nodes,
        model.sample_neighbor_strategy,
        model.sampler_seed,
        model.time_scaling_factor,
    )
    edge_features = torch.from_numpy(
        np.concatenate(
            [np.zeros((1, model.dimension), dtype=np.float32), full_stream.features],
            axis=0,
        )
    ).to(model._device_anchor.device)
    # Upstream DyGLib modules keep their own reference to the feature table
    # (e.g. the memory model's embedding module), so update every holder.
    for module in backbone.modules():
        if hasattr(module, "edge_raw_features"):
            module.edge_raw_features = edge_features


@torch.no_grad()
def evaluate_under_noise(
    name: str,
    model: nn.Module,
    graph,
    split: TemporalWindowSplit,
    rate: float,
    *,
    noise_seed: int,
    feature_mode: str,
    training: dict,
) -> dict[str, float]:
    """Score the trained ``model`` on the clean test positives with ``rate``
    noisy events inserted into its history."""
    noisy_snapshots, masks = inject_noise(
        graph.snapshots,
        rate,
        num_nodes=graph.num_nodes,
        num_source_nodes=graph.num_source_nodes,
        seed=noise_seed,
        feature_mode=feature_mode,
    )
    noisy_by_time = {int(snapshot.time): snapshot for snapshot in noisy_snapshots}
    test_query_seed = int(training.get("test_query_seed", 2))
    pair_batch_size = training.get("pair_batch_size")
    model.eval()
    if isinstance(model, DyGLibLinkBaseline):
        target_times = {int(s.time) for s in unique_snapshots(split.test, targets_only=True)}
        _attach_noisy_streams(model, noisy_snapshots, masks, target_times)
        return model.evaluate_protocol(
            split.test, [*split.train, *split.validation], query_seed=test_query_seed
        )
    if isinstance(model, TGATLinkBaseline):
        noisy_train = [noisy_by_time[int(s.time)] for s in unique_snapshots(split.train)]
        model.prepare_streams(noisy_snapshots, noisy_train)
        return model.evaluate_protocol(
            split.test, [*split.train, *split.validation], query_seed=test_query_seed
        )
    windows = noisy_windows(split.test, noisy_by_time)
    if isinstance(model, RCPSJEPA):
        model.prepare_causal_history(noisy_snapshots)
        return model.evaluate_windows(
            windows, pair_batch_size=pair_batch_size, query_seed=test_query_seed
        )
    if isinstance(model, SnapshotSSLLinkBaseline):
        return model.evaluate_windows(
            windows,
            pair_batch_size=int(training.get("pair_batch_size", 512)),
            query_seed=test_query_seed,
        )
    raise TypeError(f"no noisy-evaluation path for {type(model).__name__}")


def _training_section(name: str, config: dict) -> dict:
    shared = dict(config.get("training", {}))
    if name in SNAPSHOT_SSL_MODELS:
        return {
            **dict(config.get("snapshot_ssl_training", {})),
            **dict(config.get(f"{name}_training", {})),
        }
    if name == "rcps_jepa":
        ablation = _rcps_ablation(config)
        return {
            **shared,
            **dict(config.get("rcps_training", {})),
            **dict(ablation.get("rcps_training", {})),
        }
    return {**shared, **dict(config.get(f"{name}_training", {}))}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(config: dict) -> dict:
    seed = int(config["seed"])
    dataset = str(config.get("dataset_name", "single"))
    robustness = dict(config.get("robustness", {}))
    rates = [float(r) for r in robustness.get("noise_rates", DEFAULT_RATES)]
    noise_seed = int(robustness.get("noise_seed", 0))
    feature_mode = str(robustness.get("noise_features", "resample"))
    if feature_mode not in NOISE_FEATURE_MODES:
        raise ValueError(f"robustness.noise_features must be one of {NOISE_FEATURE_MODES}")
    wandb_logging.configure(config, task="link", dataset=dataset)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(config.get("device", "auto"))
    print(f"runtime_device={device_description(device)}", flush=True)
    graph = _build_graph(config, seed).to(device)
    common = dict(config["common_model"])
    split_cfg = config.get("split", {})
    split = temporal_window_split(
        graph.snapshots,
        int(common["window_size"]),
        float(split_cfg.get("train_ratio", 0.6)),
        float(split_cfg.get("validation_ratio", 0.2)),
    )
    link_cfg = _link_cfg(config, graph)
    requested = _requested_models(config)
    models = [name for name in SUPPORTED_MODELS if requested is None or name in requested]
    if requested is not None:
        unknown = sorted(set(requested) - set(SUPPORTED_MODELS))
        if unknown:
            raise ValueError(f"models without a noisy-evaluation path: {unknown}")

    result = {
        "dataset": dataset,
        "seed": seed,
        "noise_rates": rates,
        "noise_seed": noise_seed,
        "noise_features": feature_mode,
        "models": {},
    }
    for name in models:
        model, validation, clean_test = build_and_train(
            name, config, graph, split, link_cfg, seed, device
        )
        training = _training_section(name, config)
        by_rate: dict[str, dict[str, float]] = {}
        for rate in rates:
            metrics = evaluate_under_noise(
                name,
                model,
                graph,
                split,
                rate,
                noise_seed=noise_seed,
                feature_mode=feature_mode,
                training=training,
            )
            by_rate[f"{rate:.2f}"] = {k: float(v) for k, v in metrics.items()}
            print(
                json.dumps(
                    {
                        "model": name,
                        "noise_rate": rate,
                        "test_ap": metrics["ap"],
                        "test_auc": metrics["auc"],
                        "clean_test_ap": clean_test["ap"],
                    }
                ),
                flush=True,
            )
        if "0.00" in by_rate:
            drift = abs(by_rate["0.00"]["ap"] - float(clean_test["ap"]))
            if drift > 1e-6:
                print(
                    json.dumps(
                        {
                            "model": name,
                            "warning": "rate-0 evaluation differs from the training test pass",
                            "difference": drift,
                        }
                    ),
                    flush=True,
                )
        result["models"][name] = {
            "validation": validation,
            "clean_test": clean_test,
            "by_rate": by_rate,
        }
        del model
        release_device_memory(device)

    output_path = config.get("output_path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        write_csv(result, path.with_suffix(".csv"))
        try:
            plot(result, path.parent / "figures" / f"{dataset}_noise_ap")
        except Exception as error:  # pragma: no cover - plotting is optional
            print(f"figure skipped: {error}")
    return result


def write_csv(result: dict, path: Path) -> None:
    rows = []
    for name, entry in result["models"].items():
        clean_ap = float(entry["clean_test"]["ap"])
        for rate, metrics in entry["by_rate"].items():
            rows.append(
                {
                    "dataset": result["dataset"],
                    "model": name,
                    "noise_rate": float(rate),
                    "ap": metrics["ap"],
                    "auc": metrics["auc"],
                    "ap_drop_pct": 100.0 * (metrics["ap"] - clean_ap) / clean_ap,
                }
            )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)


LABELS = {
    "tgn": "TGN",
    "tgat": "TGAT",
    "dvgmae": "DVGMAE",
    "rcps_jepa": "DyGJEPA",
    "dygformer": "DyGFormer",
    "graphmixer": "GraphMixer",
    "cawn": "CAWN",
    "tcl": "TCL",
    "jodie": "JODIE",
    "dyrep": "DyRep",
    "cldg": "CLDG",
    "maskdgnn": "MaskDGNN",
}
# Fixed categorical order; ours first.  Same palette as the efficiency figures.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948"]
MARKERS = ["s", "o", "x", "^", "D", "v", "P", "*"]


def plot(result: dict, stem: Path) -> list[Path]:
    """AP versus noise rate, one line per model, drop at the last rate annotated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stem.parent.mkdir(parents=True, exist_ok=True)
    names = ["rcps_jepa"] + [n for n in result["models"] if n != "rcps_jepa"]
    names = [n for n in names if n in result["models"]]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for index, name in enumerate(names):
        entry = result["models"][name]
        rates = sorted(float(r) for r in entry["by_rate"])
        aps = [entry["by_rate"][f"{r:.2f}"]["ap"] for r in rates]
        color = COLORS[index % len(COLORS)]
        ax.plot(
            rates,
            aps,
            marker=MARKERS[index % len(MARKERS)],
            markersize=4,
            linewidth=1.4,
            color=color,
            label=LABELS.get(name, name),
        )
        drop = 100.0 * (aps[-1] - aps[0]) / max(aps[0], 1e-12)
        ax.annotate(
            f"{drop:+.2f}%",
            (rates[-1], aps[-1]),
            xytext=(-2, -9 if drop < 0 else 4),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color=color,
        )
    ax.set_xlabel(f"noise rate on {result['dataset']}")
    ax.set_ylabel("Average Precision")
    ax.set_xticks(sorted({float(r) for e in result["models"].values() for r in e["by_rate"]}))
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=7, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    written = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=200)
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Noise-robustness evaluation of temporal link predictors"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/link_comparison_all_robustness.yaml")
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None, help="one model seed (default: first of config)")
    parser.add_argument("--epochs", type=int, default=None, help="override training epochs (smoke tests)")
    parser.add_argument("--rates", nargs="+", type=float, default=None, help="noise rates, e.g. 0 0.2 0.4 0.6")
    parser.add_argument("--output", type=Path, default=None, help="override results directory")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.datasets:
        wanted = {name.lower() for name in args.datasets}
        entries = [e for e in config.get("datasets", []) if str(e["name"]).lower() in wanted]
        missing = wanted - {str(e["name"]).lower() for e in entries}
        if missing:
            raise ValueError(f"unknown datasets {sorted(missing)}")
        config["datasets"] = entries
    elif config.get("datasets"):
        raise SystemExit(
            "name the datasets explicitly (--datasets wikipedia ...); the overlay "
            "inherits every dataset entry of link_comparison_all.yaml"
        )
    if args.models:
        config["models"] = [name.lower() for name in args.models]
    if args.rates:
        config.setdefault("robustness", {})["noise_rates"] = [float(r) for r in args.rates]
    if args.epochs is not None:
        for key, settings in list(config.items()):
            if key.endswith("_training") and isinstance(settings, dict):
                if "epochs" in settings:
                    settings["epochs"] = args.epochs
                if "pretrain_epochs" in settings:
                    settings["pretrain_epochs"] = args.epochs
                if "probe_epochs" in settings:
                    settings["probe_epochs"] = args.epochs
        if "epochs" in config.get("training", {}):
            config["training"]["epochs"] = args.epochs
    if args.output is not None:
        config["output_dir"] = str(args.output)
        config["summary_output_path"] = str(args.output / "link_robustness_all.json")
    raw_seed = config.get("seed", 42)
    seed = args.seed if args.seed is not None else (raw_seed[0] if isinstance(raw_seed, list) else raw_seed)

    summaries = {}
    for dataset_name, dataset_config in _dataset_configs(config):
        dataset_config["seed"] = int(seed)
        # link_comparison_<ds>.json -> link_robustness_<ds>.json in output_dir.
        output = Path(dataset_config["output_path"])
        dataset_config["output_path"] = str(
            output.with_name(output.name.replace("link_comparison_", "link_robustness_"))
        )
        print(json.dumps({"dataset": dataset_name, "seed": int(seed), "output": dataset_config["output_path"]}))
        summaries[dataset_name] = run(dataset_config)
    summary_path = config.get("summary_output_path")
    if summary_path and config.get("datasets"):
        path = Path(summary_path)
        path = path.with_name(path.name.replace("link_comparison_all", "link_robustness_all"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
